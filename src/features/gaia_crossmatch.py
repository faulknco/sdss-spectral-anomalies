"""Gaia DR3 cross-match for astrometric anomaly detection.

Stars with PBH companions should exhibit astrometric signatures: excess noise
in the Gaia astrometric solution (from unmodeled orbital motion) and/or proper
motion anomalies.  This module cross-matches SDSS spectroscopic candidates with
Gaia DR3 to retrieve astrometric quality indicators.

CAVEAT: Astrometric excess noise has many mundane causes (binaries, extended
sources, crowded fields, scan-angle coverage).  High RUWE or excess noise
does NOT confirm a PBH companion.  These are additional screening criteria
for candidates warranting follow-up.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default cross-match radius in arcseconds
_MATCH_RADIUS_ARCSEC = 1.5


def _build_gaia_cone_query(
    ra: float,
    dec: float,
    radius_arcsec: float,
) -> str:
    """Build an ADQL cone-search query for a single position."""
    radius_deg = radius_arcsec / 3600.0
    return f"""
    SELECT TOP 1
        source_id, ra, dec, parallax, parallax_error,
        pmra, pmdec, phot_g_mean_mag,
        astrometric_excess_noise, astrometric_excess_noise_sig,
        ruwe, ipd_gof_harmonic_amplitude,
        visibility_periods_used, astrometric_params_solved
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    ORDER BY
        DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec}))
    """


def _build_gaia_batch_query(
    ra_values: np.ndarray,
    dec_values: np.ndarray,
    radius_arcsec: float,
) -> str:
    """Build an ADQL query that retrieves Gaia sources near multiple positions.

    Uses a bounding-box pre-filter for efficiency, then refines per-source.
    """
    radius_deg = radius_arcsec / 3600.0
    ra_min = float(np.nanmin(ra_values)) - radius_deg - 0.01
    ra_max = float(np.nanmax(ra_values)) + radius_deg + 0.01
    dec_min = float(np.nanmin(dec_values)) - radius_deg - 0.01
    dec_max = float(np.nanmax(dec_values)) + radius_deg + 0.01

    return f"""
    SELECT
        source_id, ra, dec, parallax, parallax_error,
        pmra, pmdec, phot_g_mean_mag,
        astrometric_excess_noise, astrometric_excess_noise_sig,
        ruwe, ipd_gof_harmonic_amplitude,
        visibility_periods_used, astrometric_params_solved
    FROM gaiadr3.gaia_source
    WHERE ra BETWEEN {ra_min} AND {ra_max}
      AND dec BETWEEN {dec_min} AND {dec_max}
    """


def query_gaia_for_candidates(
    metadata: pd.DataFrame,
    candidate_indices: np.ndarray | list[int] | None = None,
    match_radius_arcsec: float = _MATCH_RADIUS_ARCSEC,
    timeout: int = 120,
) -> pd.DataFrame:
    """Cross-match SDSS positions with Gaia DR3.

    If ``candidate_indices`` is provided, only those rows are queried
    (recommended to limit to PBH candidates rather than the full sample).

    Returns a DataFrame indexed by SDSS spectrum index with Gaia columns.
    If the Gaia TAP service is unreachable, returns an empty DataFrame
    and logs a warning.
    """
    if candidate_indices is not None:
        meta_subset = metadata.iloc[candidate_indices].copy()
        meta_subset["_sdss_idx"] = candidate_indices
    else:
        meta_subset = metadata.copy()
        meta_subset["_sdss_idx"] = np.arange(len(metadata))

    ra_values = meta_subset["ra"].values.astype(np.float64)
    dec_values = meta_subset["dec"].values.astype(np.float64)
    valid = np.isfinite(ra_values) & np.isfinite(dec_values)
    if valid.sum() == 0:
        logger.warning("No valid RA/Dec coordinates for Gaia cross-match")
        return _empty_result()

    try:
        from astroquery.gaia import Gaia
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

        # Check sky spread to decide strategy
        ra_range = float(np.nanmax(ra_values[valid]) - np.nanmin(ra_values[valid]))
        dec_range = float(np.nanmax(dec_values[valid]) - np.nanmin(dec_values[valid]))
        sky_area = ra_range * dec_range

        results = []

        if sky_area < 100 and valid.sum() > 10:
            # Compact footprint: batch query is efficient
            query = _build_gaia_batch_query(
                ra_values[valid], dec_values[valid], match_radius_arcsec,
            )
            logger.info("Querying Gaia DR3 (batch) for %d candidate positions...", valid.sum())
            job = Gaia.launch_job(query)
            gaia_table = job.get_results()
            gaia_df = gaia_table.to_pandas()

            if len(gaia_df) > 0:
                gaia_ra = gaia_df["ra"].values.astype(np.float64)
                gaia_dec = gaia_df["dec"].values.astype(np.float64)
                match_radius_deg = match_radius_arcsec / 3600.0

                for _, row in meta_subset[valid].iterrows():
                    sdss_idx = int(row["_sdss_idx"])
                    cos_dec = np.cos(np.radians(row["dec"]))
                    dra = (gaia_ra - row["ra"]) * cos_dec
                    ddec = gaia_dec - row["dec"]
                    sep = np.sqrt(dra**2 + ddec**2)
                    best = np.argmin(sep)
                    if sep[best] > match_radius_deg:
                        continue
                    match = gaia_df.iloc[best].to_dict()
                    match["sdss_idx"] = sdss_idx
                    match["separation_arcsec"] = float(sep[best] * 3600.0)
                    results.append(match)
        else:
            # Wide sky spread: use individual cone searches
            logger.info(
                "Querying Gaia DR3 (per-source cones) for %d candidates across %.0f sq deg...",
                valid.sum(), sky_area,
            )
            for _, row in meta_subset[valid].iterrows():
                sdss_idx = int(row["_sdss_idx"])
                query = _build_gaia_cone_query(
                    float(row["ra"]), float(row["dec"]), match_radius_arcsec,
                )
                try:
                    job = Gaia.launch_job(query)
                    table = job.get_results()
                    if len(table) > 0:
                        match = table.to_pandas().iloc[0].to_dict()
                        # Compute separation
                        cos_dec = np.cos(np.radians(row["dec"]))
                        dra = (match["ra"] - row["ra"]) * cos_dec
                        ddec = match["dec"] - row["dec"]
                        sep = np.sqrt(dra**2 + ddec**2)
                        match["sdss_idx"] = sdss_idx
                        match["separation_arcsec"] = float(sep * 3600.0)
                        results.append(match)
                except Exception as cone_err:
                    logger.debug("Cone query failed for idx %d: %s", sdss_idx, cone_err)
                    continue

    except Exception as e:
        logger.warning("Gaia cross-match failed (offline or service error): %s", e)
        return _empty_result()

    if not results:
        return _empty_result()

    result_df = pd.DataFrame(results)
    logger.info("Matched %d / %d candidates to Gaia DR3", len(result_df), valid.sum())
    return result_df


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "source_id", "ra", "dec", "parallax", "parallax_error",
        "pmra", "pmdec", "phot_g_mean_mag",
        "astrometric_excess_noise", "astrometric_excess_noise_sig",
        "ruwe", "ipd_gof_harmonic_amplitude",
        "visibility_periods_used", "astrometric_params_solved",
        "sdss_idx", "separation_arcsec",
    ])


def score_astrometric_anomalies(gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Score Gaia matches for astrometric anomalies suggesting a dark companion.

    Adds columns:
        - ruwe_flag: True if RUWE > 1.4 (standard threshold for poor single-star fit)
        - excess_noise_flag: True if astrometric_excess_noise_sig > 2
        - astrometric_anomaly_score: combined score (higher = more anomalous)
    """
    if gaia_df.empty:
        for col in ["ruwe_flag", "excess_noise_flag", "astrometric_anomaly_score"]:
            gaia_df[col] = pd.Series(dtype=float)
        return gaia_df

    df = gaia_df.copy()

    ruwe = df["ruwe"].values.astype(np.float64)
    excess_noise_sig = df["astrometric_excess_noise_sig"].values.astype(np.float64)
    excess_noise = df["astrometric_excess_noise"].values.astype(np.float64)
    ipd_amp = df["ipd_gof_harmonic_amplitude"].values.astype(np.float64)

    # Standard thresholds from Gaia documentation
    df["ruwe_flag"] = np.where(np.isfinite(ruwe), ruwe > 1.4, False)
    df["excess_noise_flag"] = np.where(
        np.isfinite(excess_noise_sig), excess_noise_sig > 2.0, False
    )

    # Composite score: weighted sum of normalized indicators
    ruwe_component = np.where(np.isfinite(ruwe), np.maximum(ruwe - 1.0, 0.0), 0.0)
    noise_component = np.where(np.isfinite(excess_noise_sig), np.maximum(excess_noise_sig, 0.0), 0.0)
    ipd_component = np.where(np.isfinite(ipd_amp), np.maximum(ipd_amp, 0.0), 0.0)

    df["astrometric_anomaly_score"] = (
        0.4 * ruwe_component
        + 0.4 * noise_component
        + 0.2 * ipd_component
    )

    return df
