"""SDSS photometric cross-match for PBH candidate color analysis.

Retrieves ugriz PSF magnitudes from SDSS PhotoObj for spectroscopic targets
and tests whether candidates have unusual colors:
- Microlensing should be achromatic → normal colors for the spectral type
- Accretion should produce UV/blue excess → anomalous u-g or g-r colors

CAVEAT: Color anomalies have many mundane causes — reddening, metallicity
variations, photometric errors, unresolved companions.  These are screening
criteria, not confirmations.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate main-sequence color ranges by broad spectral class.
# (u-g, g-r) median values from Covey et al. 2007 / SDSS stellar locus.
_EXPECTED_COLORS = {
    "O": {"u_g": 0.0, "g_r": -0.3},
    "B": {"u_g": 0.2, "g_r": -0.2},
    "A": {"u_g": 0.8, "g_r": 0.0},
    "F": {"u_g": 1.0, "g_r": 0.3},
    "G": {"u_g": 1.3, "g_r": 0.5},
    "K": {"u_g": 1.8, "g_r": 0.8},
    "M": {"u_g": 2.5, "g_r": 1.4},
}


def _parse_broad_class(subclass: str) -> str:
    """Extract broad spectral class, ignoring WD-type strings."""
    s = str(subclass).strip()
    # Skip white dwarf classifications entirely
    if s.upper().startswith("WD"):
        return "unknown"
    s = s.upper()
    for char in s:
        if char in "OBAFGKM":
            return char
    return "unknown"


def _build_photometry_query(plate: int, mjd: int, fiberid: int) -> str:
    """Build SDSS SQL to fetch ugriz for a single spectrum via PhotoObj join."""
    return f"""
    SELECT TOP 1
        p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z,
        p.psfMagErr_u, p.psfMagErr_g, p.psfMagErr_r, p.psfMagErr_i, p.psfMagErr_z,
        p.extinction_u, p.extinction_g, p.extinction_r, p.extinction_i, p.extinction_z
    FROM SpecObj AS s
    JOIN PhotoObj AS p ON s.bestobjid = p.objid
    WHERE s.plate = {plate} AND s.mjd = {mjd} AND s.fiberid = {fiberid}
    """


def _build_batch_photometry_query(
    plates: list[int],
    mjds: list[int],
    fiberids: list[int],
) -> str:
    """Build SDSS SQL to fetch ugriz for multiple spectra in one query."""
    conditions = []
    for plate, mjd, fiberid in zip(plates, mjds, fiberids):
        conditions.append(f"(s.plate={plate} AND s.mjd={mjd} AND s.fiberid={fiberid})")

    where_clause = " OR ".join(conditions)
    return f"""
    SELECT
        s.plate, s.mjd, s.fiberid,
        p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z,
        p.psfMagErr_u, p.psfMagErr_g, p.psfMagErr_r, p.psfMagErr_i, p.psfMagErr_z,
        p.extinction_u, p.extinction_g, p.extinction_r, p.extinction_i, p.extinction_z
    FROM SpecObj AS s
    JOIN PhotoObj AS p ON s.bestobjid = p.objid
    WHERE {where_clause}
    """


def query_sdss_photometry(
    metadata: pd.DataFrame,
    candidate_indices: np.ndarray | list[int] | None = None,
    batch_size: int = 50,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch ugriz PSF magnitudes from SDSS for given spectra.

    Queries in batches to stay within SDSS SQL size limits.
    Returns a DataFrame with plate/mjd/fiberid + ugriz columns.
    """
    if candidate_indices is not None:
        subset = metadata.iloc[candidate_indices].copy()
        subset["_sdss_idx"] = candidate_indices
    else:
        subset = metadata.copy()
        subset["_sdss_idx"] = np.arange(len(metadata))

    required = ["plate", "mjd", "fiberid"]
    for col in required:
        if col not in subset.columns:
            logger.warning("Metadata missing %s column for photometric query", col)
            return _empty_photometry()

    try:
        from astroquery.sdss import SDSS
    except ImportError:
        logger.warning("astroquery not available for photometric cross-match")
        return _empty_photometry()

    all_results = []
    rows = subset[required + ["_sdss_idx"]].dropna().reset_index(drop=True)

    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start:start + batch_size]
        plates = batch["plate"].astype(int).tolist()
        mjds = batch["mjd"].astype(int).tolist()
        fiberids = batch["fiberid"].astype(int).tolist()
        sdss_indices = batch["_sdss_idx"].tolist()

        query = _build_batch_photometry_query(plates, mjds, fiberids)
        try:
            prev_timeout = getattr(SDSS, "TIMEOUT", None)
            SDSS.TIMEOUT = timeout
            try:
                result = SDSS.query_sql(query)
            finally:
                if prev_timeout is not None:
                    SDSS.TIMEOUT = prev_timeout

            if result is not None and len(result) > 0:
                batch_df = result.to_pandas()
                # Map back to sdss_idx
                for _, brow in batch_df.iterrows():
                    match_mask = (
                        (batch["plate"].astype(int) == int(brow["plate"]))
                        & (batch["mjd"].astype(int) == int(brow["mjd"]))
                        & (batch["fiberid"].astype(int) == int(brow["fiberid"]))
                    )
                    matched = batch[match_mask]
                    if len(matched) > 0:
                        row_dict = brow.to_dict()
                        row_dict["sdss_idx"] = int(matched.iloc[0]["_sdss_idx"])
                        all_results.append(row_dict)
        except Exception as e:
            logger.warning("Photometry batch query failed: %s", e)
            continue

        logger.info(
            "Photometry batch %d/%d: %d results",
            start // batch_size + 1,
            (len(rows) + batch_size - 1) // batch_size,
            len(all_results),
        )

    if not all_results:
        logger.info("No photometric matches found")
        return _empty_photometry()

    return pd.DataFrame(all_results)


def _empty_photometry() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "plate", "mjd", "fiberid", "sdss_idx",
        "psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z",
        "psfMagErr_u", "psfMagErr_g", "psfMagErr_r", "psfMagErr_i", "psfMagErr_z",
        "extinction_u", "extinction_g", "extinction_r", "extinction_i", "extinction_z",
    ])


def compute_dereddened_colors(phot_df: pd.DataFrame) -> pd.DataFrame:
    """Compute extinction-corrected colors from ugriz magnitudes.

    Adds columns: u_g, g_r, r_i, i_z (dereddened colors).
    """
    if phot_df.empty:
        for col in ["u_g", "g_r", "r_i", "i_z"]:
            phot_df[col] = pd.Series(dtype=float)
        return phot_df

    df = phot_df.copy()

    # Deredden: corrected = observed - extinction
    for band in "ugriz":
        mag_col = f"psfMag_{band}"
        ext_col = f"extinction_{band}"
        corr_col = f"mag_{band}"
        if mag_col in df.columns and ext_col in df.columns:
            df[corr_col] = df[mag_col] - df[ext_col]
        elif mag_col in df.columns:
            df[corr_col] = df[mag_col]
        else:
            df[corr_col] = np.nan

    df["u_g"] = df["mag_u"] - df["mag_g"]
    df["g_r"] = df["mag_g"] - df["mag_r"]
    df["r_i"] = df["mag_r"] - df["mag_i"]
    df["i_z"] = df["mag_i"] - df["mag_z"]

    return df


def score_color_anomalies(
    phot_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Score photometric candidates for color anomalies.

    Compares each star's dereddened u-g and g-r to expectations for its
    spectral class.  Adds columns:
        - u_g_excess: observed u-g minus expected (negative = bluer than expected)
        - g_r_excess: observed g-r minus expected
        - color_anomaly_score: combined deviation (higher = more anomalous)
        - blue_excess_flag: True if u-g is significantly bluer than expected
    """
    df = compute_dereddened_colors(phot_df)

    if df.empty:
        for col in ["u_g_excess", "g_r_excess", "color_anomaly_score", "blue_excess_flag"]:
            df[col] = pd.Series(dtype=float)
        return df

    u_g_excess = []
    g_r_excess = []

    for _, row in df.iterrows():
        sdss_idx = int(row.get("sdss_idx", -1))
        if sdss_idx >= 0 and sdss_idx < len(metadata):
            subclass = str(metadata.iloc[sdss_idx].get("subclass", ""))
        else:
            subclass = ""

        broad = _parse_broad_class(subclass)
        expected = _EXPECTED_COLORS.get(broad)

        if expected is not None and np.isfinite(row.get("u_g", np.nan)):
            u_g_excess.append(row["u_g"] - expected["u_g"])
        else:
            u_g_excess.append(np.nan)

        if expected is not None and np.isfinite(row.get("g_r", np.nan)):
            g_r_excess.append(row["g_r"] - expected["g_r"])
        else:
            g_r_excess.append(np.nan)

    df["u_g_excess"] = u_g_excess
    df["g_r_excess"] = g_r_excess

    # Combined score: sum of absolute deviations
    u_g_arr = np.array(u_g_excess, dtype=np.float64)
    g_r_arr = np.array(g_r_excess, dtype=np.float64)
    df["color_anomaly_score"] = np.where(
        np.isfinite(u_g_arr) & np.isfinite(g_r_arr),
        np.abs(u_g_arr) + np.abs(g_r_arr),
        np.where(np.isfinite(u_g_arr), np.abs(u_g_arr), 0.0),
    )

    # Blue excess: u-g significantly bluer (more negative) than expected
    # Threshold: 0.5 mag bluer than class expectation
    df["blue_excess_flag"] = np.where(
        np.isfinite(u_g_arr), u_g_arr < -0.5, False
    )

    return df
