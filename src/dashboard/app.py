# src/dashboard/app.py
"""Streamlit dashboard for exploring spectral anomalies."""
import json
import math
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import DEFAULT_GRID, load_or_fetch_processed_spectrum
from src.data.preprocess import build_metadata_features, METADATA_FEATURE_COLS
from src.features.color import spectrum_to_rgb, rgb_to_hex

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
SPECTRAL_LINES = [
    ("Ca K", 3933.7),
    ("Ca H", 3968.5),
    ("Hγ", 4340.5),
    ("Hβ", 4861.3),
    ("Na D", 5892.0),
    ("Hα", 6562.8),
]


@st.cache_data
def load_data():
    spectra_path = PROCESSED_DIR / "spectra.npy"
    spectra = np.load(spectra_path) if spectra_path.exists() else None
    metadata = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet")
    comparison = pd.read_parquet(RESULTS_DIR / "comparison.parquet")
    pca_components = np.load(RESULTS_DIR / "pca_components.npy")
    if_scores = np.load(RESULTS_DIR / "if_scores.npy")
    ae_scores = np.load(RESULTS_DIR / "ae_scores.npy")
    ocsvm_scores = np.load(RESULTS_DIR / "ocsvm_scores.npy")
    dagmm_scores = np.load(RESULTS_DIR / "dagmm_scores.npy")
    if_stability_std = np.load(RESULTS_DIR / "if_stability_std.npy")
    categories = list(np.load(RESULTS_DIR / "categories.npy"))
    wavelength_grid = DEFAULT_GRID if spectra is None else np.linspace(3800, 9200, spectra.shape[1])

    param_counts = {}
    param_path = RESULTS_DIR / "param_counts.json"
    if param_path.exists():
        with open(param_path) as f:
            param_counts = json.load(f)
    comparison_config = {}
    comparison_config_path = RESULTS_DIR / "comparison_config.json"
    if comparison_config_path.exists():
        with open(comparison_config_path) as f:
            comparison_config = json.load(f)

    cond_ae_scores_path = RESULTS_DIR / "conditional_ae_scores.npy"
    cond_ae_pvalues_path = RESULTS_DIR / "conditional_ae_pvalues.npy"
    cond_ae_calibration_mask_path = RESULTS_DIR / "conditional_ae_calibration_mask.npy"
    focused_review_path = RESULTS_DIR / "focused_review.parquet"
    cond_ae_scores = np.load(cond_ae_scores_path) if cond_ae_scores_path.exists() else np.zeros(len(if_scores))
    cond_ae_pvalues = np.load(cond_ae_pvalues_path) if cond_ae_pvalues_path.exists() else np.ones(len(if_scores))
    cond_ae_calibration_mask = (
        np.load(cond_ae_calibration_mask_path)
        if cond_ae_calibration_mask_path.exists()
        else np.zeros(len(if_scores), dtype=bool)
    )
    focused_review = (
        pd.read_parquet(focused_review_path)
        if focused_review_path.exists()
        else pd.DataFrame()
    )

    return (spectra, metadata, comparison, pca_components,
            if_scores, ae_scores, ocsvm_scores, dagmm_scores,
            if_stability_std, categories, param_counts, wavelength_grid,
            cond_ae_scores, cond_ae_pvalues, cond_ae_calibration_mask, comparison_config,
            focused_review)


@st.cache_data(show_spinner=False)
def load_spectrum_for_row(metadata_row: dict) -> np.ndarray:
    return load_or_fetch_processed_spectrum(metadata_row, RAW_DIR)


def plot_spectrum(spectra, wavelength_grid, idx, title="Spectrum", show_lines: bool = False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelength_grid, y=spectra[idx],
        mode="lines", name="Observed", line=dict(color="steelblue", width=1),
    ))
    if show_lines:
        y_min = float(np.nanmin(spectra[idx]))
        y_max = float(np.nanmax(spectra[idx]))
        for name, wavelength in SPECTRAL_LINES:
            fig.add_vline(x=wavelength, line_dash="dot", line_color="rgba(255,255,255,0.22)")
            fig.add_annotation(
                x=wavelength,
                y=y_max,
                text=name,
                showarrow=False,
                yshift=8,
                font=dict(size=10, color="rgba(235,240,248,0.78)"),
            )
    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (Angstroms)",
        yaxis_title="Normalized Flux",
        height=350,
    )
    return fig


def get_display_spectrum(spectra, metadata_row, idx):
    if spectra is not None and idx < len(spectra):
        return spectra[idx]
    return load_spectrum_for_row(dict(metadata_row))


def radius_proxy_rsun(logg: float, subclass: str | None = None) -> float | None:
    """Very rough radius proxy assuming ~1 solar mass.

    This is intentionally conservative: for white dwarfs and missing values we
    do not show a proxy because the assumption is usually misleading.
    """
    if not np.isfinite(logg):
        return None
    subclass = (subclass or "").upper()
    if "WD" in subclass:
        return None
    # g ~ GM/R^2, with log g in cgs and solar log g ~ 4.44.
    value = math.sqrt(10 ** (4.44 - float(logg)))
    if not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def stellar_regime_label(candidate: pd.Series) -> tuple[str, str]:
    subclass = str(candidate.get("subclass", "")).upper()
    logg = candidate.get("elodie_logg", np.nan)
    teff = candidate.get("elodie_teff", np.nan)
    if "WD" in subclass:
        return "White Dwarf-like", "#7cc7ff"
    if np.isfinite(logg) and logg < 3.5:
        return "Giant-like", "#ffb66c"
    if np.isfinite(logg) and logg > 4.1:
        if np.isfinite(teff) and teff > 8000:
            return "Hot Dwarf-like", "#b9d9ff"
        return "Main Sequence-like", "#ffd479"
    if np.isfinite(teff) and teff > 9000:
        return "Hot Star", "#a9cbff"
    return "Ambiguous Regime", "#b7c0cf"


def plot_temperature_gauge(teff: float, color_hex: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(teff) if np.isfinite(teff) else 0.0,
            number={"suffix": " K"},
            title={"text": "Effective Temperature"},
            gauge={
                "axis": {"range": [2500, 12000]},
                "bar": {"color": color_hex},
                "steps": [
                    {"range": [2500, 4000], "color": "#8f4f2a"},
                    {"range": [4000, 5500], "color": "#d39b58"},
                    {"range": [5500, 7000], "color": "#f2e7c9"},
                    {"range": [7000, 9000], "color": "#bfd8ff"},
                    {"range": [9000, 12000], "color": "#94bfff"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_kiel_context(metadata: pd.DataFrame, candidate: pd.Series) -> go.Figure:
    context = metadata.copy()
    for col in ["elodie_teff", "elodie_logg"]:
        if col not in context.columns:
            context[col] = np.nan
    context = context[np.isfinite(context["elodie_teff"]) & np.isfinite(context["elodie_logg"])]

    fig = go.Figure()
    if len(context) > 0:
        fig.add_trace(
            go.Scatter(
                x=context["elodie_teff"],
                y=context["elodie_logg"],
                mode="markers",
                name="Local sample",
                marker=dict(color="rgba(120,140,170,0.25)", size=5),
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[candidate.get("elodie_teff", np.nan)],
            y=[candidate.get("elodie_logg", np.nan)],
            mode="markers",
            name="Candidate",
            marker=dict(color="#ff6b35", size=14, line=dict(color="white", width=1)),
            text=[candidate.get("filename", "candidate")],
        )
    )
    fig.update_layout(
        title="Teff vs log g Context",
        xaxis_title="Teff (K)",
        yaxis_title="log g",
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def nearest_neighbor_indices(metadata: pd.DataFrame, candidate_idx: int, k: int = 3) -> list[int]:
    if len(metadata) <= 1:
        return []
    usable = metadata.copy()
    for col in METADATA_FEATURE_COLS:
        if col not in usable.columns:
            usable[col] = np.nan
    features = build_metadata_features(usable)
    candidate_vec = features[candidate_idx]
    distances = np.linalg.norm(features - candidate_vec, axis=1)
    distances[candidate_idx] = np.inf
    order = np.argsort(distances)
    return [int(idx) for idx in order[: min(k, len(order))] if np.isfinite(distances[idx])]


def plot_neighbor_spectra(
    candidate_spectrum: np.ndarray,
    neighbor_spectra: list[tuple[str, np.ndarray]],
    wavelength_grid: np.ndarray,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for name, spectrum in neighbor_spectra:
        fig.add_trace(go.Scatter(
            x=wavelength_grid,
            y=spectrum,
            mode="lines",
            name=name,
            line=dict(color="rgba(160,170,185,0.55)", width=1),
        ))
    fig.add_trace(go.Scatter(
        x=wavelength_grid,
        y=candidate_spectrum,
        mode="lines",
        name="Candidate",
        line=dict(color="#ff6b35", width=2.5),
    ))
    for name, wavelength in SPECTRAL_LINES:
        fig.add_vline(x=wavelength, line_dash="dot", line_color="rgba(255,255,255,0.15)")
    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (Angstroms)",
        yaxis_title="Normalized Flux",
        height=360,
        legend=dict(orientation="h"),
    )
    return fig


def stellar_profile_card(candidate: pd.Series, color_hex: str, radius_proxy: float | None) -> str:
    radius_text = (
        f"{radius_proxy:.2f} Rsun (1 Msun proxy)"
        if radius_proxy is not None
        else "Not shown for this class"
    )
    subclass = candidate.get("subclass", "unknown")
    regime_label, regime_color = stellar_regime_label(candidate)
    return (
        "<div style='padding:18px;border:1px solid #2b3445;border-radius:18px;"
        "background:linear-gradient(160deg,#07111f,#0d1f35);color:#f3f6fb;'>"
        f"<div style='display:flex;align-items:center;gap:16px;'>"
        f"<div style='width:92px;height:92px;border-radius:50%;background:{color_hex};"
        "box-shadow:0 0 24px rgba(255,255,255,0.2), 0 0 60px rgba(255,255,255,0.08);"
        "border:2px solid rgba(255,255,255,0.35);'></div>"
        "<div>"
        f"<div style='font-size:1.2rem;font-weight:700'>{candidate.get('filename', 'candidate')}</div>"
        f"<div style='margin-top:6px'><span style='display:inline-block;padding:4px 10px;"
        f"border-radius:999px;background:{regime_color};color:#07111f;font-weight:700;font-size:0.85rem'>"
        f"{regime_label}</span></div>"
        f"<div style='opacity:0.85;margin-top:4px'>Subclass: <strong>{subclass}</strong></div>"
        f"<div style='opacity:0.85;margin-top:4px'>Approx radius: <strong>{radius_text}</strong></div>"
        f"<div style='opacity:0.85;margin-top:4px'>Metallicity [Fe/H]: "
        f"<strong>{candidate.get('elodie_feh', float('nan')):.2f}</strong></div>"
        "</div></div></div>"
    )


def main():
    st.set_page_config(page_title="SDSS Spectral Anomalies", layout="wide")
    st.title("SDSS Spectral Anomaly Explorer")

    (spectra, metadata, comparison, pca_components,
     if_scores, ae_scores, ocsvm_scores, dagmm_scores,
     if_stability_std, categories, param_counts, wl,
     cond_ae_scores, cond_ae_pvalues, cond_ae_calibration_mask, comparison_config,
     focused_review) = load_data()

    # Sidebar with parameter counts
    st.sidebar.header("Model Parameters")
    for model_name, count in param_counts.items():
        st.sidebar.metric(model_name, f"{count:,}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Anomaly Browser",
        "Model Comparison",
        "PCA Explorer",
        "Model Stability",
        "Conformal P-Values",
        "Focused Review",
    ])

    # --- Tab 1: Anomaly Browser ---
    with tab1:
        st.subheader("Top Anomalies")
        sort_by = st.selectbox("Sort by", ["Combined", "Isolation Forest", "Autoencoder", "OC-SVM", "DAGMM", "Conditional AE", "Conformal P-Value (most anomalous)"])
        top_n = st.slider("Show top N", 10, 500, 100)

        if sort_by == "Combined":
            order = np.argsort(-(if_scores + ae_scores + ocsvm_scores + dagmm_scores + cond_ae_scores))
        elif sort_by == "Isolation Forest":
            order = np.argsort(-if_scores)
        elif sort_by == "Autoencoder":
            order = np.argsort(-ae_scores)
        elif sort_by == "OC-SVM":
            order = np.argsort(-ocsvm_scores)
        elif sort_by == "DAGMM":
            order = np.argsort(-dagmm_scores)
        elif sort_by == "Conditional AE":
            order = np.argsort(-cond_ae_scores)
        else:  # Conformal P-Value (most anomalous)
            valid_idx = np.where(cond_ae_calibration_mask)[0]
            invalid_idx = np.where(~cond_ae_calibration_mask)[0]
            order = np.concatenate([
                valid_idx[np.argsort(cond_ae_pvalues[valid_idx])],
                invalid_idx,
            ])

        top_indices = order[:top_n]
        table_data = metadata.iloc[top_indices].copy()
        table_data["if_score"] = if_scores[top_indices]
        table_data["ae_score"] = ae_scores[top_indices]
        table_data["ocsvm_score"] = ocsvm_scores[top_indices]
        table_data["dagmm_score"] = dagmm_scores[top_indices]
        table_data["cond_ae_score"] = cond_ae_scores[top_indices]
        table_data["cond_ae_pvalue"] = cond_ae_pvalues[top_indices]
        table_data["conformal_valid"] = cond_ae_calibration_mask[top_indices]
        table_data["category"] = [categories[i] for i in top_indices]
        table_data["index"] = top_indices

        selected = st.dataframe(
            table_data.reset_index(drop=True),
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if selected and selected.selection and selected.selection.rows:
            row_idx = selected.selection.rows[0]
            spectrum_idx = int(table_data.iloc[row_idx]["index"])
            meta = table_data.iloc[row_idx]
            display_spectrum = get_display_spectrum(spectra, meta, spectrum_idx)
            st.plotly_chart(
                plot_spectrum(np.array([display_spectrum]), wl, 0, f"Spectrum: {meta.get('filename', spectrum_idx)}"),
                use_container_width=True,
            )
            # Perceived star color swatch
            r, g, b = spectrum_to_rgb(wl, display_spectrum)
            hex_color = rgb_to_hex(r, g, b)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0">'
                f'<div style="width:48px;height:48px;border-radius:50%;'
                f'background:{hex_color};border:2px solid #444"></div>'
                f'<span style="font-size:1.1em">Perceived color: '
                f'<strong>{hex_color}</strong> (R={r} G={g} B={b})</span></div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("IF Score", f"{if_scores[spectrum_idx]:.4f}")
            col2.metric("AE Score", f"{ae_scores[spectrum_idx]:.4f}")
            col3.metric("OC-SVM Score", f"{ocsvm_scores[spectrum_idx]:.4f}")
            col4.metric("DAGMM Score", f"{dagmm_scores[spectrum_idx]:.4f}")

            col5, col6 = st.columns(2)
            col5.metric("Cond AE Score", f"{cond_ae_scores[spectrum_idx]:.4f}")
            col6.metric("Conformal p-value", f"{cond_ae_pvalues[spectrum_idx]:.4f}")
            if cond_ae_calibration_mask[spectrum_idx]:
                st.caption("Conformal p-value is valid for this held-out calibration row.")
            else:
                st.caption("Conformal p-value shown for reference only; this row was used in training.")

    # --- Tab 2: Model Comparison ---
    with tab2:
        st.subheader("Model Score Comparison (Scatter Matrix)")
        comparison_top_n = int(comparison_config.get("top_n", min(100, len(comparison))))
        scatter_df = pd.DataFrame({
            "IF Score": if_scores,
            "AE Score": ae_scores,
            "OC-SVM Score": ocsvm_scores,
            "DAGMM Score": dagmm_scores,
            "Conditional AE Score": cond_ae_scores,
        })
        fig = px.scatter_matrix(
            scatter_df,
            dimensions=["IF Score", "AE Score", "OC-SVM Score", "DAGMM Score", "Conditional AE Score"],
            opacity=0.3, height=800,
            title="Pairwise Model Score Comparisons",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "n_models_agreed" in comparison.columns:
            n_models = len([col for col in comparison.columns if col.endswith("_rank")])
            all_agreed = int((comparison["n_models_agreed"] == n_models).sum())
            st.metric(
                f"Spectra flagged by ALL {n_models} models (top {comparison_top_n})",
                all_agreed,
            )
        elif "agreed" in comparison.columns:
            n_agreed = int(comparison["agreed"].sum())
            st.metric(f"Spectra flagged by BOTH models (top {comparison_top_n})", n_agreed)

    # --- Tab 3: PCA Explorer ---
    with tab3:
        st.subheader("PCA Latent Space")
        color_by = st.selectbox("Color by", ["IF Score", "AE Score", "OC-SVM Score", "DAGMM Score", "Spectral Type"])

        pca_df = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "PC3": pca_components[:, 2] if pca_components.shape[1] > 2 else 0,
            "IF Score": if_scores,
            "AE Score": ae_scores,
            "OC-SVM Score": ocsvm_scores,
            "DAGMM Score": dagmm_scores,
            "Spectral Type": metadata.get("subclass", "unknown"),
        })

        dim_choice = st.radio("Dimensions", ["2D", "3D"], horizontal=True)
        if dim_choice == "2D":
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color=color_by,
                opacity=0.5, height=600, title="PCA 2D Projection",
            )
        else:
            fig = px.scatter_3d(
                pca_df, x="PC1", y="PC2", z="PC3", color=color_by,
                opacity=0.4, height=700, title="PCA 3D Projection",
            )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 4: Model Stability ---
    with tab4:
        st.subheader("Model Stability (IF multi-seed)")
        sorted_idx = np.argsort(-if_scores)[:200]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(sorted_idx))),
            y=if_scores[sorted_idx],
            error_y=dict(type="data", array=if_stability_std[sorted_idx]),
            name="IF Score +/- std",
        ))
        fig.update_layout(
            title="Top 200 Anomalies: IF Score with Stability Error Bars",
            xaxis_title="Rank", yaxis_title="IF Score", height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Std heatmap for top anomalies
        st.subheader("Score Stability (Standard Deviation)")
        std_df = pd.DataFrame({
            "Spectrum Index": sorted_idx,
            "IF Score": if_scores[sorted_idx],
            "IF Std": if_stability_std[sorted_idx],
            "CV (Std/Score)": if_stability_std[sorted_idx] / (np.abs(if_scores[sorted_idx]) + 1e-8),
        })
        st.dataframe(std_df, use_container_width=True)

    # --- Tab 5: Conformal P-Values ---
    with tab5:
        st.subheader("Conditional AE: Conformal P-Value Distribution")
        valid_pvalues = cond_ae_pvalues[cond_ae_calibration_mask]
        if len(valid_pvalues) == 0:
            st.info("No held-out calibration mask found. Conformal p-values are unavailable for calibrated browsing.")
        else:
            st.caption("Only held-out calibration rows are shown here. Training-row p-values are not presented as calibrated.")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=valid_pvalues, nbinsx=50, name="p-values",
                                       marker_color="steelblue", opacity=0.75))
            fig.update_layout(xaxis_title="Conformal p-value", yaxis_title="Count",
                              title="Uniform = well-calibrated; spike near 0 = anomalies", height=400)
            st.plotly_chart(fig, use_container_width=True)

            alpha = st.slider("Flag anomalies at p-value <=", 0.01, 0.20, 0.05, step=0.01)
            flagged_idx = np.where(cond_ae_calibration_mask & (cond_ae_pvalues <= alpha))[0]
            n_flagged = int(len(flagged_idx))
            st.metric(f"Held-out rows flagged at alpha={alpha}", n_flagged)

            if len(flagged_idx) > 0:
                flagged_df = metadata.iloc[flagged_idx].copy()
                flagged_df["cond_ae_score"] = cond_ae_scores[flagged_idx]
                flagged_df["cond_ae_pvalue"] = cond_ae_pvalues[flagged_idx]
                st.dataframe(flagged_df.sort_values("cond_ae_pvalue").reset_index(drop=True),
                             use_container_width=True)

    # --- Tab 6: Focused Review ---
    with tab6:
        st.subheader("Focused Review: Highest-Consensus Candidates")
        if focused_review.empty:
            st.info("No focused review file found yet. Rerun the pipeline to generate one.")
        else:
            st.caption("These are the current top consensus candidates ranked by model agreement first, then combined rank.")
            st.dataframe(focused_review, use_container_width=True)

            focus_idx = st.selectbox(
                "Inspect candidate",
                options=list(focused_review["focus_rank"]),
                format_func=lambda value: (
                    f"#{value} - {focused_review.loc[focused_review['focus_rank'] == value, 'filename'].iloc[0]}"
                ),
            )
            candidate = focused_review.loc[focused_review["focus_rank"] == focus_idx].iloc[0]
            spectrum_idx = int(metadata.index[metadata["filename"] == candidate["filename"]][0])
            display_spectrum = get_display_spectrum(spectra, candidate, spectrum_idx)
            r, g, b = spectrum_to_rgb(wl, display_spectrum)
            color_hex = rgb_to_hex(r, g, b)
            radius_proxy = radius_proxy_rsun(candidate.get("elodie_logg", np.nan), candidate.get("subclass"))

            st.plotly_chart(
                plot_spectrum(
                    np.array([display_spectrum]),
                    wl,
                    0,
                    f"Focused Review: {candidate['filename']}",
                    show_lines=True,
                ),
                use_container_width=True,
            )

            v1, v2 = st.columns([1.15, 1])
            with v1:
                st.markdown(
                    stellar_profile_card(candidate, color_hex, radius_proxy),
                    unsafe_allow_html=True,
                )
            with v2:
                st.plotly_chart(
                    plot_temperature_gauge(candidate.get("elodie_teff", np.nan), color_hex),
                    use_container_width=True,
                )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_kiel_context(metadata, candidate), use_container_width=True)
            with c2:
                metric_df = pd.DataFrame({
                    "quantity": ["Agreement", "Combined Rank", "S/N", "Radius Proxy"],
                    "value": [
                        f"{int(candidate['n_models_agreed'])}/5",
                        f"{candidate['combined_rank']:.1f}",
                        f"{candidate.get('sn_median', float('nan')):.2f}",
                        f"{radius_proxy:.2f} Rsun" if radius_proxy is not None else "not shown",
                    ],
                })
                st.dataframe(metric_df, use_container_width=True, hide_index=True)

            p1, p2, p3 = st.columns(3)
            p1.metric("Teff", f"{candidate.get('elodie_teff', float('nan')):.0f}")
            p2.metric("log g", f"{candidate.get('elodie_logg', float('nan')):.3f}")
            p3.metric("[Fe/H]", f"{candidate.get('elodie_feh', float('nan')):.2f}")

            rank_df = pd.DataFrame({
                "model": ["IF", "AE", "OC-SVM", "DAGMM", "Cond AE"],
                "rank": [
                    int(candidate["if_rank"]),
                    int(candidate["ae_rank"]),
                    int(candidate["ocsvm_rank"]),
                    int(candidate["dagmm_rank"]),
                    int(candidate["cond_ae_rank"]),
                ],
                "score": [
                    candidate["if_score"],
                    candidate["ae_score"],
                    candidate["ocsvm_score"],
                    candidate["dagmm_score"],
                    candidate["cond_ae_score"],
                ],
            }).sort_values("rank")
            st.dataframe(rank_df.reset_index(drop=True), use_container_width=True)

            neighbor_idx = nearest_neighbor_indices(metadata, spectrum_idx, k=3)
            if neighbor_idx:
                neighbor_spectra = []
                neighbor_rows = []
                for idx in neighbor_idx:
                    neighbor_row = metadata.iloc[idx]
                    neighbor_rows.append({
                        "filename": neighbor_row.get("filename", f"idx-{idx}"),
                        "subclass": neighbor_row.get("subclass", "unknown"),
                        "sn_median": neighbor_row.get("sn_median", np.nan),
                        "elodie_teff": neighbor_row.get("elodie_teff", np.nan),
                        "elodie_logg": neighbor_row.get("elodie_logg", np.nan),
                        "elodie_feh": neighbor_row.get("elodie_feh", np.nan),
                    })
                    neighbor_spectra.append(
                        (str(neighbor_row.get("filename", f"idx-{idx}")), get_display_spectrum(spectra, neighbor_row, idx))
                    )

                st.plotly_chart(
                    plot_neighbor_spectra(
                        display_spectrum,
                        neighbor_spectra,
                        wl,
                        "Nearest Metadata Neighbors",
                    ),
                    use_container_width=True,
                )
                st.dataframe(pd.DataFrame(neighbor_rows), use_container_width=True)

            st.markdown(
                f"RA/Dec: `{candidate.get('ra', float('nan')):.6f}`, `{candidate.get('dec', float('nan')):.6f}`  "
                f"Plate/MJD/Fiber: `{int(candidate.get('plate', -1))}` / `{int(candidate.get('mjd', -1))}` / `{int(candidate.get('fiberid', -1))}`"
            )
            st.caption(
                "Radius proxy assumes roughly one solar mass and is suppressed for white-dwarf-like subclasses. "
                "Treat it as a visual cue, not a measured radius."
            )


if __name__ == "__main__":
    main()
