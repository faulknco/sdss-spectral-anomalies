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
from src.features.line_windows import DISPLAY_LINES
from datetime import datetime, timezone
from src.features.review_labels import LABEL_CODES, LABEL_DISPLAY_NAMES, load_labels, save_labels

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


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

    ml_scores_path = RESULTS_DIR / "microlensing_scores.npy"
    acc_scores_path = RESULTS_DIR / "accretion_scores.npy"
    pbh_categories_path = RESULTS_DIR / "pbh_categories.npy"
    microlensing_scores = np.load(ml_scores_path) if ml_scores_path.exists() else None
    accretion_scores = np.load(acc_scores_path) if acc_scores_path.exists() else None
    pbh_categories = list(np.load(pbh_categories_path)) if pbh_categories_path.exists() else None
    asym_scores_path = RESULTS_DIR / "line_asymmetry_scores.npy"
    line_asymmetry_scores = np.load(asym_scores_path) if asym_scores_path.exists() else None

    cvae_scores_path = RESULTS_DIR / "cvae_scores.npy"
    cvae_pvalues_path = RESULTS_DIR / "cvae_pvalues.npy"
    flow_scores_path = RESULTS_DIR / "flow_scores.npy"
    flow_pvalues_path = RESULTS_DIR / "flow_pvalues.npy"
    eval_results_path = RESULTS_DIR / "evaluation_results.json"

    cvae_scores = np.load(cvae_scores_path) if cvae_scores_path.exists() else np.zeros(len(if_scores))
    cvae_pvalues = np.load(cvae_pvalues_path) if cvae_pvalues_path.exists() else np.ones(len(if_scores))
    flow_scores = np.load(flow_scores_path) if flow_scores_path.exists() else np.zeros(len(if_scores))
    flow_pvalues = np.load(flow_pvalues_path) if flow_pvalues_path.exists() else np.ones(len(if_scores))
    eval_results = None
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            eval_results = json.load(f)

    multiepoch_path = RESULTS_DIR / "multiepoch_variability.parquet"
    multiepoch_scores_path = RESULTS_DIR / "multiepoch_scores.npy"
    gaia_path = RESULTS_DIR / "gaia_crossmatch.parquet"
    multiepoch_df = pd.read_parquet(multiepoch_path) if multiepoch_path.exists() else pd.DataFrame()
    multiepoch_scores = np.load(multiepoch_scores_path) if multiepoch_scores_path.exists() else None
    gaia_df = pd.read_parquet(gaia_path) if gaia_path.exists() else pd.DataFrame()
    phot_path = RESULTS_DIR / "photometric_crossmatch.parquet"
    phot_df = pd.read_parquet(phot_path) if phot_path.exists() else pd.DataFrame()

    return (spectra, metadata, comparison, pca_components,
            if_scores, ae_scores, ocsvm_scores, dagmm_scores,
            if_stability_std, categories, param_counts, wavelength_grid,
            cond_ae_scores, cond_ae_pvalues, cond_ae_calibration_mask, comparison_config,
            focused_review, microlensing_scores, accretion_scores, pbh_categories,
            cvae_scores, cvae_pvalues, flow_scores, flow_pvalues, eval_results,
            multiepoch_df, multiepoch_scores, gaia_df, line_asymmetry_scores, phot_df)


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
        for name, wavelength in DISPLAY_LINES:
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
    for name, wavelength in DISPLAY_LINES:
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
     focused_review, microlensing_scores, accretion_scores, pbh_categories,
     cvae_scores, cvae_pvalues, flow_scores, flow_pvalues, eval_results,
     multiepoch_df, multiepoch_scores, gaia_df) = load_data()

    labels_path = RESULTS_DIR / "review_labels.parquet"
    if "review_labels" not in st.session_state:
        labels_df = load_labels(labels_path)
        st.session_state["review_labels"] = {
            row["filename"]: {"label": row["label"], "notes": row["notes"], "timestamp": row["timestamp"]}
            for _, row in labels_df.iterrows()
        }

    pbh_available = microlensing_scores is not None and accretion_scores is not None

    # Sidebar with parameter counts
    st.sidebar.header("Model Parameters")
    for model_name, count in param_counts.items():
        st.sidebar.metric(model_name, f"{count:,}")

    st.sidebar.divider()
    st.sidebar.header("Review Labels")
    if st.sidebar.button("Save All Labels"):
        rows = [{"filename": fn, **data} for fn, data in st.session_state.get("review_labels", {}).items()]
        if rows:
            save_labels(pd.DataFrame(rows), labels_path)
            st.sidebar.success(f"Saved {len(rows)} labels")
        else:
            st.sidebar.info("No labels to save")

    tab_names = [
        "Anomaly Browser",
        "Model Comparison",
        "PCA Explorer",
        "Model Stability",
        "Conformal P-Values",
        "Focused Review",
        "Evaluation",
    ]
    if pbh_available:
        tab_names.append("PBH Candidates")
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs[:7]
    tab_pbh = tabs[7] if pbh_available else None

    # --- Tab 1: Anomaly Browser ---
    with tab1:
        st.subheader("Top Anomalies")
        sort_options = ["Combined", "Isolation Forest", "Autoencoder", "OC-SVM", "DAGMM", "Conditional AE", "CVAE", "Conditional Flow", "Conformal P-Value (most anomalous)"]
        if pbh_available:
            sort_options.extend(["Microlensing Score", "Accretion Score"])
        if line_asymmetry_scores is not None:
            sort_options.append("Line Asymmetry Score")
        sort_by = st.selectbox("Sort by", sort_options)
        top_n = st.slider("Show top N", 10, 500, 100)

        if sort_by == "Combined":
            order = np.argsort(-(if_scores + ae_scores + ocsvm_scores + dagmm_scores + cond_ae_scores + cvae_scores + flow_scores))
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
        elif sort_by == "CVAE":
            order = np.argsort(-cvae_scores)
        elif sort_by == "Conditional Flow":
            order = np.argsort(-flow_scores)
        elif sort_by == "Microlensing Score" and pbh_available:
            order = np.argsort(-microlensing_scores)
        elif sort_by == "Accretion Score" and pbh_available:
            order = np.argsort(-accretion_scores)
        elif sort_by == "Line Asymmetry Score" and line_asymmetry_scores is not None:
            order = np.argsort(-line_asymmetry_scores)
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
        table_data["cvae_score"] = cvae_scores[top_indices]
        table_data["flow_score"] = flow_scores[top_indices]
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

            col7, col8 = st.columns(2)
            col7.metric("CVAE Score", f"{cvae_scores[spectrum_idx]:.4f}")
            col8.metric("Flow Score", f"{flow_scores[spectrum_idx]:.4f}")
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
            "CVAE Score": cvae_scores,
            "Flow Score": flow_scores,
        })
        fig = px.scatter_matrix(
            scatter_df,
            dimensions=list(scatter_df.columns),
            opacity=0.3, height=900,
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
        review_labels = st.session_state.get("review_labels", {})
        if not focused_review.empty:
            n_candidates = len(focused_review)
            n_labeled = sum(1 for fn in focused_review["filename"] if fn in review_labels)
            st.progress(n_labeled / max(n_candidates, 1))
            label_counts = {}
            for fn in focused_review["filename"]:
                if fn in review_labels:
                    lbl = review_labels[fn]["label"]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            summary_parts = [f"{n_labeled}/{n_candidates} labeled"]
            for code in LABEL_CODES:
                if code in label_counts:
                    summary_parts.append(f"{label_counts[code]} {LABEL_DISPLAY_NAMES[code]}")
            st.caption(" | ".join(summary_parts))

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
                        f"{int(candidate['n_models_agreed'])}/{len([c for c in candidate.index if c.endswith('_rank')])}",
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

            models = [("IF", "if"), ("AE", "ae"), ("OC-SVM", "ocsvm"),
                      ("DAGMM", "dagmm"), ("Cond AE", "cond_ae"),
                      ("CVAE", "cvae"), ("Flow", "flow")]
            rank_rows = []
            for label, key in models:
                rank_col = f"{key}_rank"
                score_col = f"{key}_score"
                if rank_col in candidate.index and score_col in candidate.index:
                    rank_rows.append({"model": label, "rank": int(candidate[rank_col]), "score": candidate[score_col]})
            rank_df = pd.DataFrame(rank_rows).sort_values("rank")
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

            st.divider()
            st.subheader("Label This Candidate")
            candidate_filename = candidate["filename"]
            current = review_labels.get(candidate_filename, {})
            current_label = current.get("label", "")
            display_options = ["(unlabeled)"] + [LABEL_DISPLAY_NAMES[c] for c in LABEL_CODES]
            code_for_display = {v: k for k, v in LABEL_DISPLAY_NAMES.items()}
            current_display = LABEL_DISPLAY_NAMES.get(current_label, "(unlabeled)")
            current_index = display_options.index(current_display) if current_display in display_options else 0
            selected_display = st.selectbox("Classification", display_options, index=current_index, key=f"label_{candidate_filename}")
            notes = st.text_input("Notes", value=current.get("notes", ""), key=f"notes_{candidate_filename}")
            if selected_display == "(unlabeled)":
                if candidate_filename in review_labels:
                    del st.session_state["review_labels"][candidate_filename]
            else:
                selected_code = code_for_display[selected_display]
                if selected_code != current.get("label") or notes != current.get("notes", ""):
                    st.session_state["review_labels"][candidate_filename] = {"label": selected_code, "notes": notes, "timestamp": datetime.now(timezone.utc).isoformat()}

    # --- Tab 7: Evaluation ---
    with tab7:
        st.subheader("Semi-Synthetic Evaluation Results")
        if eval_results is None:
            st.info("No evaluation results found. Run the pipeline to generate them.")
        else:
            model_names = list(eval_results.keys())

            st.subheader("Overall Metrics")
            metrics_df = pd.DataFrame({
                "Model": model_names,
                "AUROC": [eval_results[m]["auroc"] for m in model_names],
                "AUPRC": [eval_results[m]["auprc"] for m in model_names],
            }).sort_values("AUROC", ascending=False)
            st.dataframe(metrics_df.reset_index(drop=True), use_container_width=True)

            st.subheader("Precision @ k")
            k_values = sorted(eval_results[model_names[0]]["precision_at_k"].keys(), key=lambda x: int(x))
            prec_data = []
            for m in model_names:
                for k in k_values:
                    prec_data.append({"Model": m, "k": int(k), "Precision": eval_results[m]["precision_at_k"][str(k)]})
            prec_df = pd.DataFrame(prec_data)
            fig = px.bar(prec_df, x="k", y="Precision", color="Model", barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Per Anomaly Type Recall")
            type_data = {}
            for m in model_names:
                ptr = eval_results[m].get("per_type_recall_at_k", {})
                for t, val in ptr.items():
                    type_data.setdefault(t, {})[m] = val
            if type_data:
                type_df = pd.DataFrame(type_data).T
                type_df.index.name = "Anomaly Type"
                fig = px.imshow(
                    type_df.values,
                    x=list(type_df.columns),
                    y=list(type_df.index),
                    color_continuous_scale="Blues",
                    text_auto=".2f",
                    height=400,
                    title="Recall by Anomaly Type and Model",
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Tab 8: PBH Candidates ---
    if tab_pbh is not None:
        with tab_pbh:
            st.subheader("PBH Candidate Screening")
            st.caption(
                "High scores do NOT confirm PBH signatures. Many mundane astrophysical "
                "phenomena (binaries, chromospheric activity, misclassified subclass, poor "
                "sky subtraction) produce similar spectral features. Single-epoch "
                "median-normalized spectra cannot measure absolute magnification. These "
                "are exploratory screens for candidates warranting follow-up observation."
            )

            pbh_col1, pbh_col2 = st.columns(2)

            # --- Microlensing candidates table ---
            with pbh_col1:
                st.markdown("### Microlensing Candidates")
                ml_order = np.argsort(-microlensing_scores)
                ml_top_n = st.slider("Top N microlensing", 10, 200, 50, key="ml_top")
                ml_idx = ml_order[:ml_top_n]
                ml_df = metadata.iloc[ml_idx].copy()
                ml_df["microlensing_score"] = microlensing_scores[ml_idx]
                ml_df["if_score"] = if_scores[ml_idx]
                ml_df["ae_score"] = ae_scores[ml_idx]
                ml_df["category"] = [categories[i] for i in ml_idx]
                if pbh_categories is not None:
                    ml_df["pbh_category"] = [pbh_categories[i] for i in ml_idx]
                st.dataframe(ml_df.reset_index(drop=True), use_container_width=True)

            # --- Accretion candidates table ---
            with pbh_col2:
                st.markdown("### Accretion Candidates")
                acc_order = np.argsort(-accretion_scores)
                acc_top_n = st.slider("Top N accretion", 10, 200, 50, key="acc_top")
                acc_idx = acc_order[:acc_top_n]
                acc_df = metadata.iloc[acc_idx].copy()
                acc_df["accretion_score"] = accretion_scores[acc_idx]
                acc_df["if_score"] = if_scores[acc_idx]
                acc_df["ae_score"] = ae_scores[acc_idx]
                acc_df["category"] = [categories[i] for i in acc_idx]
                if pbh_categories is not None:
                    acc_df["pbh_category"] = [pbh_categories[i] for i in acc_idx]
                st.dataframe(acc_df.reset_index(drop=True), use_container_width=True)

            # --- Cross-correlation scatter ---
            st.markdown("### Microlensing vs Accretion Score")
            scatter_df = pd.DataFrame({
                "Microlensing Score": microlensing_scores,
                "Accretion Score": accretion_scores,
                "Category": categories,
            })
            fig = px.scatter(
                scatter_df,
                x="Microlensing Score",
                y="Accretion Score",
                color="Category",
                opacity=0.5,
                height=500,
                title="Microlensing vs Accretion Score (colored by anomaly category)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Spectrum detail with template overlay ---
            st.markdown("### Spectrum Detail with Subclass Template")
            pbh_inspect_options = ["Microlensing", "Accretion"]
            pbh_inspect_type = st.radio("Inspect from", pbh_inspect_options, horizontal=True)
            if pbh_inspect_type == "Microlensing":
                inspect_order = ml_order
                inspect_scores = microlensing_scores
                score_label = "Microlensing Score"
            else:
                inspect_order = acc_order
                inspect_scores = accretion_scores
                score_label = "Accretion Score"

            inspect_rank = st.slider("Select rank", 1, min(100, len(inspect_order)), 1, key="pbh_rank")
            pbh_idx = int(inspect_order[inspect_rank - 1])
            pbh_meta = metadata.iloc[pbh_idx]
            pbh_spectrum = get_display_spectrum(spectra, pbh_meta, pbh_idx)

            # Build subclass population median template
            broad_class = str(pbh_meta.get("subclass", ""))[:1].upper()
            same_class_mask = metadata["subclass"].str[:1].str.upper() == broad_class
            template_spectrum = None
            if spectra is not None and same_class_mask.sum() >= 5:
                template_spectrum = np.median(spectra[same_class_mask], axis=0)

            fig = go.Figure()
            if template_spectrum is not None:
                fig.add_trace(go.Scatter(
                    x=wl, y=template_spectrum, mode="lines",
                    name=f"{broad_class}-class median",
                    line=dict(color="rgba(160,170,185,0.6)", width=1.5, dash="dash"),
                ))
            fig.add_trace(go.Scatter(
                x=wl, y=pbh_spectrum, mode="lines",
                name="Observed",
                line=dict(color="#ff6b35", width=2),
            ))
            for name, wavelength in DISPLAY_LINES:
                fig.add_vline(x=wavelength, line_dash="dot", line_color="rgba(255,255,255,0.15)")
            fig.update_layout(
                title=f"Rank #{inspect_rank}: {pbh_meta.get('filename', pbh_idx)} "
                      f"({score_label}: {inspect_scores[pbh_idx]:.3f})",
                xaxis_title="Wavelength (Angstroms)",
                yaxis_title="Normalized Flux",
                height=400,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Microlensing", f"{microlensing_scores[pbh_idx]:.4f}")
            mc2.metric("Accretion", f"{accretion_scores[pbh_idx]:.4f}")
            mc3.metric("Line Asymmetry", f"{line_asymmetry_scores[pbh_idx]:.4f}" if line_asymmetry_scores is not None else "N/A")
            mc4.metric("IF Score", f"{if_scores[pbh_idx]:.4f}")
            mc5.metric("AE Score", f"{ae_scores[pbh_idx]:.4f}")
            if pbh_categories is not None:
                st.info(f"PBH category: **{pbh_categories[pbh_idx]}**")

            # --- Multi-epoch variability ---
            st.markdown("---")
            st.markdown("### Multi-Epoch Variability")
            if not multiepoch_df.empty:
                st.caption(
                    f"Found {len(multiepoch_df)} multi-epoch groups from repeat SDSS observations. "
                    "Achromatic (wavelength-independent) variability is consistent with "
                    "gravitational microlensing, while chromatic variability suggests "
                    "intrinsic stellar changes."
                )
                me_show_n = st.slider("Show top N groups", 5, min(50, len(multiepoch_df)), 10, key="me_top")
                me_display = multiepoch_df.head(me_show_n).copy()
                # Drop member_indices for display (it's a list)
                me_cols = [c for c in me_display.columns if c != "member_indices"]
                st.dataframe(me_display[me_cols].reset_index(drop=True), use_container_width=True)

                if multiepoch_scores is not None and multiepoch_scores[pbh_idx] > 0:
                    st.success(
                        f"This candidate (idx {pbh_idx}) has multi-epoch data! "
                        f"Variability score: {multiepoch_scores[pbh_idx]:.4f}"
                    )
                elif multiepoch_scores is not None:
                    st.info("This candidate has no repeat observations in the dataset.")
            else:
                st.info("No multi-epoch groups found. Run the pipeline with more spectra to find repeat observations.")

            # --- Gaia DR3 cross-match ---
            st.markdown("### Gaia DR3 Astrometric Cross-Match")
            if not gaia_df.empty:
                st.caption(
                    f"Matched {len(gaia_df)} PBH candidates to Gaia DR3. "
                    "RUWE > 1.4 or high astrometric excess noise may indicate "
                    "an unseen massive companion, but many mundane causes exist "
                    "(binaries, crowded fields, extended sources)."
                )
                gaia_display_cols = [
                    "sdss_idx", "separation_arcsec", "phot_g_mean_mag",
                    "ruwe", "ruwe_flag", "astrometric_excess_noise",
                    "astrometric_excess_noise_sig", "excess_noise_flag",
                    "astrometric_anomaly_score", "parallax", "pmra", "pmdec",
                ]
                gaia_show = gaia_df[[c for c in gaia_display_cols if c in gaia_df.columns]]
                st.dataframe(
                    gaia_show.sort_values("astrometric_anomaly_score", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                )

                # Check if current candidate has Gaia match
                candidate_gaia = gaia_df[gaia_df["sdss_idx"] == pbh_idx]
                if len(candidate_gaia) > 0:
                    g = candidate_gaia.iloc[0]
                    gc1, gc2, gc3 = st.columns(3)
                    gc1.metric("RUWE", f"{g.get('ruwe', np.nan):.2f}")
                    gc2.metric("Excess Noise Sig", f"{g.get('astrometric_excess_noise_sig', np.nan):.1f}")
                    gc3.metric("Astro Score", f"{g.get('astrometric_anomaly_score', 0):.2f}")
                    if g.get("ruwe_flag", False):
                        st.warning("RUWE > 1.4: astrometric solution is poor for a single star")
                    if g.get("excess_noise_flag", False):
                        st.warning("Significant astrometric excess noise detected")
                else:
                    st.info("No Gaia match found for this candidate.")
            else:
                st.info(
                    "No Gaia cross-match data available. The pipeline queries Gaia DR3 "
                    "for PBH candidates when online."
                )

            # --- SDSS Photometric Cross-Match ---
            st.markdown("### SDSS Photometric Colors")
            if not phot_df.empty and "u_g" in phot_df.columns:
                st.caption(
                    f"Matched {len(phot_df)} PBH candidates to SDSS ugriz photometry. "
                    "Microlensing should produce normal colors (achromatic magnification). "
                    "Accretion should show blue/UV excess (anomalous u-g)."
                )

                # Color-color diagram
                phot_valid = phot_df[
                    np.isfinite(phot_df["u_g"]) & np.isfinite(phot_df["g_r"])
                ].copy()
                if len(phot_valid) > 0:
                    phot_valid["Blue Excess"] = phot_valid["blue_excess_flag"].map(
                        {True: "Blue excess", False: "Normal"}
                    )
                    fig = px.scatter(
                        phot_valid, x="g_r", y="u_g",
                        color="Blue Excess",
                        hover_data=["sdss_idx", "color_anomaly_score"],
                        opacity=0.7, height=450,
                        title="Color-Color Diagram (dereddened ugriz)",
                        labels={"g_r": "g - r", "u_g": "u - g"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                phot_display_cols = [
                    "sdss_idx", "u_g", "g_r", "r_i",
                    "u_g_excess", "g_r_excess",
                    "color_anomaly_score", "blue_excess_flag",
                ]
                phot_show = phot_df[[c for c in phot_display_cols if c in phot_df.columns]]
                st.dataframe(
                    phot_show.sort_values("color_anomaly_score", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                )

                # Check current candidate
                candidate_phot = phot_df[phot_df["sdss_idx"] == pbh_idx]
                if len(candidate_phot) > 0:
                    cp = candidate_phot.iloc[0]
                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("u - g", f"{cp.get('u_g', np.nan):.2f}")
                    pc2.metric("g - r", f"{cp.get('g_r', np.nan):.2f}")
                    pc3.metric("Color Score", f"{cp.get('color_anomaly_score', 0):.2f}")
                    if cp.get("blue_excess_flag", False):
                        st.warning("Blue excess detected: u-g significantly bluer than expected for spectral class")
                    elif np.isfinite(cp.get("u_g_excess", np.nan)):
                        excess = cp["u_g_excess"]
                        if abs(excess) < 0.3:
                            st.success("Colors consistent with spectral class (achromatic, consistent with microlensing)")
                        else:
                            st.info(f"u-g excess: {excess:+.2f} mag from class expectation")
                else:
                    st.info("No photometric match for this candidate.")
            else:
                st.info(
                    "No photometric cross-match data available. The pipeline queries "
                    "SDSS PhotoObj for PBH candidates when online."
                )


if __name__ == "__main__":
    main()
