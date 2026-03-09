# src/dashboard/app.py
"""Streamlit dashboard for exploring spectral anomalies."""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.features.color import spectrum_to_rgb, rgb_to_hex

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


@st.cache_data
def load_data():
    spectra = np.load(PROCESSED_DIR / "spectra.npy")
    metadata = pd.read_parquet(PROCESSED_DIR / "spectra_metadata.parquet")
    comparison = pd.read_parquet(RESULTS_DIR / "comparison.parquet")
    pca_components = np.load(RESULTS_DIR / "pca_components.npy")
    if_scores = np.load(RESULTS_DIR / "if_scores.npy")
    ae_scores = np.load(RESULTS_DIR / "ae_scores.npy")
    ocsvm_scores = np.load(RESULTS_DIR / "ocsvm_scores.npy")
    dagmm_scores = np.load(RESULTS_DIR / "dagmm_scores.npy")
    if_stability_std = np.load(RESULTS_DIR / "if_stability_std.npy")
    categories = list(np.load(RESULTS_DIR / "categories.npy"))
    wavelength_grid = np.linspace(3800, 9200, spectra.shape[1])

    param_counts = {}
    param_path = RESULTS_DIR / "param_counts.json"
    if param_path.exists():
        with open(param_path) as f:
            param_counts = json.load(f)

    cond_ae_scores_path = RESULTS_DIR / "conditional_ae_scores.npy"
    cond_ae_pvalues_path = RESULTS_DIR / "conditional_ae_pvalues.npy"
    cond_ae_scores = np.load(cond_ae_scores_path) if cond_ae_scores_path.exists() else np.zeros(len(if_scores))
    cond_ae_pvalues = np.load(cond_ae_pvalues_path) if cond_ae_pvalues_path.exists() else np.ones(len(if_scores))

    return (spectra, metadata, comparison, pca_components,
            if_scores, ae_scores, ocsvm_scores, dagmm_scores,
            if_stability_std, categories, param_counts, wavelength_grid,
            cond_ae_scores, cond_ae_pvalues)


def plot_spectrum(spectra, wavelength_grid, idx, title="Spectrum"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelength_grid, y=spectra[idx],
        mode="lines", name="Observed", line=dict(color="steelblue", width=1),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (Angstroms)",
        yaxis_title="Normalized Flux",
        height=350,
    )
    return fig


def main():
    st.set_page_config(page_title="SDSS Spectral Anomalies", layout="wide")
    st.title("SDSS Spectral Anomaly Explorer")

    (spectra, metadata, comparison, pca_components,
     if_scores, ae_scores, ocsvm_scores, dagmm_scores,
     if_stability_std, categories, param_counts, wl,
     cond_ae_scores, cond_ae_pvalues) = load_data()

    # Sidebar with parameter counts
    st.sidebar.header("Model Parameters")
    for model_name, count in param_counts.items():
        st.sidebar.metric(model_name, f"{count:,}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Anomaly Browser", "Model Comparison", "PCA Explorer", "Model Stability", "Conformal P-Values"])

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
            order = np.argsort(cond_ae_pvalues)  # smallest p-value first

        top_indices = order[:top_n]
        table_data = metadata.iloc[top_indices].copy()
        table_data["if_score"] = if_scores[top_indices]
        table_data["ae_score"] = ae_scores[top_indices]
        table_data["ocsvm_score"] = ocsvm_scores[top_indices]
        table_data["dagmm_score"] = dagmm_scores[top_indices]
        table_data["cond_ae_score"] = cond_ae_scores[top_indices]
        table_data["cond_ae_pvalue"] = cond_ae_pvalues[top_indices]
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
            st.plotly_chart(
                plot_spectrum(spectra, wl, spectrum_idx, f"Spectrum: {meta.get('filename', spectrum_idx)}"),
                use_container_width=True,
            )
            # Perceived star color swatch
            r, g, b = spectrum_to_rgb(wl, spectra[spectrum_idx])
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

    # --- Tab 2: Model Comparison ---
    with tab2:
        st.subheader("Model Score Comparison (Scatter Matrix)")
        scatter_df = pd.DataFrame({
            "IF Score": if_scores,
            "AE Score": ae_scores,
            "OC-SVM Score": ocsvm_scores,
            "DAGMM Score": dagmm_scores,
        })
        fig = px.scatter_matrix(
            scatter_df,
            dimensions=["IF Score", "AE Score", "OC-SVM Score", "DAGMM Score"],
            opacity=0.3, height=800,
            title="Pairwise Model Score Comparisons",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "n_models_agreed" in comparison.columns:
            all_agreed = int((comparison["n_models_agreed"] == 4).sum())
            st.metric("Spectra flagged by ALL 4 models (top 100)", all_agreed)
        elif "agreed" in comparison.columns:
            n_agreed = int(comparison["agreed"].sum())
            st.metric("Spectra flagged by BOTH models (top 100)", n_agreed)

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
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=cond_ae_pvalues, nbinsx=50, name="p-values",
                                   marker_color="steelblue", opacity=0.75))
        fig.update_layout(xaxis_title="Conformal p-value", yaxis_title="Count",
                          title="Uniform = well-calibrated; spike near 0 = anomalies", height=400)
        st.plotly_chart(fig, use_container_width=True)

        alpha = st.slider("Flag anomalies at p-value <=", 0.01, 0.20, 0.05, step=0.01)
        n_flagged = int((cond_ae_pvalues <= alpha).sum())
        st.metric(f"Spectra flagged at alpha={alpha}", n_flagged)

        flagged_idx = np.where(cond_ae_pvalues <= alpha)[0]
        if len(flagged_idx) > 0:
            flagged_df = metadata.iloc[flagged_idx].copy()
            flagged_df["cond_ae_score"] = cond_ae_scores[flagged_idx]
            flagged_df["cond_ae_pvalue"] = cond_ae_pvalues[flagged_idx]
            st.dataframe(flagged_df.sort_values("cond_ae_pvalue").reset_index(drop=True),
                         use_container_width=True)


if __name__ == "__main__":
    main()
