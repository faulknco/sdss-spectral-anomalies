# src/dashboard/app.py
"""Streamlit dashboard for exploring spectral anomalies."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

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
    wavelength_grid = np.linspace(3800, 9200, spectra.shape[1])
    return spectra, metadata, comparison, pca_components, if_scores, ae_scores, wavelength_grid


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

    spectra, metadata, comparison, pca_components, if_scores, ae_scores, wl = load_data()

    tab1, tab2, tab3 = st.tabs(["Anomaly Browser", "Model Comparison", "PCA Explorer"])

    # --- Tab 1: Anomaly Browser ---
    with tab1:
        st.subheader("Top Anomalies")
        sort_by = st.selectbox("Sort by", ["Combined", "Isolation Forest", "Autoencoder"])
        top_n = st.slider("Show top N", 10, 500, 100)

        if sort_by == "Combined":
            order = np.argsort(-(if_scores + ae_scores))
        elif sort_by == "Isolation Forest":
            order = np.argsort(-if_scores)
        else:
            order = np.argsort(-ae_scores)

        top_indices = order[:top_n]
        table_data = metadata.iloc[top_indices].copy()
        table_data["if_score"] = if_scores[top_indices]
        table_data["ae_score"] = ae_scores[top_indices]
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
            col1, col2, col3 = st.columns(3)
            col1.metric("IF Score", f"{if_scores[spectrum_idx]:.4f}")
            col2.metric("AE Score", f"{ae_scores[spectrum_idx]:.4f}")
            col3.metric("Spectral Type", meta.get("subclass", "N/A"))

    # --- Tab 2: Model Comparison ---
    with tab2:
        st.subheader("Isolation Forest vs Autoencoder Scores")
        scatter_df = pd.DataFrame({
            "IF Score": if_scores,
            "AE Score": ae_scores,
            "Spectral Type": metadata.get("subclass", "unknown"),
        })
        fig = px.scatter(
            scatter_df, x="IF Score", y="AE Score", color="Spectral Type",
            opacity=0.5, height=600,
            title="Model Agreement: IF vs AE Anomaly Scores",
        )
        st.plotly_chart(fig, use_container_width=True)

        n_agreed = int(comparison["agreed"].sum()) if "agreed" in comparison.columns else 0
        st.metric("Spectra flagged by BOTH models (top 100)", n_agreed)

    # --- Tab 3: PCA Explorer ---
    with tab3:
        st.subheader("PCA Latent Space")
        color_by = st.selectbox("Color by", ["IF Score", "AE Score", "Spectral Type"])

        pca_df = pd.DataFrame({
            "PC1": pca_components[:, 0],
            "PC2": pca_components[:, 1],
            "PC3": pca_components[:, 2] if pca_components.shape[1] > 2 else 0,
            "IF Score": if_scores,
            "AE Score": ae_scores,
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


if __name__ == "__main__":
    main()
