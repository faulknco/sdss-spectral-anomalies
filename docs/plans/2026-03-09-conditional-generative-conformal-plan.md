# Conditional Generative + Conformal Anomaly Detection Plan

**Goal:** Add a novel anomaly detector that scores spectra by how unusual they are *given* their stellar parameters, then calibrate those scores into statistically meaningful p-values.

**Why this direction:** The current pipeline is strong on unconditional outlier detection, but it mostly asks whether a spectrum is rare in the full sample. It does not ask whether a spectrum is strange for its expected stellar class or inferred atmospheric parameters. That is the scientific gap worth attacking.

## Strategic Question

Can we detect spectra that are anomalous relative to their expected stellar physics rather than only anomalous relative to the population average?

## Current State

- The pipeline trains unconditional models on resampled, median-normalized spectra.
- The SDSS query already pulls `elodieTEff`, `elodieLogG`, and `elodieFeH`.
- Those conditioning variables are not currently used during preprocessing, scoring, or ranking.
- Rank aggregation is heuristic. It does not provide a calibrated false alarm interpretation.

Relevant code:
- [src/data/download.py](./src/data/download.py)
- [src/data/preprocess.py](./src/data/preprocess.py)
- [src/run_pipeline.py](./src/run_pipeline.py)
- [src/models/compare.py](./src/models/compare.py)

## Proposed Method

### Core Model

Train a conditional generative model for

`p(flux | wavelength grid, Teff, log g, [Fe/H], subclass, S/N)`

and use two anomaly signals:

1. `conditional_nll`
The negative log-likelihood under the conditional model.

2. `structured_residual_score`
A residual score that upweights localized line-profile mismatches over broad continuum mismatch.

The key idea is that a hot A-type spectrum should not be compared to the same reference distribution as a cool M dwarf. The model should learn what is normal for each region of stellar-parameter space.

### Calibration Layer

Apply split conformal calibration to the final anomaly score to produce:

- `anomaly_p_value`
- a threshold for a target false discovery rate or tail probability

This does not prove astrophysical novelty. It does provide a finite-sample guarantee that the calibration set controls how extreme a score must be before we flag it.

## Model Options

### Option A: Conditional Normalizing Flow

**Recommendation:** Start here.

Pros:
- Direct likelihood for each spectrum
- Clean conditional story
- Strong fit with the "weird given metadata" objective

Cons:
- More engineering than an autoencoder
- Sensitive to preprocessing and tail behavior

Implementation sketch:
- Input: normalized flux vector
- Conditioner: MLP over `Teff`, `log g`, `[Fe/H]`, `sn_median`, subclass embedding
- Backbone: RealNVP or masked autoregressive flow over a PCA-compressed spectrum
- Score: conditional NLL + residual score

### Option B: Conditional Denoising Autoencoder

Pros:
- Faster to prototype in the current codebase
- Reuses existing PyTorch training structure

Cons:
- Reconstruction error is a weaker density proxy
- Easier to overfit broad continuum than narrow anomalies

Implementation sketch:
- Concatenate a learned metadata embedding to latent and decoder inputs
- Train with masking or denoising corruption
- Score with weighted residuals, not plain MSE

### Option C: Functional / Signature Isolation Forest Baseline

Pros:
- More novel than plain PCA + IF
- Good fallback if generative modeling takes longer than expected
- Strong mathematical literature for curves and path geometry

Cons:
- Less physically interpretable than the conditional route
- Still not a calibrated probabilistic model on its own

## Recommended Build Order

### Phase 1: Metadata-Aware Baseline

Deliver a cheap baseline before the full generative model.

Steps:
- Persist `elodieTEff`, `elodieLogG`, `elodieFeH` into processed metadata.
- Build local reference neighborhoods in parameter space.
- Score each spectrum by distance to its k nearest neighbors in `(Teff, log g, [Fe/H])` plus weighted flux residual distance.

Expected value:
- Fast check that conditional scoring adds signal before we invest in flows.

### Phase 2: Conditional Autoencoder

Use this as the first trainable conditional model.

Steps:
- Add `src/models/conditional_autoencoder.py`
- Feed metadata embeddings into encoder and decoder
- Replace plain MSE with a line-aware residual loss
- Save `conditional_ae_scores.npy`

Success criterion:
- Top anomalies are less dominated by spectral subclass edges and low-S/N artifacts than the current AE.

### Phase 3: Split Conformal Calibration

Steps:
- Split training data into train and calibration partitions
- Fit model on train
- Compute scores on calibration set
- Convert each test score into a conformal p-value
- Save `conditional_ae_pvalues.npy`

Success criterion:
- The dashboard can rank by both raw score and calibrated p-value.

### Phase 4: Conditional Flow

Steps:
- Add `src/models/conditional_flow.py`
- Start on PCA-compressed spectra to reduce dimensionality
- Compare conditional NLL against the conditional autoencoder score

Success criterion:
- Better ranking stability and better concentration of interesting spectra in the top 50 to 100 reviewed items.

## Data and Preprocessing Changes

### Metadata Retention

Carry these fields from query output into `spectra_metadata.parquet`:

- `elodieTEff`
- `elodieLogG`
- `elodieFeH`
- `snmedian`
- `subclass`

### Residual-Aware Preprocessing

Add optional variants:

- continuum-normalized spectra
- derivative spectra `dF/dlambda`
- line-window masks around Balmer, Ca II, Na D, and molecular bands

Reason:
- Many astrophysically interesting anomalies are local line-shape failures, not global continuum shifts.

## Scoring Formula

Start with:

`final_score = z(conditional_nll) + 0.5 * z(line_residual_score) + 0.25 * z(stability_std)`

where:

- `conditional_nll` measures mismatch to expected spectrum
- `line_residual_score` emphasizes local astrophysical features
- `stability_std` comes from multi-seed or perturbation runs and penalizes fragile detections

Then conformalize `final_score` on a held-out calibration set.

## Monte Carlo: Where It Helps and Where It Does Not

### Use Monte Carlo For

- perturbing flux by inverse-variance noise if available
- perturbing continuum normalization choices
- small radial-velocity or wavelength-shift perturbations
- estimating anomaly persistence under nuisance uncertainty

### Do Not Use Monte Carlo As The Main Detector

A brute-force Monte Carlo search over anomaly heuristics is unlikely to be novel enough on its own. The stronger research contribution is conditional modeling plus calibrated ranking. Monte Carlo should support robustness, not define the method.

## Evaluation Plan

Because there is no complete ground-truth anomaly catalog, use a mixed protocol.

### Quantitative

- rank stability across seeds
- rank stability across preprocessing variants
- enrichment of top-ranked spectra with low-density regions in parameter space
- agreement and disagreement with current IF, OC-SVM, AE, and DAGMM models
- calibration diagnostics for conformal p-values

### Semi-Synthetic

Inject controlled anomalies into normal spectra:

- localized emission line insertion
- line broadening or splitting
- continuum tilt
- wavelength shift
- missing band segments

Measure retrieval at top-k and area under precision-recall for the injected cases.

### Human Review

For the top 100 spectra from each method, label:

- likely instrument or reduction artifact
- low-S/N nuisance
- astrophysically plausible oddity
- known rare subtype
- unclear

The winning method is the one that improves the fraction of plausible oddities per reviewed hour.

## Minimal Code Changes

### New Files

- `src/models/conditional_autoencoder.py`
- `src/models/conformal.py`
- `tests/test_conditional_autoencoder.py`
- `tests/test_conformal.py`

### Existing Files To Update

- `src/data/preprocess.py`
- `src/run_pipeline.py`
- `src/models/compare.py`
- `src/dashboard/app.py`

## First Milestone

Implement the smallest version that can fail or succeed quickly:

1. keep and standardize stellar-parameter metadata
2. add a metadata-conditioned autoencoder
3. add conformal calibration on a held-out set
4. expose raw score, p-value, and stability in the dashboard

This is enough to test the main hypothesis without committing to the heavier flow model.

## Success Metrics

- Top anomalies are less clustered at spectral class boundaries than the current unconditional models.
- Manual review finds a higher fraction of spectra with localized or physically interpretable oddities.
- Calibrated p-values remain stable across reruns and moderate preprocessing changes.
- Conditional model disagreement with unconditional models reveals distinct candidates rather than pure noise artifacts.

## Research Anchors

- MCLOF on SDSS stellar spectra: https://academic.oup.com/mnras/article/431/2/1800/1466626
- Functional Isolation Forest: https://proceedings.mlr.press/v101/staerman19a.html
- Signature Isolation Forest: https://arxiv.org/abs/2403.04405
- Conformalized Functional Data Outlier Detection: https://proceedings.mlr.press/v266/adams25b.html
- Testing for Outliers with Conformal p-values: https://arxiv.org/abs/2104.08279
- Astronomaly: https://arxiv.org/abs/2010.11202
- SpectraFM: https://arxiv.org/abs/2411.04750
- OmniSpectra: https://arxiv.org/abs/2601.15351
- SpecCLIP: https://arxiv.org/abs/2507.01939
- Autoencoder-based stellar-spectrum anomalies from March 4, 2026: https://arxiv.org/abs/2603.03734

## Recommendation

Do not start with brute-force Monte Carlo. Start with a conditional autoencoder plus conformal calibration as the fastest serious test of the idea. If it works, upgrade the density model to a conditional flow. If the conditional route stalls, ship a functional/signature isolation forest as the mathematically novel baseline.
