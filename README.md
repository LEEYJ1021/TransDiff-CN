# Knowledge-Diffusion Trajectory Diagnostics in Chinese Transportation R&D (2007–2023)
### A Three-Step Bayesian–Empirical Pipeline for Distinguishing Pathway Reconfiguration from Output Expansion

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyMC](https://img.shields.io/badge/PyMC-5.x-orange)](https://www.pymc.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This repository contains the full analysis pipeline accompanying the paper:

> **"Pathway Reconfiguration, Not Output Expansion: Forecasting How Infrastructure Mega-Programs Restructure Knowledge-Diffusion Trajectories in National R&D Systems"**

The study asks whether large-scale infrastructure programs alter the *trajectories* through which academic knowledge diffuses into industrial innovation—or merely expand output volumes within pre-existing pathways. This distinction is critical for technology forecasting: conventional indicators (publication counts, patent grants, R&D expenditure) detect output expansion but are systematically blind to pathway reconfiguration.

Using patent-citation intensity as a codified-transfer proxy, the pipeline tests ten pre-registered hypotheses across 33 Chinese transportation research institutions (2007–2023, 528 institution-year observations), identifying a structural break coinciding with the Belt and Road Initiative (BRI) launch in 2013—a break absent for three contemporaneous policy events—and documenting rapid post-2013 institutional convergence (σ: 0.47 → 0.11; half-life ≈ 3.1 years).

The pipeline is organized into **three sequential steps**:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_btm_nlp_v6.py` | NLP feature engineering + Bayesian Transition Model |
| 2 | `step2_empirical_tra_v4.py` | Empirical hypothesis testing (H1–H10) |
| 3 | `step3_supplementary_v2.py` | Robustness checks, convergence diagnostics, and governance dashboard |

---

## Repository Structure

```
.
├── data/
│   └── scholarly_works.csv           # Input dataset (see Data section below)
├── step1_btm_nlp_v6.py               # Step 1: NLP pipeline + Bayesian Transition Model
├── step2_empirical_tra_v4.py         # Step 2: Mundlak–Chamberlain RE + hypothesis tests
├── step3_supplementary_v2.py         # Step 3: Convergence clubs + robustness + dashboard
├── outputs/
│   ├── btm_nlp_v6/                   # Step 1 outputs (figures, MCMC traces, CSVs)
│   ├── empirical_tra_v4/             # Step 2 outputs (regression tables, figures, JSON)
│   └── supplementary_v2/             # Step 3 outputs (convergence, sensitivity, dashboard)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Input Data

**File:** `data/scholarly_works.csv`

A bibliometric dataset of scholarly works affiliated with **33 major Chinese research institutions** active in transportation-related research, covering publication years 2007–2023. The dataset was exported from [Lens.org](https://www.lens.org) and contains the following key columns:

| Column | Description |
|--------|-------------|
| `Lens ID` | Unique paper identifier |
| `Title` | Paper title |
| `Publication Year` | Year of publication (2007–2023) |
| `Abstract` | Full abstract text (truncated to 800 characters in NLP preprocessing) |
| `Keywords` | Author-supplied keywords |
| `Fields of Study` | Discipline tags (semicolon-separated) |
| `Author/s` | Author list (semicolon-separated) |
| `Institution` | Corresponding or primary institution name |
| `Citing Patents Count` | Number of patents citing this paper (**primary outcome variable**) |
| `Citing Works Count` | Number of academic works citing this paper |
| `Is Open Access` | Boolean: open-access status |
| `References` | Reference list (semicolon-separated) |
| `Funding` | Funding acknowledgement text |

> **Note:** Raw export contains approximately 13,350 rows and 33 columns. After year filtering (2007–2023) and exclusion of rows with missing patent-citation data, the working dataset contains **11,271 papers** across **33 institutions**.

### Institutions Covered

The 33 institutions span five institutional-type categories and six geographic macro-regions of China, ensuring adequate representation of absorptive capacity heterogeneity:

| Type | Count | Example Institutions |
|------|-------|----------------------|
| C9 League (elite research universities) | 18 | Tsinghua University, Peking University, Zhejiang University |
| Transport-specialized universities | 5 | Beijing Jiaotong University, Tongji University, Southwest Jiaotong University, Wuhan University of Technology, Central South University |
| National research institutes | 2 | Chinese Academy of Sciences, University of Chinese Academy of Sciences |
| C7 League (national defense universities) | 2 | Beihang University, Beijing Institute of Technology |
| Teaching-focused universities | 6 | Shenzhen University, Soochow University, Shanghai University, Beijing University of Technology |

---

## Step 1 — NLP-Augmented Bayesian Transition Model (`step1_btm_nlp_v6.py`)

### What it does

This step constructs all NLP-derived covariates and estimates Bayesian models of knowledge-transfer state dynamics.

**Stage 1 — Text Processing and Feature Engineering**

1. Cleans and encodes title + abstract (truncated to 800 characters) + keywords using `sentence-transformers/paraphrase-MiniLM-L6-v2` (Siamese BERT architecture). Falls back to TF-IDF + SVD if the transformer model is unavailable.
2. Applies K-Means clustering with Gap Statistic selection (minimum K = 5) to identify transport research sub-domains. Topics are labeled via c-TF-IDF, PMI scoring, and a 10-category transport taxonomy with guaranteed cosine-similarity matching.
3. Constructs four institution-level spatial weight matrices:
   - `W_geo`: Geographic inverse-distance decay based on institutional geocoordinates
   - `W_type`: Institutional-type homophily (same organizational class = higher weight)
   - `W_region`: Shared administrative macro-region membership
   - `W_text`: Pairwise cosine similarity between institutional BERT embedding centroids
4. All matrices are row-normalized. `W_text` is assigned weight 0 in the composite `W_combo` due to negligible discriminative power (discrimination ratio = 0.002) at this sample size.

**Stage 2 — Institution–Year Panel Construction**

Aggregates paper-level NLP features into an institution × year panel (N = 528 observations). Four covariates are constructed:

| Covariate | Formula | Interpretation |
|-----------|---------|----------------|
| `tech_prox` | cos(TF-IDF centroid, patent frontier vector) | Baseline absorption probability |
| `research_novelty` | 1 − cos(BERT_t, BERT rolling 3yr mean) | Semantic departure from prior trajectory |
| `topic_entropy` | −Σ π_k log(π_k) over 5 topic clusters | Portfolio diversification |
| `intra_cohesion` | Mean pairwise BERT cosine similarity within institution-year | Internal semantic coherence |

**Stage 3 — Bayesian Models**

| Model | Specification | Purpose |
|-------|---------------|---------|
| M1 | Hierarchical Negative Binomial with geographic spatial lag | Baseline spatial diffusion |
| M2 | Bayesian Markov Transition Model (states: Low / Medium / High) | Performance state dynamics |
| M3 | Spatiotemporal NegBin with time-varying ρ(t) via Gaussian random walk | Temporal spillover evolution |
| M4v2 | NLP × Spatial Interaction NegBin (four channels + two interaction terms) | NLP-spatial channel integration |
| M5 | Bayesian mediation (C9 → NLP features → patent citations) | Indirect effect estimation |
| M6 | Multi-state degradation / reliability analysis (CTMC) | Long-run state persistence |

Model comparison uses LOO-CV and WAIC via ArviZ. MCMC settings: 2,000 draws, 2,000 tuning steps, 4 chains, `target_accept = 0.90`.

### Key outputs

| File | Description |
|------|-------------|
| `panel_data_v6.csv` | Institution × year panel with all NLP covariates (primary input for Steps 2–3) |
| `W_geo.npy` / `W_type.npy` / `W_region.npy` / `W_text.npy` / `W_combo.npy` | Spatial weight matrices |
| `T_bayesian.csv` | Bayesian transition probability matrix (3 × 3) |
| `trace_M1.nc` … `trace_M5_mediation.nc` | ArviZ InferenceData traces (NetCDF format) |
| `full_results_v6.0.json` | Master results JSON |
| `all_posterior_estimates_v6.csv` | Posterior means, HDIs, and P(>0) for all model parameters |
| `Fig1_panel_overview_v6.0.png` … `FigZ_china_map_v6.0.png` | Publication-quality figures (9 panels) |
| `topic_descriptions_v6.json` | Topic labels, c-TF-IDF terms, and transport taxonomy assignments |

### Key findings (Step 1)

- **K = 5** sub-domains identified by Gap Statistic (Gap(5) = 1.6634 > Gap(4) = 1.6476); all assigned to transport taxonomy via guaranteed cosine-similarity matching.
- **Moran's I** (permutation test, 4,993 resamples): I = −0.278, z = −2.32, p = 0.018 — negative spatial autocorrelation, indicating competitive specialization rather than passive imitation among proximate institutions.
- **ΔELPD(M4v2 − M1) = +4.10** (SE = 2.94) — marginal predictive improvement from NLP-spatial interaction; consistent with the 9.1% imputation rate compressing discriminative signal.
- **Steady-state distribution** (M2 Markov model): Low = 0.308, Medium = 0.409, High = 0.283.
- **Reliability** (M6 CTMC): Availability A = 0.277; Mean Time to Failure = 4.93 yr; Mean Time to Recovery = 5.30 yr.
- All MCMC chains converged: max R̂ < 1.006 across M1–M4v2.

---

## Step 2 — Empirical Hypothesis Testing (`step2_empirical_tra_v4.py`)

### What it does

Tests **ten pre-registered hypotheses** using the institution-year panel produced in Step 1, organized across three theoretical layers: policy regime change, absorptive capacity mechanisms, and spatial differentiation dynamics.

### Hypothesis register

| ID | Hypothesis | Mechanism | Empirical Test | Result |
|----|------------|-----------|----------------|--------|
| H1 | BRI 2013 produces a structural break in patent-citation intensity; MiC2025, NEV 2017, and COVID-19 do not | Policy trajectory switch | Chow–permutation + Bai–Perron | **Strongly supported** (F = 8.03***) |
| H2 | Tech-proximity elasticity differs by institution type | Heterogeneous absorptive capacity | Within-FE interaction | Partially supported |
| H3 | Text-similarity spatial channel evolves post-MiC2025 | Relational proximity shift | Pre/post Pearson correlation | Partially supported (p = 0.065) |
| H4 | Eastern institutions disproportionately benefit from BRI | Regional policy heterogeneity | Difference-in-differences | Rejected (convergence artifact) |
| H5 | Inverted-U relationship between research novelty and patent citations | Optimal recombination | Quadratic MC-RE + IV + SIMEX | **Supported** (β² = −0.763**; optimum = 0.472) |
| H6 | High-state persistence > low-state persistence (Matthew effect) | Desorptive capacity compounding | Markov transition + HMM | Partially supported |
| H7 | Post-BRI β-convergence in institutional patent-citation performance | Interface-driven equalization | σ-OLS + β-regression + Phillips–Sul | **Strongly supported** (β = −0.895***; half-life = 3.1 yr) |
| H8 | Topic entropy mediates C9 status → patent citations | Portfolio breadth channel | Bayesian mediation (ACME) | Partially supported (Sobel p = 0.297) |
| H9 | Open-access share amplifies patent-citation yield | Access-barrier removal | MC-RE + IPTW + Journal FE | **Strongly supported** (β = +0.326***) |
| H10 | Institutional prestige rank moderates novelty–citation curvature | Rank-conferred network access | MC-RE interaction | Rejected (β = +0.143, n.s.) |

### Estimation approach

- **Primary estimator**: Mundlak–Chamberlain Correlated Random Effects (MC-RE), augmenting conventional random effects with institution-level means of time-varying regressors. Asymptotically equivalent to fixed effects while retaining time-invariant institutional characteristics (Bellemare & Millimet, 2025; Wooldridge, 2025).
- **Robustness estimators**: Within Fixed Effects (entity-demeaning), Poisson QMLE, First Differences, sub-period splits.
- **Structural break identification**: Chow (1960) tests at four candidate policy dates, supplemented by Bai–Perron (1998) multiple-break search and permutation null distributions (4,993 resamples; ±2-year exclusion windows).
- **Spatial modeling**: Bayesian spatial negative-binomial (SAR, SEM, SDM specifications) using composite proximity matrix W_combo.
- **Convergence diagnostics**: Phillips–Sul (2007) log-t test + club-clustering algorithm; segmented OLS on σ-dispersion; quantile-β regression (Koenker–Bassett).
- **OOS validation**: 5-fold walk-forward cross-validation with expanding windows; temporal-split deviation strategy; GBM ensemble with Bayesian model averaging across pre/post-BRI regimes.
- **Multiple testing**: Holm–Bonferroni correction across H1–H10; Westfall–Young FWER correction within the H1 multi-event family.

### Key outputs

| File | Description |
|------|-------------|
| `Table1_descriptives.csv` | Summary statistics (N = 528) |
| `Table2_MC_coefs.csv` / `Table2_latex.tex` | MC-RE regression table (6 nested specifications) |
| `Table3_hypothesis_summary.csv` / `Table3_latex.tex` | Hypothesis register with empirical outcomes |
| `empirical_master_results_v4.json` | Full results JSON |
| `Fig1_H1_policy_shocks.png` … `Fig11_diagnostics.png` | Publication-quality figures (11 panels) |
| `OOS_model_comparison.csv` / `OOS_walkforward.csv` | Walk-forward predictive performance |
| `H1_chow_tests.csv` … `H8_mediation_3strategy.csv` | Hypothesis-specific output tables |

### Key findings (Step 2)

- **H1 (Structural Break)**: BRI 2013 is the only policy event producing a significant structural break in patent-citation intensity (Chow F = 8.03, ×3.95 vs. permutation placebo mean, p < 0.001; Bai–Perron BIC-minimizing break: 2013, 95% CI [2012–2014]). MiC2025 (F = 2.14), NEV (F = 1.93), and COVID-19 (F = 2.67) all fall below the Chow critical threshold and are non-significant under Westfall–Young correction. Pre-BRI event-study coefficients (τ = −5 to −1) are jointly insignificant (F = 0.412, p = 0.841), validating parallel pre-trends.
- **H5 (Novelty Optimum)**: Estimated optimal research novelty = 0.472 (95% bootstrap CI: [0.38, 0.56]); current system mean = 0.239, implying approximately 20–22% unrealized patent-citation yield addressable through portfolio reallocation without additional funding.
- **H7 (Convergence)**: σ-convergence slope = −0.028*** (full period); BRI→COVID sub-period slope = −0.034** (the sharpest compression phase); β-convergence = −0.895***; Phillips–Sul log-t b = 1.249*** (single convergence club confirmed, n = 33). Implied post-2013 half-life ≈ 3.1 years.
- **H9 (Open Access)**: OA share positively associated with patent-citation yield across all six MC-RE specifications (β ≈ +0.326***; CV = 0.329; IPTW-ATT = 0.290). Interaction with technological proximity significant (oa_x_tech = +7.06, p = 0.003), indicating that OA amplification is strongest when institutional research portfolios are closely aligned with the industrial patent frontier.
- **OOS Performance**: Walk-forward Spearman ρ = 0.313 (rank-stable across folds); NDCG ≥ 0.811; Precision@10 = 0.50. Level R² is uniformly negative (mean = −1.037), confirming that the models reliably recover institutional rankings but do not predict absolute citation magnitudes—consistent with structural non-stationarity introduced by the 2013 regime shift and the 2020 COVID disruption. Policy implications are therefore grounded in structural parameter estimates rather than point forecasts.

---

## Step 3 — Supplementary Analysis (`step3_supplementary_v2.py`)

### What it does

Provides robustness checks, convergence club diagnostics, and an integrated governance dashboard.

| Module | Method | Key Result |
|--------|--------|------------|
| A | Chow F-test with permutation null (4,993 resamples; ±2-year exclusion) | BRI: F = 8.03, ×3.95 vs. placebo mean, p < 0.001 |
| B | Walk-forward OOS: Spearman ρ, NDCG, QWK | Pre-COVID ρ = 0.295; NDCG ≥ 0.811 across all folds |
| C | Domain-threshold sensitivity sweep (β_OA, β_tech) | OA share CV = 0.33 (stable); tech proximity CV = 1.01 (unstable — reflects measurement artifact for narrow-domain institutions) |
| D | σ-convergence segmented OLS | BRI→COVID slope = −0.034**; Post-COVID slope = −0.033** |
| E | Pairwise spatial differentiation | ρ(distance, \|Δpat\|) = −0.097** (larger citation gaps among proximate institution pairs) |
| SUP-6 | Phillips–Sul log-t test + club-clustering algorithm | Global b = 1.249*** (transitional convergence); single club confirmed (n = 33) |
| SUP-8 | Integrated results dashboard | All panels computed directly from panel data |
| SUP-9 | Narrative strategy guide | Three submission-type strategies (journal-agnostic) |

### Bug fixes applied

- **BUG-FIX-1** (`phillips_sul_logt`): Replaced `res.params["x1"]` with `res.params[1]` (integer index) to resolve `statsmodels` OLS + NumPy array incompatibility when the design matrix is passed as a plain array rather than a DataFrame.
- **BUG-FIX-2** (`form_clubs`): Replaced `np.ix_()` with direct fancy indexing `mat[np.array(idx, dtype=int), :]` to resolve index type mismatch during club sub-matrix extraction.

### Key outputs

```
outputs/supplementary_v2/
├── A_event_study_chow.csv          # Chow F statistics + permutation p-values (4 policy events)
├── B_oos_walk_forward.csv          # Walk-forward CV metrics (5 folds)
├── C_domain_sensitivity.csv        # β coefficient CV across threshold definitions
├── D_sigma_convergence.csv         # Segmented σ-OLS results by sub-period
├── E_spatial_pairs.csv             # Pairwise distance–citation gap estimates
├── SUP6_convergence_clubs_v3.csv   # Phillips–Sul log-t statistics + club assignments
├── SUP6_convergence_clubs_v3.png   # Convergence club visualization
├── SUP8_results_dashboard_v2.png   # Integrated governance dashboard (all 10 hypotheses)
└── SUP9_narrative_strategy.json    # Submission narrative strategies
```

---

## Forecasting Toolkit: Transferability

A central contribution of this repository is a **replicable diagnostic toolkit** for detecting knowledge-diffusion trajectory switches in national R&D systems. The toolkit requires only institutional publication metadata and patent-citation records available from open bibliographic sources (Lens.org, OpenAlex, Dimensions), making it transferable beyond the Chinese transport context.

| Diagnostic | Script Location | Transfer Requirement |
|------------|-----------------|----------------------|
| Permutation-Chow structural-break test | `step3_supplementary_v2.py` Module A | T ≥ 10 pre-break observations; policy dates known a priori |
| Phillips–Sul log-t convergence test | `step3_supplementary_v2.py` SUP-6 | T ≥ 15; balanced or near-balanced panel |
| NLP novelty + proximity indices | `step1_btm_nlp_v6.py` Stage 2 | Abstract text + keyword metadata available |
| Bayesian spatial NegBin | `step1_btm_nlp_v6.py` Stage 3 | Institution geocoordinates; institutional-type classification |
| Walk-forward rank validation | `step2_empirical_tra_v4.py` OOS module | Minimum 3 forecast folds; Spearman ρ and NDCG as primary metrics |

**Scope conditions for generalization:** The mechanism-specificity finding (interface-creating programs produce structural breaks; funding-scale programs do not) is most likely to replicate in national R&D systems where (a) initial cross-institutional dispersion σ > 0.3, (b) the candidate program embeds formal university–industry collaboration mechanisms at the project level, and (c) patent-citation indexing coverage is sufficient for reliable measurement (estimated minimum: ≥ 500 patent-cited publications per institutional panel).

---

## Setup and Requirements

### Installation

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn tqdm
pip install pymc pytensor arviz
pip install sentence-transformers  # optional; TF-IDF + SVD fallback used if absent
pip install lightgbm               # optional; sklearn GBM fallback used if absent
```

**Python ≥ 3.9** is required. All three scripts are self-contained and can be run independently, provided the outputs of earlier steps are available in the expected directories.

### Path configuration

Each script exposes path constants at the top of the file. Update these before running:

```python
# step1_btm_nlp_v6.py
FILE_PATH = "data/scholarly_works.csv"    # input CSV
OUT_DIR   = "outputs/btm_nlp_v6"          # Step 1 output directory

# step2_empirical_tra_v4.py
V6_OUT  = "outputs/btm_nlp_v6"           # Step 1 outputs (panel, spatial matrices)
RAW_CSV = "data/scholarly_works.csv"     # original CSV (for sub-domain analysis)
OUT_DIR = "outputs/empirical_tra_v4"

# step3_supplementary_v2.py
V6_OUT    = "outputs/btm_nlp_v6"
STEP2_OUT = "outputs/empirical_tra_v4"
OUT_DIR   = "outputs/supplementary_v2"
```

### Execution order

```bash
python step1_btm_nlp_v6.py       # ~30–60 minutes depending on hardware (MCMC sampling)
python step2_empirical_tra_v4.py
python step3_supplementary_v2.py
```

> **Performance note:** Step 1 caches BERT embeddings and K-Means results in `outputs/btm_nlp_v6/nlp_cache/`. Subsequent runs load the cache automatically and skip recomputation, reducing runtime to approximately 5–10 minutes for Steps 2 and 3.

---

## Reproducibility

- `np.random.seed(42)` and `random_seed=42` are set consistently throughout all scripts.
- **MCMC settings:** 2,000 draws, 2,000 tuning steps, 4 chains, `target_accept = 0.90`. All chains converged with max R̂ < 1.006 across M1–M4v2.
- **Imputation:** 9.1% of institution-year cells (48 observations, concentrated in teaching universities during 2007–2010) were imputed using prior-year institution-specific distributions. Imputed rows are flagged via the `imputed_share` column in `panel_data_v6.csv`. The `imputed_share` contamination control is statistically insignificant across all six MC-RE specifications (|t| < 1.12), confirming that imputation does not systematically bias primary estimates. Complete-case analysis (n = 480) replicates all primary findings with negligible attenuation.
- **W_text exclusion:** The text-similarity spatial weight matrix is assigned weight 0 in `W_combo` due to a discrimination ratio of 0.002, indicating negligible informational gain relative to a uniform weight structure at this sample size.
- **Observational design:** All results are observational. The structural break, convergence patterns, and governance-lever associations are consistent with the proposed theoretical mechanisms but do not constitute causal identification. The appropriate inferential claim is that the observed patterns are consistent with, but do not causally establish, the mechanism-specificity interpretation.

---

## Data Availability

Replication data, NLP pipeline code, permutation-Chow scripts, IPTW weighting routines, regime-switching ensemble specifications, and all Bayesian model files are publicly available at:

**[https://github.com/LEEYJ1021/TransDiff-CN](https://github.com/LEEYJ1021/TransDiff-CN)**

The input dataset (`scholarly_works.csv`) was exported from [Lens.org](https://www.lens.org) under standard academic access terms. Users should obtain their own export for replication; the pipeline will reproduce all results from any export matching the column schema described in the Input Data section.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
