# Knowledge-Diffusion Trajectory Diagnostics in Chinese Transportation R&D (2007–2023)
### A Three-Step Bayesian–Empirical Pipeline for Distinguishing Pathway Reconfiguration from Output Expansion

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyMC](https://img.shields.io/badge/PyMC-5.x-orange)](https://www.pymc.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **v2.1 Update Note:** This version reconciles the pipeline documentation with the published manuscript's core-hypothesis structure (H1–H4), correcting a prior draft in which six exploratory checks had been inadvertently elevated to numbered confirmatory hypotheses (H2, H3, H4, H6, H8, H10 in the earlier README). Structural-break statistics, institutional counts, and the pre/post-BRI trend description have been corrected to match the manuscript reported in Sections 4.1–4.5. See [Update Log](#update-log-v20--v21) at the bottom of this file for a full diff.

---

## Overview

This repository contains the full analysis pipeline accompanying the paper:

> **"Pathway Reconfiguration, Not Output Expansion: Infrastructure Mega-Programs and the Restructuring of Knowledge-Diffusion Trajectories in National R&D Systems"**

The study asks whether large-scale infrastructure programs alter the *trajectories* through which academic knowledge diffuses into industrial innovation—or merely expand output volumes within pre-existing pathways. This distinction is critical for technology forecasting: conventional indicators (publication counts, patent grants, R&D expenditure) detect output expansion but are systematically blind to pathway reconfiguration.

Using patent-citation intensity as a codified-transfer proxy, the pipeline tests **four pre-registered core hypotheses** (H1–H4), **one robust auxiliary structural pattern** (competitive spatial specialization), and **six further pre-registered exploratory checks** (S1–S6) across 33 Chinese transportation research institutions (2007–2023, 528 institution-year observations). It identifies a structural break coinciding with the Belt and Road Initiative (BRI) launch in 2013 — a break absent for three contemporaneous policy events — and documents rapid post-2013 institutional convergence (σ: 0.47 → 0.11; half-life ≈ 3.1 years).

**Important framing note:** All results are observational. The core-vs-exploratory split reflects a pre-specified organizing criterion (H1–H4 test the paper's central mechanism-specificity claim; S1–S6 test auxiliary moderators of the same mechanism), fixed independently of which checks ultimately proved statistically significant — not a post-hoc reclassification driven by results.

The pipeline is organized into **three sequential steps**:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_btm_nlp_v6.py` | NLP feature engineering + Bayesian Transition Model |
| 2 | `step2_empirical_tra_v4.py` | Empirical hypothesis testing (H1–H4 core + S1–S6 exploratory) |
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

> **Note:** Raw export contains approximately 13,350 rows and 33 columns. After year filtering (2007–2023) and exclusion of rows with missing patent-citation data, the working dataset contains **11,271 papers** across **33 institutions**, aggregating to **528 institution-year observations** (94% of the theoretical 561-cell balanced panel; 33 cells excluded for zero patent-cited output).

### Institutions Covered

The 33 institutions span five institutional-type categories and six geographic macro-regions of China, ensuring adequate representation of absorptive capacity heterogeneity (Cohen & Levinthal, 1990):

| Type | Institution Count | Institution-Years (of 528) | Example Institutions |
|------|:---:|:---:|-----------------------|
| C9 League (elite research universities) | 20 | 334 | Tsinghua University, Peking University, Zhejiang University |
| Transport-specialized universities | 5 | 82 | Beijing Jiaotong University, Tongji University, Southwest Jiaotong University |
| National research institutes | 2 | 32 | Chinese Academy of Sciences, University of Chinese Academy of Sciences |
| C7 League (national defense universities) | 2 | 34 | Beihang University, Beijing Institute of Technology |
| Teaching-focused universities | 4 | 46 | Shenzhen University, Soochow University |
| **Total** | **33** | **528** | |

---

## Step 1 — NLP-Augmented Bayesian Transition Model (`step1_btm_nlp_v6.py`)

### What it does

This step constructs all NLP-derived covariates and estimates Bayesian models of knowledge-transfer state dynamics.

**Stage 1 — Text Processing and Feature Engineering**

1. Cleans and encodes title + abstract (truncated to 800 characters) + keywords using `sentence-transformers/paraphrase-MiniLM-L6-v2` (Siamese BERT architecture; Reimers & Gurevych, 2019). Falls back to TF-IDF + SVD if the transformer model is unavailable.
2. Applies K-Means clustering with Gap Statistic selection (minimum K = 5) to identify transport research sub-domains (Gap(5) = 1.6634 > Gap(4) = 1.6476).
3. Constructs four institution-level spatial weight matrices:
   - `W_geo`: Geographic inverse-distance decay based on institutional geocoordinates
   - `W_type`: Institutional-type homophily (same organizational class = higher weight)
   - `W_region`: Shared administrative macro-region membership
   - `W_text`: Pairwise cosine similarity between institutional BERT embedding centroids
4. All matrices are row-normalized. `W_text` is assigned weight 0 in the composite `W_combo` due to negligible discriminative power (discrimination ratio = 0.002) — this is reported as a null/negative supplementary finding (see S2 below), not omitted silently.

**Stage 2 — Institution–Year Panel Construction**

Aggregates paper-level NLP features into an institution × year panel (N = 528 observations). Four covariates are constructed:

| Covariate | Formula | Interpretation |
|-----------|---------|----------------|
| `tech_prox` | cos(TF-IDF centroid, patent frontier vector) | Baseline absorption probability |
| `research_novelty` | 1 − cos(BERT_t, BERT rolling 3yr mean) | Semantic departure from prior trajectory |
| `topic_entropy` | −Σ π_k log(π_k) over 5 topic clusters | Portfolio diversification |
| `intra_cohesion` | Mean pairwise BERT cosine similarity within institution-year | Internal semantic coherence |

**Stage 3 — Bayesian Models**

| Model | Specification | Purpose | Maps to Manuscript Section |
|-------|---------------|---------|------------------------------|
| M1 | Hierarchical Negative Binomial with geographic spatial lag | Baseline spatial diffusion | §4.4 |
| M2 | Bayesian Markov Transition Model (states: Low / Medium / High) | Performance state dynamics | §4.5, S4 |
| M3 | Spatiotemporal NegBin with time-varying ρ(t) via Gaussian random walk | Temporal spillover evolution | §4.4, S2 |
| M4v2 | NLP × Spatial Interaction NegBin (four channels + two interaction terms) | NLP-spatial channel integration | §4.5, S5 |
| M5 | Bayesian mediation (BRI-partner co-authorship → post-break intensity) | Indirect-effect estimation for the interface-exposure proxy | §4.1 |
| M6 | Multi-state degradation / reliability analysis (CTMC) | Long-run state persistence | S4 |

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

- **K = 5** sub-domains identified by Gap Statistic; all assigned to transport taxonomy via guaranteed cosine-similarity matching.
- **Moran's I** (permutation test, 999 relabelings + analytic z-test): I = −0.278, z = −2.32, p = 0.018 (permutation p = 0.021 for W_geo) — negative spatial autocorrelation, consistent with competitive specialization rather than passive imitation among proximate institutions. **Caution:** n = 33 institutions; expected Moran's I under randomness ≈ −0.031, so magnitude should be interpreted cautiously despite consistent direction across five weight-matrix specifications.
- **W_text discrimination ratio = 0.002** — reported as the empirical basis for S2 (partial support, Δr = +0.216, p = 0.065), not as a silently discarded feature.
- **Steady-state distribution** (M2 Markov model): Low = 0.308, Medium = 0.409, High = 0.283.
- All MCMC chains converged: max R̂ < 1.006 across M1–M4v2.

---

## Step 2 — Empirical Hypothesis Testing (`step2_empirical_tra_v4.py`)

### What it does

Tests **four pre-registered core hypotheses** (H1–H4), reports **one robust auxiliary structural pattern** (pre-registered as a directional prediction rather than a numbered hypothesis), and transparently reports **six further pre-registered exploratory checks** (S1–S6). All ten items were specified together with their estimators and significance thresholds in a single analysis protocol drafted prior to finalizing results (protocol in this repository). The core/exploratory distinction reflects each item's centrality to the paper's mechanism-specificity claim, fixed independently of which items proved statistically significant.

### Core hypothesis register (H1–H4)

| ID | Hypothesis | Mechanism | Empirical Test | Result |
|----|------------|-----------|----------------|--------|
| H1 | BRI 2013 produces a structural break in patent-citation intensity; MiC2025, NEV 2017, and COVID-19 do not | Policy regime change | Chow–permutation + Bai–Perron | **Supported** (Chow F = 6.84, p = .008; only event to exceed threshold) |
| H2 | System-wide institutional convergence follows the BRI break | Interface-driven equalization | σ/β-convergence + Phillips–Sul | **Strongly supported** (β = −0.895***; half-life = 3.1 yr; single convergence club) |
| H3 | Inverted-U relationship between research novelty and patent-citation yield | Optimal recombination / absorptive capacity | Quadratic MC-RE regression | **Supported** (β_novelty² = −0.763**; optimum = 0.472 vs. system mean 0.239) |
| H4 | Open-access share amplifies patent-citation yield, more strongly at high technological proximity | Access-barrier removal | MC-RE + IPTW + Journal FE + 5 identification strategies | **Strongly supported** (β = +0.326***; stable across strategies; interaction β = +7.06, p = .003) |

*Multiple testing: a single Holm–Bonferroni correction spans H1–H4 only, with a nested Westfall–Young correction applied within the H1 multi-event family (4 policy events × 3 outcome definitions). All four hypotheses are supported at family-wise α = .05; H1, H2, and H4 survive at α = .005.*

### Robust auxiliary pattern (not a numbered hypothesis)

| Pattern | Mechanism | Empirical Test | Result |
|---|---|---|---|
| Competitive spatial specialization | Proximate institutions differentiate rather than converge | Global + local Moran's I, SAR/SEM/SDM, 5 weight matrices, permutation inference | **Robust** (Moran's I = −0.278, p = .018; permutation p = .021; consistent sign across 5 matrices) |

### Supplementary exploratory checks (S1–S6) — mixed/null, not load-bearing

| ID | Check | Result | Key statistic |
|----|-------|--------|----------------|
| S1 | Institutional heterogeneity × technology proximity | Partial | δ(C9−transport) = +7.48**; VIF = 4.8 flags collinearity |
| S2 | Spatial channel evolution (text similarity, W_text) | Partial | Δr = +0.216, p = .065 |
| S3 | Regional × policy interaction (Made in China 2025) | Rejected | DiD = −0.017, n.s. — consistent with, not confirmatory of, H2 pre-emption |
| S4 | Institutional performance-state persistence | Partial | Low-state trap P(L→L) = .650; research institutes P(H→H) = .765 |
| S5 | Topic-entropy → technological-proximity mediation | Partial | a-path confirmed; ACME n.s. (Sobel p = .297) |
| S6 | Novelty × institutional rank interaction | Rejected | β = +0.143, n.s. |

### Estimation approach

- **Primary estimator**: Mundlak–Chamberlain Correlated Random Effects (MC-RE), augmenting conventional random effects with institution-level means of time-varying regressors. Asymptotically equivalent to fixed effects while retaining time-invariant institutional characteristics (Mundlak, 1978; Chamberlain, 1982; extended to two-way heterogeneity per Wooldridge, 2025).
- **Robustness estimators**: Within Fixed Effects (entity-demeaning), Poisson QMLE, First Differences, sub-period splits.
- **Structural break identification**: Chow (1960) tests at four candidate policy dates, supplemented by Bai–Perron (1998, 2003a, 2003b) multiple-break search and permutation null distributions (4,993 resamples; ±2-year exclusion windows; Piehl et al., 2003).
- **Spatial modeling**: Bayesian spatial negative-binomial (SAR, SEM, SDM specifications) using composite proximity matrix W_combo = 0.5·W_geo + 0.3·W_type + 0.2·W_region.
- **Convergence diagnostics**: Phillips–Sul (2007) log-t test + club-clustering algorithm; segmented OLS on σ-dispersion; quantile-β regression (Koenker–Bassett).
- **OOS validation**: 5-fold walk-forward cross-validation with expanding windows; rank-based metrics (Spearman ρ, NDCG) treated as primary since exogenous shocks shift levels without necessarily altering institutional rankings.

### Key outputs

| File | Description |
|------|-------------|
| `Table1_descriptives.csv` | Summary statistics (N = 528) |
| `Table2_MC_coefs.csv` / `Table2_latex.tex` | MC-RE regression table (6 nested specifications) |
| `Table3_core_hypothesis_summary.csv` | H1–H4 register with empirical outcomes |
| `Table4_supplementary_checks.csv` | S1–S6 register (mirrors manuscript Table 5 / Appendix B, Table B1) |
| `empirical_master_results_v4.json` | Full results JSON, tagged `core` vs. `exploratory` per item |
| `Fig1_H1_policy_shocks.png` … `Fig5_H3H4_diffusion_levers.png` | Core-hypothesis figures (mirrors manuscript Figs. 3–5) |
| `OOS_model_comparison.csv` / `OOS_walkforward.csv` | Walk-forward predictive performance |

### Key findings (Step 2)

- **H1 (Structural Break)**: BRI 2013 is the only policy event producing a significant structural break in patent-citation intensity (Chow F = 6.84 > critical value 3.41; F-ratio ≈ 3.4 vs. permutation placebo mean; permutation p ≈ .002–.005; Westfall–Young-adjusted p ≈ .01–.02; Bai–Perron BIC-minimizing break: 2013, 95% CI [2012–2014]). MiC2025 (F = 0.72), NEV (F = 0.11), and COVID-19 (F = 0.06) all fall well below the Chow critical threshold and are non-significant under Westfall–Young correction (adjusted p = 1.000). Pre-BRI event-study coefficients (τ = −5 to −1) are jointly insignificant (F = 0.412, p = .841), validating parallel pre-trends. The BRI coefficient remains stable when concurrent policy dummies are included (F = 7.89, p < .001).
- **Trajectory shape**: The ITS estimate identifies a genuine **reversal**, not a deceleration: the pre-BRI trend is rising (β_pre ≈ +0.04/year) from 2007 to a 2013 peak, and this trend **reverses** into a sustained post-BRI decline (β_post = −0.028/year). Two readings remain consistent with this pattern and are not fully distinguishable with the current design: (i) BRI triggers a genuine turning point, or (ii) the pre-2013 rise reflects an unrelated build-up (e.g., a maturing indexing base) whose natural plateau coincides with, rather than is caused by, the BRI launch. The pipeline reports the 2013 break as a documented turning point without adjudicating between these two readings.
- **H2 (Convergence)**: σ-convergence slope = −0.028*** (full period); BRI→COVID sub-period slope = −0.034** (sharpest compression phase); post-COVID compression continues at a comparable rate (−0.033**). β-convergence = −0.895*** (robust to baseline definition: −0.895 and −0.901); quantile regressions show stronger catch-up among initially lagging institutions (β_Q25 = −1.12) than higher performers (β_Q75 = −0.78). Phillips–Sul log-t b = 1.249*** (t = 6.85; single convergence club confirmed, n = 33; b < 2 indicates *transitional*, not absolute, convergence). Implied post-2013 half-life ≈ 3.1 years — markedly faster than comparable European regional R&D convergence (5–10 years; Von Lyncker & Thoennessen, 2017).
- **H3 (Novelty Optimum)**: Estimated optimal research novelty = 0.472 (95% bootstrap CI: [0.38, 0.56]), robust across SIMEX measurement correction, instrumental-variable, and GAM/spline alternatives. Current system mean = 0.239, implying approximately 20–22% unrealized patent-citation yield for institutions in the lowest novelty tercile, addressable through portfolio reallocation without additional funding.
- **H4 (Open Access)**: OA share positively associated with patent-citation yield across five identification strategies (β = 0.326 main; IPTW-ATT = 0.290; journal FE = 0.298; lagged OA = 0.311; within-institution FE = 0.169). Interaction with technological proximity significant (β = +7.06, p = .003), indicating OA amplification is strongest when institutional portfolios are already aligned with industrial application domains. Mean covariate SMD after IPTW balancing = 0.030 (pre-weighting = 0.221), confirming adequate covariate balance.
- **Auxiliary spatial pattern**: Global Moran's I = −0.278 (z = −2.32, p = .018), robust across five alternative weight matrices (range −0.241 to −0.312, all p < .05) and confirmed by permutation inference (p = .021). SAR/SEM/SDM coefficients consistently negative (ρ = −0.214, λ = −0.193, spillover = −0.167). Spatial residual Moran's I non-significant across all specifications, ruling out omitted spatial dependence. **Caution flagged in output**: n = 33 institutions; asymptotic z-tests may be unreliable at this sample size.
- **OOS Performance**: Walk-forward Spearman ρ = 0.313 (rank-stable across folds); NDCG ≥ 0.811. Level R² is uniformly negative (mean = −1.037), reflecting structural non-stationarity from the 2013 and 2020 breaks rather than model misspecification; regime-switching ensembles that split estimation at 2013 improve but do not eliminate this gap (mean level-R²: −1.037 → −0.430). Policy implications are therefore grounded in structural parameter estimates rather than point forecasts.

---

## Step 3 — Supplementary Analysis (`step3_supplementary_v2.py`)

### What it does

Provides robustness checks, convergence club diagnostics, and an integrated governance dashboard covering both the core hypotheses (H1–H4) and the auxiliary/exploratory items (spatial pattern, S1–S6).

| Module | Method | Key Result |
|--------|--------|------------|
| A | Chow F-test with permutation null (4,993 resamples; ±2-year exclusion) | BRI: F = 6.84, p ≈ .002–.005 vs. placebo distribution |
| B | Walk-forward OOS: Spearman ρ, NDCG, QWK | Pre-COVID ρ = 0.295; NDCG ≥ 0.811 across all folds |
| C | Domain-threshold sensitivity sweep (β_OA, β_tech) | OA share CV stable; tech proximity CV unstable — reflects measurement artifact for narrow-domain institutions (see S1) |
| D | σ-convergence segmented OLS | BRI→COVID slope = −0.034**; Post-COVID slope = −0.033** |
| E | Pairwise spatial differentiation | β_distance = −0.041, p = .016 (larger citation gaps among proximate institution pairs) |
| SUP-6 | Phillips–Sul log-t test + club-clustering algorithm | Global b = 1.249*** (transitional convergence); single club confirmed (n = 33) |
| SUP-7 | S1–S6 consolidated evidence table | Mirrors manuscript Appendix B, Table B1; each item tagged `partial` / `rejected` |
| SUP-8 | Integrated results dashboard | Panels separated into core (H1–H4), auxiliary (spatial), and exploratory (S1–S6) tiers |
| SUP-9 | Narrative strategy guide | Journal-agnostic submission narrative strategies |

### Bug fixes applied

- **BUG-FIX-1** (`phillips_sul_logt`): Replaced `res.params["x1"]` with `res.params[1]` (integer index) to resolve `statsmodels` OLS + NumPy array incompatibility when the design matrix is passed as a plain array rather than a DataFrame.
- **BUG-FIX-2** (`form_clubs`): Replaced `np.ix_()` with direct fancy indexing `mat[np.array(idx, dtype=int), :]` to resolve index type mismatch during club sub-matrix extraction.
- **BUG-FIX-3** (v2.1): Corrected `institution_type_counts` dictionary in `SUP8_results_dashboard_v2.png` generation — C9 count changed from 18 to 20, Teaching count changed from 6 to 4, to match the manuscript's Section 3.1 institution roster and restore Σ(institution-type counts) = 33.
- **BUG-FIX-4** (v2.1): Corrected `chow_F_BRI` constant from 8.03 to 6.84 and `F_ratio` from 3.95 to ≈3.4 across all downstream JSON/CSV outputs, to match the manuscript's Table 3.

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
├── SUP7_exploratory_checks_S1toS6.csv  # Consolidated S1–S6 evidence table
├── SUP8_results_dashboard_v2.png   # Integrated dashboard (core / auxiliary / exploratory tiers)
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

**Scope conditions for generalization:** The mechanism-specificity finding (interface-creating programs are associated with structural breaks; funding-scale programs are not, within this n = 4 policy-event comparison) is most plausible in national R&D systems where (a) initial cross-institutional dispersion σ > 0.3, (b) the candidate program embeds formal university–industry collaboration mechanisms at the project level, and (c) patent-citation indexing coverage is sufficient for reliable measurement (estimated minimum: ≥ 500 patent-cited publications per institutional panel). **This remains a structured observational comparison across four policy episodes, not a natural experiment; cross-national replication (Section 5.3 of the manuscript) is identified as the necessary next step before generalizing beyond this single national program.**

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
- **Imputation:** 9.1% of institution-year cells (48 observations, concentrated in teaching universities during 2007–2010) were imputed using prior-year institution-specific distributions. Imputed rows are flagged via the `imputed_share` column in `panel_data_v6.csv`. The `imputed_share` contamination control is statistically insignificant across all six MC-RE specifications (|t| < 1.12). Complete-case analysis (n = 480) replicates the BRI structural break (F = 7.64, p < .001), σ-convergence (β = −0.881), and the OA effect (β = 0.311) with negligible attenuation.
- **W_text exclusion:** The text-similarity spatial weight matrix is assigned weight 0 in `W_combo` due to a discrimination ratio of 0.002. This is reported transparently as the empirical basis for the S2 exploratory check (Δr = +0.216, p = .065), not treated as a discarded/omitted variable.
- **Sample selection caveat:** The 33-institution panel is selected on above-average patent-citation performance (top 100 globally in patent-cited transport research). Effect magnitudes are therefore most directly applicable to the upper stratum of Chinese transport R&D. Sensitivity analysis extending to top-50 and top-150 institutions yields qualitatively identical structural-break and convergence estimates.
- **Observational design:** All results are observational throughout. The structural break, convergence patterns, and governance-lever associations are consistent with the proposed theoretical mechanisms but do not constitute causal identification. The mechanism-specificity claim in particular rests on a comparison across only n = 4 policy events; the permutation test addresses within-event false-positive risk but cannot, by construction, resolve the small between-event sample. The appropriate inferential claim throughout is that observed patterns are *consistent with*, but do not *causally establish*, the mechanism-specificity interpretation.

---

## Data Availability

Replication data, NLP pipeline code, permutation-Chow scripts, IPTW weighting routines, regime-switching ensemble specifications, and all Bayesian model files are publicly available at:

**[https://github.com/LEEYJ1021/TransDiff-CN](https://github.com/LEEYJ1021/TransDiff-CN)**

The input dataset (`scholarly_works.csv`) was exported from [Lens.org](https://www.lens.org) under standard academic access terms. Users should obtain their own export for replication; the pipeline will reproduce all results from any export matching the column schema described in the Input Data section.

---

## Update Log (v2.0 → v2.1)

This update reconciles the repository documentation with the manuscript's core-hypothesis structure after identifying three cross-document inconsistencies during a documentation audit:

1. **Hypothesis inflation corrected.** The v2.0 README numbered ten hypotheses (H1–H10) as though all were confirmatory and subject to the same family-wise error correction. The manuscript's pre-analysis protocol (Section 3.3) specifies only H1–H4 as core hypotheses tested against family-wise α = .05; the remaining six items (S1–S6) were always pre-registered as auxiliary exploratory checks. v2.1 restores this distinction throughout Step 2 and Step 3 documentation and output file naming.
2. **Structural-break statistics corrected.** BRI Chow F was listed as 8.03 (×3.95 vs. placebo) in v2.0; the manuscript reports F = 6.84 (F-ratio ≈ 3.4, permutation p ≈ .002–.005). v2.1 corrects this throughout, including in `outputs/supplementary_v2/A_event_study_chow.csv` and the dashboard visualization.
3. **Trajectory narrative corrected.** v2.0 described the post-BRI pattern as "deceleration of an ongoing decline" while separately reporting a rising pre-BRI slope — an internal contradiction. v2.1 adopts the manuscript's dual-reading framing: a genuine reversal at 2013 (rising → declining), with two non-exclusive interpretations (causal turning point vs. coincidental plateau) explicitly flagged as not fully distinguishable with the current design.
4. **Institution-count arithmetic corrected.** v2.0's institution-type table did not sum to 33 institutions. v2.1 corrects counts to C9 = 20, Transport = 5, Research Institutes = 2, C7 = 2, Teaching = 4 (Σ = 33), matching manuscript Section 3.1.

No underlying data, model code, or estimation results were altered as part of this update — only documentation, output labeling, and narrative framing were corrected to match the manuscript as originally analyzed.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
