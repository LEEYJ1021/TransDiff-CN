# Knowledge Diffusion in Chinese Transportation Research (2007–2023)
### A Three-Step Bayesian–Empirical Pipeline

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyMC](https://img.shields.io/badge/PyMC-5.x-orange)](https://www.pymc.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This repository contains the full analysis pipeline for studying **scientific knowledge diffusion patterns** among Chinese transportation research institutions (2007–2023). The study examines how NLP-derived research characteristics, institutional attributes, policy shocks, and spatial proximity jointly shape patent citation impact — a proxy for research-to-industry knowledge transfer.

The pipeline is organized into **three sequential steps**:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_btm_nlp_v6.py` | NLP feature engineering + Bayesian Transition Model |
| 2 | `step2_empirical_tra_v4.py` | Empirical hypothesis testing (H1–H10) |
| 3 | `step3_supplementary_v2.py` | Supplementary robustness and convergence analysis |

---

## Repository Structure

```
.
├── data/
│   └── scholarly_works.csv          # Input dataset (see Data section below)
├── step1_btm_nlp_v6.py              # Step 1: Bayesian Transition Model + NLP
├── step2_empirical_tra_v4.py        # Step 2: Mundlak–Chamberlain RE + hypothesis tests
├── step3_supplementary_v2.py        # Step 3: Convergence clubs + dashboard
├── outputs/
│   ├── btm_nlp_v6/                  # Step 1 outputs (figures, traces, CSVs)
│   ├── empirical_tra_v4/            # Step 2 outputs (tables, figures, JSON)
│   └── supplementary_v2/            # Step 3 outputs
├── .gitignore
├── LICENSE
└── README.md
```

---

## Input Data

**File:** `data/scholarly_works.csv`

A bibliometric dataset of scholarly works affiliated with **33 major Chinese research institutions** active in transportation-related research, covering publication years 2007–2023. The dataset was exported from a bibliographic database and contains the following key columns:

| Column | Description |
|--------|-------------|
| `Lens ID` | Unique paper identifier |
| `Title` | Paper title |
| `Publication Year` | Year of publication (2007–2023) |
| `Abstract` | Full abstract text |
| `Keywords` | Author-supplied keywords |
| `Fields of Study` | Discipline tags (semicolon-separated) |
| `Author/s` | Author list (semicolon-separated) |
| `Institution` | Corresponding or primary institution name |
| `Citing Patents Count` | Number of patents citing this paper (**outcome variable**) |
| `Citing Works Count` | Number of academic works citing this paper |
| `Is Open Access` | Boolean: open-access status |
| `References` | Reference list (semicolon-separated) |
| `Funding` | Funding acknowledgement text |

> **Note:** Raw counts after loading: ~13,350 rows, 33 columns. After year filtering (2007–2023) and dropping rows with missing patent citation data, the working dataset contains **11,271 papers** across **33 institutions**.

### Institutions Covered

The 33 institutions span five institutional-type categories and six geographic regions of China:

- **C9 League** (elite research universities): Tsinghua University, Peking University, Zhejiang University, and 14 others
- **C7 League** (national defence universities): Beihang University, Beijing Institute of Technology
- **Transport-specialized**: Beijing Jiaotong University, Tongji University, Southwest Jiaotong University, Wuhan University of Technology, Central South University
- **Research institutes**: Chinese Academy of Sciences, University of Chinese Academy of Sciences
- **Teaching-focused**: Shenzhen University, Soochow University, Shanghai University, Beijing University of Technology

---

## Step 1 — NLP-Augmented Bayesian Transition Model (`step1_btm_nlp_v6.py`)

### What it does

1. **Text processing & BERT embeddings**: Cleans and encodes title + abstract + keywords using `sentence-transformers/paraphrase-MiniLM-L6-v2` (falls back to TF-IDF + SVD if unavailable).
2. **Topic clustering**: Applies K-Means with Gap Statistic selection (minimum K = 5) to identify transport research sub-domains. Topics are labeled via c-TF-IDF, PMI, and a 10-category transport taxonomy.
3. **Spatial weight matrices**: Constructs four institution-level spatial weights — geographic inverse-distance (`W_geo`), institutional-type homophily (`W_type`), administrative region (`W_region`), and text-similarity (`W_text`).
4. **Panel construction**: Aggregates paper-level NLP features (tech proximity, research novelty, topic entropy, intra-cohesion) into an institution × year panel.
5. **Bayesian models**:
   - **M1**: Hierarchical Negative Binomial with geographic spatial lag
   - **M2**: Bayesian Markov Transition Model (state ∈ {Low, Medium, High})
   - **M3**: Spatiotemporal NegBin with time-varying ρ(t) (Gaussian random walk)
   - **M4v2**: NLP × Spatial Interaction NegBin (four spatial channels + two interaction terms)
   - **M5**: Bayesian mediation (C9 → NLP features → patent citations)
   - **M6**: Multi-state degradation / reliability analysis (CTMC)
6. **Model comparison**: LOO-CV and WAIC via ArviZ.

### Key outputs

| File | Description |
|------|-------------|
| `panel_data_v6.csv` | Institution × year panel with all NLP covariates |
| `W_geo.npy` / `W_type.npy` / `W_region.npy` / `W_text.npy` / `W_combo.npy` | Spatial weight matrices |
| `T_bayesian.csv` | Bayesian transition probability matrix |
| `trace_M1.nc` … `trace_M5_mediation.nc` | ArviZ InferenceData traces (NetCDF) |
| `full_results_v6.0.json` | Master results JSON |
| `all_posterior_estimates_v6.csv` | Posterior means, HDIs, P(>0) for all models |
| `Fig1_panel_overview_v6.0.png` … `FigZ_china_map_v6.0.png` | 9 publication-quality figures |
| `topic_descriptions_v6.json` | Topic labels, c-TF-IDF terms, domain assignments |

### Key findings (Step 1)

- **K = 5** sub-domains identified by Gap Statistic; all assigned to transport taxonomy via guaranteed cosine-similarity matching.
- **Moran's I** (permutation, n = 999): I = −0.288, z = −2.32, p = 0.018 — negative spatial autocorrelation (dispersion pattern).
- **ΔELPD(M4v2 − M1) = +4.10** (SE = 2.94) — marginal improvement; consistent with 9.1% raw imputation rate.
- **Steady-state** (M2): Low = 0.308, Medium = 0.409, High = 0.283.
- **Reliability** (M6): A = 0.277, MTTF = 4.93 yr, MTTR = 5.30 yr.
- All chains converged: max R̂ < 1.006 across M1–M4v2.

---

## Step 2 — Empirical Analysis (`step2_empirical_tra_v4.py`)

### What it does

Tests **10 pre-registered hypotheses** using the panel produced in Step 1.

| Hypothesis | Description | Result |
|------------|-------------|--------|
| H1 | Policy shocks (BRI 2013, MiC2025 2015, NEV 2017, COVID 2020) → structural breaks | **Supported** *** |
| H2 | Transport-specialized > C9 in tech-proximity → patent elasticity | Partially supported |
| H3 | Geographic diffusion channel weakens vs. type-homophily post-2015 | See Fig. 7a |
| H4 | East/South disproportionately absorbs EV/AV knowledge post-MiC2025 | Not supported |
| H5 | Inverted-U relationship: research novelty² < 0 | **Supported** ** |
| H6 | Matthew effect: high-state persistence > low-state | Partially supported |
| H7 | β-convergence in research impact post-BRI | **Supported** *** |
| H8 | Topic entropy mediates C9 → patent citations | **Supported** ** |
| H9 | Open-access share amplifies research-to-industry transfer | **Supported** *** |
| H10 | Top-ranked institutions amplify novelty–impact relationship | Not supported |

### Estimation approach

- **Main estimator**: Mundlak–Chamberlain Random Effects (MC-RE), preserving time-invariant institutional variables while relaxing strict RE exogeneity.
- **Auxiliary estimator**: Within Fixed Effects (entity-demeaning) for robustness.
- **H8 mediation**: Three-strategy approach (Between-FE, Pooled+ClusterSE, TV Prestige) to handle the time-invariant `is_c9` instrument.
- **OOS validation**: Temporal-Split Deviation + GBM ensemble + 5-fold walk-forward CV.
- **Robustness**: Poisson QMLE, First-Differences, sub-period splits, VIF, Breusch-Pagan.

### Key outputs

| File | Description |
|------|-------------|
| `Table1_descriptives.csv` | Summary statistics |
| `Table2_MC_coefs.csv` / `Table2_latex.tex` | MC-RE regression table (6 specs) |
| `Table3_hypothesis_summary.csv` / `Table3_latex.tex` | Hypothesis results |
| `empirical_master_results_v4.json` | Full results JSON |
| `Fig1_H1_policy_shocks.png` … `Fig11_diagnostics.png` | 11 publication-quality figures |
| `OOS_model_comparison.csv` / `OOS_walkforward.csv` | Predictive performance |
| `H1_chow_tests.csv` … `H8_mediation_3strategy.csv` | Hypothesis-specific tables |

### Key findings (Step 2)

- **H1**: BRI 2013 is the strongest structural break (Chow F = 6.84, p = 0.008).
- **H5**: Optimal research novelty ≈ 0.472 (inverted-U confirmed; β_nov² = −0.763**).
- **H7**: σ-convergence slope = −0.028*** and β-convergence = −0.895*** — institutional knowledge gap narrowing.
- **H9**: OA share positively significant across all MC-RE specifications (β ≈ +0.33***).
- **OOS**: Walk-forward Spearman ρ ≈ 0.40; NDCG ≥ 0.85 across all folds; Precision@10 = 0.50.

---

## Step 3 — Supplementary Analysis (`step3_supplementary_v2.py`)

### What it does

Provides additional robustness checks and convergence analysis:

| Module | Method | Key result |
|--------|--------|------------|
| A | Chow F-test with permutation (placebo) test | BRI: F = 8.03, ×3.95 placebo mean, p < 0.001 |
| B | OOS walk-forward: Spearman ρ, NDCG, QWK | Pre-COVID ρ = 0.295; NDCG ≥ 0.811 |
| C | Domain-threshold sensitivity sweep (β_OA, β_tech) | OA share CV = 0.33 (STABLE); tech prox CV = 1.01 (UNSTABLE) |
| D | σ-convergence segmented OLS | BRI→COVID slope = −0.034** |
| E | Pairwise spatial differentiation | ρ(distance, \|Δpat\|) = −0.097** |
| SUP-6 | Phillips-Sul log-t + club formation | Global b = 1.25 (full convergence); 1 club |
| SUP-8 | Integrated results dashboard | All panels computed from data |
| SUP-9 | Narrative strategy guide | 3 submission-type strategies (no journal names) |

### Bug fixes applied

- **BUG-FIX-1**: `phillips_sul_logt` — replaced `res.params["x1"]` with `res.params[1]` (integer index) to fix `statsmodels` OLS + NumPy array incompatibility.
- **BUG-FIX-2**: `form_clubs` — replaced `np.ix_()` with direct fancy indexing `mat[np.array(idx, dtype=int), :]`.

### Key outputs

```
outputs/supplementary_v2/
├── A_event_study_chow.csv
├── B_oos_walk_forward.csv
├── C_domain_sensitivity.csv
├── D_sigma_convergence.csv
├── E_spatial_pairs.csv
├── SUP6_convergence_clubs_v3.csv
├── SUP6_convergence_clubs_v3.png
├── SUP8_results_dashboard_v2.png
└── SUP9_narrative_strategy.json
```

---

## Setup & Requirements

```bash
pip install numpy pandas scipy statsmodels scikit-learn matplotlib seaborn tqdm
pip install pymc pytensor arviz
pip install sentence-transformers  # optional; TF-IDF fallback used if absent
pip install lightgbm               # optional; sklearn GBM fallback used if absent
```

**Python ≥ 3.9** is required. All three scripts are self-contained and can be run independently, provided the outputs of earlier steps are available.

### Path configuration

Each script exposes path constants at the top. Update these before running:

```python
# step1_btm_nlp_v6.py
FILE_PATH = "data/scholarly_works.csv"   # input CSV
OUT_DIR   = "outputs/btm_nlp_v6"         # Step 1 output directory

# step2_empirical_tra_v4.py
V6_OUT  = "outputs/btm_nlp_v6"          # Step 1 outputs (panel, spatial matrices)
RAW_CSV = "data/scholarly_works.csv"    # original CSV (for sub-domain analysis)
OUT_DIR = "outputs/empirical_tra_v4"

# step3_supplementary_v2.py
V6_OUT    = "outputs/btm_nlp_v6"
STEP2_OUT = "outputs/empirical_tra_v4"
OUT_DIR   = "outputs/supplementary_v2"
```

### Execution order

```bash
python step1_btm_nlp_v6.py     # ~30–60 min depending on hardware (MCMC)
python step2_empirical_tra_v4.py
python step3_supplementary_v2.py
```

> **Tip**: Step 1 caches BERT embeddings and K-Means results in `outputs/btm_nlp_v6/nlp_cache/`. Re-runs will load these automatically and skip recomputation.

---

## Reproducibility Notes

- `np.random.seed(42)` and `random_seed=42` are set throughout.
- MCMC settings: 2,000 draws, 2,000 tune steps, 4 chains, `target_accept = 0.90`. All chains converged with max R̂ < 1.006.
- Imputation: ~9.1% of papers had no institution label and were assigned via prior-frequency sampling; imputed rows are flagged in `panel_data_v6.csv` (`imputed_share` column).
- The `W_text` channel was assigned weight 0 in `W_combo` due to a low discrimination ratio (0.002), reflecting that confirmed-paper TF-IDF is near-uniform at this sample size.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
