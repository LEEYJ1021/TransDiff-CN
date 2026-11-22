# Convergence or Compression? Measurement Sensitivity and Spatial Scale in Patent-Citation Indicators of Chinese Research

### Replication materials, version 2 (revision R1)

[![Python 3](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Version note.** This is **v2 (revision R1)**. It replaces the first-submission repository (v1, README v2.1), which is preserved in `legacy_v1/` and in git tag `v1.0-original`. Several headline results of v1 could not be reproduced and are **withdrawn**. [Section 2](#2-what-changed-from-v1-to-v2) lists, item by item, what v1 reported and what v2 reports. The v1 results are not used in the revised manuscript.

---

## Contents

1. [Overview](#1-overview)
2. [What changed from v1 to v2](#2-what-changed-from-v1-to-v2)
3. [Repository structure](#3-repository-structure)
4. [Data](#4-data)
5. [Analysis modules and where results are stored](#5-analysis-modules-and-where-results-are-stored)
6. [Main results (revised manuscript)](#6-main-results-revised-manuscript)
7. [Setup and re-running](#7-setup-and-re-running)
8. [Reproducibility](#8-reproducibility)
9. [Known limitations](#9-known-limitations)
10. [Data availability](#10-data-availability)
11. [Update log](#11-update-log)
12. [License](#12-license)

---

## 1. Overview

A fall in the spread of patent-citation performance across institutions is often read as catch-up, and large transport programmes such as China's Belt and Road Initiative (BRI) are expected to produce it. The revised manuscript asks whether that reading survives once the measurement of the indicator is made explicit.

**Units and data.** 213,328 scholarly works (2007-2023) returned by a "Transportation" query in Lens.org, restricted to authors affiliated with 89 candidate Chinese institutions (including works never cited by patents; 16,170 works, 7.6%, have at least one citing patent), linked to citing-patent records, and a 33-institution subsample selected on patent-cited output.

**Three research questions.**

| ID | Question |
|---|---|
| RQ1 | Does the dispersion of patent-citation performance fall once citation exposure, sample selection and sampling noise are handled? |
| RQ2 | Is anything specific to the BRI period? |
| RQ3 | How does spatial dependence in outcomes depend on spatial scale, the weight matrix, and whether neighbours are defined by distance or by co-authorship? |

**What the paper does not claim.** The revised paper does not evaluate the BRI (the data contain no measure of programme exposure), does not claim institutional convergence, and does not claim competitive specialisation or institutional lock-in. The BRI date is used only as the temporal context of the observation window.

**Framing.** All results are observational and associational. The contribution is a measurement framework, condensed into four checks (manuscript Section 5.2): (1) fixed citation windows or equal-exposure comparisons, with cell counts; (2) dispersion of institution means reported together with the paper-level distribution, the between-institution variance share and a sampling-noise benchmark; (3) spatial autocorrelation reported with same-city exclusion and alternative weights; (4) period-specific claims tested against placebo event years and random topic groups.

---

## 2. What changed from v1 to v2

| Item | v1 (first submission) | v2 (this revision) |
|---|---|---|
| **Question** | Four core hypotheses (H1-H4) and six exploratory checks (S1-S6): BRI-specific structural break, convergence, novelty optimum, open access, competitive spatial specialisation | Three questions (RQ1-RQ3, above). No claim about a BRI effect, institutional convergence or competitive specialisation |
| **Data** | 33 institutions, 11,271 institution-work rows, 528 usable institution-year observations. An earlier panel file contained 33 zero-filled cells (561 rows against 528 usable) and lacked 13 cells of one institution | Population of 213,328 works from 89 candidate institutions, including uncited works. 33-institution subsample: 11,271 institution-work rows (9,294 unique works), **541 non-empty institution-year cells, none filled with zero** |
| **Outcome** | Cumulative citing-patent count | Cumulative count (extracted November 2026), **fixed 3- and 5-year windows** (priority date of the citing family), and an equal-exposure comparison across cohorts |
| **Patent-paper links** | Not documented | 30,087 citing patents linked to works through resolved non-patent citations (32,538 document links collapsed to 25,982 work-family links); 99.8% of cited works have a resolved link, 99.6% match the Lens citing-patent count exactly (correlation 0.9997) |
| **Citation exposure** | Not addressed | Cumulative cited share peaks at 11.5% (2017 cohort) and falls to 5.5% (2023); at equal exposure the latest cohorts match 2016-18 (ratios 1.14, 1.01, 0.94) |
| **Dispersion** | sigma 0.471 to 0.111, read as convergence | sigma 0.478 (2007-12) to 0.126 (2020-23), a decline of about 74% (73.6%; 78.0% using 2021-23 alone). Read as compression plus sampling noise (Section 6) |
| **Structural break** | Chow F = 6.84, Bai-Perron break at 2013, parallel pre-trends | **Removed.** Replaced by a difference-in-differences of a fixed BRI-related keyword group against 1,000 random topic groups, with placebo event years 2010-2016 |
| **Convergence tests** | beta-convergence (-0.895), Phillips-Sul b = 1.249 ("single club"), half-life 3.1 years | **Removed.** No convergence claim (the earlier "rejects club formation" reading of the Phillips-Sul test was incorrect, since its null is convergence) |
| **Spatial dependence** | Moran's I = -0.278 (p = .018), composite weight matrix | **Not reproduced; withdrawn.** Reconstructed distance-based values are not significant at 5% and depend on the weights (Section 6) |
| **Novelty optimum, open access, policy recommendations** | Optimum 0.472; open-access beta = 0.326; funding and portfolio recommendations | **Removed** |
| **Reproducibility** | Scripts `step1`-`step3`; several headline values not generated by a single script; figures and tables disagreed | Analysis code in `code/` (exported from notebooks); one ledger comparing manuscript and recomputed values; manifest with fixed seed |

**Corrections disclosed in the response to reviewers (Section 7 of that letter).** An earlier panel file contained zero-filled and missing institution-year cells; the original Moran's I of -0.278, the alternative-matrix table and the pairwise results built on that panel could not be reproduced; the original dispersion statistics and several headline values were not generated by a single script.

---

## 3. Repository structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── code/
│   ├── 01_numbers_v7.py              # link construction and cohort numbers (stage v7)
│   ├── 02_analysis_v7_v8.py          # cohort compression, placebo / random-topic-group analysis (stages v7-v8)
│   └── 03_recompute_v9_v12.py        # recomputation of all reported numbers, ledger, manifest (stages v9-v12)
├── data/
│   └── derived/
│       ├── paper_family_links.csv.gz # paper-patent family links
│       └── inst_coords_33.csv        # institution coordinates (33 institutions, 18 cities)
├── outputs/
│   ├── tables/                       # result tables of the final stage (v9) and earlier stages
│   │   ├── v7/                       # link QA, cohort series, accrual curves, placebo (v7)
│   │   └── v8/                       # cohort compression, contrasts, placebo (v8)
│   └── final_numbers/                # final table/text fills, consistency ledger, manifest
├── figures/                          # Figures 1-7 (PDF and PNG) and figure data
├── docs/
│   ├── captions_final.txt            # figure captions
│   ├── captions_fig5to7.txt
│   └── lens_query.md                 # description of the Lens export
└── legacy_v1/                        # first-submission scripts and README (superseded)
```

---

## 4. Data

### 4.1 Raw data (not redistributed)

Raw Lens.org exports are **not** included in this repository. `docs/lens_query.md` describes the export.

| Item | Description |
|---|---|
| Source | Lens.org scholarly works and citing patents (non-patent-citation links) |
| Extraction | November 2026 |
| Query | "Transportation", restricted to authors affiliated with any of 89 candidate Chinese institutions; run year by year, publication years 2007-2023 |
| Population | 213,328 works (including works never cited by patents) |
| Cited works | 16,170 works (7.6%) with at least one citing patent |

### 4.2 Derived files included here

| File | Content |
|---|---|
| `data/derived/paper_family_links.csv.gz` | Links between works and citing-patent families (25,982 work-family links) |
| `data/derived/inst_coords_33.csv` | Columns: institution, city, latitude, longitude, region, type, order. Used for all distance-based statistics. 33 institutions in 18 cities, 8 of them in Beijing. Coordinates were carried over from the first-submission scripts and were not independently re-verified |

### 4.3 Samples

| Sample | Definition |
|---|---|
| Population | 213,328 works, 89 candidate institutions, 2007-2023 |
| 33-institution subsample | The Chinese institutions among the global top 100 in patent-cited output from the original query (selected on the outcome). 11,271 institution-work rows, 9,294 unique works; 32.2% of rows belong to works with several institutions. Institution-year panel: 541 non-empty cells, none filled with zero. Covers 92% of cited works in the candidate pool in 2007 and 35% in 2023 |
| Transport tag | Keyword match on Lens field-of-study terms (manuscript Appendix Table A5). Covers 42.6% of works in 2007, 18.7% in 2023 (23.6% pooled); 22.2% of the institution-work rows of the 33-institution subsample. A keyword screen, not a classification of research content; no conclusion depends on it |

Institution types (C9, C7, transport, research, teaching) in `inst_coords_33.csv` are descriptive labels and do not follow official lists.

---

## 5. Analysis modules and where results are stored

The mapping below follows file names and the manuscript's table and figure numbering. Please verify a given number against `outputs/final_numbers/H_ledger.csv` before citing it.

### 5.1 Code

| Script | Content |
|---|---|
| `code/01_numbers_v7.py` | Link construction between works and citing patents, cohort series (stage v7) |
| `code/02_analysis_v7_v8.py` | Cohort compression, compression contrasts, placebo analysis against random topic groups (stages v7-v8) |
| `code/03_recompute_v9_v12.py` | Recomputation of every number reported in the manuscript from one place; writes the consistency ledger and manifest (stages v9-v12) |

The scripts were exported from Jupyter notebooks (code cells only, outputs removed, user paths replaced by `.`). Run them in numeric order.

### 5.2 Manuscript item to file

| Manuscript item | Main file(s) |
|---|---|
| Table A1 (link quality) | `outputs/tables/v7/00_link_QA.csv`, `00_link_coverage_by_year.csv` |
| Figure 1 (truncation, cohort shares) | `outputs/tables/A3_cohort_curves.csv`, `C_same_exposure.csv`, `C_same_exposure_summary.csv`, `figures/fig1_*` |
| Figure 2 (paper-level SD, ICC, upper tail) | `outputs/tables/A_paper_level_SD_33.csv`, `A_sigma_definitions.csv`, `figures/fig2_*` |
| Figure 3 (sigma vs sampling noise) | `outputs/tables/E_sigma_vs_noise.csv`, `E_2012_loo.csv`, `E_2012_leave_one_institution_out.csv`, `figures/fig3_*` |
| Table 1 (compression contrasts) | `outputs/tables/B_table1_contrast.csv`, `outputs/final_numbers/H_table1.md` |
| Table 2, Figure 4 (BRI-related group vs random groups) | `outputs/tables/D_table2_final.csv`, `D_group_cells_primary.csv`, `D_null_lr.npz`, `outputs/final_numbers/H_table2.md`, `figures/fig4_*` |
| Table A2 (sensitivity) | `outputs/tables/D_sensitivity.csv` |
| Table A3 (trend comparison, AIC) | `outputs/final_numbers/H_tableA3_recomputed.csv` |
| Table A7 (placebo event years) | `outputs/tables/D_event_year_scan.csv` |
| P-value calibration, nominal significance counts | `outputs/tables/B1_significance_counts.csv`, `B1b_nominal_significant.csv`, `B2_pvalue_calibration.csv` |
| Date missingness and agreement | `outputs/tables/A1_missingness_by_year.csv`, `A1b_by_type.csv`, `A2_agreement.csv`, `A3b_contrast_DID.csv` |
| Table 3, Figures 5-6 (Moran's I, weights, Beijing group) | `outputs/tables/F_table3_definition_check.csv`, `outputs/final_numbers/H_table3.md`, `figures/fig5_*`, `fig6_*` |
| Section 4.5, Figure 7, Table A6 (co-authorship) | `outputs/tables/F_noise_adjusted_moran.csv`, `F_coauthor_robustness.csv`, `outputs/final_numbers/H_coauthor.md`, `figures/fig7_*` |
| Text numbers | `outputs/final_numbers/H_text_fill.md`, `H_numbers.json` |
| Manuscript vs recomputed values | `outputs/final_numbers/H_ledger.csv`, `outputs/tables/G_consistency_report.csv` |
| Run manifests | `outputs/final_numbers/manifest_v12.json`, `outputs/tables/analysis_manifest_v9.json`, `v7/analysis_manifest_v7.json`, `v8/analysis_manifest_v8.json` |
| Transport-tag validation sample | `outputs/tables/v8/validation_sample_200.csv`, `v8_08_transport_tag_share_by_year.csv` |

### 5.3 Measures

| Measure | Definition |
|---|---|
| Cumulative metric | Number of citing patents at extraction (November 2026) |
| Fixed-window metrics | Citing families whose priority date falls within 3 or 5 years of the publication year, for cohorts with closed windows (up to 2020 and 2018) |
| Dispersion (sigma) | Standard deviation across institutions of log(1 + mean citing patents), computed year by year and averaged over the years of a period; at paper level, the SD of log(1 + count) among cited works |
| Between-institution share | Intraclass correlation (ICC) of paper-level log counts, by year |
| Spatial outcome | Period-average deviation of an institution's yearly log(1 + mean citing patents) from the year mean across institutions (2013-19) |
| Noise-adjusted outcome z | Institution mean of single-institution works standardised by its sampling error (used for the co-authorship analysis) |

### 5.4 Tests

| Test | Purpose |
|---|---|
| Truncation: cohort shares under cumulative and fixed windows; same-exposure comparison (first citing patent within 0, 1, 2 years) | RQ1 |
| Selection and compression: original selection rule applied to the full candidate pool; bootstrap contrasts of paper-level SD and of the share with exactly one citing patent; sensitivity to dropping 2012 and to the transport tag | RQ1 |
| Sampling noise: observed sigma vs sigma under random reassignment of institution labels preserving cell sizes | RQ1 |
| Period specificity: BRI-related keyword groups vs 1,000 random topic groups of similar prevalence (difference-in-differences of log ratios); permutation p-values p_emp and p_adj; Benjamini-Hochberg correction; drop-2012/2013, start/end-year and placebo-event-year checks; linear, quadratic and hinge trends (AIC) | RQ2 |
| Space: Moran's I with permutation inference, alternative weights (inverse distance with and without floors, k nearest neighbours, same-city exclusion, city-level aggregation), pairwise gap-versus-distance tests, exploratory co-authorship weights | RQ3 |

---

## 6. Main results (revised manuscript)

### 6.1 Truncation (Section 4.1, Figure 1)

- The cumulative cited share rises to 11.5% for the 2017 cohort and falls to 5.5% for 2023. Under the 5-year window it rises to about 10% for 2017-2018 and does not fall in closed cohorts; under the 3-year window it is 8-9% for 2017-2020.
- At equal exposure, the share with a first citing patent within 0, 1 and 2 years for the latest observable cohorts is similar to 2016-18 (ratios 1.14, 1.01, 0.94). The decline of the cumulative share is largely lack of exposure time.
- Among cited works, the share with exactly one citing patent stays between 0.63 and 0.72 under the 5-year window but rises from 0.47 to 0.83 under the cumulative metric.

### 6.2 Selection, compression and sampling noise (Section 4.2, Figures 2-3, Table 1)

- Applying the same selection rule to the full candidate pool reproduces the pattern: the paper-level SD of log counts among unique cited works falls from 0.719 to 0.232 (2007 to 2023) in the pool and from 0.675 to 0.222 in the 33 institutions.
- sigma of institution means: 0.478 (2007-12), 0.266 (2013-19), 0.126 (2020-23), a decline of about 74% (73.6%). The decline is 71.0% when works shared among institutions are excluded.
- Almost all variance lies within institutions: ICC between 0 and 0.10 in every year. The paper-level SD (institution-work rows) falls about 65% (0.650 to 0.225); the share of works with three or more citing patents falls from 0.38 to 0.03.
- Observed sigma lies inside its label-permutation 95% interval in 16 of 17 years (exception: 2020), so the decline in sigma is largely the decline of sampling noise.
- Under the 5-year window a smaller compression remains (paper-level SD change -0.107, 95% CI -0.173 to -0.040, between 2007-12 and 2013-18). It depends on the 2012 cohort (excluding 2012: -0.064, -0.137 to +0.003) and is not detected among transport-tagged works (limited power). It cannot be separated from changes in the citing environment (China-only share of citing families: 13% for the 2007 cohort, 37% for 2018).

### 6.3 Period specificity (Section 4.3, Table 2, Figure 4)

- Primary BRI-related group, difference-in-differences of log ratios: 0.586 (cumulative), 0.521 (5-year), 0.547 (3-year).
- Cumulative: p_emp = .027, p_adj = .263; neither survives Benjamini-Hochberg correction (.086 and .300). Fixed windows: p_emp .10-.14 (all works), .25-.33 (transport-tagged).
- 2013 does not stand out among placebo event years 2010-2016 (rank 4-6 of 7 by |z|). A gradual trend fits about as well as a break at 2013.
- The primary group has 7-20 cited works before 2013 under fixed windows (25-47 cumulative). The design is informative about changes of about 1.5 to 2.2 times or more in relative citation probability, not about moderate ones.
- Nominal p < .05 under fixed windows appears in 2 of 32 specifications, both dropping the 2013 cohort (3 cited works). These are exploratory and uncorrected for the number of specifications.

### 6.4 Spatial scale and weights (Section 4.4, Table 3, Figures 5-6)

| Weights (row-standardised, 2013-19, n = 33) | I | p |
|---|---|---|
| Inverse distance, no floor | -0.262 | .051 |
| Inverse distance, 25 km floor | -0.142 | .161 |
| Inverse distance, 100 km floor | -0.076 | .345 |
| Inverse distance, excluding same-city links (< 30 km) | 0.000 | .507 |
| k nearest neighbours, k = 3 | -0.195 | .174 |
| k nearest neighbours, k = 5 | -0.098 | .443 |
| City-level aggregation, 25 km floor (18 cities) | -0.093 | .651 |

- No specification is significant at 5%. Same-city links carry 56.8% of the total weight in the no-floor matrix.
- Pairwise outcome gaps are unrelated to distance (Spearman rho = -0.02, p = .80). Within the Beijing group, dispersion falls about as fast as outside it, and the Beijing-minus-others mean gap is close to zero after 2013 with a bootstrap interval that includes zero.

### 6.5 Co-authorship ties (Section 4.5, Figure 7, Table A6; exploratory)

- Co-authorship weights (noise-adjusted outcome, single-institution works, 32 institutions): I = 0.101 (p = .008, counts), 0.068 (.015, log), 0.044 (.049, binary), 0.215 (< .001, excluding same-city pairs); distance weights: -0.184 (.068), 0.024 (.259) excluding same-city pairs.
- I is positive in all six co-authorship variants and p < .05 in five. Pairwise gaps are unrelated to the number of co-authored works (rho = 0.01, p = .83).
- This is an exploratory, descriptive association. It does not measure knowledge flows, several specifications were examined without correction for multiplicity, and similarity may partly reflect institution type.

---

## 7. Setup and re-running

### 7.1 Requirements

Python 3 with the scientific stack used by the scripts (numpy, pandas, scipy, statsmodels, matplotlib; the maps use a GADM province-boundary file). See the import lines at the top of each script for the complete list. The v1 Bayesian pipeline (PyMC, sentence-transformers) is **not** needed for v2.

### 7.2 Expected inputs (not included)

The scripts were written for the working directories used during the revision. They expect:

- raw Lens exports (yearly text exports, "all" and "cited") and the 89-institution candidate list, in the directories set by the path constants at the top of each script (`lens_export/`, `data_R1/` in the original setup);
- the original 33-institution export, used for the comparison between the candidate pool and the original sample;
- a GADM province boundary file for the maps.

Update the path constants before running. Output folders in the original setup were `outputs_R1_v7` to `outputs_R1_final`; `outputs/` in this repository holds copies of the resulting tables.

### 7.3 Order

```bash
python code/01_numbers_v7.py
python code/02_analysis_v7_v8.py
python code/03_recompute_v9_v12.py
```

The derived files in `data/derived/` are provided so that downstream tables can be inspected without the raw exports.

> **Status.** The scripts were exported from notebooks and have **not been re-run end to end in a clean directory**. Treat `outputs/final_numbers/H_ledger.csv` (manuscript vs recomputed values) as the record of what was verified.

---

## 8. Reproducibility

- **Single recomputation stage.** `03_recompute_v9_v12.py` recomputes the numbers quoted in the manuscript and writes `H_ledger.csv`, which lists each manuscript value, its recomputed value and whether they match. Where text and recomputation differed, the recomputed value replaced the text. Some ledger rows record earlier wording that the final manuscript already corrects; check the `note` column.
- **Seed.** Fixed seed 20261122, recorded in `outputs/final_numbers/manifest_v12.json`.
- **Resampling sizes.** Table 3 and Appendix Table A6: 49,999 permutations. Annual Moran's I (Figure 5a): 4,999 permutations. Random topic groups: 1,000. Fixed cell-size resampling (Figure 3a): 300 resamples. Label permutations for sigma (Figure 3b): 500.
- **Run-to-run variation.** Permutation p-values based on 49,999 permutations vary by about +/-.005 across runs; other p-values rest on fewer permutations or random groups and vary by about +/-.01. The Appendix Table A2 sensitivity runs use an independent Monte Carlo draw, so their p_emp differ slightly from Table 2 (cumulative .034 vs .027; 5-year .111 vs .103; 3-year .133 vs .139).
- **Panel.** 541 non-empty institution-year cells; no empty cell is filled with zero and nothing is imputed.
- **Link quality.** Internal consistency of the patent-paper links is documented (Appendix Table A1); links were not compared with Reliance on Science (Marx and Fuegi, 2020).

---

## 9. Known limitations

Details are in manuscript Section 5.1.

- **Selection.** The 33 institutions are selected on patent-cited output. Institution-level statements apply only to this sample; institution-level analysis of uncited works would require affiliation fields that the population export lacks.
- **Exposure.** The export gives no citing date; priority dates stand in for it. Paper dates are missing for 41% of cited works (38% in the full population), so fixed windows carry an error of about one year. Closed windows end in 2018 (5-year) and 2020 (3-year); the lag-0 cell of the equal-exposure comparison includes families with priority dates before the publication year.
- **Policy attribution.** One policy date, no comparison group, no exposure measure. Period-specific patterns are associational; the design detects only large changes.
- **Spatial statistics.** With 33 institutions, local statistics are exploratory. Outcome similarity does not measure knowledge flows. The co-authorship network is measured only among patent-cited works, and permutation benchmarks treat institution-work rows as units.
- **Measurement.** The transport tag was not validated against a content-based classification. The BRI-related keyword dictionary (Appendix Table A5) was fixed before results were examined.
- **Coordinates** were carried over from the first-submission scripts and were not independently re-verified.
- **Code.** Exported from notebooks; not re-run end to end in a clean directory (Section 7).

---

## 10. Data availability

The institution-year panel, linked paper-patent file, institution coordinate file, analysis code with its manifest and fixed seed, and all output tables are provided as Supplementary Material and in this repository (anonymous repository link as given in the manuscript's Declarations). Raw Lens.org exports are not redistributed; users should obtain their own export (see `docs/lens_query.md`).

---

## 11. Update log

**v2.0 (revision R1)**

- Reconstructed the analysis from the data export in response to the Editor's and Reviewers' comments; the paper became a measurement paper (title changed accordingly).
- Added: population including uncited works; fixed citation windows and equal-exposure comparison; selection rule applied to the candidate pool; sampling-noise benchmark; random-topic-group placebo design; alternative weights and co-authorship weights; link-quality documentation; ledger and manifest.
- Removed: hypotheses H1-H6 / S1-S6 structure; Chow and Bai-Perron tests; Phillips-Sul and beta-convergence; Moran's I of -0.278 and the composite weight matrix; novelty optimum; open-access analysis; policy recommendations; Bayesian transition and spatial negative-binomial models.
- Corrected: panel file with zero-filled and missing cells replaced by a 541-cell panel with no zero filling.
- Repository: first-submission files moved to `legacy_v1/` (author identifiers redacted; large and LFS-tracked result files are not duplicated there and remain in tag `v1.0-original`); Git LFS rules removed.

**v2.1 (first-submission repository; superseded)** Documentation reconciliation of the first submission. See `legacy_v1/README_v2.1.md`. Its results are not used in the revised manuscript.

---

## 12. License

MIT License. See [LICENSE](LICENSE) for details.
