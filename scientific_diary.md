# SECOM Case Study – Scientific Notebook (working draft)

> Style rule from the manual: scientific language, no "we"/"I", bullet points, short but precise.
> Copy these entries into the Google Docs template (2nd column = answers). Dates are real work dates.
> Structure follows CRISP-DM. Every dataset-changing decision goes into the SIP table (end of file).

---

## Entry 001 – 2026-10-27 – Project set-up and source review

**Phase:** Business Understanding / Data Understanding
**Goal:** Establish project structure, read brief and papers, verify the data.

**Observations**
- Deliverables (manual): (1) reproducible Python script, (2) scientific notebook with SIP tables, (3) MPMD presentation (PDF).
- Script constraints: dataset downloaded at runtime (UCI, id 179), automatic package installation, run in VS Code, header comment per function, no manual changes needed.
- Dataset verified locally: 1567 rows × 590 sensor features; labels file with pass/fail and timestamp (19/07/2008 – 17/10/2008 range in file order).
- Target coding in raw file: −1 = pass, +1 = fail. Recoded to fail = 1 (positive class) so that sensitivity refers to detecting faults.
- Class balance: 1463 pass / 104 fail → 6.64 % fail (ratio ≈ 1 : 14). The dataset is strongly imbalanced.
- Missing values: 41 951 NaN cells in the raw matrix (≈ 4.5 % of all cells), flagged explicitly on import (`na_values=["NaN"]`).

**Decisions**
- Fail = positive class (1). Cost model: Type I (false alarm) = 100, Type II (missed fault) = 400 (manual example values; to be justified in the notebook).
- Random seed fixed (42) for all stochastic steps.

**Open questions**
- Business meaning of cost ratio 1:4 – justify with wafer-sort logic (late detection = costlier).
- Sensor order / process step unknown (manual, "Challenges of the SECOM dataset") – limits interpretability of selected features.

---

## Entry 002 – 2026-10-27 – Critical reading of the two required papers

**Purpose:** Identify what to reproduce and what to avoid.

### Paper 1 – Munirathinam & Ramadoss (2016)
- Describes a CRISP-DM-based workflow: cleaning → scaling → feature reduction → feature selection (SME knowledge, correlation |r| ≥ 0.8, VIA via chi-square / decision tree / random forest) → models (DT, NB, kNN, LR, ANN, SVM).
- 55 % missing threshold, constant-column removal, min–max scaling to [−1, 1].
- **Methodological concerns (to discuss, not to copy):**
  - Missing values replaced by the *class mean* → uses the label to impute (leakage) and, per manual, mean imputation is prohibited.
  - Correlation pruning removes one of two correlated features without knowing which is more informative (manual week 2, step 7).
  - Internal inconsistencies: dataset described as 1115 instances in one place and 1567 in another; "452 instances discarded" for missing values; figures referenced by wrong numbers in the text; ANN/SVM results table incomplete.
  - Oversampling by duplication is described before evaluation; split and selection order is not stated clearly enough to exclude leakage.

### Paper 2 – Kerdprasop & Kerdprasop
- Proposes MeanDiff (cluster-mean difference) feature ranking and rare-case boosting (duplicate fail cases until classes are balanced).
- Reported: decision tree TP rate rises to 1.0, FP 0.161; kNN (k = 1) TP 0.983.
- **Methodological concerns:**
  - Feature selection (PCA, chi-square, gain ratio, MeanDiff) is applied before the train/test separation in the described step order → possible information leakage into evaluation.
  - Test set of 468 rows contains 59 fails (12.6 %), i.e. not stratified to the 6.6 % base rate; precision (0.37–0.47) is therefore not comparable to the real deployment prevalence.
  - A TP rate of 1.0 with 59 test fails is a small-sample result (59 positives); confidence intervals are not reported.
  - Citation check required: manual lists 2003, the supplied manuscript is dated 2010/2011 (IAENG proceedings). Verify bibliographic data before citing.

### Takeaways for the own pipeline
- Split first (stratified 80/20), all learned parameters from training data only.
- Use 60 % NaN threshold first, compare with the papers' 55 %.
- Replace mean imputation with kNN imputation (scaled data).
- Balance only the training data (SMOTE/ROSE); never the test data.
- Report sensitivity, specificity, precision, F1, ROC-AUC and cost – not accuracy alone.

---

## Entry 003 – 2026-10-27 – Pipeline skeleton and smoke test

**Phase:** Data Preparation / Modelling (technical)
- Implemented `src/secom_analysis.py` with `Preprocessor` class (fit on train, transform for test), Boruta selection, SMOTE, random forest, cost-based metrics, grid search with stratified 3-fold CV on training data only.
- Stratified split check: fail share all = 6.64 %, train = 6.62 %, test = 6.69 %.
- **Smoke test result (quick settings, k = 5, cap outliers, 15 Boruta iterations, 100 trees, local data copy):**

| Step | Columns before | Columns after |
|---|---|---|
| Constant columns removed | 590 | 474 |
| Duplicate columns removed | 474 | 474 |
| > 60 % NaN removed | 474 | 450 |
| 3σ cap + min-max + kNN (k=5) | 450 | 450 |
| Boruta (confirmed + tentative) | 450 | 10 |

- Quick-mode test performance (single run, **not a final result**): sensitivity 0.29, specificity 0.94, AUC 0.71, cost 7 700 (TP 6, FN 15, FP 17).
- Interpretation: with 21 fails in the test set, metrics fluctuate strongly; conclusions require full settings, repeated stratified CV and confidence intervals.
- Note: the manual expects ≈ 470 columns after rough reduction and 15–20 Boruta features; 450 and 10 deviate slightly → check variance computation on NaN-containing columns and Boruta settings (`max_iter`, `perc`) in full run.

**To do next:** run in full mode, inspect the missing-value Pareto chart to justify the threshold, document 3σ options.

---

## Step Impact Protocol (SIP)

| # | Options / approaches | Pros & cons | Decision | Consequence (before → after) |
|---|---|---|---|---|
| 1 | Keep all / drop zero-variance columns | Zero-variance columns carry no information; dropping is lossless | Drop variance = 0 | 590 → 474 (116 removed; quick run, train set) |
| 2 | Keep / drop duplicate columns | Duplicates add redundancy only | Check & drop | 474 → 474 (none found) |
| 3 | NaN threshold 55 % (papers) / 60 % (manual start) / 40 % | Lower threshold removes more columns but loses possibly useful sensors; higher keeps more imputation uncertainty | 60 % (to be confirmed with Pareto chart) | 474 → 450 (24 removed) |
| 4 | Outliers (3σ): do nothing / cap at ±3σ / set NaN | Nothing: robust to nothing; cap: keeps rows, shrinks tails; NaN: removes information but may add gaps | Start with cap; compare "nan" in grid search | 450 → 450 columns; outlier counts to be logged |
| 5 | Imputation: mean / kNN / MICE | Mean destroys variance (prohibited); kNN uses neighbours (needs scaling); MICE slower | kNN, k = 5 (grid: 5, 10) | remaining NaN: 0 |
| 6 | Feature selection: Boruta / RFE / L1 / MeanDiff | Boruta all-relevant, stable but slow | Boruta (confirmed + tentative) | 450 → ~10–20 (full run pending) |
| 7 | Balancing: none / SMOTE / ROSE / ADASYN / undersampling | SMOTE synthetic, risk of noise; ROSE smoothed bootstrap | SMOTE on training only (ROSE as comparison) | train fail share ≈ 6.6 % → 50 % |

---

## Weekly log template (copy per week)

**Week N – date**
- Planned:
- Done (with script version / commit):
- Results (numbers, figures):
- Problems / deviations:
- Decisions (→ SIP row #):
- Literature used:
- Next week:
