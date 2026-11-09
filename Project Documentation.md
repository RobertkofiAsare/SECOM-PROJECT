# SECOM Case Study – Project Documentation

Fault detection in semiconductor manufacturing using the UCI SECOM
dataset  
Master level, second-semester case study · Version 0.1 (working draft) ·
29 October 2026

## 1. Purpose and scope

This document describes the SECOM case study: its business context,
data, methodology, software implementation, evaluation framework,
preliminary findings and open work. It serves three audiences: the
examiners (traceability of every decision), the project author (single
reference while writing the notebook and presentation), and any future
reader who needs to reproduce the analysis.

**Project goal.** Build a reproducible machine-learning pipeline that
predicts, from process sensor data, whether a wafer lot will fail the
final functionality test, and quantify the economic value of doing so.

**Deliverables required by the course manual**

- A reproducible Python script (runs in VS Code, downloads the data at
  runtime, installs missing packages, documents every function in a
  header comment).

- A scientific notebook (CRISP-DM structure, scientific language, Step
  Impact Protocol tables).

- A presentation exported as PDF using the course template.

**Out of scope.** Deployment, real-time monitoring, root-cause analysis
of individual sensors (sensor names and process steps are anonymised in
SECOM).

## 2. Business and process context

Non-volatile memory (NAND flash) retains data without power and is
manufactured from ultra-pure silicon derived from quartz sand. The
process chain relevant to this project is:

1.  **Silicon preparation.** Purification yields polycrystalline silicon
    of about 99.9999999 % purity, melted above 1400 °C. The Czochralski
    process draws a single-crystal ingot (typically 300 mm diameter,
    1.5–2 m long, more than 100 kg).

2.  **Wafer production.** Diamond saws slice the ingot into wafers about
    0.775 mm thick; grinding and polishing make the surface almost
    atomically smooth.

3.  **Layer build-up.** In 3D NAND, dozens to hundreds of alternating
    silicon nitride and silicon dioxide layers are deposited (typically
    by chemical vapour deposition). Vertical channels are etched through
    the stack and filled to form the memory cells.

4.  **Photolithography.** Photoresist is applied, exposed to UV light
    through a mask and selectively etched to create horizontal
    structures. The step can be repeated hundreds of times.

5.  **Wafer sort / probe.** Probe needles contact the pads of each die
    and test electrical performance; defective dies are marked. The
    share of fully functional dies (yield) usually lies between about 70
    % and more than 95 % and depends strongly on process maturity. Newer
    technologies with more layers typically start with lower yield.

**Why prediction matters.** Hundreds of process steps and many sensors
generate data long before the final test. Detecting a probable failure
early allows engineers to intervene, reduce scrap and limit the cost of
processing wafers that will fail anyway. Missed failures (Type II
errors) are more expensive than false alarms (Type I errors) because a
faulty product continues through costly steps or reaches the customer.

**Mapping of SECOM to the process.** Each SECOM row is one production
entity (wafer/lot) with 590 sensor-type measurements and one pass/fail
label from an in-house functionality test, together with a timestamp.
Sensors and process steps are anonymised, so features cannot be assigned
to a specific process step above.

## 3. Data description

| Property       | Value                                                                        |
|----------------|------------------------------------------------------------------------------|
| Source         | UCI Machine Learning Repository, dataset id 179 (McCann & Johnston, 2008)    |
| Files          | secom.data (features), secom_labels.data (label + timestamp), secom.names    |
| Size           | 1567 rows × 590 features                                                     |
| Target         | Raw coding −1 = pass, +1 = fail; recoded to fail = 1, pass = 0               |
| Class balance  | 1463 pass / 104 fail (6.64 % fail, about 1 : 14)                             |
| Missing values | 41 951 NaN cells (about 4.5 % of all cells), concentrated in some columns    |
| Time span      | 19 July 2008 11:55 to 17 October 2008 06:07; rows are in chronological order |

**Challenges**

- Strong class imbalance: a model that always predicts "pass" reaches
  about 93 % accuracy and detects no failure. Accuracy is therefore not
  used as a selection criterion.

- Very few positives: with an 80/20 split the test set contains only
  about 21 fails, so single-split metrics are noisy.

- High dimensionality relative to the number of failures (590 features,
  104 failures).

- Many constant columns, many NaN-heavy columns and probable outliers.

- Anonymised features limit interpretation.

## 4. Methodology

### 4.1 Process model

The project follows CRISP-DM (business understanding, data
understanding, data preparation, modelling, evaluation, deployment). The
cycle is iterated: findings from modelling feed back into preparation.
Deployment is not required.

### 4.2 Pipeline

**Split → Preprocessing → Feature selection → Balancing → Training →
Evaluation**

### 4.3 Governing rules (derived from the course manual)

| Rule                                                                                                                                                    | Reason                                                              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| Stratified 80/20 split is the first operation after loading                                                                                             | Preserves the 6.6 % fail share in both parts; protects the test set |
| Every parameter (thresholds, 3σ limits, scaler, imputer, selected features) is learned on training data only and then applied unchanged to the test set | Prevents information leakage and over-optimistic results            |
| No mean imputation                                                                                                                                      | Mean imputation shrinks variance and distorts correlations          |
| Outliers by 3σ rule, not IQR                                                                                                                            | Course specification; limits learned on training data               |
| Balancing (SMOTE or ROSE) on training data only                                                                                                         | Test data must reflect real prevalence                              |
| Hyper-parameters chosen by cross-validation on training data, never by test results                                                                     | Keeps the test set an unbiased final check                          |
| Success measured by cost, sensitivity, precision, F1 and ROC-AUC                                                                                        | Accuracy is misleading under imbalance                              |
| Every dataset-changing decision is logged in the Step Impact Protocol (SIP)                                                                             | Traceability                                                        |

## 5. Preprocessing and modelling design

### 5.1 Preprocessing steps (class Preprocessor)

1.  **Remove constant columns** (variance 0 or entirely missing).

2.  **Remove duplicate columns.**

3.  **Remove columns with too many NaNs.** Starting threshold 60 %
    (course suggestion); the two reference papers use 55 %. The
    threshold is justified with a missing-value histogram and a Pareto
    chart.

4.  **Outlier handling (3σ).** Limits are mean ± 3 standard deviations
    from the training set. Option cap replaces values by the boundary;
    option nan sets them to NaN and re-imputes; both are compared in the
    grid search.

5.  **Min–max scaling to \[0, 1\]**, fitted on training data. Scaling
    precedes imputation because kNN is distance-based.

6.  **kNN imputation** (k = 5, grid 5 and 10), fitted on training data.

The fitted object stores all parameters, so transform() reproduces
exactly the same treatment for the test set.

### 5.2 Feature selection

Boruta (all-relevant selection with random-forest importance and shadow
features) is applied to the preprocessed training data. Confirmed and
tentative features are retained; if none are found, the 15 best-ranked
features are used. The course expects roughly 15–20 features.

### 5.3 Balancing

SMOTE creates synthetic minority samples in the training data. ROSE
(smoothed bootstrap), class weights and under-sampling are planned
comparison variants. Test data are never resampled.

### 5.4 Model

Random forest (default 300 trees). It needs no normality assumption,
tolerates mixed scales and gives importance information.
Hyper-parameters tuned in grid search: outlier mode, kNN k and tree
depth. Further candidates (logistic regression, gradient boosting, kNN
as in the papers) can be added for comparison.

## 6. Evaluation framework

**Confusion matrix convention.** Positive class = fail (1).

|             | Predicted fail           | Predicted pass             |
|-------------|--------------------------|----------------------------|
| Actual fail | TP                       | FN (Type II, missed fault) |
| Actual pass | FP (Type I, false alarm) | TN                         |

**Metrics**

- Sensitivity (recall) = TP / (TP + FN)

- Specificity = TN / (TN + FP)

- Precision = TP / (TP + FP)

- F1 = 2TP / (2TP + FP + FN)

- ROC-AUC (threshold-free ranking quality)

- Accuracy (reported for completeness only)

**Cost model.** Total cost = 100 × FP + 400 × FN (example values from
the manual: a false alarm costs 100 units, a missed fault 400). The
ratio 1 : 4 must be justified in the notebook and tested for sensitivity
(for example 1 : 2, 1 : 10). The decision threshold (default 0.5) is a
candidate for cost-based optimisation on out-of-fold predictions.

**Uncertainty.** With about 21 test failures, one additional correct
detection changes sensitivity by almost 5 percentage points. Final
reporting should include repeated stratified cross-validation and
confidence intervals (for example Wilson or bootstrap) alongside the
single test-set result.

## 7. Software documentation

**Repository layout**

| Path                            | Content                                                                  |
|---------------------------------|--------------------------------------------------------------------------|
| src/secom_analysis.py           | Complete pipeline                                                        |
| notebooks/scientific_diary.md   | Diary entries and SIP table                                              |
| docs/project_plan.md            | Weekly plan and integrity checklist                                      |
| results/figures, results/tables | Generated outputs                                                        |
| requirements.txt                | numpy, pandas, scipy, matplotlib, scikit-learn, imbalanced-learn, boruta |

**Function reference**

| Function / class           | Input                                           | Output                        | Purpose                                                              |
|----------------------------|-------------------------------------------------|-------------------------------|----------------------------------------------------------------------|
| ensure_packages            | dict of import name → pip name                  | none                          | Installs missing packages automatically                              |
| load_secom                 | none (or env SECOM_LOCAL_DIR for offline tests) | X, y, timestamps              | Downloads and parses SECOM, flags NaN, recodes target                |
| Preprocessor.fit_transform | training features                               | processed training features   | Learns and applies cleaning, 3σ, scaling, kNN imputation             |
| Preprocessor.transform     | new features                                    | processed features            | Applies fitted steps unchanged                                       |
| run_pipeline               | Xtrain, ytrain, Xtest, parameters               | fail probabilities, artefacts | Preprocessing, Boruta, SMOTE, random forest, prediction              |
| total_cost                 | true labels, predicted labels                   | cost                          | 100 × FP + 400 × FN                                                  |
| metrics                    | true labels, probabilities, threshold           | dict of metrics               | Confusion matrix and all scores                                      |
| eda_figures                | training features                               | PNG files                     | NaN histogram, Pareto chart, correlation heatmap                     |
| grid_search                | training data, quick flag                       | trial table                   | Stratified 3-fold CV over parameter grid, ranked by cost             |
| main                       | command-line flags                              | result files                  | Orchestrates split, optional grid search, final fit, test evaluation |

**Execution**

- python src/secom_analysis.py – full run

- python src/secom_analysis.py --quick – fast smoke test

- python src/secom_analysis.py --grid – with grid search

**Reproducibility.** Random seed 42 everywhere; the dataset is
downloaded at runtime; the script writes all tables and figures to
results/. No manual edits are needed between runs.

**Known limitations of the code**

- Grid search is small and slow because Boruta is refitted inside every
  fold; reduce iterations or extend only when time allows.

- The data download needs internet access to archive.ics.uci.edu.

- Hyper-parameters of Boruta (max_iter, perc) and the random forest are
  not yet tuned.

## 8. Preliminary findings

These come from a single quick-mode run (15 Boruta iterations, 100
trees, k = 5, outlier mode cap) on a local copy of the data. They
demonstrate that the pipeline works; they are not final results.

| Step (training set)             | Columns before | Columns after |
|---------------------------------|----------------|---------------|
| Constant columns removed        | 590            | 474           |
| Duplicate columns removed       | 474            | 474           |
| More than 60 % NaN removed      | 474            | 450           |
| 3σ cap, min–max, kNN imputation | 450            | 450           |
| Boruta (confirmed + tentative)  | 450            | 10            |

**Split check:** fail share 6.64 % overall, 6.62 % train, 6.69 % test.

**Test result (quick run):** TP 6, FN 15, FP 17, TN 276; sensitivity
0.29, specificity 0.94, precision 0.26, F1 0.27, AUC 0.71, cost 7 700
units. Three-fold cross-validated cost on the training set was 12 000
with AUC 0.65.

**Interpretation.** Specificity is high but most faults are missed. With
the 1 : 4 cost ratio, catching more faults at the price of more false
alarms is likely cheaper. Threshold optimisation, full Boruta settings
and comparison of balancing methods are expected to change the picture.
Deviations from the manual's expectations (450 columns instead of about
470; 10 features instead of 15–20) must be re-checked in the full run.

## 9. Literature review summary

| Aspect          | Munirathinam & Ramadoss (2016)                                                                                                                                                                         | Kerdprasop & Kerdprasop (conference paper)                                                                                                                   |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Approach        | CRISP-DM workflow; SME knowledge, correlation (                                                                                                                                                        | r                                                                                                                                                            |
| Preprocessing   | 55 % NaN threshold; scaling to \[−1, 1\]; missing values replaced by class mean                                                                                                                        | 55 % NaN threshold; constant-column removal                                                                                                                  |
| Headline result | Naïve Bayes high detection with high false-alarm rate; kNN and decision tree summarised in a table                                                                                                     | Boosted decision tree TP rate 1.0, FP rate 0.161; kNN TP rate 0.983                                                                                          |
| Concerns        | Class-mean imputation uses the label (leakage); correlation pruning ignores relevance to the target; inconsistent counts (1115 vs 1567 instances; 452 instances discarded); incomplete ANN/SVM results | Feature selection described before the split; test set with 12.6 % failures instead of 6.6 %; perfect TP rate based on 59 positives, no confidence intervals |

**Consequences for this project:** split first, learn on training data
only, replace class-mean imputation by kNN, stratify the test set,
balance only the training data, and report uncertainty. A citation check
is required for the second paper (the course manual and the uploaded
manuscript show different years).

## 10. Risks and mitigation

| Risk                                      | Effect                        | Mitigation                                                             |
|-------------------------------------------|-------------------------------|------------------------------------------------------------------------|
| Too few failures in test set              | Unstable metrics              | Repeated stratified CV, confidence intervals, report both              |
| Information leakage                       | Over-optimistic results       | Single run_pipeline for fit and transform; checklist before submission |
| Over-fitting through SMOTE or grid search | Poor generalisation           | Resample only inside training folds; select by CV cost                 |
| Arbitrary cost ratio                      | Questionable optimum          | Justify and run sensitivity analysis                                   |
| Temporal drift (July–October 2008)        | Random split too optimistic   | Optional time-ordered validation                                       |
| Runtime of Boruta                         | Slow iteration                | Quick mode; limited grid; cached results                               |
| Anonymised features                       | No engineering interpretation | State as limitation; discuss importance statistically                  |

## 11. Decision log (Step Impact Protocol, current state)

| \#  | Decision                                               | Alternatives                 | Effect                              |
|-----|--------------------------------------------------------|------------------------------|-------------------------------------|
| 1   | Drop zero-variance columns                             | Keep all                     | 590 → 474                           |
| 2   | Drop duplicate columns                                 | Keep                         | no duplicates found                 |
| 3   | NaN threshold 60 % (to be confirmed with Pareto chart) | 40 %, 55 %                   | 474 → 450                           |
| 4   | 3σ rule with capping (compare setting to NaN)          | None, IQR (not allowed)      | outlier counts to be logged         |
| 5   | kNN imputation, k = 5 (compare 10)                     | Mean, MICE                   | 0 NaN remaining                     |
| 6   | Boruta selection                                       | RFE, L1, MeanDiff            | 450 → about 10–20                   |
| 7   | SMOTE on training only                                 | ROSE, ADASYN, under-sampling | train fail share about 6.6 % → 50 % |

The full SIP with rationale is maintained in the scientific diary.

## 12. Roadmap

| Week    | Focus                                                | Documentation output                |
|---------|------------------------------------------------------|-------------------------------------|
| 1       | Papers, set-up, wafer Q&A                            | Diary entries, Q&A answers          |
| 2       | Split, cleaning, NaN analysis, 3σ                    | SIP rows 1–4, NaN diagrams, heatmap |
| 3       | Imputation theory, kNN, first random forest          | SIP row 5                           |
| 4       | Boruta, SMOTE/ROSE, metrics theory                   | SIP rows 6–7                        |
| 5       | Scaling check, test-set application, process map     | Process map, test confusion matrix  |
| 6       | Balancing comparison, grid search, cost optimisation | Trial log, cost curves              |
| 7       | Final pipeline and test-set evaluation               | Results chapter                     |
| Wrap-up | Consistency of script, notebook and slides           | PDF presentation, final script      |

## 13. Glossary

| Term           | Meaning                                                                |
|----------------|------------------------------------------------------------------------|
| SECOM          | Semiconductor manufacturing dataset (UCI)                              |
| Wafer sort     | Electrical die test before dicing                                      |
| Yield          | Share of functional dies                                               |
| Type I error   | False alarm: good wafer flagged as fail                                |
| Type II error  | Missed fault: failing wafer classified as pass                         |
| Sensitivity    | Share of actual failures detected                                      |
| Specificity    | Share of good wafers correctly passed                                  |
| SMOTE          | Synthetic Minority Over-sampling Technique                             |
| ROSE           | Random Over-Sampling Examples (smoothed bootstrap)                     |
| Boruta         | Wrapper feature selection comparing features with random shadow copies |
| kNN imputation | Fills a missing value from the k most similar rows                     |
| SIP            | Step Impact Protocol: decision log with alternatives and consequences  |
| CRISP-DM       | Cross-industry standard process for data mining                        |

## 14. References

- McCann, M., Li, Y., Maguire, L., Johnston, A. (2008). Causality
  challenge: benchmarking relevant signal components for effective
  monitoring and process control. JMLR Workshop and Conference
  Proceedings, pp. 277–288.

- McCann, M., Johnston, A. (2008). SECOM. UCI Machine Learning
  Repository, dataset 179.

- Munirathinam, S., Ramadoss, B. (2016). Predictive models for equipment
  fault detection in the semiconductor manufacturing process. IACSIT
  International Journal of Engineering and Technology, 8(4), 273–285.

- Kerdprasop, K., Kerdprasop, N. Feature selection and boosting
  techniques to improve fault detection accuracy in the semiconductor
  manufacturing process. IAENG conference paper (verify year and
  proceedings).

- Course manual: SECOM Case Study – Student's Manual.

- Kursa, M. B., Rudnicki, W. R. (2010). Feature selection with the
  Boruta package. Journal of Statistical Software, 36(11).

- Chawla, N. V. et al. (2002). SMOTE: Synthetic minority over-sampling
  technique. Journal of Artificial Intelligence Research, 16, 321–357.
