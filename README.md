# SECOM PROJECT
# SECOM Case Study (MSc Project Management & Data Science)

Fault detection on semiconductor sensor data (UCI SECOM: 1567 wafers, 590 sensors, 6.6 % fails).

## Run
```
pip install -r requirements.txt
python src/secom_analysis.py            # full run, downloads data from UCI at runtime
python src/secom_analysis.py --quick    # fast smoke test
python src/secom_analysis.py --grid     # cost-based grid search (CV on training set only)
```
Outputs: `results/figures/*.png`, `results/tables/*.csv`. Evaluation figures include
the ROC curve, precision-recall curve, confusion matrix, and cost by classification
threshold, all generated from the held-out test set. Treat the cost curve as diagnostic;
select any operating threshold using training data or cross-validation, not the test set.

## Docs
- `docs/project_plan.md` – week-by-week plan and test-set integrity checklist
- `notebooks/scientific_diary.md` – scientific diary entries + SIP table (copy into Google Docs template)

