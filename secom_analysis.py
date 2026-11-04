"""
SECOM case study - end-to-end pipeline (CRISP-DM, HTW Berlin MPMD).

Pipeline: Split -> Preprocessing -> Feature selection -> Balancing -> Training -> Evaluation.
Rule: every parameter is learned on the TRAINING set only and applied unchanged to the test set.

Usage:  python src/secom_analysis.py [--quick] [--grid]
Env:    SECOM_LOCAL_DIR=<folder with secom.data / secom_labels.data> (offline testing only;
        the submitted run downloads the data from UCI at runtime)
"""
# ---------------------------------------------------------------- 0. Setup
import subprocess, sys, importlib, os, io, zipfile, argparse, itertools, json, urllib.request
from pathlib import Path


def ensure_packages(pkgs):
    """Install missing packages. Input: {import_name: pip_name}. Output: none."""
    for mod, pip_name in pkgs.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


ensure_packages({"numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
                 "sklearn": "scikit-learn", "imblearn": "imbalanced-learn", "boruta": "boruta"})

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy

# ---------------------------------------------------------------- 1. Configuration
SEED = 42
URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
COST_TYPE_I, COST_TYPE_II = 100, 400   # Type I = false alarm (pass flagged fail); Type II = missed fail
MISSING_THRESHOLD = 0.60               # manual: start with 60 % (papers use 55 %)
DEFAULTS = dict(outlier="cap", knn_k=5, n_trees=300, max_depth=None, boruta_iter=50)


# ---------------------------------------------------------------- 2. Data loading
def load_secom():
    """Download SECOM at runtime (UCI).
    Output: X (DataFrame 1567x590), y (Series, 1 = fail, 0 = pass), ts (timestamps)."""
    local = os.environ.get("SECOM_LOCAL_DIR")
    if local:
        raw, lab = Path(local) / "secom.data", Path(local) / "secom_labels.data"
    else:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
        raw, lab = z.open("secom.data"), z.open("secom_labels.data")
    X = pd.read_csv(raw, sep=r"\s+", header=None, na_values=["NaN"])      # every NaN flagged explicitly
    L = pd.read_csv(lab, sep=" ", header=None, names=["label", "time"])
    X.columns = [f"f{i:03d}" for i in range(X.shape[1])]
    y = (L["label"] == 1).astype(int)          # original coding: -1 pass, 1 fail -> fail = positive class = 1
    ts = pd.to_datetime(L["time"].str.strip('"'), format="%d/%m/%Y %H:%M:%S")
    return X, y, ts


# ---------------------------------------------------------------- 3. Preprocessing (fit on train only)
class Preprocessor:
    """Learns all preprocessing parameters on the training set; transform() applies them unchanged."""

    def __init__(self, outlier="cap", knn_k=5, miss_thr=MISSING_THRESHOLD):
        self.outlier, self.knn_k, self.miss_thr, self.sip = outlier, knn_k, miss_thr, []

    def _log(self, step, before, after):
        self.sip.append(dict(step=step, before=before, after=after))

    def fit_transform(self, X):
        """Input: training features (with NaNs). Output: scaled, outlier-treated, imputed DataFrame."""
        n0 = X.shape[1]
        X = X.loc[:, X.var(skipna=True) > 0]                              # constant / all-NaN columns
        self._log("remove constant columns", n0, X.shape[1])
        n1 = X.shape[1]
        X = X.loc[:, ~X.T.duplicated().values]                            # duplicated columns
        self._log("remove duplicate columns", n1, X.shape[1])
        n2 = X.shape[1]
        X = X.loc[:, X.isna().mean() <= self.miss_thr]                    # too many NaNs
        self._log(f"remove columns with >{self.miss_thr:.0%} NaN", n2, X.shape[1])
        self.cols_ = X.columns
        mu, sd = X.mean(), X.std()                                        # 3-sigma limits from TRAIN
        self.lo_, self.hi_ = mu - 3 * sd, mu + 3 * sd
        X = self._outliers(X)
        self.scaler_ = MinMaxScaler().fit(X)                              # scale BEFORE kNN (distance based)
        Xs = pd.DataFrame(self.scaler_.transform(X), columns=self.cols_, index=X.index)
        self.imputer_ = KNNImputer(n_neighbors=self.knn_k).fit(Xs)
        out = pd.DataFrame(self.imputer_.transform(Xs), columns=self.cols_, index=X.index)
        self._log(f"3-sigma ({self.outlier}) + min-max + kNN impute (k={self.knn_k})", X.shape[1], out.shape[1])
        return out

    def _outliers(self, X):
        """3-sigma rule. 'cap' = replace by boundary, 'nan' = set to NaN (re-imputed later), 'none' = keep."""
        if self.outlier == "cap":
            return X.clip(self.lo_, self.hi_, axis=1)
        if self.outlier == "nan":
            return X.where((X >= self.lo_) & (X <= self.hi_))
        return X

    def transform(self, X):
        """Apply the fitted steps to new data (test set). Output: imputed, scaled DataFrame."""
        X = self._outliers(X[self.cols_])
        Xs = pd.DataFrame(self.scaler_.transform(X), columns=self.cols_, index=X.index)
        return pd.DataFrame(self.imputer_.transform(Xs), columns=self.cols_, index=X.index)


# ---------------------------------------------------------------- 4. Model run (train -> predict)
def run_pipeline(Xtr, ytr, Xte, params):
    """Fit preprocessing, Boruta, SMOTE and RF on (Xtr, ytr); predict Xte.
    Output: (predicted probability of fail, dict of fitted artefacts)."""
    p = {**DEFAULTS, **params}
    prep = Preprocessor(p["outlier"], p["knn_k"])
    A = prep.fit_transform(Xtr)
    B = prep.transform(Xte)
    rf0 = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    bor = BorutaPy(rf0, n_estimators="auto", max_iter=p["boruta_iter"], random_state=SEED, verbose=0)
    bor.fit(A.values, ytr.values)
    sel = A.columns[bor.support_ | bor.support_weak_]                     # confirmed + tentative
    if len(sel) == 0:                                                     # fallback: best-ranked 15
        sel = A.columns[np.argsort(bor.ranking_)[:15]]
    Asel, Bsel = A[sel], B[sel]
    Xb, yb = SMOTE(random_state=SEED).fit_resample(Asel, ytr)             # balance TRAIN only
    rf = RandomForestClassifier(n_estimators=p["n_trees"], max_depth=p["max_depth"],
                                n_jobs=-1, random_state=SEED).fit(Xb, yb)
    return rf.predict_proba(Bsel)[:, 1], dict(prep=prep, selected=list(sel), model=rf)


def total_cost(y, pred):
    """Cost = COST_TYPE_I * FP + COST_TYPE_II * FN. Input: true labels, predicted labels."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return COST_TYPE_I * fp + COST_TYPE_II * fn


def metrics(y, proba, thr=0.5):
    """Confusion matrix, accuracy, sensitivity, specificity, precision, F1, AUC and cost."""
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(TN=tn, FP=fp, FN=fn, TP=tp, accuracy=(tp + tn) / len(y),
                sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                precision=tp / max(tp + fp, 1), F1=f1_score(y, pred, zero_division=0),
                AUC=roc_auc_score(y, proba), cost=COST_TYPE_I * fp + COST_TYPE_II * fn, threshold=thr)


def evaluation_figures(y, proba, threshold=0.5):
    """Save complementary test-set classification plots."""
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pass", "Fail"],
           yticklabels=["Pass", "Fail"], xlabel="Predicted label",
           ylabel="True label", title=f"Confusion matrix (threshold={threshold:.2f})")
    fig.colorbar(image, ax=ax, label="Wafers")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.savefig(FIG / "confusion_matrix_test.png", dpi=150, bbox_inches="tight")

    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision)
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall (sensitivity)",
           ylabel="Precision", title=f"Precision-recall, AP={ap:.2f}")
    fig.savefig(FIG / "precision_recall_test.png", dpi=150, bbox_inches="tight")

    thresholds = np.linspace(0, 1, 201)
    costs = [total_cost(y, proba >= t) for t in thresholds]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(thresholds, costs)
    ax.axvline(threshold, color="r", ls="--", label=f"Current threshold={threshold:.2f}")
    ax.set(xlabel="Classification threshold", ylabel="Misclassification cost",
           title="Test-set cost by threshold (diagnostic)")
    ax.legend()
    fig.savefig(FIG / "cost_by_threshold_test.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------- 5. EDA figures (training set only)
def eda_figures(Xtr):
    """Missing-value histogram, Pareto chart and correlation heatmap (manual week 2)."""
    miss = Xtr.isna().mean() * 100
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(miss)), miss.values)
    ax.axhline(MISSING_THRESHOLD * 100, color="r", ls="--")
    ax.set(xlabel="column", ylabel="% missing")
    fig.savefig(FIG / "missing_per_column.png", dpi=150, bbox_inches="tight")
    thr = np.arange(0, 101, 5)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(thr, [(miss > t).sum() for t in thr], width=4)
    ax.set(xlabel="NaN threshold (%)", ylabel="columns removed")
    fig.savefig(FIG / "pareto_missing.png", dpi=150, bbox_inches="tight")
    corr = Xtr.loc[:, Xtr.var() > 0].corr().abs()                         # visualise only, nothing removed
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.colorbar(ax.imshow(corr, cmap="coolwarm", vmin=0, vmax=1))
    fig.savefig(FIG / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------- 6. Grid search (CV on TRAIN only, cost-based)
def grid_search(Xtr, ytr, quick):
    """Stratified 3-fold CV over outlier handling, kNN k, RF depth.
    Output: DataFrame of all trials (kept as documentation), best (lowest CV cost) first."""
    grid = dict(outlier=["cap", "nan"], knn_k=[5, 10], max_depth=[None, 10])
    if quick:
        grid = dict(outlier=["cap"], knn_k=[5], max_depth=[None])
    rows = []
    for vals in itertools.product(*grid.values()):
        params = dict(zip(grid, vals))
        costs, aucs = [], []
        for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xtr, ytr):
            pr, _ = run_pipeline(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b],
                                 {**params, "boruta_iter": 20, "n_trees": 150})
            costs.append(total_cost(ytr.iloc[b], pr >= 0.5))
            aucs.append(roc_auc_score(ytr.iloc[b], pr))
        rows.append({**params, "cv_cost": np.mean(costs), "cv_auc": np.mean(aucs)})
        print(rows[-1])
    return pd.DataFrame(rows).sort_values("cv_cost")


# ---------------------------------------------------------------- 7. Main
def main():
    """Split -> (optional grid search) -> final fit on train -> evaluation on untouched test set."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    X, y, ts = load_secom()
    print(f"shape={X.shape}, fail share={y.mean():.3%}, NaNs={int(X.isna().sum().sum())}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"fail share  all={y.mean():.4f}  train={ytr.mean():.4f}  test={yte.mean():.4f}")
    eda_figures(Xtr)
    params = dict(DEFAULTS)
    if a.quick:
        params.update(boruta_iter=15, n_trees=100)
    if a.grid:
        trials = grid_search(Xtr, ytr, a.quick)
        trials.to_csv(TAB / "grid_search.csv", index=False)
        best = trials.iloc[0]
        params.update(outlier=best["outlier"], knn_k=int(best["knn_k"]),
                      max_depth=None if pd.isna(best["max_depth"]) else int(best["max_depth"]))
    proba, art = run_pipeline(Xtr, ytr, Xte, params)
    pd.DataFrame(art["prep"].sip).to_csv(TAB / "sip_counts.csv", index=False)
    m = metrics(yte, proba)
    print(json.dumps({k: float(v) for k, v in m.items()}, indent=1))
    print("selected features:", art["selected"])
    pd.DataFrame([m]).to_csv(TAB / "test_metrics.csv", index=False)
    fpr, tpr, _ = roc_curve(yte, proba)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="FPR (1-specificity)", ylabel="TPR (sensitivity)", title=f"ROC, AUC={m['AUC']:.2f}")
    fig.savefig(FIG / "roc_test.png", dpi=150, bbox_inches="tight")
    evaluation_figures(yte, proba, threshold=m["threshold"])


if __name__ == "__main__":
    main()
