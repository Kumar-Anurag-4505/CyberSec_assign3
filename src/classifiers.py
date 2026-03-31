"""
RENTAKA v2 — Classifiers & Evaluation
========================================
All five classifiers from the RENTAKA paper plus comprehensive evaluation.
Handles the class imbalance present in the real khas-ccip dataset.
"""

import os, sys, time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, confusion_matrix,
                                      roc_auc_score, f1_score,
                                      precision_score, recall_score,
                                      classification_report)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RANDOM_STATE = 42
CV_FOLDS     = 10
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__))), "results")

# Paper-reported results for comparison (RENTAKA, Zakaria et al. 2024)
PAPER_RESULTS = {
    "SVM":                 {"Accuracy": 97.0492, "TPR": 0.995, "FPR": 0.071, "AUC": 0.979},
    "Random Forest":       {"Accuracy": 96.3934, "TPR": 0.984, "FPR": 0.071},
    "kNN":                 {"Accuracy": 96.0656, "TPR": 0.979, "FPR": 0.071},
    "J48 (Decision Tree)": {"Accuracy": 94.7541, "TPR": 0.979, "FPR": 0.106},
    "Naive Bayes":         {"Accuracy": 80.9836, "TPR": 0.781, "FPR": 0.142},
}

COLOR_MAP = {
    "SVM":                 "#2196F3",
    "Random Forest":       "#4CAF50",
    "Naive Bayes":         "#FF9800",
    "kNN":                 "#9C27B0",
    "J48 (Decision Tree)": "#F44336",
}


def build_classifiers(class_weight: str = "balanced") -> Dict:
    """
    Instantiate all five classifiers.
    class_weight='balanced' handles the imbalanced dataset automatically.
    """
    return {
        "Naive Bayes": GaussianNB(),

        "kNN": KNeighborsClassifier(n_neighbors=5, metric="euclidean",
                                     weights="uniform", n_jobs=-1),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, class_weight=class_weight,
                        random_state=RANDOM_STATE)),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=100, criterion="gini",
            class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1
        ),

        "J48 (Decision Tree)": DecisionTreeClassifier(
            criterion="entropy", class_weight=class_weight,
            random_state=RANDOM_STATE
        ),
    }


def train_evaluate_all(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Tuple[Dict, Dict]:
    """
    Train all 5 classifiers using 10-fold stratified CV.
    Returns results dict and predictions dict.
    """
    classifiers = build_classifiers()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    results      = {}
    all_preds    = {}

    n_ransom = (y == 1).sum()
    n_other  = (y == 0).sum()

    print(f"\n{'='*72}")
    print(f"  RENTAKA — {CV_FOLDS}-Fold Stratified Cross-Validation")
    print(f"  Dataset : {X.shape[0]} samples | {X.shape[1]} API features")
    print(f"  Class 1 (Ransomware): {n_ransom} | Class 0 (Other): {n_other}")
    print(f"{'='*72}")

    for name, clf in classifiers.items():
        print(f"\n▶ [{name}]  training...", flush=True)
        t0 = time.time()

        y_pred  = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
        y_proba = cross_val_predict(clf, X, y, cv=cv,
                                     method="predict_proba", n_jobs=-1)[:, 1]
        elapsed = time.time() - t0

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        acc  = accuracy_score(y, y_pred) * 100
        tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr  = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0
        prec = precision_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        auc  = roc_auc_score(y, y_proba)

        p = PAPER_RESULTS.get(name, {})

        results[name] = {
            "Accuracy (%)":   round(acc, 4),
            "TPR":            round(tpr, 3),
            "TNR":            round(tnr, 3),
            "FPR":            round(fpr, 3),
            "FNR":            round(fnr, 3),
            "Precision":      round(prec, 3),
            "F1-Score":       round(f1, 3),
            "AUC-ROC":        round(auc, 3),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Train Time (s)": round(elapsed, 1),
            "Paper Acc (%)":  p.get("Accuracy", "N/A"),
            "Paper TPR":      p.get("TPR", "N/A"),
            "Paper FPR":      p.get("FPR", "N/A"),
            "Paper AUC":      p.get("AUC", "N/A"),
        }

        all_preds[name] = {"y_pred": y_pred, "y_proba": y_proba}

        print(f"   Accuracy : {acc:>8.4f}%  │  Paper: {p.get('Accuracy','N/A')}%")
        print(f"   TPR      : {tpr:>8.3f}   │  Paper: {p.get('TPR','N/A')}")
        print(f"   FPR      : {fpr:>8.3f}   │  Paper: {p.get('FPR','N/A')}")
        print(f"   AUC-ROC  : {auc:>8.3f}   │  Paper: {p.get('AUC','N/A')}")
        print(f"   F1-Score : {f1:>8.3f}   │  Time : {elapsed:.1f}s")

    return results, all_preds


def print_comparison_table(results: Dict):
    """Print results comparison table."""
    print(f"\n{'='*80}")
    print("  FINAL RESULTS — Our Implementation vs RENTAKA Paper (Zakaria et al., 2024)")
    print(f"{'='*80}")
    print(f"{'Classifier':<22} {'OurAcc':>8} {'PaperAcc':>9} {'Δ':>6}"
          f" {'TPR':>6} {'FPR':>6} {'AUC':>7} {'F1':>7}")
    print("-" * 80)

    for name, r in sorted(results.items(), key=lambda x: -x[1]["Accuracy (%)"]):
        pa = r["Paper Acc (%)"]
        pa_str = f"{pa:.2f}" if isinstance(pa, float) else str(pa)
        diff   = (r["Accuracy (%)"] - pa) if isinstance(pa, float) else 0
        d_str  = f"{diff:+.2f}" if isinstance(pa, float) else " N/A"
        print(f"{name:<22} {r['Accuracy (%)']:>8.4f} {pa_str:>9} {d_str:>6}"
              f" {r['TPR']:>6.3f} {r['FPR']:>6.3f} {r['AUC-ROC']:>7.3f}"
              f" {r['F1-Score']:>7.3f}")
    print("=" * 80)


def save_models(X: np.ndarray, y: np.ndarray):
    """Train on full dataset and save models."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    clfs = build_classifiers()
    for name, clf in clfs.items():
        clf.fit(X, y)
        safe = name.replace(" ", "_").replace("(","").replace(")","")
        p = os.path.join(RESULTS_DIR, f"{safe}.pkl")
        joblib.dump(clf, p)
        print(f"  Saved: {p}")
    return clfs


def save_results_csv(results: Dict) -> str:
    """Save results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for name, r in results.items():
        rows.append({"Classifier": name, **r})
    path = os.path.join(RESULTS_DIR, "full_results.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path
