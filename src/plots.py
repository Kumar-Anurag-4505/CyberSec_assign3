"""
RENTAKA v2 — All Visualizations
==================================
Generates all plots for the assignment report and demo.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
                  os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

COLOR_MAP = {
    "SVM":                 "#2196F3",
    "Random Forest":       "#4CAF50",
    "Naive Bayes":         "#FF9800",
    "kNN":                 "#9C27B0",
    "J48 (Decision Tree)": "#F44336",
}


def plot_dataset_overview(df: pd.DataFrame, api_cols: list) -> str:
    """3-panel dataset overview: family distribution, class balance, feature sparsity."""
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Panel 1 — family bar chart
    ax1 = fig.add_subplot(gs[0])
    fam_counts = df["family"].value_counts()
    colors = ["#E53935" if f == "Ransomware" else "#1E88E5"
              for f in fam_counts.index]
    ax1.barh(fam_counts.index[::-1], fam_counts.values[::-1],
             color=colors[::-1], alpha=0.85)
    ax1.set_xlabel("Sample count")
    ax1.set_title("Malware family distribution\n(VirusShare dataset)")
    # Custom legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color="#E53935", label="Ransomware (class 1)"),
        Patch(color="#1E88E5", label="Other malware (class 0)"),
    ], fontsize=9, loc="lower right")
    ax1.grid(axis="x", alpha=0.3)

    # Panel 2 — class balance pie
    ax2 = fig.add_subplot(gs[1])
    n_r  = (df["label"] == 1).sum()
    n_o  = (df["label"] == 0).sum()
    ax2.pie([n_r, n_o], labels=[f"Ransomware\n({n_r})", f"Other malware\n({n_o})"],
            colors=["#E53935", "#1E88E5"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
    ax2.set_title("Binary classification\nclass distribution")

    # Panel 3 — API import density by class
    ax3 = fig.add_subplot(gs[2])
    X    = df[api_cols].values
    y    = df["label"].values
    r_density = X[y == 1].sum(axis=1)
    o_density = X[y == 0].sum(axis=1)
    ax3.hist(r_density, bins=30, alpha=0.6, color="#E53935", label="Ransomware",
             density=True)
    ax3.hist(o_density, bins=30, alpha=0.6, color="#1E88E5", label="Other malware",
             density=True)
    ax3.set_xlabel("Number of imported APIs per sample")
    ax3.set_ylabel("Density")
    ax3.set_title("API import count distribution\nper class")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.suptitle("VirusShare Dataset Overview — khas-ccip/api_sequences_malware_datasets",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "dataset_overview.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_top_features(X: np.ndarray, y: np.ndarray,
                      feature_names: list, top_n: int = 20) -> str:
    """Horizontal bar chart of top discriminative API calls."""
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, random_state=42)

    top_idx   = np.argsort(mi)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    ran_pct   = [X[y==1, i].mean() * 100 for i in top_idx]
    oth_pct   = [X[y==0, i].mean() * 100 for i in top_idx]
    mi_scores = [mi[i] for i in top_idx]

    y_pos = np.arange(top_n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    gridspec_kw={"width_ratios": [3, 1]})

    ax1.barh(y_pos - 0.2, ran_pct, 0.38, label="Ransomware", color="#E53935", alpha=0.85)
    ax1.barh(y_pos + 0.2, oth_pct, 0.38, label="Other malware", color="#1E88E5", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Prevalence in class (%)")
    ax1.set_title(f"Top {top_n} discriminative API calls\n"
                  f"(RENTAKA pre-encryption features)")
    ax1.legend(fontsize=10)
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(y_pos, mi_scores, 0.6, color="#37474F", alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * top_n)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mutual Info\nscore")
    ax2.set_title("Feature\nimportance")
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("RENTAKA Feature Analysis — Windows API Call Importance",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "top_api_features.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_confusion_matrices(y_true: np.ndarray, all_preds: dict) -> str:
    """2×3 grid of confusion matrices."""
    names  = list(all_preds.keys())
    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9))
    axes = axes.flatten()

    for i, name in enumerate(names):
        yp = all_preds[name]["y_pred"]
        cm = confusion_matrix(y_true, yp)
        tn, fp, fn, tp = cm.ravel()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Other malware", "Ransomware"]
        )
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(f"{name}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(
            f"Ransom Prec: {tp/(tp+fp)*100:.1f}%  |  Other Prec: {tn/(tn+fn)*100:.1f}%",
            fontsize=8.5
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices — All 5 RENTAKA Classifiers\n"
                 "Dataset: VirusShare (khas-ccip), 10-fold CV",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_roc_curves(y_true: np.ndarray, all_preds: dict) -> str:
    """ROC curves for all classifiers — matches Figure 8 in the paper."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1.2, label="Random (AUC=0.50)")

    for name, preds in all_preds.items():
        fpr_v, tpr_v, _ = roc_curve(y_true, preds["y_proba"])
        auc_v = auc(fpr_v, tpr_v)
        ax.plot(fpr_v, tpr_v, color=COLOR_MAP.get(name, "#888"),
                lw=2.0, label=f"{name} (AUC={auc_v:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — RENTAKA Classifiers\n"
                 "Dataset: VirusShare (khas-ccip)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_accuracy_comparison(results: dict) -> str:
    """Side-by-side bar: our results vs paper reported."""
    from src.classifiers import PAPER_RESULTS
    names    = list(results.keys())
    our_acc  = [results[n]["Accuracy (%)"] for n in names]
    paper_acc = [
        PAPER_RESULTS[n]["Accuracy"] if n in PAPER_RESULTS else None
        for n in names
    ]

    x     = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width/2, our_acc,    width, label="Our implementation",
                color="#2196F3", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2,
                [p if p else 0 for p in paper_acc], width,
                label="RENTAKA paper (Zakaria et al., 2024)",
                color="#FF9800", alpha=0.85, edgecolor="white")

    for b in b1:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", (b.get_x()+b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for b, p in zip(b2, paper_acc):
        if p:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x()+b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (Decision Tree)", "") for n in names], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim([60, 105])
    ax.set_title("Classification Accuracy — Our Results vs RENTAKA Paper\n"
                 "Dataset: VirusShare (khas-ccip)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_metrics_heatmap(results: dict) -> str:
    """Heatmap of all metrics across all classifiers."""
    metrics = ["Accuracy (%)", "TPR", "TNR", "FPR", "FNR",
               "Precision", "F1-Score", "AUC-ROC"]
    names   = list(results.keys())
    data    = []
    for n in names:
        row = [results[n].get(m, 0) for m in metrics]
        row[0] /= 100  # normalize accuracy to 0–1
        data.append(row)

    df_h = pd.DataFrame(data, index=names, columns=metrics)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(df_h, annot=True, fmt=".3f", cmap="YlOrRd",
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"label": "Score (0–1)"})
    ax.set_title("Performance Metrics — All RENTAKA Classifiers\n"
                 "Dataset: VirusShare (khas-ccip)")
    ax.set_ylabel("")
    plt.yticks(rotation=0)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "metrics_heatmap.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_svm_detail(y_true: np.ndarray, all_preds: dict) -> str:
    """Detailed SVM confusion matrix matching Table 4 in the paper."""
    if "SVM" not in all_preds:
        return ""
    yp = all_preds["SVM"]["y_pred"]
    cm = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Other malware (0)", "Ransomware (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"SVM Classifier — Detailed Confusion Matrix\n"
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}\n"
        f"Ransom Recall={tp/(tp+fn):.3f}  Ransom Precision={tp/(tp+fp):.3f}",
        fontsize=10
    )
    ax.text(0.5, -0.13, "Paper (Table 4): True-0=254, True-1=155, FP=15, FN=12",
            ha="center", transform=ax.transAxes, fontsize=8, color="gray")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "svm_detail.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return path


def generate_all_plots(y_true, X, feature_names, df, results, all_preds):
    """Run all visualization functions."""
    api_cols = [c for c in df.columns if c not in ("label", "family")]
    print(f"\n{'='*55}")
    print("  Generating all plots...")
    print(f"{'='*55}")
    plot_dataset_overview(df, api_cols)
    plot_top_features(X, y_true, feature_names, top_n=20)
    plot_confusion_matrices(y_true, all_preds)
    plot_roc_curves(y_true, all_preds)
    plot_accuracy_comparison(results)
    plot_metrics_heatmap(results)
    plot_svm_detail(y_true, all_preds)
    print(f"\nAll plots saved to: {RESULTS_DIR}/")
