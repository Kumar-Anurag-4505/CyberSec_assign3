"""
RENTAKA v2 — Main Pipeline
============================
Full implementation using the khas-ccip VirusShare dataset.

USAGE:
    # With the REAL dataset (download first):
    git clone https://github.com/khas-ccip/api_sequences_malware_datasets
    python main.py --csv api_sequences_malware_datasets/VirusShare.csv

    # With simulated data (demo mode, no download needed):
    python main.py

    # Algorithm demo only:
    python main.py --demo
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset_loader        import load_virusshare_csv, summarize_dataset, get_api_feature_names
from src.simulate_real_dataset import build_simulated_virusshare_csv, PE_IMPORT_APIS
from src.rentaka_core          import build_rentaka_feature_matrix, print_rentaka_demo
from src.classifiers           import (train_evaluate_all, print_comparison_table,
                                        save_models, save_results_csv)
from src.plots                 import generate_all_plots

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SIM_CSV     = os.path.join(DATA_DIR, "VirusShare_simulated.csv")

BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║    RENTAKA — Windows Ransomware Early Detection                      ║
║    Dataset: github.com/khas-ccip/api_sequences_malware_datasets      ║
╠═══════════════════════════════════════════════════════════════════════╣
║    Paper: Zakaria et al. (2024) ARASET 39(2):110-131                 ║
║    Topic: Windows Ransomware Early Detection (Assignment 4)          ║
╚═══════════════════════════════════════════════════════════════════════╝"""


def get_or_build_dataset(csv_path: str = None):
    """
    Load the real khas-ccip CSV if available, otherwise simulate it.
    Returns a loaded and cleaned DataFrame.
    """
    import pandas as pd

    # Priority 1: user-supplied real CSV
    if csv_path and os.path.exists(csv_path):
        print(f"\n[DATASET] Loading real khas-ccip VirusShare CSV...")
        print(f"  Path: {csv_path}")
        return load_virusshare_csv(csv_path), True

    # Priority 2: previously generated simulation
    if os.path.exists(SIM_CSV):
        print(f"\n[DATASET] Loading cached simulation: {SIM_CSV}")
        df = load_virusshare_csv(SIM_CSV)
        return df, False

    # Priority 3: generate simulation
    print(f"\n[DATASET] Real CSV not found. Building simulation...")
    print(f"  To use the real dataset:")
    print(f"    git clone https://github.com/khas-ccip/api_sequences_malware_datasets")
    print(f"    python main.py --csv api_sequences_malware_datasets/VirusShare.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    build_simulated_virusshare_csv(SIM_CSV, version="imbalanced")
    df = load_virusshare_csv(SIM_CSV)
    return df, False


def run_pipeline(csv_path: str = None):
    """Complete RENTAKA pipeline."""
    t0 = time.time()
    print(BANNER)

    # ── Step 1: Load dataset ──────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 1/5 — Dataset Loading")
    print(f"{'─'*65}")
    df, is_real = get_or_build_dataset(csv_path)
    summarize_dataset(df)
    api_cols = get_api_feature_names(df)

    # ── Step 2: RENTAKA feature extraction ────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 2/5 — RENTAKA Feature Extraction")
    print(f"{'─'*65}")
    print("Applying RENTAKA algorithm:")
    print("  • Identifying encryption-related API imports (boundary markers)")
    print("  • Extracting pre-encryption setup-phase features")
    print_rentaka_demo(df, api_cols, n_samples=2)

    # Build feature matrix (exclude crypto APIs from features — RENTAKA mode)
    X, y, feature_names = build_rentaka_feature_matrix(
        df, api_cols, exclude_crypto_from_features=True
    )

    # ── Step 3: Train classifiers ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 3/5 — Training 5 Classifiers (10-fold Stratified CV)")
    print(f"{'─'*65}")
    results, all_preds = train_evaluate_all(X, y, feature_names)

    # ── Step 4: Save models & results ────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 4/5 — Saving Models & Results")
    print(f"{'─'*65}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_models(X, y)
    save_results_csv(results)

    # ── Step 5: Visualizations ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("STEP 5/5 — Generating Visualizations")
    print(f"{'─'*65}")
    generate_all_plots(y, X, feature_names, df, results, all_preds)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_comparison_table(results)

    best = max(results.items(), key=lambda x: x[1]["Accuracy (%)"])
    elapsed = time.time() - t0

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  Dataset    : {'Real khas-ccip VirusShare' if is_real else 'Simulated (real structure)'}")
    print(f"  Samples    : {len(df):,}  ({(y==1).sum()} ransomware, {(y==0).sum()} other)")
    print(f"  Features   : {len(feature_names)} API calls (pre-encryption, RENTAKA mode)")
    print(f"  Best model : {best[0]} — {best[1]['Accuracy (%)']:.4f}% accuracy")
    print(f"  Paper best : SVM — 97.0492% accuracy")
    print(f"  Total time : {elapsed:.1f}s")
    print(f"  Results    : {RESULTS_DIR}/")
    print(f"{'='*65}")

    return results, all_preds, X, y, feature_names, df


def run_demo():
    """Algorithm demo — shows RENTAKA working on real-format data."""
    import pandas as pd
    print(BANNER)
    print("\nDEMO — RENTAKA Algorithm on VirusShare API Sequences")
    print("="*65)

    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SIM_CSV):
        build_simulated_virusshare_csv(SIM_CSV, version="imbalanced")
    df = load_virusshare_csv(SIM_CSV)
    api_cols = get_api_feature_names(df)
    print_rentaka_demo(df, api_cols, n_samples=3)
    print("\nRun without --demo for full training pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RENTAKA Ransomware Early Detection — khas-ccip VirusShare Dataset"
    )
    parser.add_argument("--csv",  type=str, default=None,
                        help="Path to VirusShare.csv from khas-ccip GitHub repo")
    parser.add_argument("--demo", action="store_true",
                        help="Algorithm demo only")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.demo:
        run_demo()
    else:
        run_pipeline(csv_path=args.csv)
