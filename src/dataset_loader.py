"""
RENTAKA v2 — Dataset Loader
==============================
Loads the real VirusShare.csv from:
  github.com/khas-ccip/api_sequences_malware_datasets
"""

import os, sys, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Dataset column constants ──────────────────────────────────────────────────
LABEL_COLUMN_CANDIDATES  = ["malware_type", "type", "label", "class",
                              "Category", "family", "Type"]
# Added "file" to handle the real dataset's ID column
HASH_COLUMN_CANDIDATES   = ["hash", "md5", "sha256", "Hash", "MD5",
                              "file_hash", "sha1", "file"]

RANSOMWARE_LABEL_VARIANTS = {
    "ransomware", "Ransomware", "RANSOMWARE",
    "ransom", "Ransom", "crypto-ransomware", "locker",
}

def detect_columns(df: pd.DataFrame):
    """Auto-detect which columns are the label and hash columns."""
    label_col = None
    hash_col  = None

    for c in LABEL_COLUMN_CANDIDATES:
        if c in df.columns:
            label_col = c
            break

    for c in HASH_COLUMN_CANDIDATES:
        if c in df.columns:
            hash_col = c
            break

    non_feat = {label_col, hash_col} - {None}
    api_cols = [c for c in df.columns if c not in non_feat]

    return label_col, hash_col, api_cols


def load_virusshare_csv(csv_path: str) -> pd.DataFrame:
    """Load and format the VirusShare dataset."""
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # ── FIX: Transform the real khas-ccip format ──────────────────────────────
    # If the dataset has ['file', 'api', 'class'], we need to expand it.
    if 'api' in df.columns and len(df.columns) <= 5:
        print("  [!] Detected 'long' sequence format. Expanding into binary features...")
        
        file_col = 'file' if 'file' in df.columns else 'hash'
        class_col = 'class' if 'class' in df.columns else 'label'
        
        # Split the comma/space separated APIs into a list
        df['api'] = df['api'].fillna('').astype(str)
        df['api_list'] = df['api'].apply(lambda x: [a.strip() for a in re.split(r'[,\s]+', x) if a.strip()])
        
        # Explode to one API per row, drop duplicates, and pivot
        df_exp = df.explode('api_list')
        df_exp = df_exp.dropna(subset=['api_list'])
        
        if not df_exp.empty:
            df_exp = df_exp.drop_duplicates(subset=[file_col, 'api_list'])
            df_exp['present'] = 1
            
            # Pivot into wide format (rows = hashes, columns = unique APIs)
            df_wide = df_exp.pivot(index=[file_col, class_col], columns='api_list', values='present')
            df_wide = df_wide.fillna(0).reset_index()
            df_wide.columns.name = None
            df = df_wide
            print(f"  [+] Expansion complete. Extracted {len(df.columns) - 2} unique APIs.")
    # ──────────────────────────────────────────────────────────────────────────

    label_col, hash_col, api_cols = detect_columns(df)

    if label_col is None:
        raise ValueError(f"Could not find a label column. Expected one of: {LABEL_COLUMN_CANDIDATES}")

    # Create binary label
    df["label"]  = df[label_col].apply(lambda x: 1 if str(x).strip() in RANSOMWARE_LABEL_VARIANTS else 0)
    df["family"] = df[label_col].astype(str)

    # Drop non-feature columns
    if hash_col and hash_col in df.columns:
        df = df.drop(columns=[hash_col])
    if label_col != "label":
        df = df.drop(columns=[label_col])

    keep_cols = [c for c in api_cols if c not in [hash_col, label_col, "api", "api_list"]]
    df = df[keep_cols + ["label", "family"]]

    # Ensure feature columns are numeric binary (0/1)
    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df[col] = df[col].clip(0, 1)

    return df.copy()  # .copy() prevents pandas fragmentation warnings


def get_api_feature_names(df: pd.DataFrame) -> list:
    """Return just the API call feature column names."""
    return [c for c in df.columns if c not in ("label", "family")]


def summarize_dataset(df: pd.DataFrame):
    """Print a comprehensive dataset summary."""
    api_cols = get_api_feature_names(df)
    X = df[api_cols].values
    y = df["label"].values

    print(f"\n{'='*60}")
    print("  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Source      : github.com/khas-ccip/api_sequences_malware_datasets")
    print(f"  File        : VirusShare.csv")
    print(f"  Total rows  : {len(df)}")
    print(f"  Features    : {len(api_cols)} Windows API calls (binary)")
    print(f"  Class 1     : Ransomware  ({(y==1).sum()} samples)")
    print(f"  Class 0     : Other malware ({(y==0).sum()} samples)")
    print(f"  Imbalance   : {(y==1).sum()/(y==0).sum():.2f} (ransom/other ratio)")
    print(f"  Sparsity    : {(1 - X.mean())*100:.1f}% zeros")
    print(f"  Avg APIs/sample (ransomware): {X[y==1].sum(axis=1).mean():.1f}")
    print(f"  Avg APIs/sample (other)     : {X[y==0].sum(axis=1).mean():.1f}")

    ransom_df = df[df["label"] == 1]
    if "family" in df.columns:
        print(f"\n  Ransomware samples by family:")
        for fam, cnt in ransom_df["family"].value_counts().items():
            print(f"    {str(fam):<25} {cnt}")

    print(f"{'='*60}")