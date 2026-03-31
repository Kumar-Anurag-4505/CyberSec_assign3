"""
RENTAKA v2 — Core Algorithm & Feature Engineering
===================================================
Implements the RENTAKA algorithm adapted for the khas-ccip dataset.

ADAPTATION NOTE:
The khas-ccip dataset uses PEFile static import analysis (not Cuckoo dynamic logs).
This means:
  - API calls are PE import entries (statically declared, not runtime-observed)
  - There is no temporal sequence — just presence/absence of each import
  - The "encryption boundary" concept still applies: we check which crypto APIs
    are statically imported and flag samples that have them vs don't

The RENTAKA adaptation for static imports:
  - Pre-encryption SETUP indicators: environment, persistence, network, evasion APIs
  - Encryption indicators: CryptEncrypt, BCryptEncrypt, etc.
  - A sample that imports BOTH setup APIs AND crypto APIs → ransomware signature
  - A sample with only setup APIs (no crypto) → uncertain
  - A sample with NO setup APIs → likely not ransomware
"""

import os, sys
import numpy as np
import pandas as pd
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Encryption-related API imports (RENTAKA boundary markers) ─────────────────
ENCRYPTION_APIS = {
    "CryptEncrypt", "CryptDecrypt", "CryptGenKey", "CryptDeriveKey",
    "CryptAcquireContextW", "CryptAcquireContextA", "CryptCreateHash",
    "CryptHashData", "CryptDestroyHash", "CryptDestroyKey",
    "CryptReleaseContext", "CryptImportKey", "CryptExportKey",
    "CryptSignHash", "BCryptEncrypt", "BCryptDecrypt",
    "BCryptGenerateSymmetricKey", "BCryptOpenAlgorithmProvider",
    "BCryptCreateHash", "BCryptHashData", "BCryptGenRandom",
    "RtlEncryptMemory", "SystemFunction036",
}

# ── Pre-encryption setup indicators ───────────────────────────────────────────
SETUP_APIS = {
    # Environment mapping (ransomware checks it's not in a sandbox)
    "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
    "NtQueryInformationProcess", "GetSystemInfo", "GetComputerNameW",
    "GetUserNameW", "GetLogicalDrives", "GetDriveTypeW",
    "GetVolumeInformationW", "GetSystemTimeAsFileTime",
    # Persistence
    "RegSetValueExW", "RegCreateKeyExW", "CreateServiceW",
    "OpenSCManagerW", "StartServiceW",
    # Privilege escalation
    "OpenProcessToken", "AdjustTokenPrivileges", "LookupPrivilegeValueW",
    "ImpersonateLoggedOnUser",
    # C2 Communication
    "WinHttpOpen", "WinHttpConnect", "WinHttpSendRequest",
    "WinHttpReceiveResponse", "InternetOpenW", "InternetConnectW",
    # File enumeration
    "FindFirstFileW", "FindNextFileW",
}


def apply_rentaka_algorithm_static(
    feature_row: pd.Series,
    api_columns: List[str],
) -> Tuple[dict, bool, List[str]]:
    """
    RENTAKA algorithm adapted for static PE import data.

    For static analysis: instead of finding a temporal boundary,
    we identify which of the feature dimensions correspond to
    'pre-encryption' vs 'encryption' APIs and analyze their pattern.

    Parameters
    ----------
    feature_row : pd.Series
        One row from the dataset (binary API call presence vector)
    api_columns : List[str]
        Names of the API call feature columns

    Returns
    -------
    pre_enc_features : dict  {api_name: value}
        API calls excluding encryption-specific ones
        (these are the "pre-encryption phase" indicators)
    has_crypto : bool
        Whether the sample imports encryption APIs
    imported_enc_apis : List[str]
        Which encryption APIs are imported (boundary markers)
    """
    present_apis = {col for col in api_columns if feature_row.get(col, 0) == 1}

    # Which encryption APIs are imported?
    imported_enc_apis = sorted(present_apis & ENCRYPTION_APIS)
    has_crypto = len(imported_enc_apis) > 0

    # Pre-encryption features = all present APIs minus crypto APIs
    pre_enc_apis = present_apis - ENCRYPTION_APIS

    # Build feature dict
    pre_enc_features = {
        col: (1 if col in pre_enc_apis else 0)
        for col in api_columns
        if col not in ENCRYPTION_APIS
    }

    return pre_enc_features, has_crypto, imported_enc_apis


def build_rentaka_feature_matrix(
    df: pd.DataFrame,
    api_columns: List[str],
    exclude_crypto_from_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build the RENTAKA feature matrix from the loaded dataset.

    When exclude_crypto_from_features=True (default):
        Features = all API columns EXCEPT encryption APIs
        This simulates RENTAKA's pre-encryption feature set.

    When exclude_crypto_from_features=False:
        Features = all API columns (standard ML baseline)

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
    y : np.ndarray (n_samples,)
    feature_names : List[str]
    """
    if exclude_crypto_from_features:
        feature_names = [col for col in api_columns
                         if col not in ENCRYPTION_APIS]
        mode = "RENTAKA (pre-encryption features only)"
    else:
        feature_names = list(api_columns)
        mode = "Baseline (all API features)"

    print(f"\nFeature matrix mode: {mode}")
    print(f"  Total API columns  : {len(api_columns)}")
    print(f"  Crypto APIs removed: {len(api_columns) - len(feature_names)}")
    print(f"  Features used      : {len(feature_names)}")

    X = df[feature_names].values.astype(np.float32)
    y = df["label"].values.astype(int)

    print(f"  Matrix shape       : {X.shape}")
    print(f"  Class 1 (Ransom)   : {(y==1).sum()}")
    print(f"  Class 0 (Other)    : {(y==0).sum()}")

    return X, y, feature_names


def get_discriminative_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Find the most discriminative API calls for ransomware detection.
    Uses mutual information to rank features.
    """
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X, y, random_state=42)
    ran_mask = y == 1
    oth_mask = y == 0

    rows = []
    for i, (name, score) in enumerate(zip(feature_names, mi)):
        ran_pct = X[ran_mask, i].mean() * 100
        oth_pct = X[oth_mask, i].mean() * 100
        is_setup   = name in SETUP_APIS
        is_enc     = name in ENCRYPTION_APIS
        category   = "Setup/Persistence" if is_setup else (
                     "Encryption" if is_enc else "General")
        rows.append({
            "Rank": 0,
            "API Call":      name,
            "MI Score":      round(score, 5),
            "Ransomware %":  round(ran_pct, 1),
            "Other %":       round(oth_pct, 1),
            "Δ (R-O)":       round(ran_pct - oth_pct, 1),
            "Category":      category,
        })

    df_out = pd.DataFrame(rows).sort_values("MI Score", ascending=False)
    df_out = df_out.head(top_n).reset_index(drop=True)
    df_out["Rank"] = df_out.index + 1
    return df_out


def print_rentaka_demo(df: pd.DataFrame, api_columns: List[str], n_samples: int = 3):
    """
    Demo the RENTAKA algorithm on a few real samples from the dataset.
    Good for demo video and presentation.
    """
    ransomware_samples = df[df["label"] == 1].head(n_samples)
    other_samples      = df[df["label"] == 0].head(n_samples)

    print("\n" + "="*65)
    print("  RENTAKA Algorithm — Static PE Import Analysis")
    print("="*65)

    for idx, row in ransomware_samples.iterrows():
        pre_enc, has_crypto, enc_apis = apply_rentaka_algorithm_static(
            row, api_columns
        )
        present = [c for c in api_columns if row.get(c, 0) == 1]
        setup   = [a for a in present if a in SETUP_APIS]
        print(f"\n[RANSOMWARE sample — family: {row.get('family','?')}]")
        print(f"  Total imports      : {len(present)}")
        print(f"  Setup APIs present : {setup[:5]}...")
        print(f"  Crypto APIs found  : {enc_apis}")
        print(f"  Has crypto imports : {has_crypto} ← KEY RANSOMWARE INDICATOR")

    for idx, row in other_samples.iterrows():
        pre_enc, has_crypto, enc_apis = apply_rentaka_algorithm_static(
            row, api_columns
        )
        present = [c for c in api_columns if row.get(c, 0) == 1]
        print(f"\n[OTHER MALWARE sample — family: {row.get('family','?')}]")
        print(f"  Total imports      : {len(present)}")
        print(f"  Crypto APIs found  : {enc_apis}")
        print(f"  Has crypto imports : {has_crypto}")

    print("\n" + "="*65)
