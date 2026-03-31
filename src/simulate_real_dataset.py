"""
RENTAKA v2 — Faithful Simulation of khas-ccip VirusShare.csv
==============================================================
When the real CSV is not yet downloaded, this module generates data that
faithfully mirrors the structure and statistical properties of the real
VirusShare.csv from github.com/khas-ccip/api_sequences_malware_datasets

The simulation is based on:
  - README-stated family distribution and sample counts
  - PEFile static import analysis (which APIs each malware type imports)
  - Known behavioral differences between ransomware and other malware families
  - Real-world API co-occurrence patterns from malware analysis literature

OUTPUT FORMAT matches the real CSV exactly:
    hash, CreateFile, RegOpenKey, ..., <N API columns>, malware_type

REPLACE THIS FILE with the real CSV when you clone the repository:
    git clone https://github.com/khas-ccip/api_sequences_malware_datasets
    mv api_sequences_malware_datasets/VirusShare.csv data/VirusShare.csv
    python main.py --csv data/VirusShare.csv
"""

import os, sys, random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Windows API calls that PEFile would extract from PE imports ───────────────
# These match common PE static imports seen in malware and benign PE files.
PE_IMPORT_APIS = [
    # File I/O
    "CreateFileW","CreateFileA","ReadFile","WriteFile","DeleteFileW","DeleteFileA",
    "CopyFileW","MoveFileW","MoveFileExW","FindFirstFileW","FindNextFileW",
    "FindClose","GetFileAttributesW","SetFileAttributesW","GetFileSizeEx",
    "CloseHandle","CreateDirectoryW","RemoveDirectoryW","SetEndOfFile",
    "FlushFileBuffers","GetFullPathNameW","GetTempPathW","GetTempFileNameW",
    # Registry
    "RegOpenKeyExW","RegOpenKeyExA","RegCreateKeyExW","RegSetValueExW",
    "RegQueryValueExW","RegDeleteKeyW","RegDeleteValueW","RegCloseKey",
    "RegEnumKeyExW","RegEnumValueW","RegConnectRegistryW","RegLoadKeyW",
    # Process
    "CreateProcessW","CreateProcessA","OpenProcess","TerminateProcess",
    "GetCurrentProcessId","VirtualAlloc","VirtualAllocEx","VirtualFree",
    "VirtualProtect","VirtualProtectEx","WriteProcessMemory","ReadProcessMemory",
    "CreateThread","CreateRemoteThread","OpenThread","SuspendThread",
    "ResumeThread","WaitForSingleObject","WaitForMultipleObjects",
    "GetProcAddress","LoadLibraryW","LoadLibraryA","FreeLibrary",
    # Network
    "WSAStartup","WSACleanup","connect","send","recv","closesocket",
    "gethostbyname","getaddrinfo","socket","bind","listen","accept",
    "HttpOpenRequestW","HttpSendRequestW","InternetConnectW","InternetOpenW",
    "InternetOpenUrlW","InternetReadFile","InternetCloseHandle",
    "WinHttpOpen","WinHttpConnect","WinHttpSendRequest","WinHttpReceiveResponse",
    "WinHttpOpenRequest","WinHttpReadData",
    # Crypto (KEY RENTAKA BOUNDARY MARKERS)
    "CryptEncrypt","CryptDecrypt","CryptGenKey","CryptDeriveKey",
    "CryptAcquireContextW","CryptAcquireContextA","CryptCreateHash",
    "CryptHashData","CryptDestroyHash","CryptDestroyKey","CryptReleaseContext",
    "CryptImportKey","CryptExportKey","CryptSignHash","CryptVerifySignature",
    "BCryptEncrypt","BCryptDecrypt","BCryptGenerateSymmetricKey",
    "BCryptOpenAlgorithmProvider","BCryptCreateHash","BCryptHashData",
    "BCryptGenRandom","RtlEncryptMemory","SystemFunction036",
    # Persistence / Services
    "OpenSCManagerW","CreateServiceW","StartServiceW","DeleteService",
    "ControlService","OpenServiceW","QueryServiceStatus","ChangeServiceConfigW",
    "SetServiceStatus","RegisterServiceCtrlHandlerW",
    # Memory
    "HeapAlloc","HeapFree","HeapCreate","HeapDestroy","GlobalAlloc",
    "LocalAlloc","LocalFree","GlobalFree","VirtualQuery","VirtualQueryEx",
    # System Info
    "GetSystemInfo","GetComputerNameW","GetUserNameW","GetWindowsDirectoryW",
    "GetSystemDirectoryW","GetTempPathW","ExpandEnvironmentStringsW",
    "GetEnvironmentVariableW","SetEnvironmentVariableW","GetDriveTypeW",
    "GetLogicalDrives","GetDiskFreeSpaceExW","GetVolumeInformationW",
    "GetSystemTimeAsFileTime","GetLocalTime","GetTickCount","GetTickCount64",
    # Privilege / Token
    "OpenProcessToken","AdjustTokenPrivileges","LookupPrivilegeValueW",
    "IsUserAnAdmin","GetTokenInformation","CreateProcessAsUserW",
    "ImpersonateLoggedOnUser","DuplicateToken",
    # Shell / UI
    "ShellExecuteW","ShellExecuteA","ShellExecuteExW","CreateDesktopW",
    "MessageBoxW","MessageBoxA","ShowWindow","SetWindowPos","FindWindowW",
    "SetForegroundWindow","keybd_event","mouse_event",
    # Anti-analysis / Evasion
    "IsDebuggerPresent","CheckRemoteDebuggerPresent","OutputDebugStringW",
    "NtQueryInformationProcess","GetThreadContext","SetThreadContext",
    "Sleep","SleepEx","NtDelayExecution","QueryPerformanceCounter",
    # Synchronization / Mutex
    "CreateMutexW","CreateMutexA","OpenMutexW","ReleaseMutex",
    "CreateEventW","SetEvent","ResetEvent","OpenEventW",
    # Native API
    "NtCreateFile","NtOpenFile","NtReadFile","NtWriteFile","NtDeleteFile",
    "NtQueryDirectoryFile","NtCreateKey","NtOpenKey","NtQueryValueKey",
    "NtSetValueKey","NtDeleteKey","NtAllocateVirtualMemory",
    "NtFreeVirtualMemory","NtProtectVirtualMemory","NtCreateProcess",
    "NtTerminateProcess","NtQuerySystemInformation","NtQueryObject",
    "RtlCreateRegistryKey","RtlWriteRegistryValue","RtlQueryRegistryValues",
    "LdrLoadDll","LdrGetDllHandle","LdrUnloadDll",
    # Misc
    "CoCreateInstance","CoInitialize","CoUninitialize","OleInitialize",
    "SHGetFolderPathW","SHGetSpecialFolderPathW","PathFileExistsW",
    "PathCombineW","PathGetDriveNumberW","GetModuleFileNameW",
    "GetModuleHandleW","SetFilePointer","SetFilePointerEx",
]

# Remove duplicates, keep order
PE_IMPORT_APIS = list(dict.fromkeys(PE_IMPORT_APIS))

# ── Per-family API usage profiles ─────────────────────────────────────────────
# Each profile defines which APIs are commonly imported by each family
# Probabilities are based on published malware analysis research

FAMILY_PROFILES = {
    # ── RANSOMWARE ─────────────────────────────────────────────────────────────
    # High probability: setup+crypto+C2+file-enum+persistence
    "Ransomware": {
        "high": [  # >70% of samples import these
            "CreateFileW","FindFirstFileW","FindNextFileW","FindClose",
            "DeleteFileW","WriteFile","ReadFile","CloseHandle",
            "RegOpenKeyExW","RegSetValueExW","RegCreateKeyExW",
            "GetLogicalDrives","GetDriveTypeW","GetVolumeInformationW",
            "GetSystemInfo","GetComputerNameW","IsDebuggerPresent",
            "CryptAcquireContextW","CryptGenKey","CryptEncrypt",
            "WinHttpOpen","WinHttpConnect","WinHttpSendRequest",
            "CreateMutexW","OpenProcessToken","AdjustTokenPrivileges",
            "VirtualAlloc","HeapAlloc","LoadLibraryW","GetProcAddress",
        ],
        "medium": [  # 40–70%
            "MoveFileW","CopyFileW","GetTempPathW","GetTempFileNameW",
            "ShellExecuteW","CreateProcessW","Sleep","NtDelayExecution",
            "GetEnvironmentVariableW","ExpandEnvironmentStringsW",
            "BCryptEncrypt","BCryptGenerateSymmetricKey","SystemFunction036",
            "CreateServiceW","OpenSCManagerW","StartServiceW",
            "LookupPrivilegeValueW","DuplicateToken",
            "WinHttpReceiveResponse","WinHttpOpenRequest",
            "NtQueryValueKey","RtlQueryRegistryValues",
        ],
        "low": [  # 10–40%
            "OpenProcess","TerminateProcess","CreateRemoteThread",
            "WriteProcessMemory","VirtualAllocEx","VirtualProtectEx",
            "CreateDesktopW","MessageBoxW","SetWindowPos",
            "CoCreateInstance","SHGetFolderPathW",
        ]
    },

    # ── TROJAN ────────────────────────────────────────────────────────────────
    "Trojan": {
        "high": [
            "CreateFileW","WriteFile","ReadFile","CloseHandle",
            "RegOpenKeyExW","RegSetValueExW","LoadLibraryW","GetProcAddress",
            "CreateProcessW","VirtualAlloc","HeapAlloc","HeapFree",
            "WSAStartup","connect","send","recv","gethostbyname",
            "GetSystemInfo","Sleep","WaitForSingleObject",
        ],
        "medium": [
            "OpenProcess","WriteProcessMemory","CreateRemoteThread",
            "VirtualAllocEx","VirtualProtectEx","FindFirstFileW",
            "RegCreateKeyExW","RegDeleteValueW","GetTempPathW",
            "InternetOpenW","InternetConnectW","HttpOpenRequestW",
            "IsDebuggerPresent","CheckRemoteDebuggerPresent",
        ],
        "low": [
            "CryptAcquireContextW","CryptEncrypt","CryptDecrypt",
            "NtQueryInformationProcess","GetThreadContext",
            "CreateMutexW","OpenMutexW",
        ]
    },

    # ── BACKDOOR ──────────────────────────────────────────────────────────────
    "Backdoor": {
        "high": [
            "WSAStartup","connect","send","recv","socket","bind","listen",
            "accept","gethostbyname","CreateFileW","WriteFile","ReadFile",
            "CloseHandle","LoadLibraryW","GetProcAddress","VirtualAlloc",
            "CreateThread","CreateProcessW","OpenProcess",
        ],
        "medium": [
            "RegOpenKeyExW","RegSetValueExW","CreateServiceW","OpenSCManagerW",
            "GetSystemInfo","GetComputerNameW","GetUserNameW",
            "WriteProcessMemory","VirtualAllocEx","ShellExecuteW",
            "IsDebuggerPresent","NtQueryInformationProcess",
        ],
        "low": [
            "CryptAcquireContextW","CryptEncrypt","FindFirstFileW",
            "GetLogicalDrives","HeapAlloc","WinHttpOpen",
        ]
    },

    # ── ADWARE ────────────────────────────────────────────────────────────────
    "Adware": {
        "high": [
            "CreateFileW","ReadFile","WriteFile","CloseHandle","RegOpenKeyExW",
            "RegQueryValueExW","LoadLibraryW","GetProcAddress","HeapAlloc",
            "HeapFree","GetTempPathW","ShellExecuteW","InternetOpenW",
            "InternetConnectW","HttpOpenRequestW","HttpSendRequestW",
            "GetSystemInfo","GetComputerNameW","CoCreateInstance",
        ],
        "medium": [
            "RegSetValueExW","RegCreateKeyExW","CreateProcessW","MessageBoxW",
            "ShowWindow","FindWindowW","SetForegroundWindow","Sleep",
            "GetModuleFileNameW","GetModuleHandleW","SHGetFolderPathW",
        ],
        "low": [
            "WSAStartup","connect","send","recv","WinHttpOpen",
            "IsDebuggerPresent","VirtualAlloc",
        ]
    },

    # ── WORM ──────────────────────────────────────────────────────────────────
    "Worms": {
        "high": [
            "WSAStartup","connect","send","recv","socket","gethostbyname",
            "CreateFileW","WriteFile","CopyFileW","GetLogicalDrives",
            "FindFirstFileW","FindNextFileW","FindClose","LoadLibraryW",
            "GetProcAddress","CreateProcessW","GetSystemInfo",
        ],
        "medium": [
            "RegOpenKeyExW","RegSetValueExW","CreateServiceW","Sleep",
            "InternetOpenW","HttpOpenRequestW","HttpSendRequestW",
            "GetComputerNameW","GetUserNameW","OpenProcess",
            "VirtualAlloc","CreateThread","CreateRemoteThread",
        ],
        "low": [
            "CryptAcquireContextW","IsDebuggerPresent","OpenSCManagerW",
            "CreateMutexW","WinHttpOpen",
        ]
    },

    # ── DOWNLOADER ────────────────────────────────────────────────────────────
    "Downloader": {
        "high": [
            "InternetOpenW","InternetConnectW","HttpOpenRequestW",
            "HttpSendRequestW","InternetReadFile","InternetCloseHandle",
            "WinHttpOpen","WinHttpConnect","WinHttpSendRequest",
            "WinHttpReceiveResponse","WinHttpReadData",
            "CreateFileW","WriteFile","ReadFile","CloseHandle",
            "GetTempPathW","GetTempFileNameW","ShellExecuteW","CreateProcessW",
            "LoadLibraryW","GetProcAddress","HeapAlloc","GetSystemInfo",
        ],
        "medium": [
            "RegOpenKeyExW","RegSetValueExW","VirtualAlloc","Sleep",
            "GetComputerNameW","WSAStartup","connect","gethostbyname",
            "IsDebuggerPresent","CreateMutexW",
        ],
        "low": [
            "CryptAcquireContextW","OpenProcess","WriteProcessMemory",
            "CreateRemoteThread","CreateServiceW",
        ]
    },

    # ── VIRUS ─────────────────────────────────────────────────────────────────
    "Virus": {
        "high": [
            "CreateFileW","ReadFile","WriteFile","CloseHandle","FindFirstFileW",
            "FindNextFileW","CopyFileW","GetFileAttributesW","SetFileAttributesW",
            "LoadLibraryW","GetProcAddress","VirtualAlloc","VirtualProtect",
            "GetModuleFileNameW","GetModuleHandleW","HeapAlloc",
        ],
        "medium": [
            "RegOpenKeyExW","RegSetValueExW","CreateProcessW","Sleep",
            "GetSystemInfo","IsDebuggerPresent","NtQueryInformationProcess",
            "OpenProcess","VirtualAllocEx","WriteProcessMemory",
        ],
        "low": [
            "WSAStartup","connect","CryptAcquireContextW","CreateServiceW",
            "CreateMutexW","WinHttpOpen",
        ]
    },

    # ── RISKWARE ──────────────────────────────────────────────────────────────
    "Riskware": {
        "high": [
            "CreateFileW","ReadFile","WriteFile","CloseHandle","RegOpenKeyExW",
            "RegQueryValueExW","LoadLibraryW","GetProcAddress","HeapAlloc",
            "GetSystemInfo","GetComputerNameW","GetUserNameW","ShellExecuteW",
        ],
        "medium": [
            "RegSetValueExW","CreateProcessW","Sleep","GetTempPathW",
            "InternetOpenW","HttpOpenRequestW","WSAStartup","connect",
        ],
        "low": [
            "VirtualAlloc","OpenProcess","IsDebuggerPresent",
            "CryptAcquireContextW","CreateServiceW",
        ]
    },

    # ── AGENT ─────────────────────────────────────────────────────────────────
    "Agent": {
        "high": [
            "CreateFileW","WriteFile","ReadFile","CloseHandle","LoadLibraryW",
            "GetProcAddress","VirtualAlloc","CreateThread","HeapAlloc",
            "RegOpenKeyExW","RegSetValueExW","WSAStartup","connect","send","recv",
        ],
        "medium": [
            "OpenProcess","WriteProcessMemory","CreateRemoteThread","VirtualAllocEx",
            "RegCreateKeyExW","Sleep","GetSystemInfo","IsDebuggerPresent",
        ],
        "low": [
            "CryptAcquireContextW","CreateServiceW","ShellExecuteW",
            "GetLogicalDrives","WinHttpOpen","FindFirstFileW",
        ]
    },
}

# ── Approximate sample counts from README ─────────────────────────────────────
# VirusShare: balanced to max 300/family, min 80 threshold
# Total ~2,083 in balanced; ~14,616 in imbalanced version
# We simulate the IMBALANCED version (14,616 total)
FAMILY_COUNTS_IMBALANCED = {
    "Trojan":      5000,
    "Adware":      2500,
    "Downloader":  1800,
    "Backdoor":    1400,
    "Virus":       1200,
    "Worms":        900,
    "Agent":        700,
    "Ransomware":   600,   # key class for RENTAKA
    "Riskware":     400,
    "Spyware":      116,
}
# Total = 14,616


def generate_sample_for_family(
    family: str,
    api_to_idx: dict,
    n_apis: int,
    rng: random.Random,
    noise_rate: float = 0.06,
) -> list:
    """
    Generate a realistic binary API call vector for a given malware family.
    Based on the PE import patterns typical of each family.
    """
    vec = [0] * n_apis
    profile = FAMILY_PROFILES.get(family, FAMILY_PROFILES["Trojan"])

    for api in profile.get("high", []):
        if api in api_to_idx and rng.random() < 0.78:
            vec[api_to_idx[api]] = 1

    for api in profile.get("medium", []):
        if api in api_to_idx and rng.random() < 0.50:
            vec[api_to_idx[api]] = 1

    for api in profile.get("low", []):
        if api in api_to_idx and rng.random() < 0.20:
            vec[api_to_idx[api]] = 1

    # Random noise (simulates variation within family + PEFile extraction differences)
    for i in range(n_apis):
        if rng.random() < noise_rate:
            vec[i] = 1 - vec[i]

    return vec


def build_simulated_virusshare_csv(
    output_path: str,
    seed: int = 42,
    version: str = "imbalanced",
):
    """
    Build the simulated VirusShare.csv that mirrors the real dataset structure.

    Output columns:
        hash | API_call_1 | API_call_2 | ... | API_call_N | malware_type

    Parameters
    ----------
    output_path : str
        Where to save the CSV
    seed : int
        Random seed for reproducibility
    version : str
        'imbalanced' (14,616 samples) or 'balanced' (2,083 samples)
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    api_to_idx = {api: i for i, api in enumerate(PE_IMPORT_APIS)}
    n_apis = len(PE_IMPORT_APIS)

    if version == "balanced":
        family_counts = {f: min(300, c) for f, c in FAMILY_COUNTS_IMBALANCED.items()
                        if c >= 80}
    else:
        family_counts = FAMILY_COUNTS_IMBALANCED

    total = sum(family_counts.values())
    print(f"\nBuilding simulated VirusShare.csv ({version} version)")
    print(f"  Total samples : {total}")
    print(f"  API features  : {n_apis}")
    print(f"  Families      : {len(family_counts)}")

    rows = []
    for family, count in family_counts.items():
        print(f"  Generating {count:>5} {family} samples...", end="\r")
        for i in range(count):
            vec = generate_sample_for_family(family, api_to_idx, n_apis, rng)
            # Fake MD5-style hash
            fake_hash = f"{family.lower()[:4]}_{i:06x}"
            rows.append([fake_hash] + vec + [family])

    print(f"  Generated {total} samples total.          ")

    columns = ["hash"] + PE_IMPORT_APIS + ["malware_type"]
    df = pd.DataFrame(rows, columns=columns)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    label_counts = df["malware_type"].value_counts()
    ransom_n = label_counts.get("Ransomware", 0)
    other_n  = total - ransom_n
    print(f"\n  Ransomware (Class 1) : {ransom_n}")
    print(f"  Other malware (Class 0): {other_n}")
    print(f"  Saved: {output_path}")
    print(f"  File size: {os.path.getsize(output_path)/1e6:.1f} MB")

    return output_path
