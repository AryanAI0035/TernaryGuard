#!/usr/bin/env python3
"""
TernaryGuard — N-BaIoT Dataset Download & Verification Pipeline

Downloads and extracts the N-BaIoT (Network-Based Detection of IoT Botnet Attacks)
dataset for training and evaluating the TernaryGuard 1.58-bit ternary neural network.

Dataset Overview:
    - 115 continuous statistical features per network packet flow
    - 7M+ total traffic instances across 9 commercial IoT devices
    - 10 distinct botnet attack vectors (Mirai & Bashlite/Gafgyt families) + benign traffic
    - Target deployment: ATmega328P microcontroller (32KB Flash, 2KB SRAM)
      (Features will be reduced to ~20-40 in downstream preprocessing for embedded RAM fit)

Download Sources:
    1. UCI Machine Learning Repository (Default direct download URL):
       https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip
    2. Kaggle Dataset:
       mkashifn/nbaiot-dataset (via kaggle CLI, with automatic UCI fallback)

Output Directory Structure:
    data/
    └── raw/
        ├── detection_of_iot_botnet_attacks_n_baiot.zip (or nbaiot-dataset.zip)
        └── nbaiot/
            ├── 1.benign.csv (or Danmini_Doorbell/...)
            ├── 1.gafgyt.combo.csv
            ├── 1.mirai.ack.csv
            └── ... (89 CSV files total across 9 devices)

Usage:
    python research/download_nbaiot.py                 # Direct UCI download (default)
    python research/download_nbaiot.py --source kaggle # Kaggle download with UCI fallback
    python research/download_nbaiot.py --source uci    # Explicit UCI download
    python research/download_nbaiot.py --verify-only   # Verify already extracted dataset
    python research/download_nbaiot.py --force         # Re-download and re-extract
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ──────────────── Project Paths & Constants ────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_EXTRACT_DIR = DEFAULT_RAW_DIR / "nbaiot"

UCI_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/442/"
    "detection+of+iot+botnet+attacks+n+baiot.zip"
)
UCI_ARCHIVE_FILENAME = "detection_of_iot_botnet_attacks_n_baiot.zip"
KAGGLE_DATASET_IDENTIFIER = "mkashifn/nbaiot-dataset"

# 9 IoT devices in N-BaIoT (Meidan et al., 2018)
DEVICE_CATALOG: Dict[str, Tuple[str, List[str]]] = {
    "1": ("Danmini Doorbell", ["danmini"]),
    "2": ("Ecobee Thermostat", ["ecobee"]),
    "3": ("Ennio Doorbell", ["ennio"]),
    "4": ("Philips B120N10 Baby Monitor", ["philips", "b120n10", "baby_monitor"]),
    "5": ("Provision PT-737E Security Camera", ["737e", "pt_737e", "provision_737e"]),
    "6": ("Provision PT-838 Security Camera", ["838", "820", "pt_838", "provision_820"]),
    "7": ("Samsung SNH-1011_N Security Camera", ["samsung", "snh_1011", "snh-1011"]),
    "8": ("SimpleHome XCS7-1002 Security Camera", ["1002", "xcs7_1002"]),
    "9": ("SimpleHome XCS7-1003 Security Camera", ["1003", "xcs7_1003"]),
}

# 10 attack classes + 1 benign class
KNOWN_ATTACK_CATEGORIES: List[str] = [
    "benign",
    "gafgyt.combo",
    "gafgyt.junk",
    "gafgyt.scan",
    "gafgyt.tcp",
    "gafgyt.udp",
    "mirai.ack",
    "mirai.scan",
    "mirai.syn",
    "mirai.udp",
    "mirai.udpplain",
]


# ──────────────── Network & Download Helpers ────────────────


def is_kaggle_available() -> bool:
    """Check if the Kaggle command-line tool is installed and available in PATH.

    Returns:
        bool: True if 'kaggle' executable was found, False otherwise.
    """
    return shutil.which("kaggle") is not None


def download_from_kaggle(raw_dir: Path) -> Optional[Path]:
    """Download the N-BaIoT dataset using Kaggle CLI.

    Args:
        raw_dir: Destination directory for the downloaded archive.

    Returns:
        Optional[Path]: Path to downloaded archive if successful, None otherwise.
    """
    if not is_kaggle_available():
        print("[-] 'kaggle' CLI not found in PATH.")
        print("    To use Kaggle: pip install kaggle && place kaggle.json in ~/.kaggle/")
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET_IDENTIFIER,
        "-p",
        str(raw_dir),
    ]

    print(f"[*] Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        print(f"[-] Kaggle download failed with returncode {exc.returncode}:")
        if exc.stderr:
            print(f"    {exc.stderr.strip()}")
        return None
    except Exception as exc:
        print(f"[-] Unexpected error running Kaggle CLI: {exc}")
        return None

    # Identify downloaded archive
    candidates = list(raw_dir.glob("*nbaiot*.zip")) + list(raw_dir.glob("*.zip"))
    if candidates:
        latest = max(candidates, key=os.path.getmtime)
        print(f"[+] Kaggle archive downloaded: {latest.name} ({latest.stat().st_size / (1024**2):.1f} MB)")
        return latest

    return None


def download_from_uci(
    raw_dir: Path,
    url: str = UCI_DOWNLOAD_URL,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download the N-BaIoT dataset directly from UCI Machine Learning Repository.

    Supports resuming partial downloads (.part file) via HTTP Range requests.

    Args:
        raw_dir: Directory where the zip file should be stored.
        url: Direct download URL for the dataset archive.
        chunk_size: Streaming chunk size in bytes (default: 1MB).

    Returns:
        Path: Path to the completed zip archive.

    Raises:
        RuntimeError: If download fails due to network or HTTP errors.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / UCI_ARCHIVE_FILENAME
    part_path = raw_dir / f"{UCI_ARCHIVE_FILENAME}.part"

    headers: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    initial_bytes = 0
    if part_path.exists():
        initial_bytes = part_path.stat().st_size
        if initial_bytes > 0:
            headers["Range"] = f"bytes={initial_bytes}-"
            print(f"[*] Resuming partial download from byte offset: {initial_bytes:,} ({initial_bytes / (1024**2):.1f} MB)")

    print(f"[*] Connecting to UCI archive: {url}")
    req = urllib.request.Request(url, headers=headers)

    try:
        # Open connection with a 60-second timeout
        with urllib.request.urlopen(req, timeout=60) as response:
            status_code = getattr(response, "status", 200)
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length is not None else None

            # Handle 206 Partial Content vs 200 OK
            if status_code == 206:
                mode = "ab"
                if total_size is not None:
                    total_size += initial_bytes
            else:
                mode = "wb"
                initial_bytes = 0

            pbar_total = total_size if total_size else None
            pbar = tqdm(
                total=pbar_total,
                initial=initial_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading UCI N-BaIoT",
                ncols=90,
            )

            with open(part_path, mode) as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()

    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # Range Not Satisfiable: file might already be complete
            print("[*] Server returned HTTP 416 (Range Not Satisfiable). Restarting download...")
            if part_path.exists():
                part_path.unlink()
            return download_from_uci(raw_dir=raw_dir, url=url, chunk_size=chunk_size)
        raise RuntimeError(f"HTTP error occurred while downloading: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network connection failed: {exc.reason}") from exc
    except (ConnectionError, TimeoutError) as exc:
        raise RuntimeError(f"Connection error during download: {exc}") from exc
    except KeyboardInterrupt:
        print("\n[!] Download interrupted by user. Partial progress saved. Re-run script to resume.")
        sys.exit(130)

    # Atomic rename from .part to final destination
    if part_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        part_path.rename(dest_path)

    print(f"[+] Download complete: {dest_path} ({dest_path.stat().st_size / (1024**2):.1f} MB)")
    return dest_path


# ──────────────── Extraction Helpers ────────────────


def extract_archive(
    archive_path: Path,
    extract_dir: Path,
    force: bool = False,
) -> None:
    """Extract zip archive safely into the destination directory.

    Includes protection against ZipSlip directory traversal attacks, progress
    reporting with tqdm, and automatic extraction of any nested zip archives.

    Args:
        archive_path: Path to the downloaded .zip file.
        extract_dir: Target directory for extracted files.
        force: If True, overwrite existing files.

    Raises:
        zipfile.BadZipFile: If archive is corrupted or not a zip file.
        RuntimeError: If security violation (path traversal) is detected.
    """
    if not zipfile.is_zipfile(archive_path):
        raise zipfile.BadZipFile(
            f"File '{archive_path}' is not a valid zip archive. The download may be incomplete or corrupted."
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_target = extract_dir.resolve()

    print(f"[*] Extracting archive: {archive_path.name} -> {extract_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Unzipping files", unit="file", ncols=90):
            # Security check against ZipSlip path traversal
            destination = (extract_dir / member.filename).resolve()
            if not str(destination).startswith(str(resolved_target)):
                raise RuntimeError(
                    f"Security violation: Archive member '{member.filename}' attempts path traversal outside {extract_dir}"
                )

            if destination.exists() and not force:
                continue

            zf.extract(member, path=extract_dir)

    # Detect and extract any nested zip files (common in UCI distributions)
    nested_zips = [z for z in extract_dir.rglob("*.zip") if z.resolve() != archive_path.resolve()]
    if nested_zips:
        print(f"[*] Detected {len(nested_zips)} nested zip archive(s). Extracting nested archives...")
        for nested in tqdm(nested_zips, desc="Nested archives", unit="zip", ncols=90):
            nested_parent = nested.parent
            with zipfile.ZipFile(nested, "r") as n_zf:
                n_members = n_zf.infolist()
                for n_member in n_members:
                    n_dest = (nested_parent / n_member.filename).resolve()
                    if not str(n_dest).startswith(str(resolved_target)):
                        continue
                    if not n_dest.exists() or force:
                        n_zf.extract(n_member, path=nested_parent)
            # Remove nested zip after extraction to conserve disk space
            try:
                nested.unlink()
            except OSError:
                pass


# ──────────────── Verification & Statistics Helpers ────────────────


def count_file_lines(file_path: Path, chunk_size: int = 1024 * 1024) -> int:
    """Count total newline characters in a file using fast binary chunk reading.

    Args:
        file_path: Path to the target text/CSV file.
        chunk_size: Read buffer size in bytes (default: 1MB).

    Returns:
        int: Total number of lines in the file.
    """
    total_lines = 0
    last_char = b"\n"
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_lines += chunk.count(b"\n")
            last_char = chunk[-1:]
    if last_char != b"\n" and total_lines > 0:
        total_lines += 1
    return total_lines


def parse_device_and_attack(file_path: Path, base_dir: Path) -> Tuple[str, str]:
    """Parse device identifier and attack classification from file path and name.

    Handles multiple naming schemas (UCI flat format vs Kaggle hierarchical folders).

    Args:
        file_path: Full path to a CSV file.
        base_dir: Base extraction directory.

    Returns:
        Tuple[str, str]: (Device label, Attack classification label).
    """
    rel = file_path.relative_to(base_dir)
    rel_str = str(rel).replace("\\", "/")
    rel_lower = rel_str.lower()
    stem = file_path.stem.lower()

    # 1. Device resolution
    device_label = "Unknown Device"
    # Case A: Numeric prefix like "1.benign.csv" or "1.gafgyt.combo.csv"
    num_match = re.match(r"^([1-9])\.", file_path.name)
    if num_match and num_match.group(1) in DEVICE_CATALOG:
        dev_id = num_match.group(1)
        device_label = f"Device {dev_id}: {DEVICE_CATALOG[dev_id][0]}"
    # Case B: First directory component is numeric "1/..."
    elif len(rel.parts) > 1 and rel.parts[0] in DEVICE_CATALOG:
        dev_id = rel.parts[0]
        device_label = f"Device {dev_id}: {DEVICE_CATALOG[dev_id][0]}"
    # Case C: Device keyword match in relative directory name
    else:
        for dev_id, (dev_name, keywords) in DEVICE_CATALOG.items():
            if any(kw in rel_lower for kw in keywords):
                device_label = f"Device {dev_id}: {dev_name}"
                break
        if device_label == "Unknown Device" and len(rel.parts) > 1:
            device_label = rel.parts[0].replace("_", " ")

    # 2. Attack category resolution
    if "benign" in rel_lower:
        attack_label = "benign"
    else:
        botnet = None
        if "mirai" in rel_lower:
            botnet = "mirai"
        elif "gafgyt" in rel_lower or "bashlite" in rel_lower:
            botnet = "gafgyt"

        stem_tokens = set(re.split(r"[^a-zA-Z0-9]+", stem))
        sub_type = None

        if "udpplain" in stem_tokens or "udp_plain" in stem:
            sub_type = "udpplain"
        else:
            candidates = ["combo", "junk", "scan", "tcp", "udp", "ack", "syn"]
            for candidate in candidates:
                if candidate in stem_tokens:
                    sub_type = candidate
                    break

        if botnet and sub_type:
            attack_label = f"{botnet}.{sub_type}"
        elif sub_type:
            attack_label = sub_type
        elif botnet:
            attack_label = botnet
        else:
            attack_label = stem

    return device_label, attack_label


def is_header_row(line: str) -> bool:
    """Check if the first line is a CSV header row (non-numeric feature names).

    Args:
        line: The first line of a CSV file.

    Returns:
        bool: True if the row contains string column names, False if numeric data.
    """
    tokens = [t.strip().strip('"') for t in line.split(",") if t.strip()]
    if not tokens:
        return False
    try:
        float(tokens[0])
        return False
    except ValueError:
        return True


def inspect_sample_csv(csv_path: Path, max_rows: int = 2) -> Dict[str, Any]:
    """Inspect CSV header, feature count, and first sample rows.

    Auto-detects whether the file has a header (Kaggle format) or starts
    immediately with numeric values (UCI format).

    Args:
        csv_path: Path to sample CSV file.
        max_rows: Number of sample data rows to read.

    Returns:
        Dict containing column names, column count, and sample row previews.
    """
    header_cols: List[str] = []
    sample_rows: List[str] = []
    has_header = False

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        if first_line:
            first_tokens = [c.strip().strip('"') for c in first_line.split(",")]
            has_header = is_header_row(first_line)
            if has_header:
                header_cols = first_tokens
            else:
                header_cols = [f"feat_{i}" for i in range(len(first_tokens))]
                sample_rows.append(first_line.strip())

        while len(sample_rows) < max_rows:
            line = f.readline()
            if not line:
                break
            sample_rows.append(line.strip())

    return {
        "columns": header_cols,
        "num_features": len(header_cols),
        "has_header": has_header,
        "sample_rows": sample_rows,
    }


def verify_dataset(extract_dir: Path) -> Dict[str, Any]:
    """Scan and verify all CSV files in the extracted dataset directory.

    Computes total lines, data sample counts, device breakdown, and attack breakdown,
    and prints a structured verification report.

    Args:
        extract_dir: Path to directory containing extracted N-BaIoT CSVs.

    Returns:
        Dict[str, Any]: Verification metrics and summary statistics.
    """
    if not extract_dir.exists():
        print(f"[-] Directory not found: {extract_dir}")
        return {"status": "not_found", "total_files": 0}

    csv_files = sorted(extract_dir.rglob("*.csv"))
    if not csv_files:
        print(f"[-] No CSV files found in {extract_dir}.")
        return {"status": "empty", "total_files": 0}

    print(f"\n[*] Scanning and verifying {len(csv_files)} CSV files in {extract_dir}...")

    total_lines = 0
    total_samples = 0
    device_stats: Dict[str, Dict[str, int]] = {}
    attack_stats: Dict[str, Dict[str, int]] = {}
    sample_info: Optional[Dict[str, Any]] = None

    # Check the first CSV file to determine if files have headers
    first_file_has_header = False
    with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        if first_line:
            first_file_has_header = is_header_row(first_line)

    for csv_file in tqdm(csv_files, desc="Verifying CSV files", unit="files", ncols=90):
        lines = count_file_lines(csv_file)
        samples = max(0, lines - (1 if first_file_has_header else 0))

        total_lines += lines
        total_samples += samples

        device, attack = parse_device_and_attack(csv_file, extract_dir)

        if device not in device_stats:
            device_stats[device] = {"files": 0, "samples": 0}
        device_stats[device]["files"] += 1
        device_stats[device]["samples"] += samples

        if attack not in attack_stats:
            attack_stats[attack] = {"files": 0, "samples": 0}
        attack_stats[attack]["files"] += 1
        attack_stats[attack]["samples"] += samples

        if sample_info is None and samples > 0:
            sample_info = inspect_sample_csv(csv_file)

    # ──────────────── Formatted Verification Output ────────────────

    sep = "─" * 72
    print(f"\n{sep}")
    print("                N-BaIoT Dataset Verification Report")
    print(sep)
    print(f"Dataset Location:         {extract_dir}")
    print(f"Total CSV Files:          {len(csv_files):,}")
    print(f"Total Lines:              {total_lines:,}")
    print(f"Total Samples (data):     {total_samples:,} (excluding CSV headers)")

    if sample_info:
        num_feats = sample_info["num_features"]
        status_feat = "OK (Standard N-BaIoT)" if num_feats == 115 else "Note: Different feature count"
        print(f"Features per Sample:      {num_feats} [{status_feat}]")

    # Devices Table
    print(f"\n{sep}")
    print(f"{'Identified IoT Device':<46} {'Files':>8} {'Samples':>15}")
    print(sep)
    for dev in sorted(device_stats.keys()):
        f_cnt = device_stats[dev]["files"]
        s_cnt = device_stats[dev]["samples"]
        print(f"  {dev:<44} {f_cnt:>8} {s_cnt:>15,}")

    # Attack Types Table
    print(f"\n{sep}")
    print(f"{'Attack Classification / Category':<46} {'Files':>8} {'Samples':>15}")
    print(sep)
    for atk in sorted(attack_stats.keys()):
        f_cnt = attack_stats[atk]["files"]
        s_cnt = attack_stats[atk]["samples"]
        print(f"  {atk:<44} {f_cnt:>8} {s_cnt:>15,}")

    # Sample Preview
    if sample_info:
        print(f"\n{sep}")
        print("Sample Data Preview (First File Features & Rows):")
        cols = sample_info["columns"]
        cols_preview = cols[:6] + (["..."] if len(cols) > 6 else [])
        print(f"  Columns (first 6): {', '.join(cols_preview)}")
        for idx, row in enumerate(sample_info["sample_rows"], 1):
            row_short = row[:85] + ("..." if len(row) > 85 else "")
            print(f"  Row {idx} (sample): {row_short}")

    print(sep)
    print("Verification complete. Dataset is ready for feature reduction & preprocessing.")
    print(f"{sep}\n")

    return {
        "status": "success",
        "total_files": len(csv_files),
        "total_lines": total_lines,
        "total_samples": total_samples,
        "features": sample_info["num_features"] if sample_info else None,
        "devices": device_stats,
        "attacks": attack_stats,
    }


# ──────────────── Main CLI Entrypoint ────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for N-BaIoT download script."""
    parser = argparse.ArgumentParser(
        description="Download and verify the N-BaIoT dataset for TernaryGuard IoT IDS research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["uci", "kaggle"],
        default="uci",
        help="Download source: 'uci' (direct URL, default) or 'kaggle' (with UCI fallback).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory for downloaded raw archives (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=DEFAULT_EXTRACT_DIR,
        help=f"Directory for extracted CSV files (default: {DEFAULT_EXTRACT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction even if data already exists.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download and extraction; only run verification on existing files.",
    )
    return parser


def main() -> int:
    """Main execution workflow."""
    parser = build_parser()
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir.resolve()
    extract_dir: Path = args.extract_dir.resolve()
    force: bool = args.force
    source: str = args.source

    print("=" * 72)
    print("      TernaryGuard: N-BaIoT Dataset Download & Verification")
    print("=" * 72)
    print(f"Target raw directory:     {raw_dir}")
    print(f"Target extract directory: {extract_dir}")
    print(f"Selected source:          {source}")
    print("=" * 72)

    # 1. Check if dataset is already extracted
    if not force and extract_dir.exists():
        existing_csvs = list(extract_dir.rglob("*.csv"))
        if existing_csvs:
            print(f"[+] Found {len(existing_csvs)} existing CSV file(s) in {extract_dir}.")
            if not args.verify_only:
                print("    Skipping download and extraction (use --force to overwrite).")
            verify_dataset(extract_dir)
            return 0

    # 2. Check verify-only mode
    if args.verify_only:
        print(f"[*] Running in --verify-only mode on {extract_dir}")
        stats = verify_dataset(extract_dir)
        return 0 if stats.get("total_files", 0) > 0 else 1

    # 3. Download workflow
    archive_path: Optional[Path] = None
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if a valid archive already exists locally
    default_uci_archive = raw_dir / UCI_ARCHIVE_FILENAME
    if not force and default_uci_archive.exists() and zipfile.is_zipfile(default_uci_archive):
        print(f"[+] Found existing valid archive: {default_uci_archive} (skipping download).")
        archive_path = default_uci_archive
    else:
        if source == "kaggle":
            print("[*] Attempting download from Kaggle dataset mkashifn/nbaiot-dataset...")
            archive_path = download_from_kaggle(raw_dir)
            if archive_path is None:
                print("[!] Kaggle download failed. Falling back to direct UCI repository...")
                try:
                    archive_path = download_from_uci(raw_dir)
                except Exception as exc:
                    print(f"[-] UCI fallback download failed: {exc}")
                    return 1
        else:  # source == "uci"
            print("[*] Attempting direct download from UCI repository...")
            try:
                archive_path = download_from_uci(raw_dir)
            except Exception as exc:
                print(f"[-] UCI download failed: {exc}")
                print("\n[TIP] If network access to UCI is restricted or timing out, you can:")
                print("      1. Try: python research/download_nbaiot.py --source kaggle")
                print("      2. Manually download the archive and place it in data/raw/:")
                print(f"         {UCI_DOWNLOAD_URL}")
                return 1

    if archive_path is None or not archive_path.exists():
        print("[-] Download failed; no archive available for extraction.")
        return 1

    # 4. Extraction workflow
    try:
        extract_archive(archive_path, extract_dir, force=force)
    except Exception as exc:
        print(f"[-] Extraction failed: {exc}")
        return 1

    # 5. Verification workflow
    stats = verify_dataset(extract_dir)
    if stats.get("total_files", 0) > 0:
        print("[+] Dataset pipeline ready for Phase 2: Feature Selection & Preprocessing.")
        return 0
    else:
        print("[-] Verification failed: No CSV files were produced.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
