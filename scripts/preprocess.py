"""Preprocess raw datasets: extract metadata, recover patient groups, and deduplicate."""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd
from fedmedlora.data.dedup import deduplicate_dataset_records
from fedmedlora.data.splits import (
    extract_patient_id_cxr,
    extract_patient_id_histo,
    extract_patient_id_oct,
)

def project_path(path: Path) -> str:
    """Path relative to the working directory (the project root), with forward slashes, so manifests are portable."""
    full = path.resolve()
    try:
        return full.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return full.as_posix()

def parse_brain_records(raw_dir: Path) -> List[Dict]:
    """Parses Brain MRI dataset."""
    records = []
    seen_files = set()
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    for split_name in ["Training", "Testing"]:
        p = raw_dir / split_name
        if not p.exists():
            continue
        for cls_name in classes:
            cls_p = p / cls_name
            if not cls_p.exists():
                continue
            for img_file in cls_p.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    norm_path = str(img_file.resolve()).lower()
                    if norm_path in seen_files:
                        continue
                    seen_files.add(norm_path)
                    records.append({
                        "client": "brain",
                        "image_path": project_path(img_file),
                        "label": cls_name,
                        "label_idx": classes.index(cls_name),
                        # No patient IDs exist: each file starts as its own group; near-duplicates are merged later.
                        "group_id": f"{split_name}/{cls_name}/{img_file.stem}",
                    })
    return records

def parse_cxr_records(raw_dir: Path) -> List[Dict]:
    """Parses Chest X-ray Pneumonia dataset."""
    records = []
    seen_files = set()
    classes = ["NORMAL", "PNEUMONIA"]
    search_dirs = [raw_dir, raw_dir / "chest_xray"]
    for s_dir in search_dirs:
        for split_dir in ["train", "test", "val"]:
            sub = s_dir / split_dir
            if not sub.exists():
                continue
            for cls_name in classes:
                cls_p = sub / cls_name
                if not cls_p.exists():
                    continue
                for img_file in cls_p.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        norm_path = str(img_file.resolve()).lower()
                        if norm_path in seen_files:
                            continue
                        seen_files.add(norm_path)
                        pid = extract_patient_id_cxr(img_file.name)
                        records.append({
                            "client": "cxr",
                            "image_path": project_path(img_file),
                            "label": cls_name,
                            "label_idx": classes.index(cls_name),
                            "group_id": pid,
                        })
    return records

def parse_oct_records(raw_dir: Path) -> List[Dict]:
    """Parses Retinal OCT dataset."""
    records = []
    seen_files = set()
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    search_dirs = [raw_dir, raw_dir / "OCT2017", raw_dir / "kermany2018"]
    for s_dir in search_dirs:
        for split_dir in ["train", "test", "val"]:
            sub = s_dir / split_dir
            if not sub.exists():
                continue
            for cls_name in classes:
                cls_p = sub / cls_name
                if not cls_p.exists():
                    continue
                for img_file in cls_p.glob("*.*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        norm_path = str(img_file.resolve()).lower()
                        if norm_path in seen_files:
                            continue
                        seen_files.add(norm_path)
                        pid = extract_patient_id_oct(img_file.name)
                        records.append({
                            "client": "oct",
                            "image_path": project_path(img_file),
                            "label": cls_name,
                            "label_idx": classes.index(cls_name),
                            "group_id": pid,
                        })
    return records

def parse_histo_records(raw_dir: Path) -> List[Dict]:
    """Parses Histopathology (BreaKHis) dataset for binary benign/malignant classification."""
    records = []
    seen_files = set()
    classes = ["benign", "malignant"]
    for img_file in raw_dir.rglob("*.*"):
        if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            norm_path = str(img_file.resolve()).lower()
            if norm_path in seen_files:
                continue
            seen_files.add(norm_path)
            path_str = str(img_file).lower()
            if "benign" in path_str:
                label = "benign"
                label_idx = 0
            elif "malignant" in path_str:
                label = "malignant"
                label_idx = 1
            else:
                continue
            pid = extract_patient_id_histo(img_file.name)
            records.append({
                "client": "histo",
                "image_path": project_path(img_file),
                "label": label,
                "label_idx": label_idx,
                "group_id": pid,
            })
    return records

def parse_derm_records(raw_dir: Path) -> List[Dict]:
    """Parses HAM10000 dermoscopy dataset using metadata CSV."""
    records = []
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    # Only the metadata CSV carries lesion_id; the hmnist_*.csv files in the same folder are pixel dumps.
    csv_candidates = list(raw_dir.rglob("HAM10000_metadata*.csv"))
    if not csv_candidates:
        print(f"Warning: No metadata CSV found in {raw_dir}")
        return records
    meta_df = pd.read_csv(csv_candidates[0])

    img_map = {}
    for img_file in raw_dir.rglob("*.jpg"):
        img_map[img_file.stem] = project_path(img_file)

    for _, row in meta_df.iterrows():
        img_id = str(row["image_id"])
        dx = str(row["dx"]).lower()
        if img_id in img_map and dx in classes:
            records.append({
                "client": "derm",
                "image_path": img_map[img_id],
                "label": dx,
                "label_idx": classes.index(dx),
                "group_id": str(row["lesion_id"]),
            })
    return records

# Brain MRI has no patient IDs, so near-duplicates are merged into groups; the other clients have real
# patient/lesion IDs, and cross-patient near-duplicates are removed instead (merging chains on OCT/CXR).
DEFAULT_NEAR_DUP_POLICY = {
    "brain": "group",
    "cxr": "remove_cross_group",
    "oct": "remove_cross_group",
    "derm": "remove_cross_group",
    "histo": "remove_cross_group",
}

PARSERS = {
    "brain": parse_brain_records,
    "cxr": parse_cxr_records,
    "oct": parse_oct_records,
    "histo": parse_histo_records,
    "derm": parse_derm_records,
}

def preprocess_client(
    client: str,
    raw_base: Path,
    processed_base: Path,
    dedup_report_path: Path,
    near_dup_policy: str = None,
    hamming_threshold: int = 4,
    num_workers: int = 8,
    cache_base: Path = None,
    cache_size: int = 512,
):
    raw_dir = raw_base / client
    if not raw_dir.exists():
        print(f"[{client}] Raw directory {raw_dir} does not exist, skipping.", flush=True)
        return

    parser = PARSERS.get(client)
    if parser is None:
        print(f"No parser defined for client {client}", flush=True)
        return

    print(f"[{client}] Parsing raw records from {raw_dir}...", flush=True)
    records = parser(raw_dir)
    print(f"[{client}] Found {len(records)} raw images in {len({r['group_id'] for r in records})} groups.", flush=True)
    if not records:
        return

    near_dup_policy = near_dup_policy or DEFAULT_NEAR_DUP_POLICY.get(client, "remove_cross_group")
    print(f"[{client}] Running exact (MD5) and perceptual (pHash) deduplication...", flush=True)
    kept, dedup_df = deduplicate_dataset_records(
        records,
        hamming_threshold=hamming_threshold,
        report_path=dedup_report_path / f"{client}_dedup_report.csv",
        near_dup_policy=near_dup_policy,
        num_workers=num_workers,
        cache_dir=(cache_base / client) if cache_base is not None else None,
        cache_size=cache_size,
    )
    print(f"[{client}] Kept {len(kept)} images ({len(records) - len(kept)} dropped).", flush=True)

    processed_df = pd.DataFrame(kept)
    out_file = processed_base / f"{client}.csv"
    processed_base.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(out_file, index=False)
    print(f"[{client}] Saved processed manifest to {out_file}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Preprocess medical imaging datasets")
    parser.add_argument("--clients", nargs="+", default=["brain", "cxr", "derm", "histo", "oct"])
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--dedup_dir", type=str, default="data/dedup_reports")
    parser.add_argument("--near_dup_policy", choices=["group", "remove", "remove_cross_group"], default=None,
                        help="Default per client: brain=group (no patient IDs), others=remove_cross_group")
    parser.add_argument("--hamming_threshold", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default="data/cache",
                        help="Resized image cache (empty string disables)")
    parser.add_argument("--cache_size", type=int, default=512, help="Max side of cached images")
    args = parser.parse_args()

    raw_base = Path(args.raw_dir)
    processed_base = Path(args.processed_dir)
    dedup_dir = Path(args.dedup_dir)
    cache_base = Path(args.cache_dir) if args.cache_dir else None

    for client in args.clients:
        preprocess_client(client, raw_base, processed_base, dedup_dir, near_dup_policy=args.near_dup_policy,
                          hamming_threshold=args.hamming_threshold, num_workers=args.num_workers,
                          cache_base=cache_base, cache_size=args.cache_size)

if __name__ == "__main__":
    main()
