"""Download medical imaging datasets for FedProtoLoRA using kagglehub.

Supports idempotent downloads, progress tracking, and symlinking/copying into data/raw/<client>.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import kagglehub

DATASET_SPECS = {
    "brain": {
        "slug": "masoudnickparvar/brain-tumor-mri-dataset",
        "modality": "Brain MRI",
        "size_gb": 0.16,
    },
    "cxr": {
        "slug": "paultimothymooney/chest-xray-pneumonia",
        "modality": "Chest X-ray",
        "size_gb": 1.15,
    },
    "derm": {
        "slug": "kmader/skin-cancer-mnist-ham10000",
        "modality": "Dermoscopy",
        "size_gb": 2.70,
    },
    "histo": {
        "slug": "ambarish/breakhis",
        "modality": "Histopathology",
        "size_gb": 3.99,
    },
    "oct": {
        "slug": "paultimothymooney/kermany2018",
        "modality": "Retinal OCT",
        "size_gb": 5.41,
    },
}

def download_dataset(client_name: str, raw_dir: Path, force: bool = False) -> Path:
    if client_name not in DATASET_SPECS:
        raise ValueError(f"Unknown client '{client_name}'. Available: {list(DATASET_SPECS.keys())}")

    spec = DATASET_SPECS[client_name]
    slug = spec["slug"]
    dest_dir = raw_dir / client_name

    if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
        print(f"[{client_name}] Raw dataset directory already exists at {dest_dir}, skipping download.")
        return dest_dir

    print(f"[{client_name}] Initiating download for '{slug}' (~{spec['size_gb']} GB)...")
    download_path = kagglehub.dataset_download(slug)
    src_path = Path(download_path)
    print(f"[{client_name}] Downloaded to cache: {src_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Link or copy into raw_dir / client_name
    # On Windows, copy or symlink
    print(f"[{client_name}] Linking/copying data into {dest_dir}...")
    for item in src_path.iterdir():
        target = dest_dir / item.name
        if not target.exists():
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    print(f"[{client_name}] Ready in {dest_dir}")
    return dest_dir

def main():
    parser = argparse.ArgumentParser(description="Download datasets for FedProtoLoRA")
    parser.add_argument("--clients", nargs="+", default=["brain", "cxr", "derm", "histo", "oct"],
                        help="List of clients to download")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Target raw directory")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for client in args.clients:
        try:
            download_dataset(client, raw_dir, force=args.force)
        except Exception as e:
            print(f"Error downloading {client}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
