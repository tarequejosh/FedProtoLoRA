"""Pre-decode every processed image once into a shared on-disk uint8 tensor cache (N, 3, 224, 224) per client.

All ~150 experiment runs then read memory-mapped tensors instead of decoding JPEGs (identical pixels to the
in-memory cache: same file, same bilinear resize). Output: data/tensor_cache/<client>_224.npy plus
<client>_224_index.csv mapping image_path -> row.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _decode(args):
    image_path, cache_path, size = args
    load = cache_path if isinstance(cache_path, str) and Path(cache_path).exists() else image_path
    with Image.open(load) as im:
        arr = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return arr.transpose(2, 0, 1)  # CHW, same layout as torchvision pil_to_tensor


def build(client: str, processed_dir: Path, out_dir: Path, size: int, workers: int):
    df = pd.read_csv(processed_dir / f"{client}.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{client}_{size}.npy"
    index_path = out_dir / f"{client}_{size}_index.csv"
    if npy_path.exists() and index_path.exists() and len(pd.read_csv(index_path)) == len(df):
        print(f"[{client}] tensor cache up to date ({len(df)} images)", flush=True)
        return
    arr = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint8, shape=(len(df), 3, size, size))
    cache_paths = df["cache_path"] if "cache_path" in df.columns else [None] * len(df)
    jobs = list(zip(df["image_path"], cache_paths, [size] * len(df)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, img in enumerate(pool.map(_decode, jobs, chunksize=64)):
            arr[i] = img
            if (i + 1) % 10000 == 0:
                print(f"[{client}] {i + 1}/{len(df)}", flush=True)
    arr.flush()
    del arr
    pd.DataFrame({"image_path": df["image_path"], "row": np.arange(len(df))}).to_csv(index_path, index=False)
    print(f"[{client}] wrote {npy_path} ({len(df)} images)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build shared uint8 tensor caches")
    parser.add_argument("--clients", nargs="+", default=["brain", "cxr", "derm", "histo", "oct"])
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/tensor_cache")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    for client in args.clients:
        build(client, Path(args.processed_dir), Path(args.out_dir), args.size, args.workers)


if __name__ == "__main__":
    main()
