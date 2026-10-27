"""Deduplication and leakage-safe grouping for medical imaging datasets.

Provides:
- Exact duplicate detection (MD5 hash); exact copies are dropped (both copies if their labels conflict)
- Near-duplicate detection (64-bit pHash, Hamming distance via packed uint64 popcount)
- Near-duplicate policies:
  * "group": near-duplicates are merged into one group (union-find together with the IDs), so they always land in
    the same split and no image is discarded. Use only when no patient IDs exist (brain MRI): on homogeneous
    modalities (OCT B-scans) 64-bit pHash links distinct patients transitively into one giant component.
  * "remove_cross_group": patient/lesion IDs already prevent within-patient leakage; an image whose near-duplicate
    belongs to a *different* patient group is dropped (removal cannot chain).
  * "remove": drop every image that is a near-duplicate of an earlier kept image.
- Optional resized image cache (decoding full-resolution chest X-rays dominates training time otherwise)
- Logging of every dropped / grouped image to <client>_dedup_report.csv
"""

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image


def compute_file_hash(image_path: Path) -> str:
    """Computes MD5 hash of an image file."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(image_path: Path, hash_size: int = 8) -> Tuple[str, np.ndarray]:
    """Computes perceptual hash string and flattened boolean array (64 bits)."""
    import imagehash  # deferred: only preprocessing needs it, training does not

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        ph = imagehash.phash(img_rgb, hash_size=hash_size)
    return str(ph), ph.hash.flatten()


def _hash_and_cache(args: Tuple[str, Optional[str], int]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Worker: (md5, phash_hex, error). Writes a resized RGB copy to `cache_path` when given."""
    import imagehash

    path, cache_path, cache_size = args
    try:
        md5 = compute_file_hash(Path(path))
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
        ph = str(imagehash.phash(img_rgb, hash_size=8))
        if cache_path is not None:
            cp = Path(cache_path)
            if not cp.exists():
                cp.parent.mkdir(parents=True, exist_ok=True)
                small = img_rgb.copy()
                small.thumbnail((cache_size, cache_size), Image.BICUBIC)  # keeps aspect ratio, never upsamples
                small.save(cp, quality=95)
        return md5, ph, None
    except Exception as e:  # unreadable / truncated file
        return None, None, f"{type(e).__name__}: {e}"


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def deduplicate_dataset_records(
    records: List[Dict],
    hamming_threshold: int = 4,
    report_path: Path = None,
    near_dup_policy: str = "group",
    num_workers: int = 8,
    cache_dir: Optional[Path] = None,
    cache_size: int = 512,
) -> Tuple[List[Dict], pd.DataFrame]:
    """Exact-duplicate removal plus near-duplicate grouping (or removal).

    Each record needs "image_path", "label" and "group_id". On return every kept record has "md5",
    "phash", "orig_group_id" and a (possibly merged) "group_id"; with `cache_dir`, also "cache_path".
    """
    assert near_dup_policy in ("group", "remove", "remove_cross_group"), near_dup_policy
    total = len(records)
    print(f"Hashing {total} records ({num_workers} workers, near_dup_policy={near_dup_policy}, "
          f"hamming_threshold={hamming_threshold}, cache={'off' if cache_dir is None else cache_dir})...", flush=True)

    jobs = []
    for item in records:
        cache_path = None
        if cache_dir is not None:
            # Stable, collision-free cache name derived from the source path.
            key = hashlib.md5(str(item["image_path"]).encode("utf-8")).hexdigest()
            cache_path = str(Path(cache_dir) / f"{key}.jpg")
            item["cache_path"] = cache_path
        jobs.append((str(item["image_path"]), cache_path, cache_size))

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(_hash_and_cache, jobs, chunksize=64))
    else:
        results = [_hash_and_cache(j) for j in jobs]

    removed_records: List[Dict] = []

    def _report(item, reason, dup_of="", distance=-1):
        removed_records.append({
            "client": item.get("client", "unknown"),
            "removed_file": str(item["image_path"]),
            "duplicate_of": dup_of,
            "reason": reason,
            "distance": distance,
        })

    # 1. Unreadable files
    readable = []
    for item, (md5, ph, err) in zip(records, results):
        if err is not None:
            _report(item, f"unreadable_image: {err}")
            continue
        item["md5"], item["phash"] = md5, ph
        readable.append(item)

    # 2. Exact duplicates: keep the first copy; if copies disagree on the label, drop all of them.
    by_md5: Dict[str, List[Dict]] = {}
    for item in readable:
        by_md5.setdefault(item["md5"], []).append(item)
    unique = []
    for md5, items in by_md5.items():
        if len({str(it["label"]) for it in items}) > 1:
            for it in items:
                _report(it, "exact_md5_duplicate_label_conflict", dup_of=str(items[0]["image_path"]), distance=0)
            continue
        unique.append(items[0])
        for it in items[1:]:
            _report(it, "exact_md5_duplicate", dup_of=str(items[0]["image_path"]), distance=0)

    # 3. Near-duplicates (packed 64-bit pHash, popcount of XOR)
    n = len(unique)
    hashes = np.array([int(it["phash"], 16) for it in unique], dtype=np.uint64)
    uf = _UnionFind(n)
    first_of_group: Dict[str, int] = {}
    for i, it in enumerate(unique):
        it["orig_group_id"] = str(it["group_id"])
        j = first_of_group.setdefault(it["orig_group_id"], i)
        uf.union(i, j)

    group_codes = pd.factorize(pd.Series([it["orig_group_id"] for it in unique]))[0] if n else np.array([], dtype=int)
    keep_mask = np.ones(n, dtype=bool)
    removing = near_dup_policy != "group"
    n_near = 0
    for i in range(1, n):
        dists = np.bitwise_count(hashes[:i] ^ hashes[i])
        if removing:
            eligible = keep_mask[:i]
            if near_dup_policy == "remove_cross_group":
                eligible = eligible & (group_codes[:i] != group_codes[i])
            dists = np.where(eligible, dists, 64)
        matches = np.nonzero(dists <= hamming_threshold)[0]
        if matches.size == 0:
            continue
        n_near += 1
        nearest = int(matches[np.argmin(dists[matches])])
        if removing:
            keep_mask[i] = False
            _report(unique[i], f"near_duplicate_phash_removed_{near_dup_policy}", dup_of=str(unique[nearest]["image_path"]),
                    distance=int(dists[nearest]))
        else:
            for j in matches:
                uf.union(i, int(j))
            _report(unique[i], "near_duplicate_phash_grouped", dup_of=str(unique[nearest]["image_path"]),
                    distance=int(dists[nearest]))
        if (i + 1) % 10000 == 0:
            print(f"  near-dup scan {i + 1}/{n} | near-duplicates so far: {n_near}", flush=True)

    kept = []
    for i, it in enumerate(unique):
        if not keep_mask[i]:
            continue
        it["group_id"] = unique[uf.find(i)]["orig_group_id"]
        kept.append(it)

    n_groups_before = len(first_of_group)
    n_groups_after = len({it["group_id"] for it in kept})
    print(f"Kept {len(kept)}/{total} images | exact/unreadable dropped: {len(readable) - len(unique) + total - len(readable)} | "
          f"near-duplicates: {n_near} | groups {n_groups_before} -> {n_groups_after}", flush=True)

    report_df = pd.DataFrame(removed_records, columns=["client", "removed_file", "duplicate_of", "reason", "distance"])
    # Always (over)write the report, even when empty, so a stale report from an earlier policy cannot survive.
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False)
        print(f"Deduplication report saved to {report_path} ({len(report_df)} rows)", flush=True)

    return kept, report_df
