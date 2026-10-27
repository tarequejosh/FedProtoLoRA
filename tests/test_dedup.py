"""Tests for exact/near-duplicate handling and leakage-safe group merging."""

import numpy as np
from PIL import Image

from fedmedlora.data.dedup import deduplicate_dataset_records


def _save(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return str(path)


def _records(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3))
    other = rng.integers(0, 255, (64, 64, 3))
    third = rng.integers(0, 255, (64, 64, 3))
    near = base.copy()
    near[0, 0] = (near[0, 0] + 1) % 255  # tiny change: different MD5, same pHash
    return [
        {"image_path": _save(tmp_path / "a.png", base), "label": "x", "group_id": "p1"},
        {"image_path": _save(tmp_path / "b.png", near), "label": "x", "group_id": "p2"},     # near-dup of a
        {"image_path": _save(tmp_path / "c.png", other), "label": "y", "group_id": "p3"},
        {"image_path": _save(tmp_path / "c_copy.png", other), "label": "y", "group_id": "p4"},  # exact copy of c
        {"image_path": _save(tmp_path / "d.png", third), "label": "x", "group_id": "p5"},
        {"image_path": _save(tmp_path / "d_conflict.png", third), "label": "y", "group_id": "p6"},  # copy, other label
    ]


def test_group_policy_merges_near_duplicates(tmp_path):
    kept, report = deduplicate_dataset_records(_records(tmp_path), num_workers=1, near_dup_policy="group")
    paths = {k["image_path"].split("\\")[-1].split("/")[-1]: k for k in kept}
    assert set(paths) == {"a.png", "b.png", "c.png"}  # exact copy dropped, label-conflicting pair dropped
    assert paths["a.png"]["group_id"] == paths["b.png"]["group_id"] == "p1"
    assert paths["b.png"]["orig_group_id"] == "p2"
    assert "exact_md5_duplicate_label_conflict" in set(report["reason"])


def test_remove_policy_drops_near_duplicates(tmp_path):
    kept, _ = deduplicate_dataset_records(_records(tmp_path), num_workers=1, near_dup_policy="remove")
    names = {k["image_path"].split("\\")[-1].split("/")[-1] for k in kept}
    assert names == {"a.png", "c.png"}


def test_remove_cross_group_keeps_same_patient_near_duplicates(tmp_path):
    recs = _records(tmp_path)
    recs[1]["group_id"] = "p1"  # near-duplicate of a.png from the SAME patient -> kept
    kept, _ = deduplicate_dataset_records(recs, num_workers=1, near_dup_policy="remove_cross_group")
    names = {k["image_path"].split("\\")[-1].split("/")[-1] for k in kept}
    assert names == {"a.png", "b.png", "c.png"}
    recs = _records(tmp_path)  # b.png belongs to another patient -> removed
    kept, _ = deduplicate_dataset_records(recs, num_workers=1, near_dup_policy="remove_cross_group")
    assert {k["image_path"].split("\\")[-1].split("/")[-1] for k in kept} == {"a.png", "c.png"}


def test_cache_is_written(tmp_path):
    cache = tmp_path / "cache"
    kept, _ = deduplicate_dataset_records(_records(tmp_path)[:1], num_workers=1, cache_dir=cache, cache_size=32)
    with Image.open(kept[0]["cache_path"]) as im:
        assert max(im.size) == 32


def test_in_memory_cache_keeps_augmentation(tmp_path):
    """The uint8 cache must hold the *un-augmented* image: random transforms still differ per access."""
    import pandas as pd
    import torch
    from fedmedlora.data.dataset import MedicalImageDataset, get_medical_transforms

    rng = np.random.default_rng(0)
    path = _save(tmp_path / "img.png", rng.integers(0, 255, (300, 260, 3)))
    df = pd.DataFrame([{"image_path": path, "label_idx": 0}])
    train = MedicalImageDataset(df, transform=get_medical_transforms(224, True), cache_in_memory=True)
    torch.manual_seed(0)
    views = [train[0][0] for _ in range(6)]
    assert any(not torch.equal(views[0], v) for v in views[1:])
    assert train.cached_images[0].dtype == torch.uint8 and train.cached_images[0].shape == (3, 224, 224)

    plain = MedicalImageDataset(df, transform=get_medical_transforms(224, False))
    cached = MedicalImageDataset(df, transform=get_medical_transforms(224, False), cache_in_memory=True)
    assert (plain[0][0] - cached[0][0]).abs().mean() < 0.1  # same image up to resampling differences


def test_gpu_batch_transform_matches_eval_pipeline(tmp_path):
    """Eval path of the batched transform must equal the per-image torchvision pipeline on the same uint8 image."""
    import torch
    from fedmedlora.data.dataset import gpu_batch_transform, get_medical_transforms, IMAGENET_MEAN, IMAGENET_STD

    img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
    ref = get_medical_transforms(224, False)(img)
    out = gpu_batch_transform(img[None], IMAGENET_MEAN, IMAGENET_STD, train=False)[0]
    assert torch.allclose(out, ref, atol=1e-5)
    torch.manual_seed(0)
    aug = gpu_batch_transform(img[None].repeat(4, 1, 1, 1), IMAGENET_MEAN, IMAGENET_STD, train=True)
    assert aug.shape == (4, 3, 224, 224) and not torch.allclose(aug[0], aug[1])


def test_tensor_cache_matches_decoded_pixels(tmp_path):
    """The shared on-disk tensor cache must return exactly the pixels of the decode-on-the-fly path."""
    import sys
    import pandas as pd
    import torch
    from fedmedlora.data.dataset import MedicalImageDataset

    sys.path.insert(0, "scripts")
    from build_tensor_cache import build

    rng = np.random.default_rng(1)
    paths = [_save(tmp_path / f"i{k}.png", rng.integers(0, 255, (180 + 10 * k, 150, 3))) for k in range(3)]
    df = pd.DataFrame({"image_path": paths, "label_idx": [0, 1, 0]})
    (tmp_path / "proc").mkdir()
    df.to_csv(tmp_path / "proc" / "toy.csv", index=False)
    build("toy", tmp_path / "proc", tmp_path / "tc", 224, workers=1)
    mm = np.load(tmp_path / "tc" / "toy_224.npy", mmap_mode="r")
    index = pd.read_csv(tmp_path / "tc" / "toy_224_index.csv")
    cached = MedicalImageDataset(df, return_uint8=True, tensor_cache=(mm, dict(zip(index.image_path, index.row))))
    plain = MedicalImageDataset(df, return_uint8=True)
    for i in range(3):
        assert torch.equal(cached[i][0], plain[i][0])
