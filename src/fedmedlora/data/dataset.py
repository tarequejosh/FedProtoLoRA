"""PyTorch Dataset and transformation pipelines for medical imaging foundation models."""

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_medical_transforms(
    image_size: int = 224,
    is_train: bool = True,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> v2.Compose:
    """Standardized preprocessing and augmentation pipeline (accepts PIL images or uint8 tensors)."""
    ops = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]
    if is_train:
        ops += [v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(degrees=10)]
    ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)]
    return v2.Compose(ops)


def gpu_batch_transform(
    images: torch.Tensor,
    mean,
    std,
    train: bool,
    rotation_degrees: float = 10.0,
    flip_p: float = 0.5,
) -> torch.Tensor:
    """Batched on-device equivalent of `get_medical_transforms` for uint8 (B, 3, H, W) batches.

    Train: per-sample random horizontal flip and rotation in [-deg, deg] (zero fill, like torchvision),
    then scaling to [0, 1] and normalisation. Eval: scaling and normalisation only.
    """
    x = images.float().div_(255.0)
    if train:
        b = x.shape[0]
        flip = torch.rand(b, device=x.device) < flip_p
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        angle = (torch.rand(b, device=x.device) * 2.0 - 1.0) * math.radians(rotation_degrees)
        cos, sin, zero = angle.cos(), angle.sin(), torch.zeros_like(angle)
        theta = torch.stack([torch.stack([cos, -sin, zero], -1), torch.stack([sin, cos, zero], -1)], 1)
        grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


class MedicalImageDataset(Dataset):
    """PyTorch Dataset loading medical images from a manifest dataframe or CSV file.

    With `cache_in_memory=True`, each image is decoded once and stored as a resized uint8 tensor
    (`cache_size` x `cache_size`, ~150 KB at 224 px); the (random) transform is applied to the cached
    tensor on every access, so augmentation is preserved while JPEG decoding happens only once.
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, Path, str],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        cache_in_memory: bool = False,
        cache_size: int = 224,
        return_uint8: bool = False,
        tensor_cache: Optional[Tuple[np.ndarray, Dict[str, int]]] = None,
    ):
        if isinstance(manifest, (str, Path)):
            self.df = pd.read_csv(manifest)
        else:
            self.df = manifest.copy()

        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.cache_size = cache_size
        # return_uint8: yield the resized, un-augmented uint8 image; augmentation/normalisation then run batched
        # on the GPU (gpu_batch_transform), which removes the CPU bottleneck.
        self.return_uint8 = return_uint8
        # Shared on-disk cache from scripts/build_tensor_cache.py: (memmap (N,3,S,S) uint8, image_path -> row).
        self.tensor_cache = tensor_cache
        self.cached_images: Dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_rgb(self, row) -> Image.Image:
        img_path = Path(row["image_path"])
        # Prefer the resized file cache written by scripts/preprocess.py (same image, max side 512 px).
        cache_path = row["cache_path"] if "cache_path" in row.index else None
        load_path = Path(cache_path) if isinstance(cache_path, str) and Path(cache_path).exists() else img_path
        with Image.open(load_path) as raw_img:
            return raw_img.convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        label_idx = int(row["label_idx"])
        path = str(row["image_path"])

        if self.tensor_cache is not None and path in self.tensor_cache[1]:
            img = torch.from_numpy(np.array(self.tensor_cache[0][self.tensor_cache[1][path]]))
            if self.return_uint8:
                return img, label_idx, path
        elif self.cache_in_memory or self.return_uint8:
            img = self.cached_images.get(idx)
            if img is None:
                pil = self._load_rgb(row).resize((self.cache_size, self.cache_size), Image.BILINEAR)
                img = v2.functional.pil_to_tensor(pil)  # uint8, un-augmented
                if self.cache_in_memory:
                    self.cached_images[idx] = img
            if self.return_uint8:
                return img, label_idx, path
        else:
            img = self._load_rgb(row)

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = v2.functional.to_dtype(v2.functional.to_image(img), torch.float32, scale=True)

        return img_tensor, label_idx, path
