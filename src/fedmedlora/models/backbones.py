"""Vision Foundation Model loaders with native LoRA injection.

Supports:
- DINOv2 (Small, Base) via Hugging Face transformers
- BiomedCLIP visual encoder via open_clip (timm ViT trunk)
- ToyViTBackbone for fast unit tests / --smoke mode without network downloads

Modes (selected by arguments):
- LoRA (r > 0): backbone frozen, LoRA adapters trainable.
- Linear probe (r == 0, full_finetune=False): everything frozen, only the head trains.
- Full fine-tuning (r == 0, full_finetune=True): whole backbone trainable (FedAvg-Full baseline).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .heads import ClassificationHead
from .lora import inject_lora_to_linear

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# open_clip / BiomedCLIP preprocessing statistics
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Candidate attention-projection names, tried in order. transformers>=5 renamed DINOv2's
# `query`/`value` to `q_proj`/`v_proj`; older versions use `query`/`value`.
DINOV2_TARGET_CANDIDATES: List[List[str]] = [["q_proj", "v_proj"], ["query", "value"]]


def get_backbone_normalization(backbone_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Returns (mean, std) used by the backbone's own preprocessing."""
    if "biomedclip" in backbone_name.lower():
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


class ToyViTBackbone(nn.Module):
    """Lightweight 2-block Vision Transformer for CPU/fast smoke testing."""

    def __init__(self, embed_dim: int = 128, num_blocks: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(embed_dim),
                        "q_proj": nn.Linear(embed_dim, embed_dim),
                        "k_proj": nn.Linear(embed_dim, embed_dim),
                        "v_proj": nn.Linear(embed_dim, embed_dim),
                        "out_proj": nn.Linear(embed_dim, embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 2),
                            nn.GELU(),
                            nn.Linear(embed_dim * 2, embed_dim),
                        ),
                    }
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), patches], dim=1)
        for block in self.blocks:
            h = block["norm1"](tokens)
            q, k, v = block["q_proj"](h), block["k_proj"](h), block["v_proj"](h)
            attn = ((q @ k.transpose(-2, -1)) * (self.embed_dim**-0.5)).softmax(dim=-1)
            tokens = tokens + block["out_proj"](attn @ v)
            tokens = tokens + block["mlp"](block["norm2"](tokens))
        return self.norm(tokens)[:, 0]


class VisionFoundationModel(nn.Module):
    """Frozen (or fully trainable) vision foundation model backbone with optional LoRA adapters."""

    def __init__(
        self,
        backbone_name: str = "dinov2_small",
        pretrained: bool = True,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        is_smoke: bool = False,
        full_finetune: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.is_smoke = is_smoke or self.backbone_name == "toy"
        self.r = r
        self.lora_alpha = lora_alpha
        self.full_finetune = full_finetune

        candidates: List[List[str]]
        if self.is_smoke:
            self.embed_dim = 128
            self.backbone = ToyViTBackbone(embed_dim=128)
            candidates = [["q_proj", "v_proj"]]
        elif "dinov2" in self.backbone_name:
            from transformers import AutoModel, Dinov2Config, Dinov2Model

            size = "base" if "base" in self.backbone_name else "small"
            self.embed_dim = 768 if size == "base" else 384
            if pretrained:
                self.backbone = AutoModel.from_pretrained(f"facebook/dinov2-{size}")
            else:
                heads = 12 if size == "base" else 6
                self.backbone = Dinov2Model(
                    Dinov2Config(hidden_size=self.embed_dim, num_hidden_layers=12, num_attention_heads=heads)
                )
            if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
                self.backbone.gradient_checkpointing_enable()
            candidates = DINOV2_TARGET_CANDIDATES
        elif "biomedclip" in self.backbone_name:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self.backbone = model.visual
            self.embed_dim = 512
            # open_clip wraps a timm ViT whose attention uses a fused `qkv` Linear.
            candidates = [["qkv"]]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        for param in self.backbone.parameters():
            param.requires_grad = full_finetune

        self.lora_layers = {}
        if r > 0:
            if full_finetune:
                raise ValueError("full_finetune=True is only supported with r=0.")
            target_lists = [target_modules] if target_modules else candidates
            for targets in target_lists:
                self.lora_layers = inject_lora_to_linear(
                    self.backbone, target_names=targets, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
                if self.lora_layers:
                    self.target_modules = targets
                    break
            # Fail loudly: silently training without adapters invalidates every LoRA result.
            if not self.lora_layers:
                raise RuntimeError(
                    f"No LoRA layers were injected into '{backbone_name}' (tried {target_lists}). "
                    "Check the attention module names of the installed library version."
                )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the CLS / pooled embedding."""
        if self.is_smoke or "biomedclip" in self.backbone_name:
            return self.backbone(x)
        return self.backbone(pixel_values=x).last_hidden_state[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)


class FederatedClientModel(nn.Module):
    """Shared encoder (frozen backbone + LoRA) combined with a client-local classification head."""

    def __init__(
        self,
        encoder: VisionFoundationModel,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(
            in_features=encoder.embed_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=head_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
