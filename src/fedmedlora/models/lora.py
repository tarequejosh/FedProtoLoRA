"""Self-contained LoRA (Low-Rank Adaptation) implementation for PyTorch.

Provides:
- LoRALinear: linear layer wrapper with explicit low-rank factor access (A, B)
- Helpers to inject LoRA into Vision Transformers (DINOv2, BiomedCLIP)
- Factor getters, setters, rank resizing, and merged weight state management.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer wrapper that injects trainable low-rank decomposition matrices A and B.

    Mathematical formulation:
        h = W_0 x + (lora_alpha / r) * (B @ A) x
    where:
        W_0 in R^{out_features x in_features} is frozen.
        A in R^{r x in_features} is initialized via Gaussian or Kaiming.
        B in R^{out_features x r} is initialized to zeros.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.base_layer = base_layer

        # Freeze base linear weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = (lora_alpha / r) if r > 0 else 1.0

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.merged = False
        self.merge_weights = merge_weights

        # Allocate LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, self.in_features), dtype=self.base_layer.weight.dtype))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, r), dtype=self.base_layer.weight.dtype))
            self.reset_parameters()
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def reset_parameters(self):
        """Initializes A with standard Kaiming uniform and B to zeros."""
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def get_lora_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns detached copies of A and B matrices on CPU as floats."""
        if self.r == 0 or self.lora_A is None or self.lora_B is None:
            raise ValueError("LoRA rank is 0; no factors available.")
        return self.lora_A.data.clone(), self.lora_B.data.clone()

    def set_lora_factors(self, A: torch.Tensor, B: torch.Tensor):
        """Sets the internal A and B parameters, supporting dynamic rank adjustment."""
        target_r = A.shape[0]
        assert B.shape[1] == target_r, f"Rank mismatch between A ({A.shape}) and B ({B.shape})"
        assert A.shape[1] == self.in_features, f"Input dim mismatch: {A.shape[1]} vs {self.in_features}"
        assert B.shape[0] == self.out_features, f"Output dim mismatch: {B.shape[0]} vs {self.out_features}"

        if target_r != self.r:
            self.r = target_r
            self.scaling = (self.lora_alpha / target_r) if target_r > 0 else 1.0
            device = self.base_layer.weight.device
            dtype = self.base_layer.weight.dtype
            self.lora_A = nn.Parameter(A.to(device=device, dtype=dtype))
            self.lora_B = nn.Parameter(B.to(device=device, dtype=dtype))
        else:
            with torch.no_grad():
                self.lora_A.copy_(A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
                self.lora_B.copy_(B.to(device=self.lora_B.device, dtype=self.lora_B.dtype))

    def get_delta_weight(self) -> torch.Tensor:
        """Computes the full weight update dW = scaling * (B @ A)."""
        if self.r == 0 or self.lora_A is None or self.lora_B is None:
            return torch.zeros((self.out_features, self.in_features), device=self.base_layer.weight.device)
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self):
        """Fuses LoRA delta into the base layer weight for zero-overhead inference."""
        if not self.merged and self.r > 0:
            delta_w = self.get_delta_weight()
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Restores base layer weight by subtracting the fused LoRA delta."""
        if self.merged and self.r > 0:
            delta_w = self.get_delta_weight()
            self.base_layer.weight.data -= delta_w
            self.merged = False

    def add_to_base_weight(self, delta_w: torch.Tensor):
        """Permanently adds an external weight delta to the frozen base weight (used by FLoRA)."""
        with torch.no_grad():
            self.base_layer.weight.add_(delta_w.to(self.base_layer.weight.device, self.base_layer.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return self.base_layer(x)

        # Standard unmerged path
        base_out = self.base_layer(x)
        lora_in = self.lora_dropout(x)
        # x: [..., in_features], lora_A.t(): [in_features, r]
        # lora_out: [...] @ lora_A.t() @ lora_B.t()
        lora_out = (lora_in @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


def inject_lora_to_linear(
    module: nn.Module,
    target_names: List[str],
    r: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> Dict[str, LoRALinear]:
    """Recursively replaces targeted nn.Linear layers with LoRALinear modules.

    Args:
        module: The root PyTorch module (e.g. Vision Transformer).
        target_names: Substrings or exact attribute names (e.g. ['q_proj', 'v_proj', 'qkv']).
        r: Rank of the low-rank adaptation.
        lora_alpha: Scaling alpha parameter.
        lora_dropout: Dropout probability.

    Returns:
        Dictionary mapping layer names to the injected LoRALinear modules.
    """
    injected_layers: Dict[str, LoRALinear] = {}

    for name, child in module.named_children():
        # Check if this child matches target names and is an nn.Linear
        is_target = any(t in name for t in target_names)
        if isinstance(child, nn.Linear) and is_target and not isinstance(child, LoRALinear):
            lora_layer = LoRALinear(
                base_layer=child,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(module, name, lora_layer)
            injected_layers[name] = lora_layer
        else:
            sub_injected = inject_lora_to_linear(
                child, target_names, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            for sub_name, sub_layer in sub_injected.items():
                injected_layers[f"{name}.{sub_name}"] = sub_layer

    return injected_layers


def extract_lora_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extracts all LoRA parameters (A, B) and metadata from a model.

    Returns:
        dict: {layer_name: {'A': Tensor, 'B': Tensor, 'r': int, 'scaling': float}}
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.r > 0:
            A, B = module.get_lora_factors()
            lora_state[name] = {
                "A": A.cpu(),
                "B": B.cpu(),
                "r": module.r,
                "scaling": module.scaling,
            }
    return lora_state


def load_lora_state(model: nn.Module, lora_state: Dict[str, Dict[str, torch.Tensor]]):
    """Loads LoRA parameters (A, B) into the model's LoRALinear layers.

    An entry without "B" (FedSA-LoRA download) updates only A and keeps the local B.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and name in lora_state:
            A = lora_state[name]["A"]
            B = lora_state[name].get("B")
            if B is None:
                B = module.lora_B.data
            module.set_lora_factors(A, B)


def make_fresh_lora_state(
    template_state: Dict[str, Dict[str, torch.Tensor]],
    seed: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Builds a freshly initialised LoRA state (Kaiming-uniform A, zero B) with the template's shapes.

    Generated from an explicit seed so that every client receives an *identical* initialisation:
    all federated LoRA variants assume a common starting point (FFA-LoRA additionally keeps A frozen).
    """
    gen = torch.Generator().manual_seed(seed)
    fresh = {}
    for name, entry in template_state.items():
        r, in_features = entry["A"].shape
        out_features = entry["B"].shape[0]
        # Same bound as nn.init.kaiming_uniform_(a=sqrt(5)) on an (r, in_features) tensor.
        bound = 1.0 / math.sqrt(in_features)
        A = (torch.rand((r, in_features), generator=gen) * 2.0 - 1.0) * bound
        fresh[name] = {
            "A": A.to(entry["A"].dtype),
            "B": torch.zeros((out_features, r), dtype=entry["B"].dtype),
            "r": r,
            "scaling": entry["scaling"],
        }
    return fresh


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Counts trainable vs total parameters in the model.

    Returns:
        (trainable_params, total_params, percentage_trainable)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    return trainable, total, pct
