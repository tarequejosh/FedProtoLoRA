"""Personalized local classification heads for heterogeneous client label spaces."""

from typing import Optional
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Local classification head for a federated client.

    Consists of LayerNorm followed by a projection to the client's class space,
    with an optional intermediate MLP expansion.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(in_features))

        if hidden_dim is not None and hidden_dim > 0:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
