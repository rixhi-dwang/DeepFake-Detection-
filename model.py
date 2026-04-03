"""
ViT model helpers with feature extraction support.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torchvision import models


def _vit_get_features(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward input through ViT encoder and return CLS embedding.
    """
    x = self._process_input(x)
    batch_size = x.shape[0]
    cls_token = self.class_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = self.encoder(x)
    return x[:, 0]


def attach_get_features(model: nn.Module) -> nn.Module:
    """
    Attach model.get_features(x) without changing parameter names.
    """
    model.get_features = MethodType(_vit_get_features, model)  # type: ignore[attr-defined]
    return model


def build_vit_model(
    pretrained: bool = True,
    dropout: float = 0.0,
    num_classes: int = 2,
) -> nn.Module:
    """
    Build torchvision ViT-B/16 for binary classification and expose get_features().
    """
    weights = None
    if pretrained:
        try:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        except Exception:
            try:
                weights = models.ViT_B_16_Weights.DEFAULT
            except Exception:
                weights = None

    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features

    if dropout > 0:
        model.heads.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.heads.head = nn.Linear(in_features, num_classes)

    return attach_get_features(model)


def _extract_state_dict_and_meta(checkpoint_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(checkpoint_obj, dict) and "model_state" in checkpoint_obj:
        state_dict = checkpoint_obj["model_state"]
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint key 'model_state' is not a state_dict.")
        return state_dict, checkpoint_obj

    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint key 'state_dict' is not a state_dict.")
        return state_dict, checkpoint_obj

    if isinstance(checkpoint_obj, dict):
        tensor_values = [torch.is_tensor(value) for value in checkpoint_obj.values()]
        if tensor_values and all(tensor_values):
            return checkpoint_obj, {}

    raise ValueError("Unsupported checkpoint format. Expected {'model_state': ...} or plain state_dict.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _infer_dropout(meta: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> float:
    args = meta.get("args", {}) if isinstance(meta, dict) else {}
    if isinstance(args, dict) and "dropout" in args:
        try:
            return float(args["dropout"])
        except Exception:
            pass

    has_sequential_head = any(key.startswith("heads.head.1.") for key in state_dict.keys())
    return 0.2 if has_sequential_head else 0.0


def load_vit_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    num_classes: int = 2,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a trained ViT checkpoint and return model with get_features() enabled.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_obj = torch.load(str(checkpoint_path), map_location=device)
    state_dict, meta = _extract_state_dict_and_meta(checkpoint_obj)
    state_dict = _strip_module_prefix(state_dict)
    dropout = _infer_dropout(meta=meta, state_dict=state_dict)

    model = build_vit_model(pretrained=False, dropout=dropout, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, meta

