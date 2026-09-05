"""Minimal K-photograph patient-level model used by the LidMerge protocol.

There is deliberately no learned fusion block. One shared encoder emits one
logit per photograph. It is trained once with mean-logit patient BCE; mean,
top-2 mean, and max are then calculated from those same image logits at
inference as the photograph budget increases from K=1 to K=4.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VALID_AGGREGATIONS = {"mean", "top2_mean", "max"}
TRAINING_AGGREGATION = "mean"
# Pillow < 9.1 exposes resampling filters on Image directly rather than
# through Image.Resampling.  Keep the locked transform identical across the
# Wku runtime without requiring an environment/package change.
_RESAMPLING = getattr(Image, "Resampling", Image)
DEFAULT_AUGMENTATION = {
    "horizontal_flip_probability": 0.5,
    "rotation_degrees": 10.0,
    "brightness_jitter": 0.12,
    "contrast_jitter": 0.12,
}


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def image_tensor(image: Image.Image):
    """Apply the locked transform shared by every model and budget."""

    from torchvision.transforms import functional as transform

    image = ImageOps.pad(
        image.convert("RGB"),
        (IMAGE_SIZE, IMAGE_SIZE),
        method=_RESAMPLING.LANCZOS,
        color=(0, 0, 0),
    )
    return transform.normalize(transform.to_tensor(image), MEAN, STD)


def augment_image(
    image: Image.Image,
    rng: np.random.Generator,
    configuration: dict[str, float] | None = None,
) -> Image.Image:
    config = {**DEFAULT_AUGMENTATION, **(configuration or {})}
    output = image.convert("RGB")
    if rng.random() < float(config["horizontal_flip_probability"]):
        output = ImageOps.mirror(output)
    degrees = float(config["rotation_degrees"])
    if degrees > 0:
        output = output.rotate(
            float(rng.uniform(-degrees, degrees)),
            resample=_RESAMPLING.BILINEAR,
            fillcolor=(0, 0, 0),
        )
    brightness = float(config["brightness_jitter"])
    contrast = float(config["contrast_jitter"])
    output = ImageEnhance.Brightness(output).enhance(float(rng.uniform(1.0 - brightness, 1.0 + brightness)))
    return ImageEnhance.Contrast(output).enhance(float(rng.uniform(1.0 - contrast, 1.0 + contrast)))


def _open_rgb(path: str) -> Image.Image:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"LidMerge image is missing: {candidate}")
    with Image.open(candidate) as handle:
        return ImageOps.exif_transpose(handle).convert("RGB").copy()


class PatientBudgetDataset:
    """One dataset item per patient_group with a uniform K-photo tensor.

    During training, the epoch-level schedule fixes K for every group so a
    standard DataLoader can collate tensors without padding or count metadata.
    During evaluation, the frozen photograph order supplies the nested prefix
    for the requested K.  The model receives only pixels and never K itself.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        orders: pd.DataFrame,
        training: bool,
        seed: int,
        view_budget: int | None = None,
        training_budget_schedule: list[int] | None = None,
        augmentation: dict[str, float] | None = None,
    ):
        import torch
        from torch.utils.data import Dataset

        required_manifest = {"patient_group", "image_id", "absolute_path", "label"}
        required_orders = {"patient_group", "image_id", "label", "rank", "photograph_order_seed"}
        missing_manifest = required_manifest.difference(manifest.columns)
        missing_orders = required_orders.difference(orders.columns)
        if missing_manifest:
            raise ValueError(f"manifest frame is missing {sorted(missing_manifest)}")
        if missing_orders:
            raise ValueError(f"order frame is missing {sorted(missing_orders)}")
        if orders.empty:
            raise ValueError("no frozen photograph orders were supplied")
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        self.augmentation = {**DEFAULT_AUGMENTATION, **(augmentation or {})}
        self.training_budget_schedule = [int(value) for value in (training_budget_schedule or [])]
        self.view_budget = int(view_budget) if view_budget is not None else None
        if self.training and not self.training_budget_schedule:
            raise ValueError("training requires a non-empty K-budget schedule")
        if not self.training and (self.view_budget is None or self.view_budget < 1):
            raise ValueError("evaluation requires one positive frozen view budget")
        self._records_by_group: dict[str, dict[str, dict[str, Any]]] = {}
        self._labels: dict[str, int] = {}
        for group, current in manifest.groupby("patient_group", sort=True):
            labels = pd.to_numeric(current["label"], errors="raise").astype(int).unique()
            if len(labels) != 1:
                raise ValueError(f"mixed label patient_group: {group}")
            group_text = str(group)
            self._labels[group_text] = int(labels[0])
            self._records_by_group[group_text] = {
                str(row.image_id): {"image_id": str(row.image_id), "absolute_path": str(row.absolute_path)}
                for row in current[["image_id", "absolute_path"]].itertuples(index=False)
            }
        self._order_by_group: dict[str, list[dict[str, Any]]] = {}
        for group, current in orders.groupby("patient_group", sort=True):
            group_text = str(group)
            if group_text not in self._records_by_group:
                raise ValueError(f"frozen order includes a patient outside the supplied manifest: {group_text}")
            labels = pd.to_numeric(current["label"], errors="raise").astype(int).unique()
            if len(labels) != 1 or int(labels[0]) != self._labels[group_text]:
                raise ValueError(f"frozen order label mismatch for patient_group: {group_text}")
            ordered = current.sort_values("rank")
            ranks = ordered["rank"].astype(int).tolist()
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError(f"frozen photograph ranks are not contiguous for patient_group: {group_text}")
            records: list[dict[str, Any]] = []
            for image_id in ordered["image_id"].astype(str):
                try:
                    records.append(self._records_by_group[group_text][image_id])
                except KeyError as exc:
                    raise ValueError(f"frozen image does not belong to patient_group {group_text}: {image_id}") from exc
            if len({record["image_id"] for record in records}) != len(records):
                raise ValueError(f"frozen photograph order repeats an image for patient_group: {group_text}")
            self._order_by_group[group_text] = records
        self._groups = [(group, self._labels[group]) for group in sorted(self._order_by_group)]
        if not self._groups:
            raise ValueError("no patient groups are represented by the frozen order")
        max_available = min(len(self._order_by_group[group]) for group, _ in self._groups)
        requested = self.training_budget_schedule if self.training else [int(self.view_budget)]
        if max(requested) > max_available:
            raise ValueError("requested photograph budget exceeds frozen photographs for at least one patient_group")

        class _Dataset(Dataset):
            def __len__(_self):
                return len(self._groups)

            def __getitem__(_self, index):
                return self._item(int(index))

        self.dataset = _Dataset()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.training:
            self.view_budget = self.training_budget_schedule[(self.epoch - 1) % len(self.training_budget_schedule)]

    def _records_for(self, group: str) -> list[dict[str, Any]]:
        if self.view_budget is None:
            raise RuntimeError("view budget is not initialized")
        source = self._order_by_group[group]
        if self.training:
            rng = np.random.default_rng(_seed("training-budget", self.seed, self.epoch, group))
            chosen = rng.choice(len(source), size=int(self.view_budget), replace=False)
            return [source[int(index)] for index in chosen]
        return source[: int(self.view_budget)]

    def _item(self, index: int) -> dict[str, Any]:
        import torch

        group, label = self._groups[index]
        records = self._records_for(group)
        rng = np.random.default_rng(_seed("augmentation", self.seed, self.epoch, group, index))
        images = [_open_rgb(str(record["absolute_path"])) for record in records]
        if self.training:
            images = [augment_image(image, rng, self.augmentation) for image in images]
        return {
            "views": torch.stack([image_tensor(image) for image in images]),
            "label": torch.tensor(float(label), dtype=torch.float32),
            "patient_group": group,
        }


def _weights_for(backbone: str, pretrained: bool, allow_weight_download: bool):
    from torchvision import models
    import torch

    options = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
        "efficientnet_v2_m": models.EfficientNet_V2_M_Weights.DEFAULT,
        "swin_t": models.Swin_T_Weights.DEFAULT,
    }
    if backbone not in options:
        raise ValueError(f"unsupported torchvision backbone: {backbone}")
    if not pretrained:
        return None
    weights = options[backbone]
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint.is_file() and not allow_weight_download:
        raise FileNotFoundError(
            f"pretrained {backbone} weights are not cached at {checkpoint}; "
            "download explicitly before a formal run or pass --allow-weight-download"
        )
    return weights


class ImageEncoder:
    """Shared image encoder used identically by every K and aggregation route."""

    def __init__(self, backbone: str, embedding_dim: int, pretrained: bool, allow_weight_download: bool, dropout: float):
        import torch.nn as nn
        from torchvision import models

        self.backbone_name = str(backbone)
        weights = _weights_for(self.backbone_name, bool(pretrained), bool(allow_weight_download))
        self.channels_last = False
        if self.backbone_name in {"resnet18", "resnet50"}:
            base = getattr(models, self.backbone_name)(weights=weights)
            self.body = nn.Sequential(*list(base.children())[:-2])
            channels = int(base.fc.in_features)
        elif self.backbone_name == "convnext_tiny":
            base = models.convnext_tiny(weights=weights)
            self.body = base.features
            channels = int(base.classifier[-1].in_features)
        elif self.backbone_name in {"efficientnet_v2_s", "efficientnet_v2_m"}:
            base = getattr(models, self.backbone_name)(weights=weights)
            self.body = base.features
            channels = int(base.classifier[-1].in_features)
        elif self.backbone_name == "swin_t":
            base = models.swin_t(weights=weights)
            self.body = base.features
            channels = int(base.head.in_features)
            self.channels_last = True
        else:
            raise AssertionError(self.backbone_name)
        self.projection = nn.Sequential(
            nn.Linear(channels, int(embedding_dim)),
            nn.LayerNorm(int(embedding_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def as_module(self):
        import torch.nn as nn

        parent = self

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = parent.body
                self.projection = parent.projection
                self.channels_last = parent.channels_last

            def forward(self, images):
                feature_map = self.body(images)
                if feature_map.ndim != 4:
                    raise RuntimeError(f"backbone returned unexpected shape {tuple(feature_map.shape)}")
                pooled = feature_map.mean(dim=(1, 2)) if self.channels_last else feature_map.mean(dim=(-2, -1))
                return self.projection(pooled)

        return _Encoder()


def make_image_encoder(backbone: str, embedding_dim: int, pretrained: bool, allow_weight_download: bool, dropout: float):
    return ImageEncoder(backbone, embedding_dim, pretrained, allow_weight_download, dropout).as_module()


def aggregate_image_logits(image_logits, aggregation: str):
    """Apply a prespecified symmetric inference aggregation to image logits.

    This function deliberately has no parameters.  Its inputs are the exact
    per-photograph logits from the single mean-trained model, making mean/max
    comparisons an aggregation contrast rather than a separately trained-model
    comparison.
    """

    if str(aggregation) not in VALID_AGGREGATIONS:
        raise ValueError(f"unknown LidMerge aggregation: {aggregation}")
    if image_logits.ndim != 2 or image_logits.shape[1] < 1:
        raise ValueError(f"LidMerge aggregation requires [batch, K] image logits, got {tuple(image_logits.shape)}")
    if str(aggregation) == "mean":
        return image_logits.mean(dim=1)
    if str(aggregation) == "top2_mean":
        if image_logits.shape[1] < 2:
            return image_logits.mean(dim=1)
        return image_logits.topk(k=min(2, image_logits.shape[1]), dim=1).values.mean(dim=1)
    return image_logits.max(dim=1).values


def make_budget_model(
    backbone: str,
    embedding_dim: int,
    dropout: float,
    pretrained: bool,
    allow_weight_download: bool,
    encoder_override=None,
):
    """Create the deliberately simple symmetric K-photograph model."""

    import torch
    import torch.nn as nn

    class PhotographBudgetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.training_aggregation = TRAINING_AGGREGATION
            self.encoder = (
                encoder_override
                if encoder_override is not None
                else make_image_encoder(backbone, embedding_dim, pretrained, allow_weight_download, dropout)
            )
            self.image_head = nn.Linear(int(embedding_dim), 1)

        def _encode(self, images):
            encoded = self.encoder(images)
            return encoded[0] if isinstance(encoded, (tuple, list)) else encoded

        def forward(self, views):
            if views.ndim != 5 or views.shape[1] < 1:
                raise ValueError(f"LidMerge requires [batch, K, channels, height, width], got {tuple(views.shape)}")
            batch, budget = int(views.shape[0]), int(views.shape[1])
            embeddings = self._encode(views.reshape((-1, *views.shape[2:]))).reshape(batch, budget, -1)
            image_logits = self.image_head(embeddings.reshape(-1, embeddings.shape[-1])).reshape(batch, budget)
            return {
                "training_logit": aggregate_image_logits(image_logits, TRAINING_AGGREGATION),
                "image_logits": image_logits,
            }

    return PhotographBudgetModel()


def patient_bce(model, batch: dict[str, Any]):
    """The sole formal objective: one binary cross-entropy value per patient."""

    import torch.nn.functional as functional

    output = model(batch["views"])
    loss = functional.binary_cross_entropy_with_logits(output["training_logit"], batch["label"].float())
    return loss, output


def budget_training_step(
    model,
    batch: dict[str, Any],
    optimizer,
    scaler=None,
    autocast_enabled: bool = False,
    max_grad_norm: float | None = 1.0,
):
    """Run one patient-level K-photo diagnostic update."""

    import torch

    device = batch["views"].device
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=bool(autocast_enabled)):
        loss, _ = patient_bce(model, batch)
    if scaler is None:
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
    detached = loss.detach()
    return detached, {"patient_bce": detached}
