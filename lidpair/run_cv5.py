#!/usr/bin/env python3
"""Run one declared LidMerge selection, confirmation, or architecture route.

The launcher owns the matrix.  This module owns one patient-level route.  A
single K-agnostic image model is trained with the frozen 1--4 photograph
schedule, then writes its ordered per-image logits and derives one patient
probability for every fixed K.  No outer-test label participates in any
backbone, threshold, or budget choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lidpair.model import (
    TRAINING_AGGREGATION,
    VALID_AGGREGATIONS,
    PatientBudgetDataset,
    aggregate_image_logits,
    budget_training_step,
    make_budget_model,
)
from lidpair.runtime import resolve_runtime_manifest
from lidpair.validate import resolve_locked_manifest_path, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT_MARKER = ROOT / "SOURCE_COMMIT.txt"
CONTRACT_PATH = Path(__file__).with_name("contract.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _code_fingerprint() -> str:
    paths = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("runtime.py")]
    payload = "\n".join(f"{path.name}:{_sha256(path)}" for path in paths)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        if SOURCE_COMMIT_MARKER.is_file():
            value = SOURCE_COMMIT_MARKER.read_text(encoding="utf-8").strip()
            if value:
                return value
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except Exception:
        # Wku executes a deliberately frozen bundle without a .git directory.
        # SOURCE_COMMIT.txt is written only after the exact committed files are
        # transferred; treating that bundle as clean preserves the provenance
        # firewall without silently accepting an unknown source.
        return False if SOURCE_COMMIT_MARKER.is_file() else None


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("LidMerge formal routes require CUDA; refusing a silent CPU fallback")
    return torch.device("cuda")


def _make_grad_scaler(torch, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _training_policy(config: dict) -> dict[str, object]:
    """Return and validate the locked complete-K-cycle training policy."""

    training = config["training"]
    schedule = tuple(int(value) for value in config["views"]["training_budget_schedule"])
    cycle_length = int(training.get("cycle_length", len(schedule)))
    early_stopping = training.get("early_stopping", {})
    minimum_cycles = int(early_stopping.get("minimum_cycles", 3))
    minimum_epochs = int(early_stopping.get("minimum_epochs", minimum_cycles * cycle_length))
    maximum_epochs = int(training["diagnostic_epochs"])
    if schedule != (1, 2, 3, 4) or cycle_length != 4:
        raise ValueError("LidMerge requires the locked training schedule K=1,2,3,4")
    if minimum_cycles < 3:
        raise ValueError("LidMerge requires at least three complete K cycles")
    if minimum_epochs != minimum_cycles * cycle_length:
        raise ValueError("minimum_epochs must equal minimum_cycles * cycle_length")
    if maximum_epochs < minimum_epochs or maximum_epochs % cycle_length != 0:
        raise ValueError("diagnostic_epochs must be a complete-K-cycle count")
    return {
        "schedule": schedule,
        "cycle_length": cycle_length,
        "minimum_cycles": minimum_cycles,
        "minimum_epochs": minimum_epochs,
        "maximum_epochs": maximum_epochs,
        "patience_cycles": int(early_stopping.get("patience_cycles", 3)),
        "min_delta": float(early_stopping.get("min_delta", 0.0)),
    }


def _validate_training_epochs(config: dict, epochs: int) -> None:
    """Reject an outer/robustness epoch lock that violates the cycle contract."""

    policy = _training_policy(config)
    value = int(epochs)
    if value < int(policy["minimum_epochs"]) or value > int(policy["maximum_epochs"]):
        raise ValueError(
            f"training_epochs must be in [{policy['minimum_epochs']}, {policy['maximum_epochs']}], got {value}"
        )
    if value % int(policy["cycle_length"]) != 0:
        raise ValueError(f"training_epochs must be a multiple of the K-cycle length {policy['cycle_length']}")


def _checkpoint_eligible(config: dict, epoch: int) -> bool:
    """Return whether an epoch may define the selected diagnostic checkpoint."""

    policy = _training_policy(config)
    value = int(epoch)
    return value >= int(policy["minimum_epochs"]) and value % int(policy["cycle_length"]) == 0


def _runtime_provenance(torch, device) -> dict[str, object]:
    import torchvision

    return {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "cuda_runtime_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.version() is not None else None,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
    }


def _artifact_record(path: Path, run_dir: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"declared run artifact is missing: {path}")
    return {"run_relative_path": path.relative_to(run_dir).as_posix(), "sha256": _sha256(path), "size_bytes": int(path.stat().st_size)}


def _required_run_artifacts(stage: str) -> set[str]:
    # image_logits.csv is the audit trail proving that mean and max use exactly
    # the same shared image-model outputs rather than separately trained routes.
    required = {"diagnostic_state.pt", "diagnostic_history.csv", "image_logits.csv", "predictions.csv"}
    if stage == "outer":
        required.update({"counterfactual_predictions.csv", "alternate_order_predictions.csv"})
    return required


def _resume_completed_run(run_dir: Path, plan: dict[str, object]) -> dict[str, object]:
    """Fail closed unless an existing route is exactly reusable on this commit.

    A resumed route must have been produced by the current code fingerprint,
    current Git commit, the same locked protocol, and intact artifact hashes.
    It is deliberately not a best-effort cache: a changed route must be
    explicitly recomputed under a new clean commit.
    """

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"cannot resume an incomplete route without run_manifest.json: {run_dir}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"cannot resume a non-complete route: {run_dir}")
    immutable_plan_keys = (
        "stage",
        "outer_fold",
        "inner_fold",
        "seed",
        "backbone",
        "training_aggregation",
        "training_route_name",
        "inference_aggregations",
        "budgets",
        "protocol_manifest_sha256",
        "contract_sha256",
        "output_directory",
        "outer_selection",
        "route_family",
        "training_epochs",
    )
    mismatch = [key for key in immutable_plan_keys if manifest.get(key) != plan.get(key)]
    if mismatch:
        raise RuntimeError(f"cannot resume route with incompatible plan fields {mismatch}: {run_dir}")
    if manifest.get("git_commit") != _git_commit() or manifest.get("code_fingerprint") != _code_fingerprint():
        raise RuntimeError(f"cannot resume route produced by a different code commit or fingerprint: {run_dir}")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise RuntimeError(f"cannot resume route without an artifact manifest: {run_dir}")
    observed: set[str] = set()
    resolved_root = run_dir.resolve()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"cannot resume route with malformed artifact record: {run_dir}")
        relative = Path(str(record.get("run_relative_path", "")))
        if relative.is_absolute() or not str(relative):
            raise RuntimeError(f"cannot resume route with unsafe artifact path: {run_dir}")
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError(f"cannot resume route with escaping artifact path: {run_dir}") from exc
        if not candidate.is_file():
            raise RuntimeError(f"cannot resume route with missing artifact {relative}: {run_dir}")
        if int(candidate.stat().st_size) != int(record.get("size_bytes", -1)) or _sha256(candidate) != record.get("sha256"):
            raise RuntimeError(f"cannot resume route with modified artifact {relative}: {run_dir}")
        observed.add(relative.as_posix())
    required = _required_run_artifacts(str(plan["stage"]))
    if not required.issubset(observed):
        raise RuntimeError(f"cannot resume route missing declared artifacts {sorted(required.difference(observed))}: {run_dir}")
    return manifest


def _load_protocol(protocol_root: Path, image_root: Path | None, check_paths: bool):
    validate_protocol(protocol_root, image_root=image_root, check_paths=check_paths)
    lock = _load_json(protocol_root / "protocol_lock.json")
    contract = lock["contract"]
    if lock.get("contract_sha256") != _sha256(CONTRACT_PATH):
        raise RuntimeError("LidMerge contract changed after protocol locking; rebuild the locked protocol explicitly")
    manifest_path = resolve_locked_manifest_path(lock, protocol_root)
    if _sha256(manifest_path) != lock.get("manifest_sha256"):
        raise RuntimeError("frozen manifest is missing or differs from protocol_lock.json")
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str, "patient_group": str})
    manifest = resolve_runtime_manifest(manifest, image_root=image_root, check_paths=False)
    primary_orders = pd.read_csv(protocol_root / str(lock["primary_order_file"]), dtype={"image_id": str, "patient_group": str})
    alternate_orders = {
        int(seed): pd.read_csv(protocol_root / str(filename), dtype={"image_id": str, "patient_group": str})
        for seed, filename in lock.get("alternate_order_files", {}).items()
    }
    outer = pd.read_csv(protocol_root / "outer_folds.csv", dtype={"patient_group": str})
    inner = pd.read_csv(protocol_root / "inner_folds.csv", dtype={"patient_group": str})
    return lock, contract, manifest, primary_orders, alternate_orders, outer, inner


def _split(
    manifest: pd.DataFrame,
    orders: pd.DataFrame,
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    stage: str,
    outer_fold: int,
    inner_fold: int | None,
):
    all_groups = set(orders["patient_group"].astype(str))
    test_groups = set(outer.loc[outer["outer_fold"].astype(int).eq(int(outer_fold)), "patient_group"].astype(str))
    if stage == "inner":
        if inner_fold is None:
            raise ValueError("--inner-fold is required for --stage inner")
        current = inner[inner["protocol_outer_fold"].astype(int).eq(int(outer_fold))]
        evaluation_groups = set(current.loc[current["inner_fold"].astype(int).eq(int(inner_fold)), "patient_group"].astype(str))
        train_groups = all_groups - test_groups - evaluation_groups
    elif stage in {"outer", "architecture_robustness"}:
        train_groups, evaluation_groups = all_groups - test_groups, test_groups
    else:
        raise ValueError(stage)
    if not train_groups or not evaluation_groups or train_groups & evaluation_groups or test_groups & train_groups:
        raise RuntimeError("invalid patient_group train/evaluation partition")
    train_frame = manifest[manifest["patient_group"].astype(str).isin(train_groups)].copy()
    evaluation_frame = manifest[manifest["patient_group"].astype(str).isin(evaluation_groups)].copy()
    train_orders = orders[orders["patient_group"].astype(str).isin(train_groups)].copy()
    evaluation_orders = orders[orders["patient_group"].astype(str).isin(evaluation_groups)].copy()
    if evaluation_orders["patient_group"].nunique() != len(evaluation_groups):
        raise RuntimeError("frozen photograph orders do not cover every evaluation patient_group")
    return train_frame, evaluation_frame, train_orders, evaluation_orders, sorted(train_groups), sorted(evaluation_groups)


def _loader(dataset: PatientBudgetDataset, batch_size: int, training: bool):
    import torch
    from torch.utils.data import DataLoader

    if not training:
        return DataLoader(dataset.dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    generator = torch.Generator()
    generator.manual_seed(int(dataset.seed))
    return DataLoader(dataset.dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)


def _validation_patient_bce_by_budget(
    model,
    evaluation_frame: pd.DataFrame,
    evaluation_orders: pd.DataFrame,
    seed: int,
    device,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate every locked K budget on the inner-validation patients.

    This is deliberately separate from outer-test prediction.  The early-stop
    signal is available only for inner routes; outer routes receive a locked
    epoch count aggregated from inner routes and never inspect outer labels.
    The returned values are used only at complete K cycles, so no checkpoint
    can be selected after seeing only the K=1 update of a cycle.
    """

    import torch

    values: dict[str, float] = {}
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    for budget in (1, 2, 3, 4):
        dataset = PatientBudgetDataset(
            evaluation_frame,
            evaluation_orders,
            training=False,
            seed=seed,
            view_budget=budget,
        )
        loader = _loader(dataset, batch_size, training=False)
        total_loss = 0.0
        total_patients = 0
        with torch.no_grad():
            for batch in loader:
                views = batch["views"].to(device)
                labels = batch["label"].to(device)
                output = model(views)
                logits = output["training_logit"]
                total_loss += float(criterion(logits, labels).detach().cpu())
                total_patients += int(labels.numel())
        if total_patients == 0:
            raise RuntimeError("inner-validation early stopping received no patient_group rows")
        values[str(budget)] = total_loss / total_patients
    return values


def _fit_diagnostic(
    train_frame: pd.DataFrame,
    train_orders: pd.DataFrame,
    config: dict,
    backbone: str,
    seed: int,
    device,
    pretrained: bool,
    allow_download: bool,
    output_dir: Path,
    evaluation_frame: pd.DataFrame | None = None,
    evaluation_orders: pd.DataFrame | None = None,
    training_epochs: int | None = None,
):
    import torch

    model = make_budget_model(
        backbone=backbone,
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"]["dropout"]),
        pretrained=pretrained,
        allow_weight_download=allow_download,
    ).to(device)
    policy = _training_policy(config)
    dataset = PatientBudgetDataset(
        train_frame,
        train_orders,
        training=True,
        seed=seed,
        training_budget_schedule=list(policy["schedule"]),
        augmentation=config["training"].get("augmentation"),
    )
    loader = _loader(dataset, int(config["training"]["batch_size"]), training=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    amp_enabled = bool(config["training"].get("mixed_precision", False) and device.type == "cuda")
    scaler = _make_grad_scaler(torch, amp_enabled)
    torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, float]] = []
    maximum_epochs = int(policy["maximum_epochs"])
    cycle_length = int(policy["cycle_length"])
    minimum_cycles = int(policy["minimum_cycles"])
    minimum_epochs = int(policy["minimum_epochs"])
    patience_cycles = int(policy["patience_cycles"])
    min_delta = float(policy["min_delta"])
    early_enabled = evaluation_frame is not None and evaluation_orders is not None
    if patience_cycles < 1:
        raise ValueError("early-stopping patience_cycles must be positive")
    requested_epochs = int(training_epochs) if training_epochs is not None else maximum_epochs
    _validate_training_epochs(config, requested_epochs)
    best_validation = float("inf")
    best_epoch = 0
    best_validation_by_budget: dict[str, float] | None = None
    cycles_without_improvement = 0
    best_state: dict[str, object] | None = None
    for epoch in range(1, requested_epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        losses: list[float] = []
        for batch in loader:
            moved = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}
            loss, _ = budget_training_step(model, moved, optimizer, scaler=scaler, autocast_enabled=amp_enabled, max_grad_norm=1.0)
            losses.append(float(loss.detach().cpu()))
        validation_by_budget: dict[str, float] | None = None
        validation_bce = None
        cycle_complete = epoch % cycle_length == 0
        if early_enabled and cycle_complete:
            validation_by_budget = _validation_patient_bce_by_budget(
                model,
                evaluation_frame,
                evaluation_orders,
                seed,
                device,
                int(config["training"]["batch_size"]),
            )
            validation_bce = float(np.mean(list(validation_by_budget.values())))
            # Do not allow the first two cycles to define a selected checkpoint.
            # They are trained for warm-up only; the locked minimum is three
            # complete K=1..4 cycles.
            checkpoint_eligible = _checkpoint_eligible(config, epoch)
            if checkpoint_eligible and validation_bce < best_validation - min_delta:
                best_validation = validation_bce
                best_epoch = int(epoch)
                best_validation_by_budget = dict(validation_by_budget)
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                cycles_without_improvement = 0
            elif checkpoint_eligible:
                cycles_without_improvement += 1
        history.append(
            {
                "epoch": int(epoch),
                "photograph_budget": int(dataset.view_budget),
                "patient_bce": float(np.mean(losses)) if losses else float("nan"),
                "validation_patient_bce_k1_mean": float(validation_by_budget["1"]) if validation_by_budget is not None else float("nan"),
                "validation_patient_bce_k2_mean": float(validation_by_budget["2"]) if validation_by_budget is not None else float("nan"),
                "validation_patient_bce_k3_mean": float(validation_by_budget["3"]) if validation_by_budget is not None else float("nan"),
                "validation_patient_bce_k4_mean": float(validation_by_budget["4"]) if validation_by_budget is not None else float("nan"),
                "validation_patient_bce_k1_k4_mean": float(validation_bce) if validation_bce is not None else float("nan"),
                "checkpoint_cycle": bool(early_enabled and cycle_complete),
            }
        )
        if early_enabled and epoch >= minimum_epochs and epoch % cycle_length == 0 and cycles_without_improvement >= patience_cycles:
            break
    stopped_epoch = len(history)
    if early_enabled:
        if best_state is None or best_epoch < 1:
            raise RuntimeError("inner-validation early stopping never recorded a valid checkpoint")
        if best_epoch < minimum_epochs or best_epoch % cycle_length != 0:
            raise RuntimeError("selected checkpoint does not satisfy the complete-K-cycle minimum")
        model.load_state_dict(best_state)
    else:
        best_epoch = stopped_epoch
    _validate_training_epochs(config, best_epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "diagnostic_state.pt"
    history_path = output_dir / "diagnostic_history.csv"
    torch.save(model.state_dict(), state_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    memory = {
        "mixed_precision": amp_enabled,
        "one_k_view_forward_backward": True,
        "training_epochs_max": maximum_epochs,
        "training_epochs_requested": requested_epochs,
        "training_epochs_ran": stopped_epoch,
        "training_epochs_realized": int(best_epoch),
        "training_cycles_realized": int(best_epoch // cycle_length),
        "cycle_length": cycle_length,
        "early_stopping_enabled": bool(early_enabled),
        "early_stopped": bool(early_enabled and stopped_epoch < requested_epochs),
        "early_stopping_monitor": "inner_validation_patient_bce_k1_k4_mean" if early_enabled else None,
        "early_stopping_minimum_cycles": minimum_cycles if early_enabled else None,
        "early_stopping_minimum_epochs": minimum_epochs if early_enabled else None,
        "early_stopping_patience_cycles": patience_cycles if early_enabled else None,
        "early_stopping_min_delta": min_delta if early_enabled else None,
        "best_validation_patient_bce_k1_k4_mean": float(best_validation) if early_enabled else None,
        "best_validation_by_budget": best_validation_by_budget if early_enabled else None,
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return model, memory, state_path, history_path


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable NumPy sigmoid used only to audit saved logits."""

    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _validate_shared_logit_prediction_trace(
    predictions: pd.DataFrame,
    image_logits: pd.DataFrame,
    budgets: list[int],
    aggregations: list[str],
) -> None:
    """Fail closed unless every reported probability rebuilds from saved logits.

    The raw trace contains exactly one four-image prefix per patient_group.  All
    K=1..4 mean, top-2 mean, and max probabilities must be deterministic aggregates of that
    trace; no separately optimized max model is allowed.
    """

    normalized_budgets = sorted({int(value) for value in budgets})
    if not normalized_budgets or normalized_budgets != list(range(1, max(normalized_budgets) + 1)):
        raise ValueError("shared-logit prediction trace requires contiguous prefix budgets beginning at K=1")
    maximum_budget = max(normalized_budgets)
    required_predictions = {"patient_group", "label", "budget", "variant", "probability", "training_aggregation"}
    required_logits = {"patient_group", "label", "view_rank", "image_logit", "training_aggregation"}
    if missing := required_predictions.difference(predictions.columns):
        raise ValueError(f"prediction artifact lacks shared-logit columns: {sorted(missing)}")
    if missing := required_logits.difference(image_logits.columns):
        raise ValueError(f"image-logit artifact lacks required columns: {sorted(missing)}")
    if not predictions["training_aggregation"].astype(str).eq(TRAINING_AGGREGATION).all():
        raise ValueError("prediction artifact does not identify the shared mean-trained model")
    if not image_logits["training_aggregation"].astype(str).eq(TRAINING_AGGREGATION).all():
        raise ValueError("image-logit artifact does not identify the shared mean-trained model")
    trace = image_logits.copy()
    trace["patient_group"] = trace["patient_group"].astype(str)
    trace["label"] = pd.to_numeric(trace["label"], errors="raise").astype(int)
    trace["view_rank"] = pd.to_numeric(trace["view_rank"], errors="raise").astype(int)
    trace["image_logit"] = pd.to_numeric(trace["image_logit"], errors="raise")
    if trace.duplicated(["patient_group", "view_rank"]).any() or trace["image_logit"].isna().any():
        raise RuntimeError("image-logit artifact has duplicate or missing patient views")
    ranks = trace.groupby("patient_group")["view_rank"].apply(lambda values: sorted(values.tolist()))
    if not ranks.apply(lambda values: values == list(range(1, maximum_budget + 1))).all():
        raise RuntimeError("image-logit artifact does not contain one ordered K=1..K=max prefix per patient_group")
    rows: list[dict[str, object]] = []
    for (patient_group, label), current in trace.groupby(["patient_group", "label"], sort=True):
        logits = current.sort_values("view_rank")["image_logit"].to_numpy(dtype=np.float64)
        for budget in normalized_budgets:
            prefix = logits[:budget]
            for aggregation in aggregations:
                if aggregation == "mean":
                    logit = float(prefix.mean())
                elif aggregation == "max":
                    logit = float(prefix.max())
                else:
                    raise ValueError(f"unsupported shared-logit aggregation: {aggregation}")
                rows.append({
                    "patient_group": str(patient_group),
                    "label": int(label),
                    "budget": int(budget),
                    "variant": str(aggregation),
                    "reconstructed_probability": float(_stable_sigmoid(np.asarray([logit]))[0]),
                })
    reconstructed = pd.DataFrame(rows)
    observed = predictions[["patient_group", "label", "budget", "variant", "probability"]].copy()
    observed["patient_group"] = observed["patient_group"].astype(str)
    observed["label"] = pd.to_numeric(observed["label"], errors="raise").astype(int)
    observed["budget"] = pd.to_numeric(observed["budget"], errors="raise").astype(int)
    observed["probability"] = pd.to_numeric(observed["probability"], errors="raise")
    merged = observed.merge(reconstructed, on=["patient_group", "label", "budget", "variant"], validate="one_to_one")
    expected_rows = trace["patient_group"].nunique() * len(normalized_budgets) * len(aggregations)
    if len(merged) != expected_rows:
        raise RuntimeError("image-logit reconstruction has missing patient-level budget predictions")
    if not np.allclose(
        merged["probability"].to_numpy(dtype=np.float64),
        merged["reconstructed_probability"].to_numpy(dtype=np.float64),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("reported mean/top2_mean/max probabilities do not reconstruct from the saved shared image logits")


def _predict_all_budgets(
    model,
    frame: pd.DataFrame,
    orders: pd.DataFrame,
    budgets: list[int],
    seed: int,
    device,
    batch_size: int,
    aggregations: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every K from one saved K=max ordered image-logit trace."""

    import torch

    normalized_budgets = sorted({int(value) for value in budgets})
    if not normalized_budgets or normalized_budgets != list(range(1, max(normalized_budgets) + 1)):
        raise ValueError("LidMerge prediction requires contiguous K=1..K=max budgets")
    maximum_budget = max(normalized_budgets)
    dataset = PatientBudgetDataset(frame, orders, training=False, seed=seed, view_budget=maximum_budget)
    loader = _loader(dataset, batch_size, training=False)
    order_seed_values = orders["photograph_order_seed"].astype(int).unique()
    if len(order_seed_values) != 1:
        raise ValueError("prediction requires exactly one frozen photograph-order seed")
    prediction_rows: list[dict[str, object]] = []
    logit_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch["views"].to(device))
            raw_logits = output["image_logits"]
            for index, patient_group in enumerate(batch["patient_group"]):
                label = int(float(batch["label"][index]))
                for view_rank, image_logit in enumerate(raw_logits[index].detach().cpu().tolist(), start=1):
                    logit_rows.append({
                        "patient_group": str(patient_group),
                        "label": label,
                        "maximum_budget": int(maximum_budget),
                        "view_rank": int(view_rank),
                        "photograph_order_seed": int(order_seed_values[0]),
                        "training_aggregation": TRAINING_AGGREGATION,
                        "image_logit": float(image_logit),
                    })
            for view_budget in normalized_budgets:
                prefix_logits = raw_logits[:, :view_budget]
                for aggregation in aggregations:
                    probability = torch.sigmoid(aggregate_image_logits(prefix_logits, aggregation)).detach().cpu().numpy()
                    for index, value in enumerate(probability):
                        prediction_rows.append({
                            "patient_group": str(batch["patient_group"][index]),
                            "label": int(float(batch["label"][index])),
                            "budget": int(view_budget),
                            "photograph_order_seed": int(order_seed_values[0]),
                            "variant": str(aggregation),
                            "training_aggregation": TRAINING_AGGREGATION,
                            "probability": float(value),
                        })
    result = pd.DataFrame(prediction_rows).sort_values(["variant", "budget", "patient_group"]).reset_index(drop=True)
    trace = pd.DataFrame(logit_rows).sort_values(["patient_group", "view_rank"]).reset_index(drop=True)
    expected_groups = orders["patient_group"].nunique()
    if result.duplicated(["patient_group", "budget", "variant"]).any() or len(result) != expected_groups * len(normalized_budgets) * len(aggregations):
        raise RuntimeError("all-budget prediction output has duplicate or missing patient groups")
    if trace.duplicated(["patient_group", "view_rank"]).any() or len(trace) != expected_groups * maximum_budget:
        raise RuntimeError("raw image-logit trace has duplicate or missing patient views")
    _validate_shared_logit_prediction_trace(result, trace, normalized_budgets, aggregations)
    return result, trace


def _predict_counterfactuals(
    model,
    frame: pd.DataFrame,
    orders: pd.DataFrame,
    budgets: list[int],
    seed: int,
    device,
    batch_size: int,
    lowpass_kernel_size: int,
    aggregations: list[str],
) -> pd.DataFrame:
    """Emit prespecified implementation and one-photo robustness controls."""

    import torch
    import torch.nn.functional as functional

    kernel_size = int(lowpass_kernel_size)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("counterfactual low-pass kernel size must be an odd integer >= 3")
    rows: list[dict[str, object]] = []
    order_seed_values = orders["photograph_order_seed"].astype(int).unique()
    if len(order_seed_values) != 1:
        raise ValueError("counterfactual prediction requires exactly one order seed")
    model.eval()
    with torch.no_grad():
        for budget in budgets:
            dataset = PatientBudgetDataset(frame, orders, training=False, seed=seed, view_budget=int(budget))
            for batch in _loader(dataset, batch_size, training=False):
                views = batch["views"].to(device)
                lowpass = views.clone()
                lowpass[:, 0] = functional.avg_pool2d(
                    views[:, 0],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    count_include_pad=False,
                )
                conditions = {
                    "true_prefix": views,
                    "reversed_prefix": views.flip(1),
                    "duplicate_view1": views[:, :1].repeat(1, int(budget), 1, 1, 1),
                    "lowpass_view1": lowpass,
                }
                for condition, values in conditions.items():
                    output = model(values)
                    for aggregation in aggregations:
                        probability = torch.sigmoid(aggregate_image_logits(output["image_logits"], aggregation)).detach().cpu().numpy()
                        for index, value in enumerate(probability):
                            rows.append({
                                "patient_group": str(batch["patient_group"][index]),
                                "label": int(float(batch["label"][index])),
                                "budget": int(budget),
                                "photograph_order_seed": int(order_seed_values[0]),
                                "condition": condition,
                                "variant": str(aggregation),
                                "training_aggregation": TRAINING_AGGREGATION,
                                "probability": float(value),
                            })
    result = pd.DataFrame(rows).sort_values(["variant", "budget", "condition", "patient_group"]).reset_index(drop=True)
    expected = len(orders["patient_group"].unique())
    counts = result.groupby(["variant", "budget", "condition"])["patient_group"].agg(["size", "nunique"])
    if not counts["size"].eq(expected).all() or not counts["nunique"].eq(expected).all():
        raise RuntimeError("counterfactual predictions contain duplicate or missing patient groups")
    return result


def _predict_alternate_orders(model, frame: pd.DataFrame, alternate_orders: dict[int, pd.DataFrame], evaluation_groups: list[str], budgets: list[int], seed: int, device, batch_size: int, aggregations: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    expected = set(evaluation_groups)
    for order_seed, all_orders in sorted(alternate_orders.items()):
        orders = all_orders[all_orders["patient_group"].astype(str).isin(expected)].copy()
        if set(orders["patient_group"].astype(str)) != expected:
            raise RuntimeError(f"alternate order draw {order_seed} does not cover the held-out patient groups")
        prediction, _ = _predict_all_budgets(model, frame, orders, budgets, seed, device, batch_size, aggregations)
        rows.append(prediction)
    if not rows:
        return pd.DataFrame(columns=["patient_group", "label", "budget", "photograph_order_seed", "variant", "training_aggregation", "probability"])
    result = pd.concat(rows, ignore_index=True).sort_values(["variant", "photograph_order_seed", "budget", "patient_group"]).reset_index(drop=True)
    if result.duplicated(["patient_group", "budget", "photograph_order_seed", "variant"]).any():
        raise RuntimeError("alternate-order predictions contain duplicate patient groups")
    return result


def _run_directory(output_root: Path, stage: str, outer_fold: int, inner_fold: int | None, seed: int, backbone: str, training_route_name: str) -> Path:
    pieces: list[Path | str] = [output_root, stage, f"outer_{outer_fold}"]
    if inner_fold is not None:
        pieces.append(f"inner_{inner_fold}")
    pieces.extend([f"seed_{seed}", f"{backbone}__{training_route_name}"])
    return Path(*pieces)


def _selection_for_outer(path: Path, lock: dict, outer_fold: int, budgets: list[int]) -> dict:
    selection = _load_json(path)
    if selection.get("selection_kind") != "variant_budget_thresholds":
        raise ValueError("outer confirmation requires a LidMerge variant-budget threshold selection")
    if int(selection.get("outer_fold", -1)) != int(outer_fold) or str(selection.get("variant")) != TRAINING_AGGREGATION:
        raise ValueError("outer confirmation must use the mean-aggregation threshold lock only to identify its selected backbone")
    if selection.get("protocol_manifest_sha256") != lock["manifest_sha256"] or selection.get("contract_sha256") != lock["contract_sha256"]:
        raise ValueError("outer selection targets a different locked protocol")
    thresholds = selection.get("budget_thresholds", {})
    if set(map(int, thresholds)) != set(map(int, budgets)):
        raise ValueError("outer selection lacks a validation-only threshold for every declared K")
    for budget in budgets:
        value = thresholds[str(int(budget))]
        if not 0.0 <= float(value["threshold"]) <= 1.0:
            raise ValueError(f"outer selection has an invalid threshold for K={budget}")
    return selection


def _validate_cuda_preflight(path: Path, lock: dict) -> dict:
    report = _load_json(path)
    if report.get("status") != "passed" or not str(report.get("device", "")).startswith("cuda"):
        raise RuntimeError("formal LidMerge routes require a passed CUDA preflight report")
    if report.get("git_commit") != _git_commit():
        raise RuntimeError("CUDA preflight report was produced from a different git commit")
    if report.get("contract_sha256") != lock["contract_sha256"] or report.get("manifest_sha256") != lock["manifest_sha256"]:
        raise RuntimeError("CUDA preflight report targets a different locked protocol")
    if not report.get("production_backward"):
        raise RuntimeError("CUDA preflight report lacks a production K=4 backward check")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--image-root", help="mounted canonical image root; operational pixel-path resolution only")
    parser.add_argument("--stage", choices=("inner", "outer", "architecture_robustness"), required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", help="required for inner and architecture-robustness routes; optional consistency check for a selected outer route")
    parser.add_argument("--selection", help="required mean-aggregation validation-only threshold JSON for an outer route; it supplies the selected backbone only")
    parser.add_argument("--cuda-preflight-report", help="required passed target-GPU report for a non-dry formal route")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="use declared ImageNet torchvision initialization")
    parser.add_argument("--allow-weight-download", action="store_true", help="explicitly permit official torchvision weight download")
    parser.add_argument("--training-epochs", type=int, help="outer/robustness epoch count locked by inner validation")
    parser.add_argument("--resume", action="store_true", help="reuse only a completed route with matching code, protocol, and artifact hashes")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.resume and args.force:
        raise ValueError("--resume and --force are mutually exclusive")
    protocol_root = Path(args.protocol_root).resolve()
    output_root = Path(args.output_root).resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else None
    lock, contract, manifest, primary_orders, alternate_orders, outer, inner = _load_protocol(protocol_root, image_root=image_root, check_paths=not args.dry_run)
    n_outer = int(contract["validation"]["outer_folds"])
    if int(args.outer_fold) not in range(n_outer):
        raise ValueError("outer fold is outside the locked range")
    if args.stage == "inner" and args.inner_fold not in range(int(contract["validation"]["inner_folds"])):
        raise ValueError("inner route requires a declared --inner-fold")
    if args.stage != "inner" and args.inner_fold is not None:
        raise ValueError("only inner routes may declare --inner-fold")
    budgets = [int(value) for value in contract["views"]["budgets"]]
    aggregation = contract["aggregation"]
    if str(aggregation["training"]) != TRAINING_AGGREGATION:
        raise ValueError("LidMerge implementation supports only the locked mean-logit training aggregation")
    inference_aggregations = [str(value) for value in aggregation["inference"]]
    if not inference_aggregations or set(inference_aggregations) - VALID_AGGREGATIONS:
        raise ValueError("contract declares unsupported inference aggregations")
    training_route_name = str(aggregation["training_route_name"])
    selection = None
    if args.stage == "inner":
        if not args.backbone:
            raise ValueError("inner route requires --backbone")
        backbone = str(args.backbone)
        if backbone not in {str(value) for value in contract["backbone_screen"]}:
            raise ValueError("inner route requested a backbone outside the locked screen")
    elif args.stage == "outer":
        if not args.selection:
            raise ValueError("outer confirmation requires --selection from inner validation only")
        selection = _selection_for_outer(Path(args.selection).resolve(), lock, int(args.outer_fold), budgets)
        backbone = str(selection["backbone"])
        if args.backbone and str(args.backbone) != backbone:
            raise ValueError("requested outer backbone disagrees with the locked inner selection")
    else:
        robustness = contract["architecture_robustness"]
        if args.selection:
            raise ValueError("architecture robustness is fixed and must not receive an inner selection file")
        if not args.backbone:
            raise ValueError("architecture robustness requires one predeclared --backbone")
        backbone = str(args.backbone)
        if backbone not in {str(value) for value in robustness["backbones"]}:
            raise ValueError("architecture robustness requested a backbone outside its locked set")
        if str(robustness["aggregation"]) != TRAINING_AGGREGATION:
            raise ValueError("architecture robustness must use the locked mean-training route")
    if args.stage == "inner":
        training_epochs = None
    else:
        if args.stage == "outer":
            if selection is None or "training_epochs" not in selection:
                raise ValueError("outer confirmation requires an inner-validation training_epochs lock")
            locked_epochs = int(selection["training_epochs"])
        else:
            if args.training_epochs is None:
                raise ValueError("architecture robustness requires the primary inner-validation training_epochs lock")
            locked_epochs = int(args.training_epochs)
        if args.training_epochs is not None and int(args.training_epochs) != locked_epochs:
            raise ValueError("requested training_epochs disagree with the inner-validation lock")
        training_epochs = locked_epochs
    if training_epochs is not None:
        _validate_training_epochs(contract, int(training_epochs))
    run_dir = _run_directory(output_root, str(args.stage), int(args.outer_fold), args.inner_fold, int(args.seed), backbone, training_route_name)
    plan = {
        "status": "dry_run" if args.dry_run else "planned",
        "stage": str(args.stage),
        "outer_fold": int(args.outer_fold),
        "inner_fold": int(args.inner_fold) if args.inner_fold is not None else None,
        "seed": int(args.seed),
        "backbone": backbone,
        "training_aggregation": TRAINING_AGGREGATION,
        "training_route_name": training_route_name,
        "inference_aggregations": inference_aggregations,
        "budgets": budgets,
        "protocol_manifest_sha256": lock["manifest_sha256"],
        "contract_sha256": lock["contract_sha256"],
        "output_directory": str(run_dir),
        "outer_selection": str(Path(args.selection).resolve()) if args.selection else None,
        "route_family": "primary" if args.stage in {"inner", "outer"} else "architecture_robustness",
        "training_epochs": int(training_epochs) if training_epochs is not None else None,
    }
    if args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return 0
    if bool(contract["execution"]["require_clean_git"]) and _git_dirty():
        raise RuntimeError("formal LidMerge route requires a clean git worktree")
    if not args.pretrained:
        raise ValueError("formal LidMerge route requires --pretrained for the declared strong backbone")
    if bool(contract["execution"]["require_production_backward_smoke"]):
        if not args.cuda_preflight_report:
            raise ValueError("formal LidMerge route requires --cuda-preflight-report")
        _validate_cuda_preflight(Path(args.cuda_preflight_report).resolve(), lock)
    device = _device()
    _set_seed(int(args.seed))
    train_frame, evaluation_frame, train_orders, evaluation_orders, train_groups, evaluation_groups = _split(
        manifest, primary_orders, outer, inner, str(args.stage), int(args.outer_fold), args.inner_fold
    )
    if args.resume:
        reused = _resume_completed_run(run_dir, plan)
        print(json.dumps({"status": "reused", "run_directory": str(run_dir), "predictions": str(run_dir / "predictions.csv"), "git_commit": reused["git_commit"]}, ensure_ascii=False))
        return 0
    if (run_dir / "predictions.csv").exists() and not args.force:
        raise RuntimeError(f"route already has predictions; use --force only to replace this declared route: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    model, memory, state_path, history_path = _fit_diagnostic(
        train_frame,
        train_orders,
        contract,
        backbone,
        int(args.seed),
        device,
        pretrained=True,
        allow_download=bool(args.allow_weight_download),
        output_dir=run_dir,
        evaluation_frame=evaluation_frame if args.stage == "inner" else None,
        evaluation_orders=evaluation_orders if args.stage == "inner" else None,
        training_epochs=training_epochs,
    )
    prediction, image_logits = _predict_all_budgets(
        model,
        evaluation_frame,
        evaluation_orders,
        budgets,
        int(args.seed),
        device,
        int(contract["training"]["batch_size"]),
        inference_aggregations,
    )
    prediction = prediction.assign(
        stage=str(args.stage),
        outer_fold=int(args.outer_fold),
        inner_fold=int(args.inner_fold) if args.inner_fold is not None else -1,
        seed=int(args.seed),
        backbone=backbone,
    )
    image_logits = image_logits.assign(
        stage=str(args.stage),
        outer_fold=int(args.outer_fold),
        inner_fold=int(args.inner_fold) if args.inner_fold is not None else -1,
        seed=int(args.seed),
        backbone=backbone,
    )
    prediction_path = run_dir / "predictions.csv"
    image_logits_path = run_dir / "image_logits.csv"
    prediction.to_csv(prediction_path, index=False)
    image_logits.to_csv(image_logits_path, index=False)
    artifacts = [state_path, history_path, image_logits_path, prediction_path]
    if args.stage == "outer":
        controls = _predict_counterfactuals(
            model,
            evaluation_frame,
            evaluation_orders,
            budgets,
            int(args.seed),
            device,
            int(contract["training"]["batch_size"]),
            int(contract["counterfactual_evaluation"]["lowpass_kernel_size"]),
            inference_aggregations,
        )
        controls = controls.assign(stage="outer", outer_fold=int(args.outer_fold), seed=int(args.seed), backbone=backbone)
        controls_path = run_dir / "counterfactual_predictions.csv"
        controls.to_csv(controls_path, index=False)
        alternate = _predict_alternate_orders(model, evaluation_frame, alternate_orders, evaluation_groups, budgets, int(args.seed), device, int(contract["training"]["batch_size"]), inference_aggregations)
        alternate = alternate.assign(stage="outer", outer_fold=int(args.outer_fold), seed=int(args.seed), backbone=backbone)
        alternate_path = run_dir / "alternate_order_predictions.csv"
        alternate.to_csv(alternate_path, index=False)
        artifacts.extend([controls_path, alternate_path])
    run_manifest = {
        **plan,
        "status": "complete",
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "code_fingerprint": _code_fingerprint(),
        "runtime": _runtime_provenance(__import__("torch"), device),
        "training_patient_groups": int(len(train_groups)),
        "evaluation_patient_groups": int(len(evaluation_groups)),
        "memory": memory,
        "selection": selection,
        "shared_logit_artifact": {
            "path": image_logits_path.name,
            "maximum_budget": int(max(budgets)),
            "prediction_reconstruction_verified": True,
        },
        "artifacts": [_artifact_record(path, run_dir) for path in artifacts],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "run_directory": str(run_dir), "predictions": str(prediction_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
