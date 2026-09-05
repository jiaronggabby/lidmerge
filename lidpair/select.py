#!/usr/bin/env python3
"""Create LidMerge backbone and K-specific threshold locks from inner data only."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from lidpair.metrics import patient_metrics, select_threshold_at_specificity
from lidpair.validate import validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_protocol(protocol_root: Path):
    validate_protocol(protocol_root, check_paths=False)
    lock = _load_json(protocol_root / "protocol_lock.json")
    inner = pd.read_csv(protocol_root / "inner_folds.csv", dtype={"patient_group": str})
    return lock, lock["contract"], inner


def _route_path(output_root: Path, outer_fold: int, inner_fold: int, seed: int, backbone: str, training_route_name: str) -> Path:
    return output_root / "inner" / f"outer_{outer_fold}" / f"inner_{inner_fold}" / f"seed_{seed}" / f"{backbone}__{training_route_name}" / "predictions.csv"


def _inner_training_epochs(
    output_root: Path,
    outer_fold: int,
    seeds: list[int],
    backbone: str,
    n_inner: int,
    training_route_name: str,
    contract: dict,
) -> int:
    """Aggregate complete-K-cycle stopping points for outer training."""

    training = contract["training"]
    early_stopping = training["early_stopping"]
    cycle_length = int(training["cycle_length"])
    minimum_cycles = int(early_stopping["minimum_cycles"])
    minimum_epochs = int(early_stopping["minimum_epochs"])
    maximum_epochs = int(training["diagnostic_epochs"])
    cycles: list[int] = []
    for inner_fold in range(n_inner):
        for seed in seeds:
            prediction_path = _route_path(output_root, outer_fold, inner_fold, int(seed), backbone, training_route_name)
            manifest_path = prediction_path.parent / "run_manifest.json"
            manifest = _load_json(manifest_path)
            memory = manifest.get("memory")
            if not isinstance(memory, dict):
                raise ValueError(f"{manifest_path} lacks training-cycle provenance")
            if "training_cycles_realized" in memory:
                cycle = int(memory["training_cycles_realized"])
            elif "training_epochs_realized" in memory:
                epochs = int(memory["training_epochs_realized"])
                if epochs % cycle_length != 0:
                    raise ValueError(f"{manifest_path} has a non-complete K cycle: {epochs} epochs")
                cycle = epochs // cycle_length
            else:
                raise ValueError(f"{manifest_path} lacks training-cycle provenance")
            if cycle < minimum_cycles:
                raise ValueError(f"{manifest_path} has too few complete K cycles: {cycle}")
            cycles.append(cycle)
    if not cycles:
        raise ValueError("no inner-validation stopping epochs were available")
    # A deterministic integer median is locked into the outer route. The
    # outer test is never used to choose or refine this value. The clamp is a
    # protocol guard, not a post-hoc performance choice.
    epochs = cycle_length * int(round(statistics.median(cycles)))
    epochs = max(minimum_epochs, min(maximum_epochs, epochs))
    if epochs % cycle_length != 0:
        raise ValueError(f"locked outer training epochs are not a complete K cycle: {epochs}")
    return int(epochs)


def _read_inner_predictions(output_root: Path, inner: pd.DataFrame, outer_fold: int, seeds: list[int], backbone: str, aggregation: str, budgets: list[int], n_inner: int, training_route_name: str) -> pd.DataFrame:
    required = {"patient_group", "label", "budget", "probability", "stage", "outer_fold", "inner_fold", "seed", "backbone", "variant", "training_aggregation"}
    rows: list[pd.DataFrame] = []
    if not seeds:
        raise ValueError("inner selection requires at least one declared prediction seed")
    for inner_fold in range(n_inner):
        expected = inner[(inner["protocol_outer_fold"].astype(int).eq(int(outer_fold))) & (inner["inner_fold"].astype(int).eq(int(inner_fold)))][["patient_group", "label"]].copy()
        seed_frames: list[pd.DataFrame] = []
        for seed in seeds:
            path = _route_path(output_root, outer_fold, inner_fold, int(seed), backbone, training_route_name)
            if not path.is_file():
                raise FileNotFoundError(f"missing required inner prediction route: {path}")
            current = pd.read_csv(path, dtype={"patient_group": str})
            if missing := required.difference(current.columns):
                raise ValueError(f"{path} lacks patient-level prediction columns: {sorted(missing)}")
            current = current[current["variant"].astype(str).eq(str(aggregation))].copy()
            if current.empty:
                raise ValueError(f"{path} lacks the requested inference aggregation: {aggregation}")
            current["budget"] = pd.to_numeric(current["budget"], errors="raise").astype(int)
            if set(current["budget"]) != set(budgets):
                raise ValueError(f"{path} does not contain every locked K budget")
            counts = current.groupby("budget")["patient_group"].agg(["size", "nunique"])
            if not counts["size"].eq(len(expected)).all() or not counts["nunique"].eq(len(expected)).all():
                raise ValueError(f"{path} does not contain one prediction per inner patient group and budget")
            if not current["stage"].astype(str).eq("inner").all():
                raise ValueError(f"{path} was not produced by an inner route")
            for column, value in (("outer_fold", outer_fold), ("inner_fold", inner_fold), ("seed", seed)):
                if not pd.to_numeric(current[column], errors="raise").astype(int).eq(int(value)).all():
                    raise ValueError(f"{path} has inconsistent {column}")
            if (
                not current["backbone"].astype(str).eq(backbone).all()
                or not current["variant"].astype(str).eq(aggregation).all()
                or not current["training_aggregation"].astype(str).eq("mean").all()
            ):
                raise ValueError(f"{path} does not match requested backbone / aggregation")
            checked = current[["patient_group", "label"]].drop_duplicates().merge(expected, on="patient_group", suffixes=("_run", "_lock"), validate="one_to_one")
            if set(checked["patient_group"]) != set(expected["patient_group"]) or checked["label_run"].astype(int).ne(checked["label_lock"].astype(int)).any():
                raise ValueError(f"{path} labels do not match the locked inner split")
            seed_frames.append(current[["patient_group", "label", "budget", "probability"]].rename(columns={"probability": f"probability_seed_{seed}"}))
        merged = seed_frames[0]
        for current in seed_frames[1:]:
            merged = merged.merge(current, on=["patient_group", "label", "budget"], validate="one_to_one")
        probability_columns = [column for column in merged.columns if column.startswith("probability_seed_")]
        merged["probability"] = merged[probability_columns].mean(axis=1)
        rows.append(merged[["patient_group", "label", "budget", "probability"]])
    result = pd.concat(rows, ignore_index=True).sort_values(["budget", "patient_group"]).reset_index(drop=True)
    if result.duplicated(["patient_group", "budget"]).any():
        raise ValueError("a patient_group appeared in more than one inner-validation prediction file")
    return result


def _score(predictions: pd.DataFrame) -> tuple[dict, tuple[float, float, float]]:
    threshold_result = select_threshold_at_specificity(predictions["label"], predictions["probability"], 0.90)
    metrics = patient_metrics(predictions["label"], predictions["probability"], threshold=threshold_result.threshold)
    metrics["selection_threshold_specificity"] = float(threshold_result.specificity)
    metrics["selection_threshold_sensitivity"] = float(threshold_result.sensitivity)
    return metrics, (float(metrics["sensitivity"]), float(metrics["auprc"]), float(metrics["auroc"]))


def _write_json(payload: dict, path: Path, force: bool) -> Path:
    if path.exists() and not force:
        existing = _load_json(path)
        if existing == payload:
            return path
        raise RuntimeError(f"selection lock already exists but does not exactly match current inner predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def select_backbone(protocol_root: Path, output_root: Path, selection_root: Path, outer_fold: int, force: bool) -> Path:
    lock, contract, inner = _load_protocol(protocol_root)
    n_inner = int(contract["validation"]["inner_folds"])
    screen_seed = int(contract["validation"]["selection_seed"])
    screen_variant = str(contract["validation"]["backbone_screen_aggregation"])
    screen_budget = int(contract["validation"]["backbone_screen_budget"])
    all_budgets = [int(value) for value in contract["views"]["budgets"]]
    candidates: list[dict] = []
    training_route_name = str(contract["aggregation"]["training_route_name"])
    for order, backbone in enumerate(contract["backbone_screen"]):
        prediction = _read_inner_predictions(output_root, inner, outer_fold, [screen_seed], str(backbone), screen_variant, all_budgets, n_inner, training_route_name)
        metrics, score = _score(prediction[prediction["budget"].eq(screen_budget)])
        training_epochs = _inner_training_epochs(output_root, outer_fold, [screen_seed], str(backbone), n_inner, training_route_name, contract)
        candidates.append({"backbone": str(backbone), "variant": screen_variant, "budget": screen_budget, "metrics": metrics, "score": list(score), "order": order, "training_epochs": training_epochs})
    chosen = max(candidates, key=lambda row: (*row["score"], -int(row["order"])))
    payload = {
        "selection_kind": "backbone_screen",
        "outer_fold": int(outer_fold),
        "selection_seed": screen_seed,
        "backbone": chosen["backbone"],
        "screen_variant": screen_variant,
        "screen_budget": screen_budget,
        "selection_metric": contract["validation"]["selection_metric"],
        "selection_input": "inner-fold patient-level K=1 predictions only",
        "prediction_seeds": [screen_seed],
        "metrics": chosen["metrics"],
        "training_epochs": int(chosen["training_epochs"]),
        "candidates": candidates,
        "protocol_manifest_sha256": lock["manifest_sha256"],
        "contract_sha256": lock["contract_sha256"],
    }
    return _write_json(payload, selection_root / f"outer_{outer_fold}" / "backbone_selection.json", force)


def select_variant(protocol_root: Path, output_root: Path, selection_root: Path, outer_fold: int, backbone_selection: Path, variant: str, force: bool) -> Path:
    lock, contract, inner = _load_protocol(protocol_root)
    if variant not in set(contract["aggregation"]["inference"]):
        raise ValueError(f"aggregation variant is not declared by the contract: {variant}")
    screen = _load_json(backbone_selection)
    if screen.get("selection_kind") != "backbone_screen" or int(screen.get("outer_fold", -1)) != int(outer_fold):
        raise ValueError("backbone selection is not for this outer fold")
    if screen.get("protocol_manifest_sha256") != lock["manifest_sha256"] or screen.get("contract_sha256") != lock["contract_sha256"]:
        raise ValueError("backbone selection targets a different locked protocol")
    budgets = [int(value) for value in contract["views"]["budgets"]]
    seeds = [int(value) for value in contract["validation"]["threshold_prediction_seeds"]]
    prediction = _read_inner_predictions(
        output_root,
        inner,
        outer_fold,
        seeds,
        str(screen["backbone"]),
        variant,
        budgets,
        int(contract["validation"]["inner_folds"]),
        str(contract["aggregation"]["training_route_name"]),
    )
    thresholds: dict[str, dict] = {}
    for budget in budgets:
        metrics, _ = _score(prediction[prediction["budget"].eq(int(budget))])
        thresholds[str(budget)] = {"threshold": float(metrics["threshold"]), "metrics": metrics}
    training_epochs = _inner_training_epochs(
        output_root,
        outer_fold,
        seeds,
        str(screen["backbone"]),
        int(contract["validation"]["inner_folds"]),
        str(contract["aggregation"]["training_route_name"]),
        contract,
    )
    payload = {
        "selection_kind": "variant_budget_thresholds",
        "outer_fold": int(outer_fold),
        "selection_seed": int(contract["validation"]["selection_seed"]),
        "backbone": str(screen["backbone"]),
        "variant": str(variant),
        "training_aggregation": str(contract["aggregation"]["training"]),
        "budgets": budgets,
        "budget_thresholds": thresholds,
        "training_epochs": int(training_epochs),
        "selection_metric": contract["validation"]["selection_metric"],
        "selection_input": "matching inner-fold patient-level seed-ensemble predictions only",
        "prediction_seeds": seeds,
        "backbone_selection": str(backbone_selection.resolve()),
        "protocol_manifest_sha256": lock["manifest_sha256"],
        "contract_sha256": lock["contract_sha256"],
    }
    return _write_json(payload, selection_root / f"outer_{outer_fold}" / f"{variant}_selection.json", force)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--selection-root", help="defaults to OUTPUT_ROOT/selection")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--mode", choices=("backbone", "variant"), required=True)
    parser.add_argument("--backbone-selection", help="required for --mode variant")
    parser.add_argument("--variant", help="required for --mode variant")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_root, output_root = Path(args.protocol_root).resolve(), Path(args.output_root).resolve()
    selection_root = Path(args.selection_root).resolve() if args.selection_root else output_root / "selection"
    if args.mode == "backbone":
        path = select_backbone(protocol_root, output_root, selection_root, int(args.outer_fold), bool(args.force))
    else:
        if not args.backbone_selection or not args.variant:
            raise ValueError("--mode variant requires --backbone-selection and --variant")
        path = select_variant(protocol_root, output_root, selection_root, int(args.outer_fold), Path(args.backbone_selection).resolve(), str(args.variant), bool(args.force))
    print(json.dumps({"status": "complete", "selection": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
