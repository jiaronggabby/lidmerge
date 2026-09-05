#!/usr/bin/env python3
"""Summarize alternate blinded photograph-order sensitivity without selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lidpair.metrics import patient_metrics
from lidpair.validate import validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _route_path(output_root: Path, outer_fold: int, seed: int, backbone: str, training_route_name: str) -> Path:
    return output_root / "outer" / f"outer_{outer_fold}" / f"seed_{seed}" / f"{backbone}__{training_route_name}" / "alternate_order_predictions.csv"


def collect_alternate_order_oof(protocol_root: Path, output_root: Path) -> tuple[pd.DataFrame, dict]:
    audit = validate_protocol(protocol_root, check_paths=False)
    lock = _load_json(protocol_root / "protocol_lock.json")
    contract = lock["contract"]
    outer = pd.read_csv(protocol_root / "outer_folds.csv", dtype={"patient_group": str})
    seeds = [int(value) for value in contract["validation"]["formal_seeds"]]
    budgets = [int(value) for value in contract["views"]["budgets"]]
    order_seeds = [int(value) for value in contract["views"]["alternate_order_seeds"]]
    rows: list[pd.DataFrame] = []
    training_route_name = str(contract["aggregation"]["training_route_name"])
    for variant in contract["aggregation"]["inference"]:
        for outer_fold in range(int(contract["validation"]["outer_folds"])):
            selection = _load_json(output_root / "selection" / f"outer_{outer_fold}" / f"{variant}_selection.json")
            if (
                selection.get("selection_kind") != "variant_budget_thresholds"
                or int(selection.get("outer_fold", -1)) != outer_fold
                or str(selection.get("variant")) != str(variant)
                or str(selection.get("training_aggregation")) != "mean"
                or selection.get("protocol_manifest_sha256") != lock["manifest_sha256"]
                or selection.get("contract_sha256") != lock["contract_sha256"]
                or [int(value) for value in selection.get("prediction_seeds", [])] != seeds
            ):
                raise ValueError("alternate-order aggregation found an invalid selection lock")
            if set(map(int, selection.get("budget_thresholds", {}))) != set(budgets):
                raise ValueError("alternate-order selection lacks a threshold for every K")
            expected = outer[outer["outer_fold"].astype(int).eq(outer_fold)][["patient_group", "label"]].copy()
            frames: list[pd.DataFrame] = []
            for seed in seeds:
                path = _route_path(output_root, outer_fold, seed, str(selection["backbone"]), training_route_name)
                current = pd.read_csv(path, dtype={"patient_group": str})
                required = {"patient_group", "label", "budget", "photograph_order_seed", "probability", "stage", "outer_fold", "seed", "backbone", "variant", "training_aggregation"}
                if missing := required.difference(current.columns):
                    raise ValueError(f"{path} lacks alternate-order prediction columns: {sorted(missing)}")
                current = current[current["variant"].astype(str).eq(str(variant))].copy()
                if current.empty:
                    raise ValueError(f"{path} lacks the requested shared-logit aggregation: {variant}")
                current["budget"] = pd.to_numeric(current["budget"], errors="raise").astype(int)
                current["photograph_order_seed"] = pd.to_numeric(current["photograph_order_seed"], errors="raise").astype(int)
                if set(current["budget"]) != set(budgets) or set(current["photograph_order_seed"]) != set(order_seeds):
                    raise ValueError(f"{path} alternate-order dimensions differ from the protocol")
                counts = current.groupby(["budget", "photograph_order_seed"])["patient_group"].agg(["size", "nunique"])
                if not counts["size"].eq(len(expected)).all() or not counts["nunique"].eq(len(expected)).all():
                    raise ValueError(f"{path} does not contain one held-out patient per K/order draw")
                labels = current[["patient_group", "label"]].drop_duplicates().merge(expected, on="patient_group", suffixes=("_run", "_lock"), validate="one_to_one")
                if set(labels["patient_group"]) != set(expected["patient_group"]) or labels["label_run"].astype(int).ne(labels["label_lock"].astype(int)).any():
                    raise ValueError(f"{path} labels differ from the locked outer fold")
                if not current["stage"].astype(str).eq("outer").all() or not current["outer_fold"].astype(int).eq(outer_fold).all() or not current["seed"].astype(int).eq(seed).all():
                    raise ValueError(f"{path} has inconsistent route provenance")
                if (
                    not current["backbone"].astype(str).eq(str(selection["backbone"])).all()
                    or not current["variant"].astype(str).eq(str(variant)).all()
                    or not current["training_aggregation"].astype(str).eq("mean").all()
                ):
                    raise ValueError(f"{path} has inconsistent selected route provenance")
                probability = pd.to_numeric(current["probability"], errors="raise")
                if probability.isna().any() or ((probability < 0.0) | (probability > 1.0)).any():
                    raise ValueError(f"{path} contains invalid alternate-order probabilities")
                frames.append(current[["patient_group", "label", "budget", "photograph_order_seed", "probability"]].rename(columns={"probability": f"probability_seed_{seed}"}))
            merged = frames[0]
            for current in frames[1:]:
                merged = merged.merge(current, on=["patient_group", "label", "budget", "photograph_order_seed"], validate="one_to_one")
            probability_columns = [column for column in merged.columns if column.startswith("probability_seed_")]
            merged["probability"] = merged[probability_columns].mean(axis=1)
            thresholds = selection["budget_thresholds"]
            merged["decision_threshold"] = merged["budget"].map(lambda budget: float(thresholds[str(int(budget))]["threshold"]))
            merged["decision"] = merged["probability"] >= merged["decision_threshold"]
            merged["outer_fold"] = outer_fold
            merged["backbone"] = str(selection["backbone"])
            merged["variant"] = str(variant)
            rows.append(merged[["patient_group", "label", "budget", "photograph_order_seed", "outer_fold", "backbone", "variant", "probability", "decision_threshold", "decision"]])
    result = pd.concat(rows, ignore_index=True).sort_values(["variant", "photograph_order_seed", "budget", "patient_group"]).reset_index(drop=True)
    return result, {"audit": audit, "lock": lock, "contract": contract}


def write_alternate_order_summary(protocol_root: Path, output_root: Path, force: bool) -> dict:
    oof, context = collect_alternate_order_oof(protocol_root, output_root)
    root = output_root / "summary"
    root.mkdir(parents=True, exist_ok=True)
    oof_path = root / "lidmerge_alternate_order_oof_patient_predictions.csv"
    report_path = root / "lidmerge_alternate_order_summary.json"
    if (oof_path.exists() or report_path.exists()) and not force:
        raise RuntimeError(f"alternate-order outputs already exist; use --force to rebuild: {root}")
    metrics = {
        # Alternate-order analysis is descriptive sensitivity analysis only;
        # the core patient metrics are sufficient and avoid repeated logistic
        # calibration fits outside the main budget curve.
        f"{variant}:draw_{draw}:K{budget}": patient_metrics(
            frame["label"], frame["probability"], decisions=frame["decision"], include_extended=False
        )
        for (variant, draw, budget), frame in oof.groupby(["variant", "photograph_order_seed", "budget"], sort=True)
    }
    oof.to_csv(oof_path, index=False)
    payload = {
        "status": "complete",
        "patient_level_only": True,
        "metrics_by_blinded_order_draw": metrics,
        "interpretation": "Alternate orders are a prespecified inference-only sensitivity analysis. They are not a selector for a preferred K, order, or model.",
        "protocol_manifest_sha256": context["lock"]["manifest_sha256"],
        "contract_sha256": context["lock"]["contract_sha256"],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"oof_path": str(oof_path), "summary_path": str(report_path), "summary": payload}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = write_alternate_order_summary(Path(args.protocol_root).resolve(), Path(args.output_root).resolve(), bool(args.force))
    print(json.dumps({"status": "complete", "oof_path": result["oof_path"], "summary_path": result["summary_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
