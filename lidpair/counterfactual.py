#!/usr/bin/env python3
"""Aggregate LidMerge inference-only implementation controls at patient level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lidpair.metrics import patient_metrics
from lidpair.validate import validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _route_path(output_root: Path, outer_fold: int, seed: int, backbone: str, training_route_name: str) -> Path:
    return output_root / "outer" / f"outer_{outer_fold}" / f"seed_{seed}" / f"{backbone}__{training_route_name}" / "counterfactual_predictions.csv"


def collect_counterfactual_oof(protocol_root: Path, output_root: Path) -> tuple[pd.DataFrame, dict]:
    audit = validate_protocol(protocol_root, check_paths=False)
    lock = _load_json(protocol_root / "protocol_lock.json")
    contract = lock["contract"]
    outer = pd.read_csv(protocol_root / "outer_folds.csv", dtype={"patient_group": str})
    budgets = [int(value) for value in contract["views"]["budgets"]]
    conditions = [str(value) for value in contract["counterfactual_evaluation"]["conditions"]]
    seeds = [int(value) for value in contract["validation"]["formal_seeds"]]
    all_rows: list[pd.DataFrame] = []
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
                raise ValueError("counterfactual aggregation found an invalid selection lock")
            thresholds = selection.get("budget_thresholds", {})
            if set(map(int, thresholds)) != set(budgets):
                raise ValueError("counterfactual aggregation selection lacks a threshold for every K")
            expected = outer[outer["outer_fold"].astype(int).eq(outer_fold)][["patient_group", "label"]].copy()
            seed_rows: list[pd.DataFrame] = []
            for seed in seeds:
                path = _route_path(output_root, outer_fold, seed, str(selection["backbone"]), training_route_name)
                frame = pd.read_csv(path, dtype={"patient_group": str, "condition": str})
                required = {"patient_group", "label", "budget", "condition", "probability", "stage", "outer_fold", "seed", "backbone", "variant", "training_aggregation"}
                if missing := required.difference(frame.columns):
                    raise ValueError(f"{path} lacks counterfactual columns: {sorted(missing)}")
                frame = frame[frame["variant"].astype(str).eq(str(variant))].copy()
                if frame.empty:
                    raise ValueError(f"{path} lacks the requested shared-logit aggregation: {variant}")
                frame["budget"] = pd.to_numeric(frame["budget"], errors="raise").astype(int)
                if set(frame["budget"]) != set(budgets) or set(frame["condition"].astype(str)) != set(conditions):
                    raise ValueError(f"{path} counterfactual dimensions differ from the locked protocol")
                counts = frame.groupby(["budget", "condition"])["patient_group"].agg(["size", "nunique"])
                if not counts["size"].eq(len(expected)).all() or not counts["nunique"].eq(len(expected)).all():
                    raise ValueError(f"{path} lacks one patient row per K/control")
                labels = frame[["patient_group", "label"]].drop_duplicates().merge(expected, on="patient_group", suffixes=("_run", "_lock"), validate="one_to_one")
                if set(labels["patient_group"]) != set(expected["patient_group"]) or labels["label_run"].astype(int).ne(labels["label_lock"].astype(int)).any():
                    raise ValueError(f"{path} labels do not match the locked outer fold")
                if not frame["stage"].astype(str).eq("outer").all() or not frame["outer_fold"].astype(int).eq(outer_fold).all() or not frame["seed"].astype(int).eq(seed).all():
                    raise ValueError(f"{path} has inconsistent route provenance")
                if (
                    not frame["backbone"].astype(str).eq(str(selection["backbone"])).all()
                    or not frame["variant"].astype(str).eq(str(variant)).all()
                    or not frame["training_aggregation"].astype(str).eq("mean").all()
                ):
                    raise ValueError(f"{path} has inconsistent selected route provenance")
                probability = pd.to_numeric(frame["probability"], errors="raise")
                if probability.isna().any() or ((probability < 0.0) | (probability > 1.0)).any():
                    raise ValueError(f"{path} contains invalid counterfactual probabilities")
                seed_rows.append(frame[["patient_group", "label", "budget", "condition", "probability"]].rename(columns={"probability": f"probability_seed_{seed}"}))
            merged = seed_rows[0]
            for current in seed_rows[1:]:
                merged = merged.merge(current, on=["patient_group", "label", "budget", "condition"], validate="one_to_one")
            probability_columns = [column for column in merged.columns if column.startswith("probability_seed_")]
            merged["probability"] = merged[probability_columns].mean(axis=1)
            merged["decision_threshold"] = merged["budget"].map(lambda budget: float(thresholds[str(int(budget))]["threshold"]))
            merged["decision"] = merged["probability"] >= merged["decision_threshold"]
            merged["outer_fold"] = outer_fold
            merged["backbone"] = str(selection["backbone"])
            merged["variant"] = str(variant)
            all_rows.append(merged[["patient_group", "label", "budget", "condition", "outer_fold", "backbone", "variant", "probability", "decision_threshold", "decision"]])
    oof = pd.concat(all_rows, ignore_index=True).sort_values(["variant", "budget", "condition", "patient_group"]).reset_index(drop=True)
    return oof, {"audit": audit, "lock": lock, "contract": contract}


def write_counterfactual_summary(protocol_root: Path, output_root: Path, force: bool) -> dict:
    oof, context = collect_counterfactual_oof(protocol_root, output_root)
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    oof_path = summary_root / "lidmerge_counterfactual_oof_patient_predictions.csv"
    report_path = summary_root / "lidmerge_counterfactual_summary.json"
    if (oof_path.exists() or report_path.exists()) and not force:
        raise RuntimeError(f"counterfactual outputs already exist; use --force to rebuild: {summary_root}")
    tolerance = float(context["contract"]["counterfactual_evaluation"]["invariance_tolerance"])
    deltas: dict[str, float] = {}
    lowpass_effect: dict[str, dict[str, float]] = {}
    for variant, current_variant in oof.groupby("variant", sort=True):
        pivot = current_variant.pivot(index=["patient_group", "budget"], columns="condition", values="probability")
        for budget in context["contract"]["views"]["budgets"]:
            if int(budget) >= 2:
                current = pivot.xs(int(budget), level="budget")
                deltas[f"{variant}:K{budget}:true_vs_reversed_max_abs"] = float((current["true_prefix"] - current["reversed_prefix"]).abs().max())
                first = pivot.xs(1, level="budget")["true_prefix"]
                deltas[f"{variant}:K{budget}:duplicate_vs_K1_max_abs"] = float((current["duplicate_view1"] - first.reindex(current.index)).abs().max())
        for budget, current_budget in current_variant.groupby("budget", sort=True):
            clean = current_budget[current_budget["condition"].eq("true_prefix")].set_index("patient_group").sort_index()
            lowpass = current_budget[current_budget["condition"].eq("lowpass_view1")].set_index("patient_group").sort_index()
            if not clean.index.equals(lowpass.index) or clean["label"].astype(int).ne(lowpass["label"].astype(int)).any():
                raise ValueError("low-pass control does not match true-prefix patient groups")
            clean_metrics = patient_metrics(clean["label"], clean["probability"], decisions=clean["decision"], include_extended=False)
            lowpass_metrics = patient_metrics(lowpass["label"], lowpass["probability"], decisions=lowpass["decision"], include_extended=False)
            lowpass_effect[f"{variant}:K{int(budget)}"] = {
                "mean_absolute_probability_shift": float((lowpass["probability"] - clean["probability"]).abs().mean()),
                **{
                    f"lowpass_minus_clean_{metric}": float(lowpass_metrics[metric]) - float(clean_metrics[metric])
                    for metric in ("auroc", "auprc", "brier", "sensitivity", "specificity", "fnr")
                },
            }
    passed = all(value <= tolerance for value in deltas.values())
    reversal_deltas = {key: value for key, value in deltas.items() if "true_vs_reversed" in key}
    duplicate_deltas = {key: value for key, value in deltas.items() if "duplicate_vs_K1" in key}
    metrics = {
        # These are implementation/robustness checks, not another calibration
        # study.  Retain the shared core metrics (including ECE) while avoiding
        # repeated descriptive logistic calibration fits for every control cell.
        f"{variant}:K{budget}:{condition}": patient_metrics(
            frame["label"], frame["probability"], decisions=frame["decision"], include_extended=False
        )
        for (variant, budget, condition), frame in oof.groupby(["variant", "budget", "condition"], sort=True)
    }
    oof.to_csv(oof_path, index=False)
    payload = {
        "status": "complete" if passed else "failed",
        "patient_level_only": True,
        "implementation_invariance_passed": passed,
        "invariance_tolerance": tolerance,
        "numerical_difference_within_tolerance": passed,
        "reversal_differences": reversal_deltas,
        "duplicate_differences": duplicate_deltas,
        "max_absolute_probability_differences": deltas,
        "lowpass_view1_supporting_effect": lowpass_effect,
        "metrics": metrics,
        "interpretation": "Reversal and duplicated-view checks validate symmetric aggregation only. lowpass_view1 is a reporting-only stress analysis for whether additional photographs buffer one impaired input. Neither analysis proves clinical complementarity.",
        "protocol_manifest_sha256": context["lock"]["manifest_sha256"],
        "contract_sha256": context["lock"]["contract_sha256"],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Counterfactual reporting must not prevent independent summaries from
    # being generated.  The report records whether the declared tolerance was
    # met; only malformed inputs remain hard errors above.  This separates a
    # numerical implementation warning from a pipeline failure.
    return {"oof_path": str(oof_path), "summary_path": str(report_path), "summary": payload}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = write_counterfactual_summary(Path(args.protocol_root).resolve(), Path(args.output_root).resolve(), bool(args.force))
    print(json.dumps({"status": "complete", "oof_path": result["oof_path"], "summary_path": result["summary_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
