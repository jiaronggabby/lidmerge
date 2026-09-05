#!/usr/bin/env python3
"""Aggregate completed LidMerge outer predictions strictly at patient level.

Each route trains one mean-logit image model. This module reconstructs mean,
top-2 mean, and max curves from the same per-image model outputs, applies only
inner-fold threshold locks, and keeps confirmatory inference deliberately
narrow: one primary high-specificity partial-AUROC comparison and three
Holm-adjusted incremental sensitivity comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lidpair.metrics import patient_metrics, stratified_patient_bootstrap
from lidpair.validate import validate_protocol


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "standardized_pauc_at_fpr_0_10",
    "auroc",
    "auprc",
    "brier",
    "ece_10",
    "sensitivity",
    "specificity",
    "fnr",
)
METRIC_LABELS = {
    "standardized_pauc_at_fpr_0_10": "standardized_pauc_fpr_0_10",
    "sensitivity": "sensitivity_at_inner_selected_spec90",
    "specificity": "achieved_specificity",
    "auroc": "auroc",
    "auprc": "auprc",
    "fnr": "malignant_fnr",
    "brier": "brier_score",
    "ece_10": "ece_10",
}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregation(contract: dict) -> tuple[str, list[str]]:
    configuration = contract["aggregation"]
    route_name = str(configuration["training_route_name"])
    aggregations = [str(value) for value in configuration["inference"]]
    if str(configuration["training"]) != "mean" or aggregations != ["mean", "top2_mean", "max"]:
        raise ValueError("LidMerge summary requires its locked shared-logit mean/top2_mean/max aggregation contract")
    return route_name, aggregations


def _route_path(output_root: Path, outer_fold: int, seed: int, backbone: str, training_route_name: str) -> Path:
    return output_root / "outer" / f"outer_{outer_fold}" / f"seed_{seed}" / f"{backbone}__{training_route_name}" / "predictions.csv"


def _selection_path(output_root: Path, outer_fold: int, aggregation: str) -> Path:
    return output_root / "selection" / f"outer_{outer_fold}" / f"{aggregation}_selection.json"


def _validate_outer_prediction(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    outer_fold: int,
    seed: int,
    backbone: str,
    aggregation: str,
    budgets: list[int],
    source: Path,
) -> pd.DataFrame:
    required = {
        "patient_group", "label", "budget", "probability", "stage", "outer_fold",
        "inner_fold", "seed", "backbone", "variant", "training_aggregation",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"{source} lacks patient-level prediction columns: {sorted(missing)}")
    current = frame[frame["variant"].astype(str).eq(str(aggregation))].copy()
    if current.empty:
        raise ValueError(f"{source} lacks the requested shared-logit aggregation: {aggregation}")
    current["patient_group"] = current["patient_group"].astype(str)
    current["label"] = pd.to_numeric(current["label"], errors="raise").astype(int)
    current["budget"] = pd.to_numeric(current["budget"], errors="raise").astype(int)
    current["probability"] = pd.to_numeric(current["probability"], errors="raise")
    if set(current["budget"]) != set(budgets):
        raise ValueError(f"{source} does not contain each declared K budget for {aggregation}")
    counts = current.groupby("budget")["patient_group"].agg(["size", "nunique"])
    if not counts["size"].eq(len(expected)).all() or not counts["nunique"].eq(len(expected)).all():
        raise ValueError(f"{source} does not contain one outer prediction per patient_group and K")
    if current.duplicated(["patient_group", "budget"]).any() or not current["stage"].astype(str).eq("outer").all():
        raise ValueError(f"{source} has invalid outer prediction provenance")
    for column, value in (("outer_fold", outer_fold), ("seed", seed)):
        if not pd.to_numeric(current[column], errors="raise").astype(int).eq(int(value)).all():
            raise ValueError(f"{source} has inconsistent {column}")
    if (
        not current["backbone"].astype(str).eq(backbone).all()
        or not current["variant"].astype(str).eq(aggregation).all()
        or not current["training_aggregation"].astype(str).eq("mean").all()
    ):
        raise ValueError(f"{source} has inconsistent selected backbone or shared-logit aggregation")
    if current["inner_fold"].astype(int).ne(-1).any():
        raise ValueError(f"{source} outer predictions must retain inner_fold=-1")
    labels = current[["patient_group", "label"]].drop_duplicates().merge(expected, on="patient_group", suffixes=("_run", "_lock"), validate="one_to_one")
    if set(labels["patient_group"]) != set(expected["patient_group"]) or labels["label_run"].ne(labels["label_lock"]).any():
        raise ValueError(f"{source} labels differ from the locked outer patient groups")
    if current["probability"].isna().any() or ((current["probability"] < 0.0) | (current["probability"] > 1.0)).any():
        raise ValueError(f"{source} contains invalid probabilities")
    return current


def _selection(lock: dict, selection_path: Path, outer_fold: int, aggregation: str, budgets: list[int], seeds: list[int]) -> dict:
    selection = _load_json(selection_path)
    if (
        selection.get("selection_kind") != "variant_budget_thresholds"
        or int(selection.get("outer_fold", -1)) != int(outer_fold)
        or str(selection.get("variant")) != str(aggregation)
    ):
        raise ValueError(f"invalid LidMerge aggregation selection lock: {selection_path}")
    if selection.get("training_aggregation") != "mean":
        raise ValueError(f"selection does not identify the shared mean-trained model: {selection_path}")
    if selection.get("protocol_manifest_sha256") != lock["manifest_sha256"] or selection.get("contract_sha256") != lock["contract_sha256"]:
        raise ValueError(f"selection lock targets a different protocol: {selection_path}")
    if [int(value) for value in selection.get("prediction_seeds", [])] != seeds:
        raise ValueError(f"selection lock does not use the declared inner seed ensemble: {selection_path}")
    thresholds = selection.get("budget_thresholds", {})
    if set(map(int, thresholds)) != set(budgets):
        raise ValueError(f"selection lock lacks a threshold for every K: {selection_path}")
    return selection


def collect_oof(protocol_root: Path, output_root: Path, aggregation: str) -> tuple[pd.DataFrame, dict]:
    audit = validate_protocol(protocol_root, check_paths=False)
    lock = _load_json(protocol_root / "protocol_lock.json")
    contract = lock["contract"]
    training_route_name, aggregations = _aggregation(contract)
    if aggregation not in aggregations:
        raise ValueError(f"aggregation is not declared: {aggregation}")
    outer = pd.read_csv(protocol_root / "outer_folds.csv", dtype={"patient_group": str})
    budgets = [int(value) for value in contract["views"]["budgets"]]
    seeds = [int(value) for value in contract["validation"]["formal_seeds"]]
    rows: list[pd.DataFrame] = []
    seed_rows: list[pd.DataFrame] = []
    selections: dict[int, dict] = {}
    for outer_fold in range(int(contract["validation"]["outer_folds"])):
        selection = _selection(lock, _selection_path(output_root, outer_fold, aggregation), outer_fold, aggregation, budgets, seeds)
        selections[outer_fold] = selection
        expected = outer[outer["outer_fold"].astype(int).eq(outer_fold)][["patient_group", "label"]].copy()
        seed_frames: list[pd.DataFrame] = []
        for seed in seeds:
            source = _route_path(output_root, outer_fold, seed, str(selection["backbone"]), training_route_name)
            current = _validate_outer_prediction(
                pd.read_csv(source, dtype={"patient_group": str}),
                expected,
                outer_fold,
                seed,
                str(selection["backbone"]),
                aggregation,
                budgets,
                source,
            )
            thresholds = selection["budget_thresholds"]
            current["decision_threshold"] = current["budget"].map(lambda budget: float(thresholds[str(int(budget))]["threshold"]))
            current["decision"] = current["probability"] >= current["decision_threshold"]
            current["outer_fold"] = outer_fold
            current["backbone"] = str(selection["backbone"])
            current["variant"] = aggregation
            seed_rows.append(current[["patient_group", "label", "budget", "outer_fold", "backbone", "variant", "seed", "probability", "decision_threshold", "decision"]])
            seed_frames.append(current[["patient_group", "label", "budget", "probability"]].rename(columns={"probability": f"probability_seed_{seed}"}))
        merged = seed_frames[0]
        for current in seed_frames[1:]:
            merged = merged.merge(current, on=["patient_group", "label", "budget"], validate="one_to_one")
        probability_columns = [column for column in merged.columns if column.startswith("probability_seed_")]
        merged["probability"] = merged[probability_columns].mean(axis=1)
        thresholds = selection["budget_thresholds"]
        merged["decision_threshold"] = merged["budget"].map(lambda budget: float(thresholds[str(int(budget))]["threshold"]))
        merged["decision"] = merged["probability"] >= merged["decision_threshold"]
        merged["outer_fold"] = outer_fold
        merged["backbone"] = str(selection["backbone"])
        merged["variant"] = aggregation
        rows.append(merged[["patient_group", "label", "budget", "outer_fold", "backbone", "variant", "probability", "decision_threshold", "decision"]])
    oof = pd.concat(rows, ignore_index=True).sort_values(["budget", "patient_group"]).reset_index(drop=True)
    seed_oof = pd.concat(seed_rows, ignore_index=True).sort_values(["seed", "budget", "patient_group"]).reset_index(drop=True)
    n_primary = int(contract["data"]["expected_primary_groups"])
    counts = oof.groupby("budget")["patient_group"].agg(["size", "nunique"])
    if not counts["size"].eq(n_primary).all() or not counts["nunique"].eq(n_primary).all():
        raise ValueError("outer OOF aggregation is incomplete")
    seed_counts = seed_oof.groupby(["seed", "budget"])["patient_group"].agg(["size", "nunique"])
    if not seed_counts["size"].eq(n_primary).all() or not seed_counts["nunique"].eq(n_primary).all():
        raise ValueError("per-seed outer OOF aggregation is incomplete")
    return oof, {"audit": audit, "lock": lock, "contract": contract, "selections": selections, "seed_oof": seed_oof}


def paired_bootstrap(candidate: pd.DataFrame, comparator: pd.DataFrame, n_bootstrap: int, seed: int, inferential_metrics: set[str] | None = None) -> dict:
    """Patient-resampled paired metric differences with fixed inner thresholds."""

    inferential_metrics = set(inferential_metrics or set())
    columns = ["patient_group", "label", "probability", "decision"]
    merged = candidate[columns].merge(comparator[columns], on=["patient_group", "label"], suffixes=("_candidate", "_comparator"), validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(comparator):
        raise ValueError("paired comparison requires exactly matched patient_group rows")
    y = merged["label"].to_numpy(dtype=int)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("paired bootstrap requires at least two patients in each class")
    candidate_values = patient_metrics(y, merged["probability_candidate"], decisions=merged["decision_candidate"], include_extended=False)
    comparator_values = patient_metrics(y, merged["probability_comparator"], decisions=merged["decision_comparator"], include_extended=False)
    point = {name: float(candidate_values[name]) - float(comparator_values[name]) for name in METRICS}
    draws = {name: [] for name in METRICS}
    rng = np.random.default_rng(int(seed))
    candidate_probability = merged["probability_candidate"].to_numpy(dtype=float)
    comparator_probability = merged["probability_comparator"].to_numpy(dtype=float)
    candidate_decision = merged["decision_candidate"].to_numpy(dtype=bool)
    comparator_decision = merged["decision_comparator"].to_numpy(dtype=bool)
    for _ in range(int(n_bootstrap)):
        index = np.concatenate([rng.choice(positive, size=len(positive), replace=True), rng.choice(negative, size=len(negative), replace=True)])
        left = patient_metrics(y[index], candidate_probability[index], decisions=candidate_decision[index], include_extended=False)
        right = patient_metrics(y[index], comparator_probability[index], decisions=comparator_decision[index], include_extended=False)
        for name in METRICS:
            draws[name].append(float(left[name]) - float(right[name]))
    result: dict[str, dict[str, float | None]] = {}
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        result[name] = {
            "difference": point[name],
            "ci_low": float(np.quantile(array, 0.025)),
            "ci_high": float(np.quantile(array, 0.975)),
            "bootstrap_p_two_sided": (
                min(1.0, 2.0 * min(float((array <= 0.0).mean()), float((array >= 0.0).mean())))
                if name in inferential_metrics else None
            ),
        }
    return {"n_patients": int(len(merged)), "metrics": result}


def _holm_adjust(named_p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(named_p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (count - index)))
        adjusted[name] = running
    return adjusted


def _metrics_by_budget(oof: pd.DataFrame) -> dict[str, dict]:
    return {
        str(int(budget)): patient_metrics(current["label"], current["probability"], decisions=current["decision"])
        for budget, current in oof.groupby("budget", sort=True)
    }


def _metric_confidence_intervals_by_budget(oof: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for offset, (budget, current) in enumerate(oof.groupby("budget", sort=True), start=1):
        point = patient_metrics(current["label"], current["probability"], decisions=current["decision"])
        draws = stratified_patient_bootstrap(
            current["label"], current["probability"], threshold=None, decisions=current["decision"], n_bootstrap=int(n_bootstrap), seed=int(seed) + offset,
        )
        result[str(int(budget))] = {
            "n_patients": int(point["n_patients"]),
            "n_malignant": int(point["n_malignant"]),
            "n_benign": int(point["n_benign"]),
            "metrics": {
                name: {"estimate": float(point[name]), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975))}
                for name, values in draws.items()
            },
        }
    return result


def _budget_curve_ci_rows(ci_by_variant: dict[str, dict]) -> pd.DataFrame:
    """Flatten per-K patient bootstrap intervals into a figure-ready table."""

    rows: list[dict[str, object]] = []
    for aggregation, budgets in ci_by_variant.items():
        for budget, payload in budgets.items():
            for metric, values in payload["metrics"].items():
                rows.append({
                    "aggregation": str(aggregation),
                    "budget": int(budget),
                    "metric": str(metric),
                    "display_metric": METRIC_LABELS[str(metric)],
                    "estimate": float(values["estimate"]),
                    "ci_low": float(values["ci_low"]),
                    "ci_high": float(values["ci_high"]),
                    "n_patients": int(payload["n_patients"]),
                    "n_malignant": int(payload["n_malignant"]),
                    "n_benign": int(payload["n_benign"]),
                    "threshold_source": "aggregation-and-K-specific inner-fold seed-ensemble selection",
                })
    return pd.DataFrame(rows).sort_values(["aggregation", "metric", "budget"]).reset_index(drop=True)


def _point_difference(candidate: pd.DataFrame, comparator: pd.DataFrame) -> dict[str, float]:
    columns = ["patient_group", "label", "probability", "decision"]
    merged = candidate[columns].merge(comparator[columns], on=["patient_group", "label"], suffixes=("_candidate", "_comparator"), validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(comparator):
        raise ValueError("point comparison requires exactly matched patient_group rows")
    candidate_metrics = patient_metrics(merged["label"], merged["probability_candidate"], decisions=merged["decision_candidate"], include_extended=False)
    comparator_metrics = patient_metrics(merged["label"], merged["probability_comparator"], decisions=merged["decision_comparator"], include_extended=False)
    return {name: float(candidate_metrics[name]) - float(comparator_metrics[name]) for name in METRICS}


def _curve_rows(frame: pd.DataFrame, group_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for values, current in frame.groupby(group_columns, sort=True):
        values_tuple = values if isinstance(values, tuple) else (values,)
        base = dict(zip(group_columns, values_tuple))
        by_budget = {int(budget): data.copy() for budget, data in current.groupby("budget", sort=True)}
        for budget, data in by_budget.items():
            metric_rows.append({**base, "budget": budget, **patient_metrics(data["label"], data["probability"], decisions=data["decision"], include_extended=False)})
        if 1 in by_budget and 4 in by_budget:
            delta_rows.append({**base, "comparison": "K4_minus_K1", **_point_difference(by_budget[4], by_budget[1])})
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def _direction_counts(values: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="raise")
    return {"positive": int((numeric > 0.0).sum()), "zero": int((numeric == 0.0).sum()), "negative": int((numeric < 0.0).sum()), "total": int(len(numeric))}


def _correction_transitions(candidate: pd.DataFrame, comparator: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = ["patient_group", "label", "probability", "decision_threshold", "decision", "outer_fold", "backbone"]
    merged = candidate[columns].merge(comparator[columns], on=["patient_group", "label"], suffixes=("_k4", "_k1"), validate="one_to_one")
    if merged["outer_fold_k4"].astype(int).ne(merged["outer_fold_k1"].astype(int)).any():
        raise ValueError("K1 and K4 correction rows must come from the same outer fold")
    if merged["backbone_k4"].astype(str).ne(merged["backbone_k1"].astype(str)).any():
        raise ValueError("K1 and K4 correction rows must use the same selected backbone")
    merged["decision_k4"] = merged["decision_k4"].astype(bool)
    merged["decision_k1"] = merged["decision_k1"].astype(bool)
    transitions: list[str] = []
    for row in merged.itertuples(index=False):
        if int(row.label) == 1:
            if row.decision_k1 and row.decision_k4:
                transitions.append("both_correct")
            elif not row.decision_k1 and row.decision_k4:
                transitions.append("K1_false_negative_K4_corrected")
            elif row.decision_k1 and not row.decision_k4:
                transitions.append("K1_correct_K4_false_negative")
            else:
                transitions.append("both_false_negative")
        else:
            if not row.decision_k1 and not row.decision_k4:
                transitions.append("both_correct")
            elif not row.decision_k1 and row.decision_k4:
                transitions.append("K4_added_false_positive")
            elif row.decision_k1 and not row.decision_k4:
                transitions.append("K4_corrected_K1_false_positive")
            else:
                transitions.append("both_false_positive")
    merged["transition"] = transitions
    merged["patient_class"] = np.where(merged["label"].eq(1), "malignant", "benign")
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for patient_class, current in merged.groupby("patient_class", sort=True):
        denominator = int(len(current))
        summary[str(patient_class)] = {
            str(transition): {"count": int(count), "proportion": float(count / denominator)}
            for transition, count in current["transition"].value_counts().sort_index().items()
        }
    merged = merged.rename(columns={
        "probability_k4": "probability_K4",
        "probability_k1": "probability_K1",
        "decision_threshold_k4": "threshold_K4",
        "decision_threshold_k1": "threshold_K1",
        "decision_k4": "decision_K4",
        "decision_k1": "decision_K1",
        "outer_fold_k4": "outer_fold",
        "backbone_k4": "backbone",
    }).drop(columns=["outer_fold_k1", "backbone_k1"])
    return merged.sort_values("patient_group").reset_index(drop=True), summary


def write_summary(protocol_root: Path, output_root: Path, n_bootstrap: int, seed: int, force: bool) -> dict:
    all_oof: list[pd.DataFrame] = []
    all_seed_oof: list[pd.DataFrame] = []
    contexts: dict[str, dict] = {}
    lock_contract = None
    locked_contract = _load_json(protocol_root / "protocol_lock.json")["contract"]
    aggregations = [str(value) for value in locked_contract["aggregation"]["inference"]]
    for aggregation in aggregations:
        oof, context = collect_oof(protocol_root, output_root, aggregation)
        all_oof.append(oof)
        all_seed_oof.append(context["seed_oof"])
        contexts[aggregation] = context
        lock_contract = context["contract"]
    if lock_contract is None:
        raise RuntimeError("no OOF prediction family was collected")
    contract = lock_contract
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof": summary_root / "lidmerge_oof_patient_predictions.csv",
        "report": summary_root / "lidmerge_summary.json",
        "transition_rows": summary_root / "lidmerge_mean_K4_vs_K1_patient_correction_transitions.csv",
        "budget_curve_ci": summary_root / "lidmerge_budget_curve_patient_ci.csv",
        "seed_metrics": summary_root / "lidmerge_seed_budget_metrics.csv",
        "seed_deltas": summary_root / "lidmerge_seed_K4_minus_K1.csv",
        "fold_metrics": summary_root / "lidmerge_outer_fold_budget_metrics.csv",
        "fold_deltas": summary_root / "lidmerge_outer_fold_K4_minus_K1.csv",
        "direction": summary_root / "lidmerge_direction_consistency.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise RuntimeError(f"summary outputs already exist; use --force only to rebuild them: {summary_root}")
    oof = pd.concat(all_oof, ignore_index=True).sort_values(["variant", "budget", "patient_group"]).reset_index(drop=True)
    seed_oof = pd.concat(all_seed_oof, ignore_index=True).sort_values(["variant", "seed", "budget", "patient_group"]).reset_index(drop=True)
    by_variant = {aggregation: _metrics_by_budget(oof[oof["variant"].eq(aggregation)]) for aggregation in aggregations}
    ci_by_variant = {
        aggregation: _metric_confidence_intervals_by_budget(oof[oof["variant"].eq(aggregation)], n_bootstrap, int(seed) + 1000 * index)
        for index, aggregation in enumerate(aggregations, start=1)
    }
    budget_curve_ci = _budget_curve_ci_rows(ci_by_variant)
    comparisons: dict[str, dict] = {}
    primary = contract["comparisons"]["primary"]
    candidate = oof[(oof["variant"].eq(primary["candidate"]["variant"])) & (oof["budget"].eq(int(primary["candidate"]["budget"])))]
    comparator = oof[(oof["variant"].eq(primary["comparator"]["variant"])) & (oof["budget"].eq(int(primary["comparator"]["budget"])))]
    comparisons["primary_mean_K4_minus_K1"] = paired_bootstrap(
        candidate,
        comparator,
        n_bootstrap,
        seed,
        inferential_metrics={"standardized_pauc_at_fpr_0_10"},
    )
    incremental_names: list[str] = []
    for offset, item in enumerate(contract["comparisons"]["secondary_incremental"], start=1):
        name = f"secondary_mean_K{item['candidate_budget']}_minus_K{item['comparator_budget']}"
        left = oof[(oof["variant"].eq(item["variant"])) & (oof["budget"].eq(int(item["candidate_budget"])))]
        right = oof[(oof["variant"].eq(item["variant"])) & (oof["budget"].eq(int(item["comparator_budget"])))]
        comparisons[name] = paired_bootstrap(left, right, n_bootstrap, seed + offset, inferential_metrics={"sensitivity"})
        incremental_names.append(name)
    for offset, budget in enumerate(contract["comparisons"]["secondary_aggregation_budgets"], start=101):
        mean_rows = oof[(oof["variant"].eq("mean")) & (oof["budget"].eq(int(budget)))]
        for variant_index, variant in enumerate(aggregations[1:], start=0):
            right = oof[(oof["variant"].eq(variant)) & (oof["budget"].eq(int(budget)))]
            comparisons[f"exploratory_mean_minus_{variant}_K{budget}"] = paired_bootstrap(
                mean_rows,
                right,
                n_bootstrap,
                seed + offset + 10 * variant_index,
            )
    incremental_p_values = {
        name: float(comparisons[name]["metrics"]["sensitivity"]["bootstrap_p_two_sided"])
        for name in incremental_names
    }
    adjusted = _holm_adjust(incremental_p_values)
    for result in comparisons.values():
        for metric in result["metrics"].values():
            metric["holm_adjusted_p"] = None
    for name, value in adjusted.items():
        comparisons[name]["metrics"]["sensitivity"]["holm_adjusted_p"] = value
    primary_candidate = oof[(oof["variant"].eq("mean")) & (oof["budget"].eq(4))]
    primary_comparator = oof[(oof["variant"].eq("mean")) & (oof["budget"].eq(1))]
    transition_rows, transitions = _correction_transitions(primary_candidate, primary_comparator)
    seed_metrics, seed_deltas = _curve_rows(seed_oof, ["variant", "seed"])
    fold_metrics, fold_deltas = _curve_rows(oof, ["variant", "outer_fold"])
    direction = {
        "primary_mean_K4_minus_K1_sensitivity": {
            "per_seed": _direction_counts(seed_deltas[seed_deltas["variant"].eq("mean")]["sensitivity"]),
            "per_outer_fold_seed_ensemble": _direction_counts(fold_deltas[fold_deltas["variant"].eq("mean")]["sensitivity"]),
        },
        "interpretation": "Positive, zero, and negative counts are descriptive direction-consistency evidence. They do not replace the prespecified patient-level primary comparison or make outer folds and seeds independent statistical samples.",
    }
    oof.to_csv(paths["oof"], index=False)
    transition_rows.to_csv(paths["transition_rows"], index=False)
    budget_curve_ci.to_csv(paths["budget_curve_ci"], index=False)
    seed_metrics.to_csv(paths["seed_metrics"], index=False)
    seed_deltas.to_csv(paths["seed_deltas"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    fold_deltas.to_csv(paths["fold_deltas"], index=False)
    paths["direction"].write_text(json.dumps(direction, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "status": "complete",
        "patient_level_only": True,
        "primary_cohort_patient_groups": int(contract["data"]["expected_primary_groups"]),
        "photographic_budgets": [int(value) for value in contract["views"]["budgets"]],
        "aggregation_rule": contract["aggregation"]["rule"],
        "metric_by_variant_and_budget": by_variant,
        "metric_ci_by_variant_and_budget": ci_by_variant,
        "budget_curve_ci_artifact": str(paths["budget_curve_ci"]),
        "metric_definitions": {
            "standardized_pauc_at_fpr_0_10": "Standardized partial AUROC over false-positive rates from 0 to 0.10, computed from pooled outer-fold patient predictions.",
            "sensitivity": "Achieved outer-test sensitivity at the aggregation- and K-specific threshold selected only from the matched inner-fold seed ensemble under a 90% specificity constraint.",
            "specificity": "Achieved outer-test specificity at that fixed inner-selected threshold.",
            "fnr": "Malignant false-negative rate at that fixed inner-selected threshold.",
            "ece_10": "Ten-bin expected calibration error, reported as an exploratory calibration metric.",
        },
        "comparisons": comparisons,
        "multiplicity": contract["comparisons"]["multiplicity"],
        "mean_K4_vs_K1_correction_transitions": transitions,
        "correction_transition_interpretation": "K1 and K4 decisions use their own aggregation- and K-specific thresholds locked from matched inner-fold seed-ensemble predictions. Categories therefore describe prespecified operating-point changes, not a same-threshold causal attribution.",
        "supplementary_curve_artifacts": {key: str(path) for key, path in paths.items() if key not in {"oof", "report"}},
        "bootstrap_replicates": int(n_bootstrap),
        "protocol_manifest_sha256": contexts["mean"]["lock"]["manifest_sha256"],
        "contract_sha256": contexts["mean"]["lock"]["contract_sha256"],
        "interpretation_boundary": "The K=1..4 curve is an internal estimate for this fixed cohort and blinded ordering rule. A non-significant or imprecise increment provides no evidence of additional benefit under this protocol; it does not prove performance saturation, equivalence, or a clinical acquisition budget.",
    }
    paths["report"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"oof_path": str(paths["oof"]), "summary_path": str(paths["report"]), "summary": payload}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_root = Path(args.protocol_root).resolve()
    contract = _load_json(protocol_root / "protocol_lock.json")["contract"]
    n_bootstrap = int(args.n_bootstrap) if args.n_bootstrap is not None else int(contract["validation"]["bootstrap_replicates"])
    if n_bootstrap < 100:
        raise ValueError("n-bootstrap must be at least 100")
    result = write_summary(protocol_root, Path(args.output_root).resolve(), n_bootstrap, int(args.seed), bool(args.force))
    print(json.dumps({"status": "complete", "oof_path": result["oof_path"], "summary_path": result["summary_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
