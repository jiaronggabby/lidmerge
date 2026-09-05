#!/usr/bin/env python3
"""Summarize LidMerge's predeclared cross-architecture K-budget replication.

The selected primary outer route supplies one architecture result in each fold;
only the two nonselected architectures require supporting outer routes.  Every
route uses the same mean-trained image model and this report uses mean
aggregation only.  It is descriptive replication, not an additional family of
significance tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lidpair.validate import validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _route_path(output_root: Path, stage: str, outer_fold: int, seed: int, backbone: str, training_route_name: str) -> Path:
    return output_root / stage / f"outer_{outer_fold}" / f"seed_{seed}" / f"{backbone}__{training_route_name}" / "predictions.csv"


def _selected_backbone(output_root: Path, lock: dict, outer_fold: int) -> str:
    path = output_root / "selection" / f"outer_{outer_fold}" / "mean_selection.json"
    selection = _load_json(path)
    if (
        selection.get("selection_kind") != "variant_budget_thresholds"
        or int(selection.get("outer_fold", -1)) != int(outer_fold)
        or str(selection.get("variant")) != "mean"
        or str(selection.get("training_aggregation")) != "mean"
        or selection.get("protocol_manifest_sha256") != lock["manifest_sha256"]
        or selection.get("contract_sha256") != lock["contract_sha256"]
    ):
        raise ValueError(f"invalid primary mean selection lock: {path}")
    return str(selection["backbone"])


def _validate_prediction(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    expected_stage: str,
    outer_fold: int,
    seed: int,
    backbone: str,
    budgets: list[int],
    source: Path,
) -> pd.DataFrame:
    required = {
        "patient_group", "label", "budget", "probability", "stage", "outer_fold",
        "inner_fold", "seed", "backbone", "variant", "training_aggregation",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"{source} lacks architecture-replication prediction columns: {sorted(missing)}")
    current = frame[frame["variant"].astype(str).eq("mean")].copy()
    if current.empty:
        raise ValueError(f"{source} lacks mean aggregation predictions")
    current["patient_group"] = current["patient_group"].astype(str)
    current["label"] = pd.to_numeric(current["label"], errors="raise").astype(int)
    current["budget"] = pd.to_numeric(current["budget"], errors="raise").astype(int)
    current["probability"] = pd.to_numeric(current["probability"], errors="raise")
    if set(current["budget"]) != set(budgets):
        raise ValueError(f"{source} does not contain every locked K budget")
    counts = current.groupby("budget")["patient_group"].agg(["size", "nunique"])
    if not counts["size"].eq(len(expected)).all() or not counts["nunique"].eq(len(expected)).all():
        raise ValueError(f"{source} does not contain one held-out patient per K")
    if current.duplicated(["patient_group", "budget"]).any():
        raise ValueError(f"{source} has duplicate patient-level K predictions")
    if not current["stage"].astype(str).eq(expected_stage).all():
        raise ValueError(f"{source} does not have the expected {expected_stage} route provenance")
    if not current["outer_fold"].astype(int).eq(int(outer_fold)).all() or not current["inner_fold"].astype(int).eq(-1).all():
        raise ValueError(f"{source} has invalid held-out-fold provenance")
    if (
        not current["seed"].astype(int).eq(int(seed)).all()
        or not current["backbone"].astype(str).eq(str(backbone)).all()
        or not current["training_aggregation"].astype(str).eq("mean").all()
    ):
        raise ValueError(f"{source} has inconsistent route provenance")
    if current["probability"].isna().any() or ((current["probability"] < 0.0) | (current["probability"] > 1.0)).any():
        raise ValueError(f"{source} contains invalid probabilities")
    labels = current[["patient_group", "label"]].drop_duplicates().merge(expected, on="patient_group", suffixes=("_run", "_lock"), validate="one_to_one")
    if set(labels["patient_group"]) != set(expected["patient_group"]) or labels["label_run"].ne(labels["label_lock"]).any():
        raise ValueError(f"{source} labels differ from the locked outer patient groups")
    return current


def _ranking_metrics(labels: np.ndarray, probabilities: np.ndarray, metrics: list[str]) -> dict[str, float | int]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    values: dict[str, float | int] = {
        "n_patients": int(len(labels)),
        "n_malignant": int((labels == 1).sum()),
        "n_benign": int((labels == 0).sum()),
    }
    if "auroc" in metrics:
        values["auroc"] = float(roc_auc_score(labels, probabilities))
    if "auprc" in metrics:
        values["auprc"] = float(average_precision_score(labels, probabilities))
    if "brier" in metrics:
        values["brier"] = float(brier_score_loss(labels, probabilities))
    return values


def _paired_ranking_bootstrap(candidate: pd.DataFrame, comparator: pd.DataFrame, metrics: list[str], n_bootstrap: int, seed: int) -> dict:
    columns = ["patient_group", "label", "probability"]
    merged = candidate[columns].merge(comparator[columns], on=["patient_group", "label"], suffixes=("_candidate", "_comparator"), validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(comparator):
        raise ValueError("architecture replication requires exactly matched patient_group predictions")
    labels = merged["label"].to_numpy(dtype=int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("architecture replication bootstrap requires at least two patients in each class")
    candidate_probability = merged["probability_candidate"].to_numpy(dtype=float)
    comparator_probability = merged["probability_comparator"].to_numpy(dtype=float)
    point_left = _ranking_metrics(labels, candidate_probability, metrics)
    point_right = _ranking_metrics(labels, comparator_probability, metrics)
    point = {metric: float(point_left[metric]) - float(point_right[metric]) for metric in metrics}
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_bootstrap)):
        index = np.concatenate([rng.choice(positive, size=len(positive), replace=True), rng.choice(negative, size=len(negative), replace=True)])
        left = _ranking_metrics(labels[index], candidate_probability[index], metrics)
        right = _ranking_metrics(labels[index], comparator_probability[index], metrics)
        for metric in metrics:
            draws[metric].append(float(left[metric]) - float(right[metric]))
    return {
        "n_patients": int(len(merged)),
        "metrics": {
            metric: {
                "difference": point[metric],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
            for metric, values in draws.items()
        },
    }


def collect_architecture_oof(protocol_root: Path, output_root: Path) -> tuple[pd.DataFrame, dict]:
    audit = validate_protocol(protocol_root, check_paths=False)
    lock = _load_json(protocol_root / "protocol_lock.json")
    contract = lock["contract"]
    configuration = contract["architecture_robustness"]
    backbones = [str(value) for value in configuration["backbones"]]
    seeds = [int(value) for value in configuration["seeds"]]
    budgets = [int(value) for value in contract["views"]["budgets"]]
    training_route_name = str(contract["aggregation"]["training_route_name"])
    outer = pd.read_csv(protocol_root / "outer_folds.csv", dtype={"patient_group": str})
    rows: list[pd.DataFrame] = []
    route_sources: dict[str, str] = {}
    for outer_fold in range(int(contract["validation"]["outer_folds"])):
        selected = _selected_backbone(output_root, lock, outer_fold)
        if selected not in backbones:
            raise ValueError(f"selected backbone lies outside the predeclared architecture family: {selected}")
        expected = outer[outer["outer_fold"].astype(int).eq(outer_fold)][["patient_group", "label"]].copy()
        for backbone in backbones:
            stage = "outer" if backbone == selected else "architecture_robustness"
            seed_frames: list[pd.DataFrame] = []
            for seed in seeds:
                source = _route_path(output_root, stage, outer_fold, seed, backbone, training_route_name)
                current = _validate_prediction(pd.read_csv(source, dtype={"patient_group": str}), expected, stage, outer_fold, seed, backbone, budgets, source)
                seed_frames.append(current[["patient_group", "label", "budget", "probability"]].rename(columns={"probability": f"probability_seed_{seed}"}))
            merged = seed_frames[0]
            for current in seed_frames[1:]:
                merged = merged.merge(current, on=["patient_group", "label", "budget"], validate="one_to_one")
            probability_columns = [column for column in merged.columns if column.startswith("probability_seed_")]
            merged["probability"] = merged[probability_columns].mean(axis=1)
            merged["outer_fold"] = int(outer_fold)
            merged["backbone"] = backbone
            rows.append(merged[["patient_group", "label", "budget", "outer_fold", "backbone", "probability"]])
            route_sources[f"outer_{outer_fold}:{backbone}"] = stage
    oof = pd.concat(rows, ignore_index=True).sort_values(["backbone", "budget", "patient_group"]).reset_index(drop=True)
    expected_count = int(contract["data"]["expected_primary_groups"])
    counts = oof.groupby(["backbone", "budget"])["patient_group"].agg(["size", "nunique"])
    if not counts["size"].eq(expected_count).all() or not counts["nunique"].eq(expected_count).all():
        raise ValueError("architecture-replication OOF predictions are incomplete")
    return oof, {"audit": audit, "lock": lock, "contract": contract, "configuration": configuration, "route_sources": route_sources}


def write_architecture_robustness_summary(protocol_root: Path, output_root: Path, n_bootstrap: int, seed: int, force: bool) -> dict:
    oof, context = collect_architecture_oof(protocol_root, output_root)
    configuration = context["configuration"]
    metrics = [str(value) for value in configuration["metrics"]]
    summary_root = output_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    oof_path = summary_root / "lidmerge_backbone_robustness_oof_patient_predictions.csv"
    report_path = summary_root / "lidmerge_backbone_robustness_summary.json"
    table_path = summary_root / "lidmerge_backbone_robustness_comparisons.csv"
    if (oof_path.exists() or report_path.exists() or table_path.exists()) and not force:
        raise RuntimeError(f"architecture-replication outputs already exist; use --force to rebuild: {summary_root}")
    metrics_by_backbone_and_budget = {
        str(backbone): {
            str(int(budget)): _ranking_metrics(current["label"].to_numpy(dtype=int), current["probability"].to_numpy(dtype=float), metrics)
            for budget, current in oof[oof["backbone"].eq(str(backbone))].groupby("budget", sort=True)
        }
        for backbone in configuration["backbones"]
    }
    comparison = configuration["comparison"]
    candidate_budget = int(comparison["candidate_budget"])
    comparator_budget = int(comparison["comparator_budget"])
    comparisons: dict[str, dict] = {}
    table_rows: list[dict[str, object]] = []
    for offset, backbone in enumerate(configuration["backbones"]):
        current = oof[oof["backbone"].eq(str(backbone))]
        result = _paired_ranking_bootstrap(
            current[current["budget"].eq(candidate_budget)],
            current[current["budget"].eq(comparator_budget)],
            metrics,
            n_bootstrap,
            seed + offset,
        )
        comparisons[str(backbone)] = result
        for metric, values in result["metrics"].items():
            table_rows.append({"backbone": backbone, "comparison": f"K{candidate_budget}_minus_K{comparator_budget}", "metric": metric, "n_patients": result["n_patients"], **values})
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(table_rows).sort_values(["backbone", "metric"]).to_csv(table_path, index=False)
    payload = {
        "status": "complete",
        "supporting_analysis_only": True,
        "selection_allowed": False,
        "thresholds_used": False,
        "patient_level_only": True,
        "architecture_robustness": configuration,
        "route_sources": context["route_sources"],
        "metrics_by_backbone_and_budget": metrics_by_backbone_and_budget,
        "paired_K4_minus_K1_by_backbone": comparisons,
        "bootstrap_replicates": int(n_bootstrap),
        "interpretation": "This predeclared supporting analysis reports CIs for whether the direction of the K=1-to-4 curve reproduces under the three fixed architecture families. It does not choose a preferred architecture, alter the primary selected-backbone comparison, or add a separate multiplicity-adjusted superiority claim.",
        "protocol_manifest_sha256": context["lock"]["manifest_sha256"],
        "contract_sha256": context["lock"]["contract_sha256"],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"oof_path": str(oof_path), "summary_path": str(report_path), "comparison_table": str(table_path), "summary": payload}


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
    lock = _load_json(Path(args.protocol_root).resolve() / "protocol_lock.json")
    n_bootstrap = int(args.n_bootstrap) if args.n_bootstrap is not None else int(lock["contract"]["validation"]["bootstrap_replicates"])
    result = write_architecture_robustness_summary(Path(args.protocol_root).resolve(), Path(args.output_root).resolve(), n_bootstrap, int(args.seed), bool(args.force))
    print(json.dumps({"status": "complete", "oof_path": result["oof_path"], "summary_path": result["summary_path"], "comparison_table": result["comparison_table"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
