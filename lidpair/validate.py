#!/usr/bin/env python3
"""Validate the frozen LidMerge protocol before any training route runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from lidpair import METHOD_ID, PROTOCOL_VERSION
from lidpair.prepare import CONTRACT_PATH, group_summary, load_manifest, read_contract, sha256_file
from lidpair.runtime import required_runtime_image_ids, resolve_runtime_manifest


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_locked_manifest_path(lock: dict, protocol_root: Path) -> Path:
    reference = Path(str(lock["manifest_relative_to_protocol_root"]))
    if reference.is_absolute():
        raise ValueError("protocol lock stores an absolute manifest reference")
    candidate = (protocol_root / reference).resolve()
    expected = Path(lock["manifest_relative_to_protocol_root"]).as_posix()
    actual = Path(os.path.relpath(candidate, protocol_root.resolve())).as_posix()
    if actual != expected:
        raise ValueError("protocol manifest reference does not resolve safely under the protocol root")
    return candidate


def _require_exact_columns(frame: pd.DataFrame, expected: list[str], source: Path) -> None:
    actual = list(frame.columns)
    if actual != expected:
        raise ValueError(f"{source.name} columns differ from lock contract: {actual}")


def _validate_orders(
    orders: pd.DataFrame,
    manifest: pd.DataFrame,
    primary: pd.DataFrame,
    max_budget: int,
    expected_seed: int,
    source: Path,
) -> None:
    _require_exact_columns(orders, ["patient_group", "label", "photograph_order_seed", "rank", "image_id"], source)
    orders["patient_group"] = orders["patient_group"].astype(str)
    orders["image_id"] = orders["image_id"].astype(str)
    orders["label"] = pd.to_numeric(orders["label"], errors="raise").astype(int)
    orders["rank"] = pd.to_numeric(orders["rank"], errors="raise").astype(int)
    if not orders["photograph_order_seed"].astype(int).eq(int(expected_seed)).all():
        raise ValueError(f"{source.name} has a wrong photograph-order seed")
    if set(orders["patient_group"]) != set(primary["patient_group"].astype(str)):
        raise ValueError(f"{source.name} does not cover exactly the primary patient groups")
    counts = orders.groupby("patient_group", sort=False).size()
    if not counts.eq(int(max_budget)).all():
        raise ValueError(f"{source.name} must contain exactly {max_budget} frozen photographs per patient_group")
    expected_ranks = list(range(1, int(max_budget) + 1))
    if any(sorted(current["rank"].tolist()) != expected_ranks for _, current in orders.groupby("patient_group", sort=False)):
        raise ValueError(f"{source.name} ranks are not the required nested prefixes")
    if orders.duplicated(["patient_group", "rank"]).any() or orders.duplicated(["patient_group", "image_id"]).any():
        raise ValueError(f"{source.name} repeats a rank or photograph within a patient_group")
    expected_labels = primary.set_index("patient_group")["label"].astype(int)
    observed = orders.groupby("patient_group")["label"].agg(["nunique", "first"])
    if observed["nunique"].ne(1).any() or any(int(observed.loc[group, "first"]) != int(expected_labels.loc[group]) for group in expected_labels.index):
        raise ValueError(f"{source.name} labels differ from primary cohort labels")
    source_pairs = manifest[["patient_group", "image_id"]].astype(str)
    checked = orders[["patient_group", "image_id"]].merge(source_pairs, on=["patient_group", "image_id"], how="left", indicator=True)
    if not checked["_merge"].eq("both").all():
        raise ValueError(f"{source.name} contains a photograph outside its patient_group")


def _validate_folds(primary: pd.DataFrame, outer: pd.DataFrame, inner: pd.DataFrame, contract: dict) -> None:
    _require_exact_columns(outer, ["patient_group", "label", "outer_fold"], Path("outer_folds.csv"))
    outer["patient_group"] = outer["patient_group"].astype(str)
    outer["label"] = pd.to_numeric(outer["label"], errors="raise").astype(int)
    outer["outer_fold"] = pd.to_numeric(outer["outer_fold"], errors="raise").astype(int)
    expected_groups = set(primary["patient_group"].astype(str))
    if outer["patient_group"].duplicated().any() or set(outer["patient_group"]) != expected_groups:
        raise ValueError("outer folds do not represent exactly one row per primary patient_group")
    n_outer = int(contract["validation"]["outer_folds"])
    if not outer["outer_fold"].between(0, n_outer - 1).all():
        raise ValueError("outer folds contain an invalid fold index")
    expected_labels = primary.set_index("patient_group")["label"].astype(int)
    if any(int(outer.set_index("patient_group").loc[group, "label"]) != int(expected_labels.loc[group]) for group in expected_labels.index):
        raise ValueError("outer-fold labels differ from primary cohort labels")
    _require_exact_columns(inner, ["patient_group", "label", "outer_fold", "protocol_outer_fold", "inner_fold"], Path("inner_folds.csv"))
    inner["patient_group"] = inner["patient_group"].astype(str)
    for column in ("label", "outer_fold", "protocol_outer_fold", "inner_fold"):
        inner[column] = pd.to_numeric(inner[column], errors="raise").astype(int)
    n_inner = int(contract["validation"]["inner_folds"])
    if len(inner) != len(primary) * n_outer:
        raise ValueError("inner folds must have one row per patient_group for every protocol outer fold")
    for outer_fold in range(n_outer):
        current = inner[inner["protocol_outer_fold"].eq(outer_fold)].copy()
        if current["patient_group"].duplicated().any() or set(current["patient_group"]) != expected_groups:
            raise ValueError(f"inner folds are incomplete for protocol outer fold {outer_fold}")
        merged = current.merge(outer, on="patient_group", suffixes=("_inner", "_outer"), validate="one_to_one")
        if merged["label_inner"].ne(merged["label_outer"]).any() or merged["outer_fold_inner"].ne(merged["outer_fold_outer"]).any():
            raise ValueError(f"inner folds disagree with outer lock for outer fold {outer_fold}")
        test = current["outer_fold"].eq(outer_fold)
        if not current.loc[test, "inner_fold"].eq(-1).all():
            raise ValueError(f"outer-test patient_group appears in an inner fold for outer fold {outer_fold}")
        if not current.loc[~test, "inner_fold"].between(0, n_inner - 1).all():
            raise ValueError(f"inner-fold index is invalid for outer fold {outer_fold}")


def _validate_architecture_robustness(contract: dict) -> None:
    configuration = contract.get("architecture_robustness")
    if not isinstance(configuration, dict):
        raise ValueError("LidMerge contract lacks architecture_robustness configuration")
    screen = [str(value) for value in contract.get("backbone_screen", [])]
    backbones = [str(value) for value in configuration.get("backbones", [])]
    primary_backbone = str(contract.get("primary_backbone", ""))
    if screen != [primary_backbone]:
        raise ValueError("the corrected protocol must use its single declared primary backbone")
    if len(backbones) < 3 or len(set(backbones)) != len(backbones) or primary_backbone not in backbones:
        raise ValueError("architecture robustness must contain the primary backbone and at least two support architectures")
    aggregation = contract.get("aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError("LidMerge contract lacks aggregation configuration")
    if str(aggregation.get("training")) != "mean" or str(configuration.get("aggregation")) != "mean":
        raise ValueError("architecture robustness must use the declared mean-training aggregation")
    if [str(value) for value in aggregation.get("inference", [])] != ["mean", "top2_mean", "max"]:
        raise ValueError("LidMerge must declare mean, top2_mean, and max as shared-logit inference aggregations")
    if str(aggregation.get("training_route_name", "")).strip() != "mean_train":
        raise ValueError("LidMerge requires the explicit mean_train route namespace")
    declared_seeds = [int(value) for value in configuration.get("seeds", [])]
    formal_seeds = [int(value) for value in contract.get("validation", {}).get("formal_seeds", [])]
    if declared_seeds != formal_seeds:
        raise ValueError("architecture robustness must use the declared formal seed ensemble")
    comparison = configuration.get("comparison", {})
    budgets = {int(value) for value in contract.get("views", {}).get("budgets", [])}
    if {int(comparison.get("candidate_budget", -1)), int(comparison.get("comparator_budget", -1))} != {1, 4} or not {1, 4}.issubset(budgets):
        raise ValueError("architecture robustness must compare the locked K=4 and K=1 budgets")
    if [str(value) for value in configuration.get("metrics", [])] != ["auroc", "auprc", "brier"]:
        raise ValueError("architecture robustness metrics must be the locked threshold-free set")


def _validate_aggregation_and_comparisons(contract: dict) -> None:
    aggregation = contract.get("aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError("LidMerge contract lacks aggregation configuration")
    if str(aggregation.get("training")) != "mean":
        raise ValueError("LidMerge must train the shared image model with mean-logit BCE only")
    if str(aggregation.get("training_route_name")) != "mean_train":
        raise ValueError("LidMerge must store the single training route under mean_train")
    if [str(value) for value in aggregation.get("inference", [])] != ["mean", "top2_mean", "max"]:
        raise ValueError("LidMerge must derive mean, top2_mean, and max from shared image logits")
    if str(aggregation.get("image_logit_artifact")) != "image_logits.csv":
        raise ValueError("LidMerge must preserve shared per-image logits in image_logits.csv")
    pilot = contract.get("pilot", {})
    if str(pilot.get("training_aggregation")) != "mean" or [str(value) for value in pilot.get("inference_aggregations", [])] != ["mean", "top2_mean", "max"]:
        raise ValueError("pilot must run one shared mean-training route and report all fixed aggregations")
    comparisons = contract.get("comparisons")
    if not isinstance(comparisons, dict):
        raise ValueError("LidMerge contract lacks comparison configuration")
    primary = comparisons.get("primary", {})
    if primary.get("candidate") != {"variant": "mean", "budget": 4} or primary.get("comparator") != {"variant": "mean", "budget": 1}:
        raise ValueError("primary comparison must remain mean K4 versus K1")
    incremental = comparisons.get("secondary_incremental", [])
    expected_incremental = [
        {"variant": "mean", "candidate_budget": 2, "comparator_budget": 1},
        {"variant": "mean", "candidate_budget": 3, "comparator_budget": 2},
        {"variant": "mean", "candidate_budget": 4, "comparator_budget": 3},
    ]
    if incremental != expected_incremental:
        raise ValueError("secondary incremental comparisons must remain the three prespecified mean sensitivity increments")
    if [int(value) for value in comparisons.get("secondary_aggregation_budgets", [])] != [1, 2, 3, 4]:
        raise ValueError("mean/max exploratory comparisons must cover the same K=1..4 curve")
    expected_pointwise_metrics = ["standardized_pauc_at_fpr_0_10", "sensitivity", "specificity", "auroc", "auprc", "fnr", "brier", "ece_10"]
    if [str(value) for value in comparisons.get("pointwise_ci_metrics", [])] != expected_pointwise_metrics:
        raise ValueError("pointwise K-budget confidence intervals must include the locked diagnostic and calibration metrics")


def _validate_cpu_logit_postprocessing(contract: dict) -> None:
    """Validate the declared, training-free learned aggregation analysis."""

    configuration = contract.get("cpu_logit_postprocessing")
    if not isinstance(configuration, dict):
        raise ValueError("LidMerge contract lacks the CPU logit postprocessing configuration")
    if str(configuration.get("status")) != "predeclared_secondary_analysis":
        raise ValueError("CPU logit postprocessing must remain a predeclared secondary analysis")
    if str(configuration.get("input_artifact")) != "image_logits.csv":
        raise ValueError("CPU logit postprocessing must use the saved image_logits.csv artifact")
    if str(configuration.get("source_backbone")) != str(contract.get("primary_backbone")):
        raise ValueError("CPU logit postprocessing must use the fixed primary backbone logits")
    if str(configuration.get("source_training_route_name")) != "mean_train":
        raise ValueError("CPU logit postprocessing must use the shared mean_train route")
    budgets = [int(value) for value in contract.get("views", {}).get("budgets", [])]
    if [int(value) for value in configuration.get("budgets", [])] != budgets:
        raise ValueError("CPU logit postprocessing budgets must match the locked K=1..K=4 budgets")
    if [str(value) for value in configuration.get("aggregators", [])] != ["xgboost", "random_forest", "logistic_regression"]:
        raise ValueError("CPU logit postprocessing must declare XGBoost, random-forest, and logistic-regression aggregators")
    for name in ("xgboost", "random_forest", "logistic_regression"):
        candidates = configuration.get(name, {}).get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError(f"CPU {name} postprocessing requires a predeclared candidate grid")
        candidate_ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
        if any(not value for value in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"CPU {name} candidate IDs must be present and unique")
    if "three-fold cross-fitted log loss" not in str(configuration.get("nested_rule", "")):
        raise ValueError("CPU logit postprocessing must select configurations using inner cross-fitted log loss")
    if "inner_cross_fitted_patient_specificity_at_least_0.90" != str(configuration.get("threshold_metric")):
        raise ValueError("CPU logit postprocessing threshold must be selected from inner cross-fitted probabilities")


def _validate_counterfactual_evaluation(contract: dict) -> None:
    configuration = contract.get("counterfactual_evaluation")
    if not isinstance(configuration, dict):
        raise ValueError("LidMerge contract lacks counterfactual evaluation configuration")
    expected_conditions = ["true_prefix", "reversed_prefix", "duplicate_view1", "lowpass_view1"]
    if [str(value) for value in configuration.get("conditions", [])] != expected_conditions:
        raise ValueError("counterfactual conditions must match the locked LidMerge controls")
    tolerance = float(configuration.get("invariance_tolerance", 0.0))
    if not 0.0 < tolerance <= 1e-2:
        raise ValueError("counterfactual invariance tolerance must be a small positive value")
    kernel_size = int(configuration.get("lowpass_kernel_size", 0))
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("counterfactual low-pass kernel size must be an odd integer >= 3")


def _validate_training_schedule(contract: dict) -> None:
    training = contract.get("training")
    if not isinstance(training, dict):
        raise ValueError("LidMerge contract lacks training configuration")
    maximum_epochs = int(training.get("diagnostic_epochs", 0))
    if maximum_epochs < 1:
        raise ValueError("diagnostic_epochs must be positive")
    early_stopping = training.get("early_stopping")
    if not isinstance(early_stopping, dict):
        raise ValueError("LidMerge contract lacks early-stopping configuration")
    if not bool(early_stopping.get("enabled_for_inner_routes", False)):
        raise ValueError("inner routes must use the locked validation-only early-stopping rule")
    if str(early_stopping.get("monitor")) != "inner_validation_patient_bce_k1_k4_mean":
        raise ValueError("early stopping must monitor the equal-weight inner-validation K=1..K=4 BCE mean")
    if str(early_stopping.get("mode")) != "min":
        raise ValueError("early-stopping mode must minimize validation BCE")
    cycle_length = int(training.get("cycle_length", 0))
    if cycle_length != 4:
        raise ValueError("the corrected protocol cycle_length must be 4")
    minimum_epochs = int(early_stopping.get("minimum_epochs", 0))
    patience_cycles = int(early_stopping.get("patience_cycles", 0))
    min_delta = float(early_stopping.get("min_delta", -1.0))
    if minimum_epochs < 12 or minimum_epochs > maximum_epochs or minimum_epochs % cycle_length != 0:
        raise ValueError("early-stopping minimum_epochs must be a complete K cycle within diagnostic_epochs and at least 12")
    if patience_cycles < 1 or min_delta < 0.0:
        raise ValueError("early-stopping patience_cycles and min_delta must be non-negative and patience positive")
    if "4 * median(best_cycle)" not in str(early_stopping.get("outer_epoch_rule", "")):
        raise ValueError("outer routes must use the predeclared complete-cycle epoch aggregation rule")


def validate_protocol(protocol_root: Path, image_root: Path | None = None, check_paths: bool = False) -> dict:
    protocol_root = Path(protocol_root).resolve()
    lock = _load_json(protocol_root / "protocol_lock.json")
    contract = read_contract()
    if lock.get("method_id") != METHOD_ID or lock.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol lock is not the active LidMerge implementation")
    if lock.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise ValueError("contract changed after protocol locking; rebuild lidmerge_protocol explicitly")
    if lock.get("contract") != contract:
        raise ValueError("embedded protocol contract differs from the active contract")
    _validate_architecture_robustness(contract)
    _validate_aggregation_and_comparisons(contract)
    _validate_cpu_logit_postprocessing(contract)
    _validate_counterfactual_evaluation(contract)
    _validate_training_schedule(contract)
    manifest_path = resolve_locked_manifest_path(lock, protocol_root)
    if sha256_file(manifest_path) != lock.get("manifest_sha256"):
        raise ValueError("frozen manifest checksum differs from protocol lock")
    manifest = load_manifest(manifest_path, contract, check_paths=False)
    runtime_ids = required_runtime_image_ids(protocol_root) if check_paths else None
    manifest = resolve_runtime_manifest(
        manifest,
        image_root=image_root,
        check_paths=check_paths,
        required_image_ids=runtime_ids,
    )
    summary = group_summary(manifest)
    if len(summary) != int(lock.get("patient_groups")):
        raise ValueError("manifest patient-group total differs from protocol lock")
    primary_path = protocol_root / "primary_patient_groups.csv"
    primary = pd.read_csv(primary_path, dtype={"patient_group": str})
    _require_exact_columns(primary, ["patient_group", "label"], primary_path)
    primary["patient_group"] = primary["patient_group"].astype(str)
    primary["label"] = pd.to_numeric(primary["label"], errors="raise").astype(int)
    if primary["patient_group"].duplicated().any() or len(primary) != int(contract["data"]["expected_primary_groups"]):
        raise ValueError("primary cohort cardinality differs from contract")
    label_counts = {"malignant": int((primary["label"] == 1).sum()), "benign": int((primary["label"] == 0).sum())}
    if label_counts != {key: int(value) for key, value in contract["data"]["expected_primary_label_counts"].items()}:
        raise ValueError("primary patient-level label counts differ from contract")
    manifest_labels = summary.set_index("patient_group")["label"].astype(int)
    if any(group not in manifest_labels.index or int(manifest_labels.loc[group]) != int(label) for group, label in primary[["patient_group", "label"]].itertuples(index=False)):
        raise ValueError("primary cohort contains an invalid patient_group or label")
    maximum_budget = int(contract["views"]["maximum_budget"])
    primary_order_file = protocol_root / str(lock["primary_order_file"])
    if sha256_file(primary_order_file) != lock.get("primary_order_sha256"):
        raise ValueError("primary photograph-order checksum differs from protocol lock")
    primary_orders = pd.read_csv(primary_order_file, dtype={"patient_group": str, "image_id": str})
    _validate_orders(primary_orders, manifest, primary, maximum_budget, int(contract["views"]["primary_order_seed"]), primary_order_file)
    for seed in contract["views"].get("alternate_order_seeds", []):
        filename = lock.get("alternate_order_files", {}).get(str(seed))
        if not filename:
            raise ValueError(f"protocol lock has no alternate order file for seed {seed}")
        path = protocol_root / str(filename)
        if sha256_file(path) != lock.get("alternate_order_sha256", {}).get(str(seed)):
            raise ValueError(f"alternate photograph-order checksum differs for seed {seed}")
        current = pd.read_csv(path, dtype={"patient_group": str, "image_id": str})
        _validate_orders(current, manifest, primary, maximum_budget, int(seed), path)
    outer = pd.read_csv(protocol_root / "outer_folds.csv", dtype={"patient_group": str})
    inner = pd.read_csv(protocol_root / "inner_folds.csv", dtype={"patient_group": str})
    _validate_folds(primary, outer, inner, contract)
    return {
        "status": "passed",
        "method_id": METHOD_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_root": str(protocol_root),
        "manifest_sha256": lock["manifest_sha256"],
        "contract_sha256": lock["contract_sha256"],
        "patient_groups": int(len(summary)),
        "primary_patient_groups": int(len(primary)),
        "primary_label_counts": label_counts,
        "budgets": [int(value) for value in contract["views"]["budgets"]],
        "path_check": bool(check_paths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--image-root", help="mounted canonical image root for operational path validation")
    parser.add_argument("--skip-path-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_protocol(
        Path(args.protocol_root),
        image_root=Path(args.image_root).resolve() if args.image_root else None,
        check_paths=not args.skip_path_check,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
