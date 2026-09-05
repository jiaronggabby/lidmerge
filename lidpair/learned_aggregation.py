#!/usr/bin/env python3
"""Run training-free patient-level aggregation on saved LidMerge image logits.

The image encoder is not retrained here.  For every outer fold, budget, and
formal seed, the module reads the corresponding cross-fitted inner
``image_logits.csv`` files, selects a small predeclared XGBoost, random-forest,
or logistic-regression configuration by inner-fold log loss, refits that
configuration on the outer-training logits, and predicts the held-out outer
patients.  Thresholds are selected from the formal-seed-averaged cross-fitted
inner probabilities.

This is a secondary CPU analysis.  It is deliberately separate from the
locked mean-aggregation primary analysis and fails closed when route
artifacts do not carry the active protocol, cohort, or complete-cycle
provenance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from lidpair.metrics import patient_metrics, select_threshold_at_specificity, stratified_patient_bootstrap
from lidpair.prepare import sha256_file
from lidpair.summarize import paired_bootstrap
from lidpair.validate import validate_protocol


ROOT = Path(__file__).resolve().parents[1]
LEARNED_AGGREGATORS = ("xgboost", "random_forest", "logistic_regression")
BASELINE_AGGREGATORS = ("mean", "top2_mean", "max")
SUMMARY_FILENAME = "learned_aggregation_summary.json"
REQUIRED_LOGIT_COLUMNS = {
    "patient_group",
    "label",
    "maximum_budget",
    "view_rank",
    "photograph_order_seed",
    "training_aggregation",
    "image_logit",
    "stage",
    "outer_fold",
    "inner_fold",
    "seed",
    "backbone",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _positive_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    classes = [int(value) for value in getattr(estimator, "classes_", [])]
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes) or set(classes) != {0, 1}:
        raise ValueError("meta-aggregator must expose probabilities for both binary classes")
    positive_column = classes.index(1)
    values = probabilities[:, positive_column]
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("meta-aggregator produced invalid probabilities")
    return values


class FrozenLogitSource:
    """Read and validate the active primary-route image-logit artifacts."""

    def __init__(
        self,
        protocol_root: Path,
        source_root: Path,
        source_contract_path: Path | None = None,
    ) -> None:
        self.protocol_root = Path(protocol_root).resolve()
        self.source_root = Path(source_root).resolve()
        self.audit = validate_protocol(self.protocol_root, check_paths=False)
        self.lock = _load_json(self.protocol_root / "protocol_lock.json")
        self.contract = self.lock["contract"]
        self.active_contract_sha256 = str(self.lock["contract_sha256"])
        self.source_contract_path = (
            Path(source_contract_path).resolve() if source_contract_path is not None else None
        )
        self.source_contract_sha256 = self.active_contract_sha256
        self.accepted_contract_sha256 = {self.active_contract_sha256}
        if self.source_contract_path is not None:
            source_contract = _load_json(self.source_contract_path)
            active_core_contract = {
                key: value
                for key, value in self.contract.items()
                if key != "cpu_logit_postprocessing"
            }
            if source_contract != active_core_contract:
                raise ValueError(
                    "source contract is not the active contract with only the declared "
                    "CPU postprocessing section removed"
                )
            self.source_contract_sha256 = sha256_file(self.source_contract_path)
            self.accepted_contract_sha256.add(self.source_contract_sha256)
        self.backbone = str(self.contract["cpu_logit_postprocessing"]["source_backbone"])
        self.training_route_name = str(self.contract["cpu_logit_postprocessing"]["source_training_route_name"])
        self.maximum_budget = int(self.contract["views"]["maximum_budget"])
        self.primary_order_seed = int(self.contract["views"]["primary_order_seed"])
        self.outer_count = int(self.contract["validation"]["outer_folds"])
        self.inner_count = int(self.contract["validation"]["inner_folds"])
        self.formal_seeds = [int(value) for value in self.contract["validation"]["formal_seeds"]]
        self.budgets = [int(value) for value in self.contract["views"]["budgets"]]
        self.outer = pd.read_csv(self.protocol_root / "outer_folds.csv", dtype={"patient_group": str})
        self.inner = pd.read_csv(self.protocol_root / "inner_folds.csv", dtype={"patient_group": str})
        self._cache: dict[tuple[str, int, int, int | None], pd.DataFrame] = {}
        self.source_records: dict[str, dict[str, Any]] = {}

    def _route_directory(self, stage: str, outer_fold: int, seed: int, inner_fold: int | None) -> Path:
        path = self.source_root / stage / f"outer_{outer_fold}"
        if stage == "inner":
            if inner_fold is None:
                raise ValueError("inner route requires inner_fold")
            path = path / f"inner_{inner_fold}"
        elif inner_fold is not None:
            raise ValueError("outer route cannot have inner_fold")
        return path / f"seed_{seed}" / f"{self.backbone}__{self.training_route_name}"

    def _expected_groups(self, stage: str, outer_fold: int, inner_fold: int | None) -> pd.DataFrame:
        if stage == "inner":
            assert inner_fold is not None
            expected = self.inner[
                self.inner["protocol_outer_fold"].astype(int).eq(int(outer_fold))
                & self.inner["inner_fold"].astype(int).eq(int(inner_fold))
            ][["patient_group", "label"]].copy()
        else:
            expected = self.outer[self.outer["outer_fold"].astype(int).eq(int(outer_fold))][["patient_group", "label"]].copy()
        expected["patient_group"] = expected["patient_group"].astype(str)
        expected["label"] = pd.to_numeric(expected["label"], errors="raise").astype(int)
        if expected.empty or expected["patient_group"].duplicated().any():
            raise ValueError(f"locked {stage} groups are empty or duplicated for outer={outer_fold}, inner={inner_fold}")
        return expected.sort_values("patient_group").reset_index(drop=True)

    @staticmethod
    def _realized_epochs(manifest: dict[str, Any]) -> int | None:
        memory = manifest.get("memory", {})
        for container in (manifest, memory):
            for key in ("training_epochs_realized", "training_epochs_ran", "training_epochs"):
                value = container.get(key)
                if value is not None:
                    return int(value)
        return None

    def _validate_route_manifest(
        self,
        manifest: dict[str, Any],
        stage: str,
        outer_fold: int,
        inner_fold: int | None,
        seed: int,
        route_directory: Path,
    ) -> None:
        if manifest.get("status") != "complete":
            raise ValueError(f"route is not complete: {route_directory}")
        expected_inner = int(inner_fold) if stage == "inner" and inner_fold is not None else -1
        checks = {
            "stage": stage,
            "outer_fold": int(outer_fold),
            "inner_fold": expected_inner,
            "seed": int(seed),
            "backbone": self.backbone,
            "training_aggregation": "mean",
            "training_route_name": self.training_route_name,
            "protocol_manifest_sha256": self.lock["manifest_sha256"],
        }
        for key, expected in checks.items():
            if key == "inner_fold" and stage == "outer" and manifest.get(key) in {None, -1, "-1"}:
                continue
            if str(manifest.get(key)) != str(expected):
                raise ValueError(f"{route_directory}/run_manifest.json has {key}={manifest.get(key)!r}; expected {expected!r}")
        manifest_contract_sha256 = str(manifest.get("contract_sha256"))
        if manifest_contract_sha256 not in self.accepted_contract_sha256:
            raise ValueError(
                f"{route_directory}/run_manifest.json has contract_sha256={manifest_contract_sha256!r}; "
                f"accepted active/source hashes are {sorted(self.accepted_contract_sha256)!r}"
            )
        epochs = self._realized_epochs(manifest)
        cycle_length = int(self.contract["training"]["cycle_length"])
        minimum_epochs = int(self.contract["training"]["early_stopping"]["minimum_epochs"])
        if epochs is None or epochs < minimum_epochs or epochs % cycle_length != 0:
            raise ValueError(
                f"{route_directory} does not carry a complete-cycle training epoch count: "
                f"realized={epochs}, minimum={minimum_epochs}, cycle_length={cycle_length}"
            )
        memory = manifest.get("memory", {})
        monitor = memory.get("early_stopping_monitor")
        expected_monitor = self.contract["training"]["early_stopping"]["monitor"]
        accepted_monitors = {str(expected_monitor), "validation_patient_bce_k1_k4_mean"}
        if monitor is not None and str(monitor) not in accepted_monitors:
            raise ValueError(f"{route_directory} uses an incompatible early-stopping monitor: {monitor!r}")
        if stage == "inner":
            history_path = route_directory / "diagnostic_history.csv"
            if not history_path.is_file():
                raise FileNotFoundError(f"inner route lacks diagnostic history for cycle verification: {history_path}")
            history = pd.read_csv(history_path)
            required_history_columns = {
                "epoch",
                "validation_patient_bce_k1_mean",
                "validation_patient_bce_k2_mean",
                "validation_patient_bce_k3_mean",
                "validation_patient_bce_k4_mean",
                "validation_patient_bce_k1_k4_mean",
                "checkpoint_cycle",
            }
            missing_history_columns = required_history_columns.difference(history.columns)
            if missing_history_columns:
                raise ValueError(
                    f"{history_path} lacks complete K1-K4 cycle evidence: {sorted(missing_history_columns)}"
                )
            checkpoint_mask = history["checkpoint_cycle"].astype(str).str.strip().str.lower().eq("true")
            checkpoint_rows = history[checkpoint_mask].copy()
            checkpoint_epochs = pd.to_numeric(checkpoint_rows["epoch"], errors="raise").astype(int)
            if checkpoint_epochs.empty or not (checkpoint_epochs % cycle_length == 0).all():
                raise ValueError(f"{history_path} contains a checkpoint outside a complete K cycle")
            if int(checkpoint_epochs.max()) < minimum_epochs or epochs not in set(checkpoint_epochs.tolist()):
                raise ValueError(
                    f"{history_path} does not document the realized complete-cycle checkpoint: "
                    f"realized={epochs}, checkpoint_epochs={checkpoint_epochs.tolist()}"
                )

    def read(self, stage: str, outer_fold: int, seed: int, inner_fold: int | None = None) -> pd.DataFrame:
        key = (str(stage), int(outer_fold), int(seed), None if inner_fold is None else int(inner_fold))
        if key in self._cache:
            return self._cache[key].copy()
        if stage not in {"inner", "outer"}:
            raise ValueError(f"unsupported route stage: {stage}")
        route_directory = self._route_directory(stage, outer_fold, seed, inner_fold)
        manifest_path = route_directory / "run_manifest.json"
        logits_path = route_directory / str(self.contract["cpu_logit_postprocessing"]["input_artifact"])
        manifest = _load_json(manifest_path)
        self._validate_route_manifest(manifest, stage, outer_fold, inner_fold, seed, route_directory)
        if not logits_path.is_file():
            raise FileNotFoundError(logits_path)
        frame = pd.read_csv(logits_path, dtype={"patient_group": str})
        missing = REQUIRED_LOGIT_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{logits_path} lacks image-logit columns: {sorted(missing)}")
        frame = frame.copy()
        frame["patient_group"] = frame["patient_group"].astype(str)
        for column in ("label", "maximum_budget", "view_rank", "photograph_order_seed", "outer_fold", "inner_fold", "seed"):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        frame["image_logit"] = pd.to_numeric(frame["image_logit"], errors="raise")
        if not np.isfinite(frame["image_logit"].to_numpy(dtype=float)).all():
            raise ValueError(f"{logits_path} contains non-finite image logits")
        if frame["maximum_budget"].nunique() != 1 or int(frame["maximum_budget"].iloc[0]) != self.maximum_budget:
            raise ValueError(f"{logits_path} has an unexpected maximum_budget")
        if not frame["view_rank"].isin(range(1, self.maximum_budget + 1)).all():
            raise ValueError(f"{logits_path} has invalid view ranks")
        if frame.duplicated(["patient_group", "view_rank"]).any():
            raise ValueError(f"{logits_path} repeats a patient_group/view_rank row")
        if not frame["photograph_order_seed"].eq(self.primary_order_seed).all():
            raise ValueError(f"{logits_path} uses a photograph order other than the locked primary order")
        for column, expected in (
            ("stage", stage),
            ("outer_fold", int(outer_fold)),
            ("inner_fold", int(inner_fold) if stage == "inner" and inner_fold is not None else -1),
            ("seed", int(seed)),
            ("backbone", self.backbone),
            ("training_aggregation", "mean"),
        ):
            if not frame[column].astype(str).eq(str(expected)).all():
                raise ValueError(f"{logits_path} has inconsistent {column}")
        expected = self._expected_groups(stage, outer_fold, inner_fold)
        expected_set = set(expected["patient_group"])
        observed_set = set(frame["patient_group"])
        if observed_set != expected_set:
            raise ValueError(f"{logits_path} patient groups differ from the locked {stage} groups")
        counts = frame.groupby("patient_group")["view_rank"].agg(["size", "nunique"])
        if not counts["size"].eq(self.maximum_budget).all() or not counts["nunique"].eq(self.maximum_budget).all():
            raise ValueError(f"{logits_path} must contain exactly one logit for every rank 1..{self.maximum_budget}")
        labels = frame[["patient_group", "label"]].drop_duplicates()
        if labels["patient_group"].duplicated().any():
            raise ValueError(f"{logits_path} contains inconsistent patient labels")
        label_check = labels.merge(expected, on="patient_group", suffixes=("_run", "_lock"), validate="one_to_one")
        if label_check["label_run"].ne(label_check["label_lock"]).any():
            raise ValueError(f"{logits_path} labels differ from the locked patient groups")
        pivot = frame.pivot(index="patient_group", columns="view_rank", values="image_logit")
        pivot = pivot.reindex(columns=list(range(1, self.maximum_budget + 1)))
        if pivot.isna().any().any():
            raise ValueError(f"{logits_path} cannot be reshaped into the complete ordered logit matrix")
        wide = expected.set_index("patient_group").join(pivot, how="left").reset_index()
        wide = wide.rename(columns={rank: f"image_logit_{rank}" for rank in range(1, self.maximum_budget + 1)})
        wide["inner_fold"] = int(inner_fold) if stage == "inner" and inner_fold is not None else -1
        wide = wide.sort_values("patient_group").reset_index(drop=True)
        self._cache[key] = wide
        record_key = "/".join([stage, str(outer_fold), str(inner_fold if inner_fold is not None else -1), str(seed)])
        self.source_records[record_key] = {
            "stage": stage,
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold) if inner_fold is not None else -1,
            "seed": int(seed),
            "backbone": self.backbone,
            "training_route_name": self.training_route_name,
            "manifest_contract_sha256": str(manifest.get("contract_sha256")),
            "manifest_early_stopping_monitor": str((manifest.get("memory") or {}).get("early_stopping_monitor")),
            "route_directory": str(route_directory),
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256_file(manifest_path),
            "image_logit_artifact": str(logits_path),
            "image_logit_sha256": sha256_file(logits_path),
            "training_epochs_realized": self._realized_epochs(manifest),
        }
        return wide.copy()

    def inner_wide(self, outer_fold: int, seed: int) -> pd.DataFrame:
        frames = [self.read("inner", outer_fold, seed, inner_fold) for inner_fold in range(self.inner_count)]
        result = pd.concat(frames, ignore_index=True).sort_values("patient_group").reset_index(drop=True)
        if result["patient_group"].duplicated().any():
            raise ValueError("inner cross-fitted logit source repeats an outer-training patient")
        return result


def _logit_features(wide: pd.DataFrame, budget: int) -> tuple[np.ndarray, list[str]]:
    columns = [f"image_logit_{rank}" for rank in range(1, int(budget) + 1)]
    values = wide[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite image logits cannot be used as meta-features")
    # Sorting makes the learned comparator symmetric to the order of the same
    # K-image set while retaining exactly the locked prefix as its input.
    features = np.sort(values, axis=1)[:, ::-1]
    return features, [f"logit_rank_{rank}" for rank in range(1, int(budget) + 1)]


def _baseline_logits(wide: pd.DataFrame, budget: int, aggregation: str) -> np.ndarray:
    values = wide[[f"image_logit_{rank}" for rank in range(1, int(budget) + 1)]].to_numpy(dtype=float)
    if aggregation == "mean":
        return values.mean(axis=1)
    if aggregation == "top2_mean":
        return np.sort(values, axis=1)[:, -min(2, int(budget)) :].mean(axis=1)
    if aggregation == "max":
        return values.max(axis=1)
    raise ValueError(f"unknown shared-logit aggregation: {aggregation}")


def _seed_ensemble(frames: list[pd.DataFrame], include_inner_fold: bool = False) -> pd.DataFrame:
    if not frames:
        raise ValueError("cannot ensemble an empty seed list")
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["patient_group", "seed"]).any():
        raise ValueError("seed ensemble contains duplicate patient predictions")
    expected_seed_count = len(frames)
    counts = combined.groupby("patient_group")["seed"].nunique()
    if not counts.eq(expected_seed_count).all():
        raise ValueError("seed ensemble is missing a formal seed for a patient")
    group_columns = ["patient_group", "label"]
    aggregated = combined.groupby(group_columns, as_index=False, sort=True)["probability"].mean()
    if include_inner_fold:
        fold_frame = combined[group_columns + ["inner_fold"]].drop_duplicates(group_columns)
        aggregated = aggregated.merge(fold_frame, on=group_columns, validate="one_to_one")
    return aggregated


def _make_estimator(aggregator: str, candidate: dict[str, Any], random_state: int) -> Any:
    parameters = {key: value for key, value in candidate.items() if key != "candidate_id"}
    if aggregator == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError("xgboost is required for the declared XGBoost CPU postprocessing analysis") from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=int(random_state),
            verbosity=0,
            **parameters,
        )
    if aggregator == "random_forest":
        return RandomForestClassifier(n_jobs=1, random_state=int(random_state), **parameters)
    if aggregator == "logistic_regression":
        return LogisticRegression(random_state=int(random_state), **parameters)
    raise ValueError(f"unknown learned aggregator: {aggregator}")


def _cross_fitted_meta_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    inner_fold: np.ndarray,
    aggregator: str,
    candidate: dict[str, Any],
    random_state: int,
    inner_fold_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    predictions = np.full(len(labels), np.nan, dtype=float)
    fold_scores: dict[str, float] = {}
    for validation_fold in range(int(inner_fold_count)):
        validation_mask = inner_fold == int(validation_fold)
        training_mask = ~validation_mask
        if not validation_mask.any() or not training_mask.any():
            raise ValueError(f"inner meta-CV fold {validation_fold} is empty")
        if len(np.unique(labels[training_mask])) < 2 or len(np.unique(labels[validation_mask])) < 2:
            raise ValueError(f"inner meta-CV fold {validation_fold} lacks both binary classes")
        estimator = _make_estimator(aggregator, candidate, _stable_seed(random_state, validation_fold, aggregator))
        estimator.fit(features[training_mask], labels[training_mask])
        held_out = _positive_probability(estimator, features[validation_mask])
        predictions[validation_mask] = held_out
        fold_scores[str(validation_fold)] = float(log_loss(labels[validation_mask], held_out, labels=[0, 1]))
    if not np.isfinite(predictions).all():
        raise ValueError("inner meta-CV did not produce one held-out probability per patient")
    return predictions, fold_scores


def _select_meta_candidate(
    features: np.ndarray,
    labels: np.ndarray,
    inner_fold: np.ndarray,
    aggregator: str,
    candidates: list[dict[str, Any]],
    random_state: int,
    inner_fold_count: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    evaluations: list[dict[str, Any]] = []
    predictions_by_index: dict[int, np.ndarray] = {}
    for index, candidate in enumerate(candidates):
        predictions, fold_scores = _cross_fitted_meta_predictions(
            features,
            labels,
            inner_fold,
            aggregator,
            candidate,
            _stable_seed(random_state, index),
            inner_fold_count,
        )
        mean_score = float(np.mean(list(fold_scores.values())))
        if not np.isfinite(mean_score):
            raise ValueError(f"meta-CV log loss is not finite for {aggregator}/{candidate.get('candidate_id')}")
        predictions_by_index[index] = predictions
        evaluations.append({
            "candidate_index": int(index),
            "candidate_id": str(candidate["candidate_id"]),
            "candidate": candidate,
            "mean_inner_log_loss": mean_score,
            "inner_log_loss_by_fold": fold_scores,
        })
    best = min(evaluations, key=lambda row: (float(row["mean_inner_log_loss"]), int(row["candidate_index"])))
    return best["candidate"], predictions_by_index[int(best["candidate_index"])], evaluations


def _fit_full_meta_and_predict(
    features: np.ndarray,
    labels: np.ndarray,
    outer_features: np.ndarray,
    aggregator: str,
    candidate: dict[str, Any],
    random_state: int,
) -> np.ndarray:
    if len(np.unique(labels)) < 2:
        raise ValueError("full meta-aggregator fit lacks both binary classes")
    estimator = _make_estimator(aggregator, candidate, random_state)
    estimator.fit(features, labels)
    return _positive_probability(estimator, outer_features)


def _collect_shared_logit_baselines(reader: FrozenLogitSource) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    for aggregation in BASELINE_AGGREGATORS:
        for budget in reader.budgets:
            for outer_fold in range(reader.outer_count):
                inner_seed_frames: list[pd.DataFrame] = []
                outer_seed_frames: list[pd.DataFrame] = []
                for seed in reader.formal_seeds:
                    inner = reader.inner_wide(outer_fold, seed)
                    outer = reader.read("outer", outer_fold, seed)
                    inner_seed_frames.append(pd.DataFrame({
                        "patient_group": inner["patient_group"],
                        "label": inner["label"],
                        "inner_fold": inner["inner_fold"],
                        "seed": int(seed),
                        "probability": _sigmoid(_baseline_logits(inner, budget, aggregation)),
                    }))
                    outer_seed_frames.append(pd.DataFrame({
                        "patient_group": outer["patient_group"],
                        "label": outer["label"],
                        "seed": int(seed),
                        "probability": _sigmoid(_baseline_logits(outer, budget, aggregation)),
                    }))
                inner_ensemble = _seed_ensemble(inner_seed_frames, include_inner_fold=True)
                threshold = select_threshold_at_specificity(inner_ensemble["label"], inner_ensemble["probability"], minimum_specificity=0.90)
                outer_ensemble = _seed_ensemble(outer_seed_frames)
                outer_ensemble["decision_threshold"] = float(threshold.threshold)
                outer_ensemble["decision"] = outer_ensemble["probability"] >= float(threshold.threshold)
                outer_ensemble["outer_fold"] = int(outer_fold)
                outer_ensemble["budget"] = int(budget)
                outer_ensemble["aggregator"] = aggregation
                outer_ensemble["model_family"] = "shared_logit_rule"
                rows.append(outer_ensemble)
                threshold_rows.append({
                    "aggregator": aggregation,
                    "budget": int(budget),
                    "outer_fold": int(outer_fold),
                    "threshold": float(threshold.threshold),
                    "inner_specificity": float(threshold.specificity),
                    "inner_sensitivity": float(threshold.sensitivity),
                    "inner_false_negatives": int(threshold.false_negatives),
                    "inner_false_positives": int(threshold.false_positives),
                    "source": "formal-seed-averaged inner cross-fitted shared image logits",
                })
    result = pd.concat(rows, ignore_index=True).sort_values(["aggregator", "budget", "patient_group"]).reset_index(drop=True)
    return result, pd.DataFrame(threshold_rows)


def _fit_learned_task(
    reader: FrozenLogitSource,
    aggregator: str,
    outer_fold: int,
    budget: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    configuration = reader.contract["cpu_logit_postprocessing"][aggregator]
    candidates = [dict(candidate) for candidate in configuration["candidates"]]
    inner_seed_frames: list[pd.DataFrame] = []
    outer_seed_frames: list[pd.DataFrame] = []
    selected_by_seed: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for seed in reader.formal_seeds:
        inner = reader.inner_wide(outer_fold, seed)
        outer = reader.read("outer", outer_fold, seed)
        features, current_feature_names = _logit_features(inner, budget)
        outer_features, outer_feature_names = _logit_features(outer, budget)
        if current_feature_names != outer_feature_names:
            raise ValueError("inner and outer meta-feature names differ")
        if feature_names is None:
            feature_names = current_feature_names
        elif feature_names != current_feature_names:
            raise ValueError("meta-feature names differ across formal seeds")
        labels = inner["label"].to_numpy(dtype=int)
        inner_fold = inner["inner_fold"].to_numpy(dtype=int)
        selected, inner_oof, evaluations = _select_meta_candidate(
            features,
            labels,
            inner_fold,
            aggregator,
            candidates,
            _stable_seed(reader.lock["contract_sha256"], aggregator, outer_fold, budget, seed),
            reader.inner_count,
        )
        outer_probability = _fit_full_meta_and_predict(
            features,
            labels,
            outer_features,
            aggregator,
            selected,
            _stable_seed(reader.lock["contract_sha256"], "final", aggregator, outer_fold, budget, seed),
        )
        inner_seed_frames.append(pd.DataFrame({
            "patient_group": inner["patient_group"],
            "label": inner["label"],
            "inner_fold": inner["inner_fold"],
            "seed": int(seed),
            "probability": inner_oof,
        }))
        outer_seed_frames.append(pd.DataFrame({
            "patient_group": outer["patient_group"],
            "label": outer["label"],
            "seed": int(seed),
            "probability": outer_probability,
        }))
        selected_by_seed.append({
            "seed": int(seed),
            "selected_candidate_id": str(selected["candidate_id"]),
            "selected_candidate": selected,
            "candidate_evaluations": evaluations,
        })
    inner_ensemble = _seed_ensemble(inner_seed_frames, include_inner_fold=True)
    threshold = select_threshold_at_specificity(
        inner_ensemble["label"],
        inner_ensemble["probability"],
        minimum_specificity=0.90,
    )
    outer_ensemble = _seed_ensemble(outer_seed_frames)
    outer_ensemble["decision_threshold"] = float(threshold.threshold)
    outer_ensemble["decision"] = outer_ensemble["probability"] >= float(threshold.threshold)
    outer_ensemble["outer_fold"] = int(outer_fold)
    outer_ensemble["budget"] = int(budget)
    outer_ensemble["aggregator"] = aggregator
    outer_ensemble["model_family"] = "nested_cpu_logit_aggregator"
    selection_payload = {
        "status": "complete",
        "postprocessing_kind": "nested_cpu_logit_aggregation",
        "aggregator": aggregator,
        "outer_fold": int(outer_fold),
        "budget": int(budget),
        "source_backbone": reader.backbone,
        "source_training_route_name": reader.training_route_name,
        "formal_seeds": reader.formal_seeds,
        "feature_names": feature_names or [],
        "feature_rule": reader.contract["cpu_logit_postprocessing"]["feature_rule"],
        "selected_by_seed": selected_by_seed,
        "threshold": {
            "value": float(threshold.threshold),
            "inner_specificity": float(threshold.specificity),
            "inner_sensitivity": float(threshold.sensitivity),
            "false_negatives": int(threshold.false_negatives),
            "false_positives": int(threshold.false_positives),
            "rule": reader.contract["cpu_logit_postprocessing"]["threshold_metric"],
        },
        "protocol_manifest_sha256": reader.lock["manifest_sha256"],
        "active_contract_sha256": reader.active_contract_sha256,
        "source_contract_sha256": reader.source_contract_sha256,
        "training_free": True,
    }
    return outer_ensemble, outer_seed_frames, selection_payload


_WORKER_READER: FrozenLogitSource | None = None


def _init_learned_worker(reader: FrozenLogitSource) -> None:
    global _WORKER_READER
    _WORKER_READER = reader


def _fit_learned_task_in_worker(task: tuple[str, int, int]) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, Any]]:
    if _WORKER_READER is None:
        raise RuntimeError("learned-aggregation worker was not initialized")
    aggregator, outer_fold, budget = task
    return _fit_learned_task(_WORKER_READER, aggregator, int(outer_fold), int(budget))


def _run_learned_aggregator(
    reader: FrozenLogitSource,
    aggregator: str,
    analysis_root: Path,
    force: bool,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    tasks = [(outer_fold, budget) for outer_fold in range(reader.outer_count) for budget in reader.budgets]
    worker_count = max(1, min(int(workers), len(tasks)))
    if worker_count == 1:
        task_results = [_fit_learned_task(reader, aggregator, outer_fold, budget) for outer_fold, budget in tasks]
    else:
        worker_tasks = [(aggregator, outer_fold, budget) for outer_fold, budget in tasks]
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_learned_worker,
            initargs=(reader,),
        ) as pool:
            task_results = list(pool.map(_fit_learned_task_in_worker, worker_tasks))
    task_results.sort(key=lambda item: (int(item[2]["outer_fold"]), int(item[2]["budget"])))
    all_oof: list[pd.DataFrame] = []
    all_seed: list[pd.DataFrame] = []
    selection_records: list[dict[str, Any]] = []
    for outer_ensemble, outer_seed_frames, selection_payload in task_results:
        outer_fold = int(selection_payload["outer_fold"])
        budget = int(selection_payload["budget"])
        threshold_value = float(selection_payload["threshold"]["value"])
        all_oof.append(outer_ensemble)
        for seed_frame, selected_by_seed_row in zip(
            outer_seed_frames,
            selection_payload["selected_by_seed"],
            strict=True,
        ):
            current = seed_frame.copy()
            current["decision_threshold"] = threshold_value
            current["decision"] = current["probability"] >= threshold_value
            current["outer_fold"] = outer_fold
            current["budget"] = budget
            current["aggregator"] = aggregator
            current["model_family"] = "nested_cpu_logit_aggregator"
            current["selected_candidate_id"] = str(selected_by_seed_row["selected_candidate_id"])
            all_seed.append(current)
        selection_path = analysis_root / "selection" / aggregator / f"outer_{outer_fold}_K{budget}.json"
        if selection_path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite learned-aggregation selection: {selection_path}")
        _atomic_json(selection_path, selection_payload)
        selection_records.append(selection_payload)
    oof = pd.concat(all_oof, ignore_index=True).sort_values(["aggregator", "budget", "patient_group"]).reset_index(drop=True)
    seed_oof = pd.concat(all_seed, ignore_index=True).sort_values(["aggregator", "seed", "budget", "patient_group"]).reset_index(drop=True)
    return oof, seed_oof, selection_records


def _metric_ci(frame: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, Any]:
    point = patient_metrics(frame["label"], frame["probability"], decisions=frame["decision"], include_extended=False)
    draws = stratified_patient_bootstrap(
        frame["label"],
        frame["probability"],
        threshold=None,
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
        decisions=frame["decision"],
    )
    metrics = {
        name: {
            "estimate": float(point[name]),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
        for name, values in draws.items()
    }
    return {
        "n_patients": int(point["n_patients"]),
        "n_malignant": int(point["n_malignant"]),
        "n_benign": int(point["n_benign"]),
        "metrics": metrics,
    }


def _summary(
    reader: FrozenLogitSource,
    baseline_oof: pd.DataFrame,
    learned_oof: pd.DataFrame,
    requested_aggregators: list[str],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    combined = pd.concat([baseline_oof, learned_oof], ignore_index=True)
    # ``run_postprocessing`` accepts a declared subset for inexpensive pilot
    # checks and resumable analyses.  Summaries must therefore describe the
    # requested subset, rather than attempting to score learned aggregators
    # that were not run (which would create an empty frame and a misleading
    # metric error).
    learned_aggregators = [str(value) for value in requested_aggregators]
    aggregators = list(BASELINE_AGGREGATORS) + learned_aggregators
    metric_ci: dict[str, dict[str, Any]] = {}
    point_metrics: dict[str, dict[str, Any]] = {}
    for aggregator_index, aggregator in enumerate(aggregators):
        metric_ci[aggregator] = {}
        point_metrics[aggregator] = {}
        for budget in reader.budgets:
            current = combined[(combined["aggregator"].eq(aggregator)) & (combined["budget"].eq(int(budget)))].copy()
            point_metrics[aggregator][str(budget)] = patient_metrics(
                current["label"], current["probability"], decisions=current["decision"], include_extended=False
            )
            metric_ci[aggregator][str(budget)] = _metric_ci(
                current,
                n_bootstrap,
                int(bootstrap_seed) + 1000 * aggregator_index + int(budget),
            )
    paired_vs_mean: dict[str, dict[str, Any]] = {}
    for learned_index, aggregator in enumerate(learned_aggregators):
        for budget in reader.budgets:
            candidate = learned_oof[(learned_oof["aggregator"].eq(aggregator)) & (learned_oof["budget"].eq(int(budget)))]
            comparator = baseline_oof[(baseline_oof["aggregator"].eq("mean")) & (baseline_oof["budget"].eq(int(budget)))]
            paired_vs_mean[f"{aggregator}_vs_mean_K{budget}"] = paired_bootstrap(
                candidate,
                comparator,
                n_bootstrap,
                int(bootstrap_seed) + 20000 + 100 * learned_index + int(budget),
                inferential_metrics=set(),
            )
    return {
        "status": "complete",
        "training_free": True,
        "patient_level_only": True,
        "role": reader.contract["cpu_logit_postprocessing"]["role"],
        "source_backbone": reader.backbone,
        "source_training_route_name": reader.training_route_name,
        "input_artifact": reader.contract["cpu_logit_postprocessing"]["input_artifact"],
        "primary_cohort_patient_groups": int(reader.contract["data"]["expected_primary_groups"]),
        "budgets": reader.budgets,
        "point_metrics_by_aggregator_and_budget": point_metrics,
        "metric_ci_by_aggregator_and_budget": metric_ci,
        "paired_comparisons_against_shared_mean": paired_vs_mean,
        "bootstrap_replicates": int(n_bootstrap),
        "bootstrap_unit": "patient_group",
        "selection_rule": reader.contract["cpu_logit_postprocessing"]["nested_rule"],
        "protocol_manifest_sha256": reader.lock["manifest_sha256"],
        "contract_sha256": reader.lock["contract_sha256"],
    }


def run_postprocessing(
    protocol_root: Path,
    source_root: Path,
    analysis_root: Path,
    aggregators: list[str],
    n_bootstrap: int,
    bootstrap_seed: int,
    force: bool,
    workers: int = 1,
    source_contract_path: Path | None = None,
) -> dict[str, Any]:
    requested = [str(value) for value in aggregators]
    if requested != [value for value in LEARNED_AGGREGATORS if value in requested]:
        raise ValueError(f"aggregators must be a subset of the declared order {LEARNED_AGGREGATORS}")
    source = Path(source_root).resolve()
    analysis = Path(analysis_root).resolve()
    if source == analysis:
        raise ValueError("analysis_root must be separate from the source route root")
    analysis.mkdir(parents=True, exist_ok=True)
    existing_manifest = analysis / "postprocessing_manifest.json"
    if existing_manifest.is_file() and not force:
        prior = _load_json(existing_manifest)
        requested_source_contract = str(Path(source_contract_path).resolve()) if source_contract_path else None
        if (
            prior.get("status") == "complete"
            and prior.get("protocol_manifest_sha256") == _load_json(Path(protocol_root).resolve() / "protocol_lock.json")["manifest_sha256"]
            and prior.get("contract_sha256") == _load_json(Path(protocol_root).resolve() / "protocol_lock.json")["contract_sha256"]
            and prior.get("source_root") == str(source)
            and prior.get("aggregators") == requested
            and prior.get("source_contract_path") == requested_source_contract
        ):
            return {"status": "reused", "manifest": str(existing_manifest), "summary": str(analysis / "summary" / SUMMARY_FILENAME)}
        raise FileExistsError(f"learned aggregation output has a different provenance: {analysis}")
    if not force:
        existing_files = [path for path in analysis.rglob("*") if path.is_file()]
        if existing_files:
            raise FileExistsError(f"learned aggregation output is non-empty; use a new analysis root or --force: {analysis}")
    reader = FrozenLogitSource(Path(protocol_root), source, source_contract_path=source_contract_path)
    declared = [str(value) for value in reader.contract["cpu_logit_postprocessing"]["aggregators"]]
    if requested != [value for value in declared if value in requested]:
        raise ValueError("requested aggregators do not match the declared CPU postprocessing contract")
    baseline_oof, baseline_thresholds = _collect_shared_logit_baselines(reader)
    learned_frames: list[pd.DataFrame] = []
    learned_seed_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, Any]] = []
    for aggregator in requested:
        oof, seed_oof, selections = _run_learned_aggregator(reader, aggregator, analysis, force, workers)
        learned_frames.append(oof)
        learned_seed_frames.append(seed_oof)
        selection_records.extend(selections)
    learned_oof = pd.concat(learned_frames, ignore_index=True).sort_values(["aggregator", "budget", "patient_group"]).reset_index(drop=True)
    learned_seed_oof = pd.concat(learned_seed_frames, ignore_index=True).sort_values(["aggregator", "seed", "budget", "patient_group"]).reset_index(drop=True)
    expected_groups = int(reader.contract["data"]["expected_primary_groups"])
    for frame, name in ((baseline_oof, "shared-logit"), (learned_oof, "learned")):
        counts = frame.groupby(["aggregator", "budget"])["patient_group"].agg(["size", "nunique"])
        if not counts["size"].eq(expected_groups).all() or not counts["nunique"].eq(expected_groups).all():
            raise ValueError(f"{name} postprocessing output is incomplete")
    summary = _summary(
        reader,
        baseline_oof,
        learned_oof,
        requested,
        int(n_bootstrap),
        int(bootstrap_seed),
    )
    summary_root = analysis / "summary"
    _atomic_csv(analysis / "shared_logit_baseline_patient_predictions.csv", baseline_oof)
    _atomic_csv(analysis / "shared_logit_baseline_thresholds.csv", baseline_thresholds)
    _atomic_csv(analysis / "learned_aggregator_outer_seed_predictions.csv", learned_seed_oof)
    _atomic_csv(analysis / "learned_aggregator_oof_patient_predictions.csv", learned_oof)
    summary_path = summary_root / SUMMARY_FILENAME
    _atomic_json(summary_path, summary)
    manifest = {
        "status": "complete",
        "postprocessing_kind": "nested_cpu_logit_aggregation",
        "training_free": True,
        "source_root": str(source),
        "analysis_root": str(analysis),
        "aggregators": requested,
        "budgets": reader.budgets,
        "source_backbone": reader.backbone,
        "source_training_route_name": reader.training_route_name,
        "protocol_manifest_sha256": reader.lock["manifest_sha256"],
        "contract_sha256": reader.lock["contract_sha256"],
        "active_contract_sha256": reader.active_contract_sha256,
        "source_contract_sha256": reader.source_contract_sha256,
        "source_contract_path": str(reader.source_contract_path) if reader.source_contract_path else None,
        "workers": int(max(1, workers)),
        "manifest_rows": int(reader.contract["data"]["expected_images"]),
        "patient_groups": int(reader.contract["data"]["expected_patient_groups"]),
        "primary_patient_groups": int(expected_groups),
        "formal_seeds": reader.formal_seeds,
        "outer_folds": reader.outer_count,
        "inner_folds": reader.inner_count,
        "input_artifact": reader.contract["cpu_logit_postprocessing"]["input_artifact"],
        "source_route_count": len(reader.source_records),
        "source_routes": list(reader.source_records.values()),
        "selection_record_count": len(selection_records),
        "selection_records": [
            {
                "aggregator": record["aggregator"],
                "outer_fold": record["outer_fold"],
                "budget": record["budget"],
                "threshold": record["threshold"]["value"],
            }
            for record in selection_records
        ],
        "outputs": {
            "shared_logit_baseline_patient_predictions": str(analysis / "shared_logit_baseline_patient_predictions.csv"),
            "shared_logit_baseline_thresholds": str(analysis / "shared_logit_baseline_thresholds.csv"),
            "learned_aggregator_outer_seed_predictions": str(analysis / "learned_aggregator_outer_seed_predictions.csv"),
            "learned_aggregator_oof_patient_predictions": str(analysis / "learned_aggregator_oof_patient_predictions.csv"),
            "summary": str(summary_path),
        },
    }
    manifest["code_sha256"] = sha256_file(Path(__file__).resolve())
    _atomic_json(existing_manifest, manifest)
    return {"status": "complete", "manifest": str(existing_manifest), "summary": str(summary_path)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--source-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--analysis-root", help="defaults to SOURCE_ROOT/learned_aggregation")
    parser.add_argument("--aggregators", nargs="+", choices=list(LEARNED_AGGREGATORS), default=list(LEARNED_AGGREGATORS))
    parser.add_argument("--bootstrap-replicates", type=int, help="defaults to the locked patient-level bootstrap count")
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--workers", type=int, default=1, help="parallel CPU workers for independent aggregator/budget/fold fits")
    parser.add_argument("--source-contract", help="optional compatible pre-postprocessing source contract for existing logits")
    parser.add_argument("--force", action="store_true", help="overwrite only this derived analysis namespace")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol_root = Path(args.protocol_root).resolve()
    source_root = Path(args.source_root).resolve()
    analysis_root = Path(args.analysis_root).resolve() if args.analysis_root else source_root / "learned_aggregation"
    lock = _load_json(protocol_root / "protocol_lock.json")
    default_bootstrap = int(lock["contract"]["validation"]["bootstrap_replicates"])
    if args.bootstrap_replicates is not None and int(args.bootstrap_replicates) < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    result = run_postprocessing(
        protocol_root,
        source_root,
        analysis_root,
        list(args.aggregators),
        int(args.bootstrap_replicates or default_bootstrap),
        int(args.bootstrap_seed),
        bool(args.force),
        max(1, int(args.workers)),
        Path(args.source_contract).resolve() if args.source_contract else None,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
