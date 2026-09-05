#!/usr/bin/env python3
"""Build the locked LidMerge patient-level photographic-budget protocol.

Only ``patient_group`` and its binary label define cohorts, splits, learning,
prediction rows, and statistics.  ``image_id`` and image paths only resolve
pixels within a patient group; they are never passed to a model or used for a
split, threshold, or statistical comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from lidpair import METHOD_ID, PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = Path(__file__).with_name("contract.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32 - 1)


def portable_relative_reference(path: Path, protocol_root: Path) -> str:
    reference = Path(os.path.relpath(path.resolve(), protocol_root.resolve()))
    if reference.is_absolute():
        raise ValueError("locked manifest reference must be relative to the protocol root")
    return reference.as_posix()


def read_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def load_manifest(path: Path, contract: dict, check_paths: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"image_id": str, "patient_group": str})
    data = contract["data"]
    require_columns(
        frame,
        {str(data["group_field"]), str(data["label_field"]), str(data["image_id_field"]),
         str(data["image_field"]), "canonical_relative_path"},
        str(path),
    )
    if len(frame) != int(data["expected_images"]):
        raise ValueError(f"manifest has {len(frame)} rows, expected {data['expected_images']}")
    frame["patient_group"] = frame[str(data["group_field"])].astype(str)
    frame["image_id"] = frame[str(data["image_id_field"])].astype(str)
    frame["label"] = pd.to_numeric(frame[str(data["label_field"])], errors="raise").astype(int)
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("manifest labels must be binary")
    if frame["image_id"].duplicated().any():
        raise ValueError("manifest image_id must be unique")
    if check_paths:
        missing = [text for text in frame[str(data["image_field"])].astype(str) if not Path(text).is_file()]
        if missing:
            raise FileNotFoundError(f"manifest contains {len(missing)} missing image paths; first={missing[0]}")
    return frame


def group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("patient_group", sort=True)["label"].agg(["nunique", "first"]).reset_index()
    if grouped["nunique"].ne(1).any():
        bad = grouped.loc[grouped["nunique"].ne(1), "patient_group"].iloc[0]
        raise ValueError(f"patient_group has inconsistent labels: {bad}")
    return grouped.rename(columns={"first": "label"})[["patient_group", "label"]].astype({"patient_group": str, "label": int})


def eligible_summary(frame: pd.DataFrame, summary: pd.DataFrame, minimum_photographs: int) -> pd.DataFrame:
    """Select the common primary cohort solely by within-group availability.

    Availability is a cohort-eligibility requirement, not a model feature,
    split stratifier, input field, or reported covariate.
    """

    availability = frame.groupby("patient_group", sort=True).size()
    eligible = summary[summary["patient_group"].map(availability).ge(int(minimum_photographs))].copy()
    return eligible.sort_values("patient_group").reset_index(drop=True)


def _label_strata(summary: pd.DataFrame) -> pd.Series:
    return summary["label"].astype(str)


def assign_outer_folds(summary: pd.DataFrame, contract: dict) -> pd.DataFrame:
    n_splits = int(contract["validation"]["outer_folds"])
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(contract["validation"]["fold_seed"]),
    )
    output = summary[["patient_group", "label"]].copy().reset_index(drop=True)
    output["outer_fold"] = -1
    for fold_id, (_, test_index) in enumerate(splitter.split(output, _label_strata(output))):
        output.loc[test_index, "outer_fold"] = fold_id
    if output["outer_fold"].lt(0).any() or output["patient_group"].duplicated().any():
        raise RuntimeError("failed to assign deterministic outer folds")
    return output.sort_values("patient_group").reset_index(drop=True)


def assign_inner_folds(outer: pd.DataFrame, contract: dict) -> pd.DataFrame:
    n_outer = int(contract["validation"]["outer_folds"])
    n_inner = int(contract["validation"]["inner_folds"])
    base_seed = int(contract["validation"]["fold_seed"])
    rows: list[pd.DataFrame] = []
    for outer_fold in range(n_outer):
        current = outer[["patient_group", "label", "outer_fold"]].copy()
        current["protocol_outer_fold"] = outer_fold
        current["inner_fold"] = -1
        train_pool = current[current["outer_fold"].ne(outer_fold)].copy().reset_index(drop=True)
        splitter = StratifiedKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=base_seed + outer_fold * 104729,
        )
        for inner_fold, (_, validation_index) in enumerate(splitter.split(train_pool, _label_strata(train_pool))):
            groups = train_pool.iloc[validation_index]["patient_group"]
            current.loc[current["patient_group"].isin(groups), "inner_fold"] = inner_fold
        if current.loc[current["outer_fold"].ne(outer_fold), "inner_fold"].lt(0).any():
            raise RuntimeError(f"outer fold {outer_fold}: missing inner validation assignment")
        rows.append(current)
    return pd.concat(rows, ignore_index=True).sort_values(["protocol_outer_fold", "patient_group"]).reset_index(drop=True)


def deterministic_order(rows: pd.DataFrame, patient_group: str, order_seed: int, maximum_budget: int) -> list[str]:
    if len(rows) < int(maximum_budget):
        raise ValueError(f"{patient_group} has fewer than {maximum_budget} eligible photographs")
    candidates = rows.sort_values("image_id")["image_id"].astype(str).tolist()
    rng = np.random.default_rng(stable_seed("photograph-order", patient_group, order_seed))
    permutation = rng.permutation(len(candidates))
    return [candidates[int(index)] for index in permutation[: int(maximum_budget)]]


def build_orders(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    contract: dict,
    order_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    max_budget = int(contract["views"]["maximum_budget"])
    for group_row in primary.itertuples(index=False):
        group = str(group_row.patient_group)
        candidates = frame[frame["patient_group"].eq(group)]
        for rank, image_id in enumerate(deterministic_order(candidates, group, int(order_seed), max_budget), start=1):
            rows.append(
                {
                    "patient_group": group,
                    "label": int(group_row.label),
                    "photograph_order_seed": int(order_seed),
                    "rank": int(rank),
                    "image_id": str(image_id),
                }
            )
    result = pd.DataFrame(rows).sort_values(["patient_group", "rank"]).reset_index(drop=True)
    expected_rows = len(primary) * max_budget
    if len(result) != expected_rows or result.duplicated(["patient_group", "rank"]).any():
        raise RuntimeError("photograph-order generation is incomplete or duplicated")
    if result.duplicated(["patient_group", "image_id"]).any():
        raise RuntimeError("a primary photograph order repeats an image within a patient_group")
    return result


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def build_protocol(manifest_path: Path, output_root: Path, contract: dict, check_paths: bool) -> dict:
    manifest = load_manifest(manifest_path, contract, check_paths=check_paths)
    summary = group_summary(manifest)
    if len(summary) != int(contract["data"]["expected_patient_groups"]):
        raise ValueError(f"manifest has {len(summary)} patient groups, expected {contract['data']['expected_patient_groups']}")
    primary = eligible_summary(manifest, summary, int(contract["views"]["maximum_budget"]))
    expected_primary = int(contract["data"]["expected_primary_groups"])
    if len(primary) != expected_primary:
        raise ValueError(f"primary cohort has {len(primary)} groups, expected {expected_primary}")
    observed_counts = {
        "malignant": int((primary["label"] == 1).sum()),
        "benign": int((primary["label"] == 0).sum()),
    }
    if observed_counts != {key: int(value) for key, value in contract["data"]["expected_primary_label_counts"].items()}:
        raise ValueError(f"primary label counts differ from contract: {observed_counts}")
    orders = build_orders(manifest, primary, contract, int(contract["views"]["primary_order_seed"]))
    alternate_orders = {
        int(seed): build_orders(manifest, primary, contract, int(seed))
        for seed in contract["views"].get("alternate_order_seeds", [])
    }
    outer = assign_outer_folds(primary, contract)
    inner = assign_inner_folds(outer, contract)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(summary, output_root / "patient_group_summary.csv")
    write_csv(primary, output_root / "primary_patient_groups.csv")
    primary_order_file = "photograph_orders_at_least_four_photographs.csv"
    write_csv(orders, output_root / primary_order_file)
    alternate_order_files: dict[str, str] = {}
    for seed, current in alternate_orders.items():
        filename = f"photograph_orders_at_least_four_photographs_draw_{seed}.csv"
        write_csv(current, output_root / filename)
        alternate_order_files[str(seed)] = filename
    write_csv(outer, output_root / "outer_folds.csv")
    write_csv(inner, output_root / "inner_folds.csv")
    payload = {
        "method_id": METHOD_ID,
        "protocol_version": PROTOCOL_VERSION,
        "prepare_code_sha256": sha256_file(Path(__file__)),
        "contract_relative_to_protocol_root": portable_relative_reference(CONTRACT_PATH, output_root),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "manifest_relative_to_protocol_root": portable_relative_reference(manifest_path, output_root),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": int(len(manifest)),
        "patient_groups": int(len(summary)),
        "primary_groups": int(len(primary)),
        "primary_label_counts": observed_counts,
        "outer_folds": int(contract["validation"]["outer_folds"]),
        "inner_folds": int(contract["validation"]["inner_folds"]),
        "primary_order_file": primary_order_file,
        "primary_order_sha256": sha256_file(output_root / primary_order_file),
        "alternate_order_files": alternate_order_files,
        "alternate_order_sha256": {seed: sha256_file(output_root / filename) for seed, filename in alternate_order_files.items()},
        "selection_rule": contract["views"]["selection_rule"],
        "contract": contract,
    }
    (output_root / "protocol_lock.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "manifest_raw.csv"))
    parser.add_argument("--output-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--force", action="store_true", help="replace a protocol directory only after an explicit request")
    parser.add_argument("--skip-path-check", action="store_true", help="only for a manifest-only CI environment")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    contract = read_contract()
    if output_root.exists() and any(output_root.iterdir()):
        lock_path = output_root / "protocol_lock.json"
        if not args.force:
            if not lock_path.is_file():
                raise RuntimeError(f"refusing to overwrite non-empty protocol root without --force: {output_root}")
            prior = json.loads(lock_path.read_text(encoding="utf-8"))
            same = prior.get("manifest_sha256") == sha256_file(manifest) and prior.get("contract_sha256") == sha256_file(CONTRACT_PATH)
            if not same:
                raise RuntimeError(f"protocol root is stale or has a different input; use a new root or --force: {output_root}")
            print(json.dumps({"status": "reused", "protocol_root": str(output_root), "primary_groups": prior.get("primary_groups")}, ensure_ascii=True))
            return 0
        shutil.rmtree(output_root)
    payload = build_protocol(manifest, output_root, contract, check_paths=not args.skip_path_check)
    # Keep the machine-readable lock UTF-8, but make Windows console output
    # independent of the active code page.
    print(json.dumps({"status": "complete", "protocol_root": str(output_root), **payload}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
