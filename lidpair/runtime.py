"""Operational image-path resolution without modifying the frozen manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def required_runtime_image_ids(protocol_root: Path) -> set[str]:
    """Return image IDs referenced by the locked primary and alternate orders."""

    root = Path(protocol_root).resolve()
    lock = json.loads((root / "protocol_lock.json").read_text(encoding="utf-8"))
    filenames = [str(lock["primary_order_file"])]
    filenames.extend(str(value) for value in lock.get("alternate_order_files", {}).values())
    image_ids: set[str] = set()
    for filename in filenames:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"image_id": str})
        if "image_id" not in frame.columns:
            raise ValueError(f"locked order file lacks image_id: {path}")
        image_ids.update(frame["image_id"].astype(str))
    if not image_ids:
        raise ValueError("locked order files contain no runtime image IDs")
    return image_ids


def resolve_runtime_manifest(
    manifest: pd.DataFrame,
    image_root: Path | None = None,
    check_paths: bool = False,
    required_image_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Return a copy whose ``absolute_path`` values point to the declared mounted image root.

    This is an I/O-only mapping.  The model receives pixels, while all learning,
    splitting, predictions, and statistics remain keyed solely by patient_group.
    The input CSV and its checksum are never changed.
    """

    if "absolute_path" not in manifest.columns:
        raise ValueError("manifest lacks the operational absolute_path column")
    frame = manifest.copy()
    if image_root is not None:
        if "canonical_relative_path" not in frame.columns:
            raise ValueError("--image-root requires canonical_relative_path in the frozen manifest")
        root = Path(image_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"declared image root does not exist: {root}")
        resolved: list[str] = []
        for relative_text in frame["canonical_relative_path"].astype(str):
            relative = Path(relative_text)
            if relative.is_absolute():
                raise ValueError(f"canonical_relative_path must be relative: {relative_text}")
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"runtime image path escapes declared image root: {relative_text}") from exc
            resolved.append(str(candidate))
        frame["absolute_path"] = resolved
    if check_paths:
        if required_image_ids is None:
            paths_to_check = frame["absolute_path"].astype(str)
        else:
            if "image_id" not in frame.columns:
                raise ValueError("required runtime image checks need image_id in the manifest")
            observed_ids = set(frame["image_id"].astype(str))
            unknown_ids = sorted(set(required_image_ids).difference(observed_ids))
            if unknown_ids:
                raise ValueError(f"locked runtime orders reference unknown image IDs: {unknown_ids[:3]}")
            paths_to_check = frame.loc[frame["image_id"].astype(str).isin(required_image_ids), "absolute_path"].astype(str)
        missing = [str(path) for path in paths_to_check if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"runtime image paths include {len(missing)} missing files; first={missing[0]}")
    return frame
