#!/usr/bin/env python3
"""Render the frozen LidMerge paired-comparison results as one CSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_comparison_table(output_root: Path, force: bool) -> Path:
    report_path = output_root / "summary" / "lidmerge_summary.json"
    if not report_path.is_file():
        raise FileNotFoundError("run lidpair.summarize before rendering the comparison table")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for comparison, result in report["comparisons"].items():
        for metric, values in result["metrics"].items():
            rows.append({"comparison": comparison, "metric": metric, "n_patients": result["n_patients"], **values})
    destination = output_root / "summary" / "lidmerge_paired_comparisons.csv"
    if destination.exists() and not force:
        raise RuntimeError(f"comparison table already exists; use --force to rebuild: {destination}")
    pd.DataFrame(rows).sort_values(["comparison", "metric"]).to_csv(destination, index=False)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = write_comparison_table(Path(args.output_root).resolve(), bool(args.force))
    print(json.dumps({"status": "complete", "comparison_table": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
