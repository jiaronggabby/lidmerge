"""Run LidMerge training, aggregation and patient-level evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _module(module: str, args: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", module, *args], cwd=ROOT)


def _run_all(args: argparse.Namespace) -> int:
    contract = json.loads((ROOT / "lidpair" / "contract.json").read_text(encoding="utf-8"))
    validation = contract["validation"]
    formal_seeds = [int(value) for value in validation["formal_seeds"]]
    outer_folds = int(validation["outer_folds"])
    inner_folds = int(validation["inner_folds"])
    primary_backbone = str(contract["primary_backbone"])
    support_backbones = [str(value) for value in contract["architecture_robustness"]["backbones"]
                         if str(value) != primary_backbone]
    protocol = str(Path(args.protocol_root).resolve())
    output = str(Path(args.output_root).resolve())
    analysis = str(Path(args.analysis_root).resolve())
    prepare_args = ["--manifest", args.manifest, "--output-root", protocol]
    if args.force:
        prepare_args.append("--force")
    code = _module("lidpair.prepare", prepare_args)
    if code:
        return code
    common = ["--protocol-root", protocol, "--image-root", args.image_root,
              "--output-root", output, "--pretrained"]
    if args.allow_weight_download:
        common.append("--allow-weight-download")
    if args.resume:
        common.append("--resume")
    if args.force:
        common.append("--force")
    for outer in range(outer_folds):
        for inner in range(inner_folds):
            code = _module("lidpair.run_cv5", common + ["--stage", "inner",
                "--outer-fold", str(outer), "--inner-fold", str(inner),
                "--seed", str(validation["selection_seed"]), "--backbone", primary_backbone])
            if code:
                return code
        select_args = ["--protocol-root", protocol,
            "--output-root", output, "--outer-fold", str(outer),
            "--mode", "backbone"]
        if args.force:
            select_args.append("--force")
        code = _module("lidpair.select", select_args)
        if code:
            return code
    for outer in range(outer_folds):
        selected = Path(output) / "selection" / f"outer_{outer}" / "backbone_selection.json"
        selected_backbone = json.loads(selected.read_text(encoding="utf-8"))["backbone"]
        for inner in range(inner_folds):
            for seed in formal_seeds:
                if seed == int(validation["selection_seed"]):
                    continue
                code = _module("lidpair.run_cv5", common + ["--stage", "inner",
                    "--outer-fold", str(outer), "--inner-fold", str(inner),
                    "--seed", str(seed), "--backbone", str(selected_backbone)])
                if code:
                    return code
        for variant in contract["aggregation"]["inference"]:
            select_args = ["--protocol-root", protocol,
                "--output-root", output, "--outer-fold", str(outer),
                "--mode", "variant", "--backbone-selection", str(selected),
                "--variant", str(variant)]
            if args.force:
                select_args.append("--force")
            code = _module("lidpair.select", select_args)
            if code:
                return code
    for outer in range(outer_folds):
        selection = Path(output) / "selection" / f"outer_{outer}" / "mean_selection.json"
        epochs = json.loads(selection.read_text(encoding="utf-8"))["training_epochs"]
        for seed in formal_seeds:
            code = _module("lidpair.run_cv5", common + ["--stage", "outer",
                "--outer-fold", str(outer), "--seed", str(seed),
                "--selection", str(selection), "--training-epochs", str(epochs)])
            if code:
                return code
    for backbone in support_backbones:
        for outer in range(outer_folds):
            selection = Path(output) / "selection" / f"outer_{outer}" / "mean_selection.json"
            epochs = json.loads(selection.read_text(encoding="utf-8"))["training_epochs"]
            for seed in formal_seeds:
                code = _module("lidpair.run_cv5", common + ["--stage", "architecture_robustness",
                    "--outer-fold", str(outer), "--seed", str(seed),
                    "--backbone", backbone, "--training-epochs", str(epochs)])
                if code:
                    return code
    postprocess_args = ["--protocol-root", protocol, "--output-root", output]
    if args.force:
        postprocess_args.append("--force")
    for module in ("lidpair.summarize", "lidpair.counterfactual", "lidpair.pair_draw"):
        code = _module(module, postprocess_args)
        if code:
            return code
    code = _module("lidpair.compare", ["--output-root", output] + (["--force"] if args.force else []))
    if code:
        return code
    robustness_args = ["--protocol-root", protocol, "--output-root", output]
    if args.force:
        robustness_args.append("--force")
    code = _module("lidpair.backbone_robustness", robustness_args)
    if code:
        return code
    learned_args = ["--protocol-root", protocol, "--source-root", output,
        "--analysis-root", analysis, "--workers", str(args.workers)]
    if args.force:
        learned_args.append("--force")
    return _module("lidpair.learned_aggregation", learned_args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="generate a local protocol from a user-supplied manifest")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--protocol-root", required=True)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--skip-path-check", action="store_true")

    train = sub.add_parser("train", help="run one explicit CUDA training route")
    train.add_argument("--protocol-root", required=True)
    train.add_argument("--image-root", required=True)
    train.add_argument("--output-root", required=True)
    train.add_argument("--stage", choices=("inner", "outer", "architecture_robustness"), required=True)
    train.add_argument("--outer-fold", type=int, default=0)
    train.add_argument("--inner-fold", type=int)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--backbone")
    train.add_argument("--selection")
    train.add_argument("--training-epochs", type=int)
    train.add_argument("--pretrained", action="store_true")
    train.add_argument("--allow-weight-download", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--force", action="store_true")

    evaluate = sub.add_parser("eval", help="summarize one patient-level prediction CSV")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--budget", type=int)
    evaluate.add_argument("--variant")
    evaluate.add_argument("--threshold", type=float,
                          help="use an externally locked patient-level threshold")

    run_all = sub.add_parser("all", help="run preparation, training, evaluation summaries, and learned aggregation")
    run_all.add_argument("--manifest", required=True)
    run_all.add_argument("--image-root", required=True)
    run_all.add_argument("--protocol-root", required=True)
    run_all.add_argument("--output-root", required=True)
    run_all.add_argument("--analysis-root", required=True)
    run_all.add_argument("--workers", type=int, default=1)
    run_all.add_argument("--allow-weight-download", action="store_true")
    run_all.add_argument("--resume", action="store_true")
    run_all.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        forwarded = ["--manifest", args.manifest, "--output-root", args.protocol_root]
        for flag in ("--force", "--skip-path-check"):
            if getattr(args, flag[2:].replace("-", "_")):
                forwarded.append(flag)
        return _module("lidpair.prepare", forwarded)
    if args.command == "train":
        forwarded = [
            "--protocol-root", args.protocol_root, "--image-root", args.image_root,
            "--output-root", args.output_root, "--stage", args.stage,
            "--outer-fold", str(args.outer_fold), "--seed", str(args.seed),
        ]
        optional = {
            "--inner-fold": args.inner_fold,
            "--backbone": args.backbone,
            "--selection": args.selection,
            "--training-epochs": args.training_epochs,
        }
        for flag, value in optional.items():
            if value is not None:
                forwarded.extend([flag, str(value)])
        for flag in ("--pretrained", "--allow-weight-download", "--resume", "--force"):
            if getattr(args, flag[2:].replace("-", "_")):
                forwarded.append(flag)
        return _module("lidpair.run_cv5", forwarded)
    if args.command == "all":
        return _run_all(args)
    import pandas as pd
    from lidpair.metrics import patient_metrics

    frame = pd.read_csv(Path(args.predictions).resolve())
    required = {"label", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        parser.error(f"prediction CSV is missing columns: {sorted(missing)}")
    for column, requested in (("budget", args.budget), ("variant", args.variant)):
        if column in frame.columns:
            if requested is None:
                parser.error(f"--{column} is required when the prediction CSV contains that column")
            frame = frame[frame[column].astype(str).eq(str(requested))].copy()
    if "seed" in frame.columns and frame["seed"].nunique() != 1:
        parser.error("eval refuses to pool multiple seeds")
    if "patient_group" in frame.columns and frame["patient_group"].duplicated().any():
        parser.error("eval requires at most one row per patient_group")
    if frame.empty:
        parser.error("the requested prediction slice is empty")
    decision_column = next((name for name in ("decision", "predicted", "patient_decision")
                            if name in frame.columns), None)
    if decision_column is not None and args.threshold is not None:
        parser.error("provide either --threshold or a saved patient decision column")
    if decision_column is not None:
        values = patient_metrics(frame["label"], frame["probability"],
                                 decisions=frame[decision_column])
    elif args.threshold is not None:
        values = patient_metrics(frame["label"], frame["probability"],
                                 threshold=args.threshold)
    else:
        # The metric helper's default threshold is validation-derived.  It is
        # intentionally not used for a transported/public evaluation slice.
        values = patient_metrics(frame["label"], frame["probability"], threshold=0.5,
                                 include_extended=False)
        for key in ("threshold", "sensitivity", "specificity", "fnr", "ppv", "npv",
                    "true_positive", "false_negative", "true_negative", "false_positive"):
            values.pop(key, None)
    print(json.dumps(values, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
