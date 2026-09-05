#!/usr/bin/env python3
"""Emit the exact LidMerge matrix without initiating any training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lidpair.validate import validate_protocol


ROOT = Path(__file__).resolve().parents[1]


def _load_lock(protocol_root: Path) -> dict:
    return json.loads((protocol_root / "protocol_lock.json").read_text(encoding="utf-8"))


def build_matrix(protocol_root: Path) -> dict:
    audit = validate_protocol(protocol_root, check_paths=False)
    lock = _load_lock(protocol_root)
    contract = lock["contract"]
    outer = int(contract["validation"]["outer_folds"])
    inner = int(contract["validation"]["inner_folds"])
    backbones = [str(value) for value in contract["backbone_screen"]]
    aggregation = contract["aggregation"]
    training_aggregation = str(aggregation["training"])
    inference_aggregations = [str(value) for value in aggregation["inference"]]
    formal_seeds = [int(value) for value in contract["validation"]["formal_seeds"]]
    screen = outer * inner * len(backbones)
    # Every mean-trained route emits mean, top-2 mean, and max inference
    # aggregations. The fixed primary backbone's seed-42 screen route is
    # reused for all aggregation threshold locks, so only its two remaining
    # formal seeds run.
    main_inner = outer * inner * (len(formal_seeds) - 1)
    main_outer = outer * len(formal_seeds)
    robustness = contract["architecture_robustness"]
    robustness_backbones = [str(value) for value in robustness["backbones"]]
    robustness_seeds = [int(value) for value in robustness["seeds"]]
    # In each outer fold the selected primary route already supplies one of the
    # three architecture results.  Only the two nonselected backbones need a
    # separate supporting route.
    architecture_robustness = outer * (len(robustness_backbones) - 1) * len(robustness_seeds)
    core = screen + main_inner + main_outer + architecture_robustness
    budgets = [int(value) for value in contract["views"]["budgets"]]
    cpu_postprocessing = (
        len(contract["cpu_logit_postprocessing"]["aggregators"])
        * len(budgets)
        * outer
        * len(formal_seeds)
    )
    return {
        "protocol_id": contract["protocol_id"],
        "objective": {
            "primary_question": "Within one fixed patient_group cohort with at least four routine photographs, how does a blinded photographic budget K=1,2,3,4 change malignant eyelid-tumor triage performance?",
            "primary_endpoint": contract["validation"]["primary_endpoint"],
            "claim_boundary": contract["claim_boundary"],
        },
        "patient_level_boundary": {
            "unit": "patient_group",
            "primary_cohort": int(contract["data"]["expected_primary_groups"]),
            "photographic_budgets": budgets,
            "common_cohort_for_every_budget": True,
            "allowed_learning_fields": ["patient_group", "label"],
            "operational_only": ["image_id and image path resolve frozen photographs; neither enters a model, split, selection, or statistic"],
            "excluded_from_model_split_selection_and_statistics": ["record_role", "source", "batch", "photograph count", "all other manifest metadata"],
        },
        "selection_firewall": {
            "backbone": f"The primary backbone is fixed to {contract['primary_backbone']}; the supporting architecture routes are descriptive and cannot select the primary model.",
            "threshold": "For each prespecified inference aggregation and each K, the threshold is selected from matching inner-fold seed-ensemble predictions only; it does not train a separate max model.",
            "outer_test": "Outer-test patient groups never participate in backbone, K, threshold, or aggregation selection.",
        },
        "model": {
            "training_aggregation": training_aggregation,
            "inference_aggregations": {"mean": "symmetric average of the shared image logits", "top2_mean": "mean of the two largest shared image logits", "max": "symmetric decisive-image comparator over those same logits"},
            "training": "One K-agnostic shared image model is trained once with mean-logit BCE and a complete-cycle K=1..4 schedule, then all fixed aggregations are evaluated from the same per-image logits.",
            "excluded": contract["model"]["explicitly_excluded"],
        },
        "phases": [
            {
                "name": "preflight",
                "required_artifacts": ["protocol lock", "patient-level fold validation", "model/unit smoke", "CUDA 224px K=4 backward smoke"],
                "training_routes": 0,
            },
            {
                "name": "development_pilot",
                "confirmatory": False,
                "outer_test_evaluated": False,
                "training_aggregation": str(contract["pilot"]["training_aggregation"]),
                "inference_aggregations": contract["pilot"]["inference_aggregations"],
                "training_routes": 1,
                "counted_in_formal_matrix": False,
            },
            {
                "name": "backbone_screen",
                "models": backbones,
                "training_aggregation": training_aggregation,
                "inference_aggregations": inference_aggregations,
                "budget_for_selection": int(contract["validation"]["backbone_screen_budget"]),
                "training_routes": screen,
                "all_budget_predictions_written": budgets,
            },
            {
                "name": "main_inner_selection",
                "training_aggregation": training_aggregation,
                "inference_aggregations": inference_aggregations,
                "seed_ensemble": formal_seeds,
                "training_routes": main_inner,
                "all_budget_predictions_written": budgets,
            },
            {
                "name": "main_outer_confirmation",
                "training_aggregation": training_aggregation,
                "inference_aggregations": inference_aggregations,
                "formal_seeds": formal_seeds,
                "training_routes": main_outer,
                "inference_only": ["alternate blinded photograph orders", "prefix reversal", "duplicate-view null"],
            },
            {
                "name": "architecture_robustness",
                "supporting_analysis_only": True,
                "models": robustness_backbones,
                "training_aggregation": str(robustness["aggregation"]),
                "formal_seeds": robustness_seeds,
                "training_routes": architecture_robustness,
                "selected_backbone_routes_reused_from_primary_outer": outer * len(robustness_seeds),
                "all_budget_predictions_written": budgets,
                "selection_allowed": False,
            },
            {
                "name": "cpu_logit_postprocessing",
                "training_routes": 0,
                "derived_jobs": cpu_postprocessing,
                "aggregators": contract["cpu_logit_postprocessing"]["aggregators"],
                "budgets": budgets,
                "source_artifact": contract["cpu_logit_postprocessing"]["input_artifact"],
                "nested_within_outer_fold": True,
                "training_free": True,
                "selection_allowed": False,
            },
        ],
        "route_counts": {
            "backbone_screen": screen,
            "main_inner_new": main_inner,
            "main_outer": main_outer,
            "architecture_robustness": architecture_robustness,
            "core_training_routes": core,
            "cpu_logit_postprocessing_jobs": cpu_postprocessing,
            "inference_budgets_do_not_multiply_training_routes": budgets,
        },
        "comparisons": contract["comparisons"],
        "required_completion_artifacts": [
            "one seed-ensemble OOF probability per patient_group, aggregation variant, and K",
            "per-outer-fold backbone and K-specific threshold locks from inner predictions",
            "patient-level bootstrap confidence intervals and paired comparisons",
            "raw ordered image logits, reconstructed patient-level predictions, run manifests, checkpoints, histories, and artifact hashes",
            "inference-only alternate-order and implementation-invariance summaries",
            "target-GPU production-backbone 224px K=4 backward-smoke report before training",
            "strictly nested CPU XGBoost, random-forest, and logistic-regression aggregation summaries from saved image logits",
        ],
        "audit": audit,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(ROOT / "lidmerge_protocol"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs_lidmerge_corrected_cycles"))
    parser.add_argument("--write", action="store_true", help="write matrix_plan.json under OUTPUT_ROOT/preflight")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matrix = build_matrix(Path(args.protocol_root).resolve())
    if args.write:
        path = Path(args.output_root).resolve() / "preflight" / "matrix_plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
        matrix["written_to"] = str(path)
    print(json.dumps(matrix, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
