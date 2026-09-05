import unittest
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import pandas as pd

from lidpair.metrics import select_threshold_at_specificity
from lidpair.model import aggregate_image_logits
from lidpair.run_cv5 import _checkpoint_eligible, _training_policy, _validate_training_epochs


class PublicCoreTests(unittest.TestCase):
    def test_aggregation_rules(self):
        values = torch.tensor([[1.0, 3.0, 2.0, -1.0]], dtype=torch.float32)
        self.assertAlmostEqual(float(aggregate_image_logits(values, "mean")[0]), 1.25, places=5)
        self.assertAlmostEqual(float(aggregate_image_logits(values, "top2_mean")[0]), 2.5, places=5)
        self.assertAlmostEqual(float(aggregate_image_logits(values, "max")[0]), 3.0, places=5)

    def test_top2_k1_equals_mean_and_max(self):
        values = torch.tensor([[2.0]], dtype=torch.float32)
        self.assertEqual(float(aggregate_image_logits(values, "mean")[0]), 2.0)
        self.assertEqual(float(aggregate_image_logits(values, "top2_mean")[0]), 2.0)
        self.assertEqual(float(aggregate_image_logits(values, "max")[0]), 2.0)

    def test_production_cycle_policy(self):
        config = json.loads((Path(__file__).parents[1] / "lidpair" / "contract.json").read_text())
        policy = _training_policy(config)
        self.assertEqual(policy["schedule"], (1, 2, 3, 4))
        self.assertEqual(policy["minimum_cycles"], 3)
        self.assertEqual(policy["patience_cycles"], 3)
        self.assertEqual(policy["min_delta"], 0.0005)
        for epoch in range(1, 12):
            self.assertFalse(_checkpoint_eligible(config, epoch))
        for epoch in (12, 16, 60):
            self.assertTrue(_checkpoint_eligible(config, epoch))
            _validate_training_epochs(config, epoch)
        with self.assertRaises(ValueError):
            _validate_training_epochs(config, 13)

    def test_threshold_selection_is_specificity_constrained(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
        selected = select_threshold_at_specificity(labels, probabilities, minimum_specificity=0.90)
        self.assertGreaterEqual(selected.specificity, 0.90)

    def test_eval_cli_roundtrip_selects_one_budget_and_variant(self):
        root = Path(__file__).parents[1]
        (root / "outputs").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "outputs", prefix="synthetic_cli_test_") as temp_dir:
            temp_root = Path(temp_dir)
            rows = []
            for budget, offset in ((1, 0.0), (4, 0.1)):
                for group, label, probability in (("a", 0, 0.1 + offset), ("b", 0, 0.2 + offset),
                                                   ("c", 1, 0.8 + offset), ("d", 1, 0.9)):
                    rows.append({"patient_group": group, "label": label, "probability": probability,
                                 "budget": budget, "variant": "mean", "seed": 42})
            path = temp_root / "predictions.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = subprocess.run([sys.executable, str(root / "run_experiment.py"), "eval",
                                     "--predictions", str(path), "--budget", "4", "--variant", "mean"],
                                    cwd=root, capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["n_patients"], 4)
            self.assertEqual(payload["n_malignant"], 2)
            self.assertNotIn("threshold", payload)
            self.assertNotIn("sensitivity", payload)
            rejected = subprocess.run([sys.executable, str(root / "run_experiment.py"), "eval",
                                       "--predictions", str(path)], cwd=root,
                                      capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)

    def test_all_dispatches_ninety_training_routes_and_full_summaries(self):
        import run_experiment

        root = Path(__file__).parents[1]
        temp_root = root / "outputs" / "synthetic_dispatch_test"
        output = temp_root / "runs"
        protocol = temp_root / "protocol"
        analysis = temp_root / "learned"
        calls = []

        def fake_module(module, argv):
            calls.append((module, list(argv)))
            if module == "lidpair.select" and "--mode" in argv:
                outer = argv[argv.index("--outer-fold") + 1]
                selection_root = output / "selection" / f"outer_{outer}"
                selection_root.mkdir(parents=True, exist_ok=True)
                if argv[argv.index("--mode") + 1] == "backbone":
                    (selection_root / "backbone_selection.json").write_text(
                        json.dumps({"backbone": "convnext_tiny"})
                    )
                else:
                    (selection_root / "mean_selection.json").write_text(
                        json.dumps({"training_epochs": 12})
                    )
            return 0

        class Args:
            manifest = "manifest.csv"
            image_root = "images"
            protocol_root = str(protocol)
            output_root = str(output)
            analysis_root = str(analysis)
            workers = 1
            allow_weight_download = True
            resume = True
            force = False

        try:
            with patch.object(run_experiment, "_module", side_effect=fake_module):
                self.assertEqual(run_experiment._run_all(Args()), 0)
            train_calls = [(m, a) for m, a in calls if m == "lidpair.run_cv5"]
            self.assertEqual(len(train_calls), 90)
            self.assertEqual(sum("--stage" in a and a[a.index("--stage") + 1] == "inner" for _, a in train_calls), 45)
            self.assertEqual(sum("--stage" in a and a[a.index("--stage") + 1] == "outer" for _, a in train_calls), 15)
            self.assertEqual(sum("--stage" in a and a[a.index("--stage") + 1] == "architecture_robustness" for _, a in train_calls), 30)
            for _, argv in train_calls:
                self.assertIn("--allow-weight-download", argv)
                self.assertIn("--resume", argv)
            self.assertEqual(sum(m == "lidpair.backbone_robustness" for m, _ in calls), 1)
            self.assertEqual(sum(m == "lidpair.learned_aggregation" for m, _ in calls), 1)
            learned = next(a for m, a in calls if m == "lidpair.learned_aggregation")
            self.assertNotIn("--force", learned)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
