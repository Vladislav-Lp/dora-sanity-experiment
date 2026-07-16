import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dora_study.digits import load_split
from dora_study.models import AdapterLinear, AdaptedMLP, BaseMLP, count_trainable_parameters
from dora_study.synthetic import constructed_dora, make_problem, relative_weight_error, svd_lora_oracle


class AdapterTests(unittest.TestCase):
    def test_lora_and_dora_are_noops_at_initialization(self) -> None:
        torch.manual_seed(7)
        weight = torch.randn(6, 9)
        for method in ["lora", "dora"]:
            layer = AdapterLinear(weight, method=method, rank=3, init_seed=99)
            self.assertTrue(torch.allclose(layer.effective_weight(), weight, atol=1e-6, rtol=1e-6))

    def test_parameter_counts(self) -> None:
        base = BaseMLP()
        lora = AdaptedMLP(base, method="lora", rank=4, init_seed=1)
        dora = AdaptedMLP(base, method="dora", rank=4, init_seed=1)
        self.assertEqual(count_trainable_parameters(lora), 1832)
        self.assertEqual(count_trainable_parameters(dora), 2034)


class SyntheticTests(unittest.TestCase):
    def test_rank_four_positive_control_is_exact(self) -> None:
        problem = make_problem(100, true_rank=4, magnitude_strength=0.0)
        lora = svd_lora_oracle(problem.base_weight, problem.target_weight, rank=4)
        dora = constructed_dora(problem, rank=4)
        self.assertLess(relative_weight_error(lora, problem.target_weight), 1e-5)
        self.assertLess(relative_weight_error(dora, problem.target_weight), 1e-5)

    def test_dora_represents_mixed_ground_truth(self) -> None:
        problem = make_problem(100, true_rank=4, magnitude_strength=0.8)
        dora = constructed_dora(problem, rank=4)
        self.assertLess(relative_weight_error(dora, problem.target_weight), 1e-5)


class DataTests(unittest.TestCase):
    def test_digits_split_is_deterministic_and_disjoint_in_size(self) -> None:
        first = load_split()
        second = load_split()
        self.assertEqual((len(first.x_train), len(first.x_val), len(first.x_test)), (1077, 360, 360))
        self.assertTrue((first.x_train == second.x_train).all())
        self.assertTrue((first.y_test == second.y_test).all())


if __name__ == "__main__":
    unittest.main()
