import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dora_study.digits import balanced_nested_indices, build_candidate, load_split, make_optimizer
from dora_study.models import (
    AdapterConv2d,
    AdapterLinear,
    AdaptedCNN,
    AdaptedMLP,
    BaseCNN,
    BaseMLP,
    count_trainable_parameters,
)
from dora_study.synthetic import (
    constructed_dora,
    direction_error,
    make_problem,
    relative_weight_error,
    svd_lora_oracle,
)


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

    def test_parameter_matched_allocations(self) -> None:
        base = BaseMLP()
        matched_lora = AdaptedMLP(base, method="lora", rank=(5, 4, 4), init_seed=1)
        budgeted_dora = AdaptedMLP(base, method="dora", rank=(4, 3, 3), init_seed=1)
        self.assertEqual(count_trainable_parameters(matched_lora), 2024)
        self.assertEqual(count_trainable_parameters(budgeted_dora), 1768)
        self.assertLessEqual(count_trainable_parameters(budgeted_dora), 1832)

    def test_convolution_adapters_are_noops_at_initialization(self) -> None:
        torch.manual_seed(8)
        weight = torch.randn(5, 3, 3, 3)
        for method in ["lora", "dora"]:
            layer = AdapterConv2d(weight, method=method, rank=2, padding=1, init_seed=44)
            self.assertTrue(torch.allclose(layer.effective_weight(), weight, atol=1e-6, rtol=1e-6))

    def test_adapted_cnn_matches_base_at_initialization(self) -> None:
        torch.manual_seed(9)
        base = BaseCNN()
        features = torch.randn(7, 64)
        for method in ["lora", "dora"]:
            adapted = AdaptedCNN(base, method=method, rank=4, init_seed=10)
            self.assertTrue(torch.allclose(adapted(features), base(features), atol=1e-5, rtol=1e-5))

    def test_lora_plus_optimizer_uses_declared_ratio(self) -> None:
        model = build_candidate(BaseMLP(), method="lora_plus", rank=4, seed=1)
        optimizer = make_optimizer(model, method="lora_plus", learning_rate=1e-3, lora_plus_ratio=16.0)
        rates = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
        self.assertAlmostEqual(rates["lora_a"], 1e-3)
        self.assertAlmostEqual(rates["lora_b"], 16e-3)


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

    def test_direction_error_respects_cosine_distance_bounds(self) -> None:
        weight = torch.randn(8, 12)
        self.assertGreaterEqual(direction_error(weight, weight), 0.0)


class DataTests(unittest.TestCase):
    def test_digits_split_is_deterministic_and_disjoint_in_size(self) -> None:
        first = load_split()
        second = load_split()
        self.assertEqual((len(first.x_train), len(first.x_val), len(first.x_test)), (1077, 360, 360))
        self.assertTrue((first.x_train == second.x_train).all())
        self.assertTrue((first.y_test == second.y_test).all())

    def test_data_regime_subsets_are_balanced_and_nested(self) -> None:
        split = load_split()
        small = balanced_nested_indices(split.y_train, per_class=5, seed=301)
        large = balanced_nested_indices(split.y_train, per_class=10, seed=301)
        self.assertEqual(len(small), 50)
        self.assertEqual(len(large), 100)
        self.assertTrue(set(small).issubset(set(large)))
        counts = torch.bincount(torch.from_numpy(split.y_train[small]))
        self.assertTrue(torch.equal(counts, torch.full((10,), 5, dtype=counts.dtype)))


if __name__ == "__main__":
    unittest.main()
