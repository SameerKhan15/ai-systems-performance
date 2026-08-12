"""Correctness tests for the prefill latency lab.

Run from this directory with:
    python -m unittest -v
"""

from __future__ import annotations

import math
import unittest

import torch

from prefill_latency_lab import (
    LabConfig,
    PrefillAttentionBlock,
    calculate_prefill_accounting,
    fit_power_law,
    make_causal_mask,
    nearest_rank_percentile,
)


class PrefillAccountingTests(unittest.TestCase):
    def test_accounting_matches_closed_form(self) -> None:
        result = calculate_prefill_accounting(
            batch_size=1,
            prompt_length=5,
            embed_dim=16,
            num_heads=2,
            dtype_bytes=4,
        )

        self.assertEqual(result.projection_macs, 4 * 5 * 16 * 16)
        self.assertEqual(result.attention_score_macs, 1 * 2 * 5 * 5 * 8)
        self.assertEqual(result.attention_value_macs, 1 * 2 * 5 * 5 * 8)
        self.assertEqual(
            result.total_macs,
            4 * 5 * 16 * 16 + 2 * 5 * 5 * 16,
        )
        self.assertEqual(result.logical_score_elements, 1 * 2 * 5 * 5)
        self.assertEqual(result.logical_score_bytes, 1 * 2 * 5 * 5 * 4)
        self.assertEqual(result.qkv_bytes, 3 * 1 * 5 * 16 * 4)

    def test_doubling_prompt_length_quadruples_attention_macs(self) -> None:
        small = calculate_prefill_accounting(
            batch_size=1,
            prompt_length=128,
            embed_dim=256,
            num_heads=8,
            dtype_bytes=4,
        )
        large = calculate_prefill_accounting(
            batch_size=1,
            prompt_length=256,
            embed_dim=256,
            num_heads=8,
            dtype_bytes=4,
        )

        self.assertEqual(large.projection_macs, 2 * small.projection_macs)
        self.assertEqual(large.attention_score_macs, 4 * small.attention_score_macs)
        self.assertEqual(large.logical_score_bytes, 4 * small.logical_score_bytes)


class ModelCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = LabConfig(
            embed_dim=16,
            num_heads=2,
            batch_size=1,
            vocab_size=128,
            dtype=torch.float32,
            device="cpu",
            attention_backend="manual",
        )
        self.model = PrefillAttentionBlock(self.cfg, seed=7)

    def test_output_shape(self) -> None:
        token_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        causal_mask = make_causal_mask(prompt_length=5, device="cpu")

        with torch.inference_mode():
            output = self.model(token_ids, causal_mask)

        self.assertEqual(tuple(output.shape), (1, 5, 16))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_future_token_does_not_change_earlier_outputs(self) -> None:
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        longer = torch.tensor([[1, 2, 3, 4, 99]], dtype=torch.long)

        with torch.inference_mode():
            prefix_output = self.model(
                prefix,
                make_causal_mask(prompt_length=4, device="cpu"),
            )
            longer_output = self.model(
                longer,
                make_causal_mask(prompt_length=5, device="cpu"),
            )

        torch.testing.assert_close(
            prefix_output,
            longer_output[:, :4, :],
            rtol=1e-5,
            atol=1e-6,
        )


class StatisticsTests(unittest.TestCase):
    def test_nearest_rank_p95(self) -> None:
        values = [float(value) for value in range(1, 21)]
        self.assertEqual(nearest_rank_percentile(values, 95.0), 19.0)

    def test_power_law_fit_recovers_quadratic_exponent(self) -> None:
        lengths = [64, 128, 256, 512]
        latencies = [0.001 * (length**2) for length in lengths]

        alpha, coefficient, r_squared = fit_power_law(lengths, latencies)

        self.assertTrue(math.isclose(alpha, 2.0, rel_tol=1e-12))
        self.assertTrue(math.isclose(coefficient, 0.001, rel_tol=1e-12))
        self.assertTrue(math.isclose(r_squared, 1.0, rel_tol=1e-12))


if __name__ == "__main__":
    unittest.main()