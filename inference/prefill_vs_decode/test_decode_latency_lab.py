"""Correctness tests for the decode latency lab.

Run from this directory with:
    python -m unittest -v test_decode_latency_lab.py
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from decode_latency_lab import (
    DecodeAttentionBlock,
    LabConfig,
    calculate_decode_accounting,
    fit_power_law,
    make_synthetic_kv_cache,
    nearest_rank_percentile,
)


class DecodeAccountingTests(unittest.TestCase):
    def test_accounting_matches_closed_form(self) -> None:
        result = calculate_decode_accounting(
            batch_size=1,
            context_length=5,
            embed_dim=16,
            num_heads=2,
            dtype_bytes=4,
        )
        self.assertEqual(result.projection_macs, 4 * 16 * 16)
        self.assertEqual(result.attention_score_macs, 1 * 2 * 5 * 8)
        self.assertEqual(result.attention_value_macs, 1 * 2 * 5 * 8)
        self.assertEqual(result.attention_macs, 2 * 5 * 16)
        self.assertEqual(result.total_macs, 4 * 16 * 16 + 2 * 5 * 16)
        self.assertEqual(result.logical_kv_elements, 2 * 1 * 5 * 16)
        self.assertEqual(result.logical_kv_bytes, 2 * 1 * 5 * 16 * 4)
        self.assertEqual(result.logical_score_elements, 1 * 2 * 5)
        self.assertEqual(result.logical_score_bytes, 1 * 2 * 5 * 4)
        self.assertEqual(result.logical_kv_write_bytes, 2 * 1 * 16 * 4)

    def test_doubling_context_doubles_attention_and_kv_not_projection(self) -> None:
        small = calculate_decode_accounting(
            batch_size=1,
            context_length=128,
            embed_dim=256,
            num_heads=8,
            dtype_bytes=2,
        )
        large = calculate_decode_accounting(
            batch_size=1,
            context_length=256,
            embed_dim=256,
            num_heads=8,
            dtype_bytes=2,
        )
        self.assertEqual(large.projection_macs, small.projection_macs)
        self.assertEqual(large.attention_macs, 2 * small.attention_macs)
        self.assertEqual(large.logical_kv_bytes, 2 * small.logical_kv_bytes)
        self.assertEqual(
            large.logical_kv_write_bytes,
            small.logical_kv_write_bytes,
        )


class DecodeModelCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = LabConfig(
            embed_dim=16,
            num_heads=2,
            batch_size=1,
            vocab_size=128,
            dtype=torch.float32,
            device="cpu",
        )
        self.model = DecodeAttentionBlock(self.cfg, seed=7)

    def test_output_shape_and_finiteness(self) -> None:
        token_ids = torch.tensor([[3]], dtype=torch.long)
        k_cache, v_cache = make_synthetic_kv_cache(
            cfg=self.cfg,
            context_length=5,
            seed=11,
        )
        with torch.inference_mode():
            output = self.model(token_ids, k_cache, v_cache)
        self.assertEqual(tuple(output.shape), (1, 1, 16))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_only_final_cache_slot_is_overwritten(self) -> None:
        token_ids = torch.tensor([[4]], dtype=torch.long)
        k_cache, v_cache = make_synthetic_kv_cache(
            cfg=self.cfg,
            context_length=5,
            seed=12,
        )
        original_k_prefix = k_cache[:, :, :-1, :].clone()
        original_v_prefix = v_cache[:, :, :-1, :].clone()

        with torch.inference_mode():
            _, expected_k_new, expected_v_new = self.model.project_new_token(token_ids)
            _ = self.model(token_ids, k_cache, v_cache)

        torch.testing.assert_close(k_cache[:, :, :-1, :], original_k_prefix)
        torch.testing.assert_close(v_cache[:, :, :-1, :], original_v_prefix)
        torch.testing.assert_close(k_cache[:, :, -1:, :], expected_k_new)
        torch.testing.assert_close(v_cache[:, :, -1:, :], expected_v_new)

    def test_forward_matches_explicit_reference(self) -> None:
        token_ids = torch.tensor([[9]], dtype=torch.long)
        k_cache, v_cache = make_synthetic_kv_cache(
            cfg=self.cfg,
            context_length=7,
            seed=13,
        )
        k_reference = k_cache.clone()
        v_reference = v_cache.clone()

        with torch.inference_mode():
            actual = self.model(token_ids, k_cache, v_cache)

            q, k_new, v_new = self.model.project_new_token(token_ids)
            k_reference[:, :, -1:, :] = k_new
            v_reference[:, :, -1:, :] = v_new
            scores = torch.matmul(
                q, k_reference.transpose(-2, -1)
            ) / math.sqrt(self.cfg.head_dim)
            weights = F.softmax(scores, dim=-1)
            attention_output = torch.matmul(weights, v_reference)
            merged = self.model.merge_heads(attention_output)
            expected = self.model.wo(merged)

        torch.testing.assert_close(actual, expected)


class StatisticsTests(unittest.TestCase):
    def test_nearest_rank_p95(self) -> None:
        values = list(range(1, 21))
        self.assertEqual(nearest_rank_percentile(values, 95.0), 19)

    def test_power_law_linear_exponent(self) -> None:
        context_lengths = [64, 128, 256, 512]
        latencies = [0.1 * value for value in context_lengths]
        alpha, coefficient, r_squared = fit_power_law(
            context_lengths,
            latencies,
        )
        self.assertAlmostEqual(alpha, 1.0, places=12)
        self.assertAlmostEqual(coefficient, 0.1, places=12)
        self.assertAlmostEqual(r_squared, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()