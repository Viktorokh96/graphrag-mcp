"""Тесты для src/color_utils.py — проекция эмбеддинга в цвет через Oklab."""

import numpy as np
import pytest

from src.color_utils import (
    _BGE_DIM,
    _linear_rgb_to_srgb,
    _oklab_to_linear_rgb,
    embedding_to_rgb,
)


def _unit_vector(seed: int, dim: int = _BGE_DIM) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


class TestOklabToLinearRgb:
    def test_black_maps_to_zero(self):
        assert _oklab_to_linear_rgb(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    def test_white_is_neutral_grey(self):
        r, g, b = _oklab_to_linear_rgb(1.0, 0.0, 0.0)
        assert r == pytest.approx(1.0, abs=1e-3)
        assert g == pytest.approx(1.0, abs=1e-3)
        assert b == pytest.approx(1.0, abs=1e-3)

    def test_positive_a_shifts_towards_red(self):
        r, g, _ = _oklab_to_linear_rgb(0.6, 0.15, 0.0)
        assert r > g


class TestLinearRgbToSrgb:
    def test_clamps_out_of_range_values(self):
        r, g, b = _linear_rgb_to_srgb(-1.0, 2.0, 0.0)
        assert (r, b) == (0.0, 0.0)
        assert g == pytest.approx(1.0)

    def test_linear_segment_for_small_values(self):
        r, _, _ = _linear_rgb_to_srgb(0.001, 0.0, 0.0)
        assert r == pytest.approx(12.92 * 0.001, rel=1e-6)

    def test_gamma_segment_for_large_values(self):
        r, _, _ = _linear_rgb_to_srgb(0.5, 0.0, 0.0)
        assert r == pytest.approx(1.055 * 0.5 ** (1 / 2.4) - 0.055, rel=1e-6)


class TestEmbeddingToRgb:
    def test_returns_three_channels_in_range(self):
        r, g, b = embedding_to_rgb(_unit_vector(1))
        for channel in (r, g, b):
            assert isinstance(channel, int)
            assert 0 <= channel <= 255

    def test_deterministic_for_same_embedding(self):
        vec = _unit_vector(2)
        assert embedding_to_rgb(vec) == embedding_to_rgb(vec)

    def test_different_embeddings_give_different_colors(self):
        assert embedding_to_rgb(_unit_vector(3)) != embedding_to_rgb(_unit_vector(4))

    def test_accepts_numpy_array(self):
        vec = np.asarray(_unit_vector(5), dtype=np.float32)
        assert embedding_to_rgb(vec) == embedding_to_rgb(vec.tolist())

    def test_close_embeddings_give_close_colors(self):
        base = np.asarray(_unit_vector(6))
        noise = np.random.default_rng(7).standard_normal(_BGE_DIM) * 0.001
        near = (base + noise) / np.linalg.norm(base + noise)
        c1 = embedding_to_rgb(base)
        c2 = embedding_to_rgb(near)
        assert max(abs(a - b) for a, b in zip(c1, c2)) <= 10

    def test_zero_vector_is_mid_grey(self):
        assert embedding_to_rgb([0.0] * _BGE_DIM) == embedding_to_rgb(np.zeros(_BGE_DIM))

    @pytest.mark.parametrize("dim", [0, 3, 64, 768, 2048])
    def test_wrong_dimension_raises(self, dim):
        with pytest.raises(ValueError, match=str(_BGE_DIM)):
            embedding_to_rgb([0.0] * dim)
