"""Embedding → RGB conversion via Oklab color space.

Отображает BGE-M3 эмбеддинг (1024D) в перцептивно равномерный цвет sRGB.
Семантически близкие эмбеддинги → визуально близкие цвета.
"""

from typing import Union

import numpy as np

# Фиксированная матрица проекции 1024 → 3.
# Генерируем один раз с фиксированным seed, чтобы один и тот же эмбеддинг
# всегда давал один и тот же цвет.
_PROJECTION_MATRIX = np.random.RandomState(42).randn(3, 1024)

_BGE_DIM = 1024


def _oklab_to_linear_rgb(L: float, a: float, b: float) -> tuple[float, float, float]:
    """Oklab → линейный RGB."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l_lin = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r = +4.0767416621 * l_lin - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l_lin + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l_lin - 0.7034186147 * m + 1.7076147010 * s

    return r, g, b


def _linear_rgb_to_srgb(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Гамма-коррекция: линейный RGB → sRGB."""
    def _clamp_and_correct(c: float) -> float:
        c = np.clip(c, 0.0, 1.0)
        return float(np.where(
            c <= 0.0031308,
            12.92 * c,
            1.055 * np.power(c, 1.0 / 2.4) - 0.055,
        ))

    return _clamp_and_correct(r), _clamp_and_correct(g), _clamp_and_correct(b)


def embedding_to_rgb(embedding: Union[list, np.ndarray]) -> tuple[int, int, int]:
    """BGE-M3 эмбеддинг (1024D) → RGB (0-255).

    Args:
        embedding: dense-вектор размерности 1024 (нормализованный).

    Returns:
        (R, G, B) — каждый канал 0-255.

    Raises:
        ValueError: если размерность не 1024.
    """
    vec = np.asarray(embedding, dtype=np.float32)

    if vec.shape[0] != _BGE_DIM:
        raise ValueError(
            f"Ожидается вектор размерности {_BGE_DIM}, получен {vec.shape[0]}"
        )

    # 1. Случайная проекция: 1024D → 3D.
    # BGE-векторы нормализованы (длина ~1), поэтому проекция даёт
    # значения со стандартным отклонением ~1, т.е. в диапазоне [-3, 3].
    proj_3d = _PROJECTION_MATRIX @ vec

    # 2. 3D координаты → Oklab.
    # L (Lightness) ∈ [0, 1], центрируем вокруг 0.6 с размахом ±0.3.
    # a и b (chroma) центрируем в 0 с размахом ±0.15 (чтобы не выйти за sRGB).
    L = 0.6 + 0.15 * float(proj_3d[0])
    a = 0.12 * float(proj_3d[1])
    b = 0.12 * float(proj_3d[2])

    # 3. Oklab → линейный RGB → sRGB
    r_lin, g_lin, b_lin = _oklab_to_linear_rgb(L, a, b)
    r_srgb, g_srgb, b_srgb = _linear_rgb_to_srgb(r_lin, g_lin, b_lin)

    # 4. Конвертация в 0-255
    return (
        int(round(r_srgb * 255)),
        int(round(g_srgb * 255)),
        int(round(b_srgb * 255)),
    )
