"""Lesson 001: vectors and data representation.

The functions in this module intentionally expose validation and Shape checks.
They are small enough to compare directly with the mathematical definitions in
``lessons/001-vectors-and-data-representation.md``.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector: TypeAlias = NDArray[np.float64]


def as_float_vector(values: ArrayLike, *, name: str = "vector") -> FloatVector:
    """Return *values* as a finite, non-empty, one-dimensional float vector.

    Args:
        values: Any array-like value accepted by NumPy.
        name: Human-readable input name used in validation errors.

    Raises:
        TypeError: If the input cannot be converted to floating-point values.
        ValueError: If the result is not one-dimensional, is empty, or contains
            NaN or infinity.
    """

    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc

    if vector.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; received shape {vector.shape}"
        )
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")

    return vector


def _matching_vectors(x: ArrayLike, y: ArrayLike) -> tuple[FloatVector, FloatVector]:
    """Validate and return two vectors with equal Shape."""

    x_vector = as_float_vector(x, name="x")
    y_vector = as_float_vector(y, name="y")

    if x_vector.shape != y_vector.shape:
        raise ValueError(
            "x and y must have the same shape; "
            f"received {x_vector.shape} and {y_vector.shape}"
        )

    return x_vector, y_vector


def vector_add(x: ArrayLike, y: ArrayLike) -> FloatVector:
    """Compute vector addition after explicit Shape validation."""

    x_vector, y_vector = _matching_vectors(x, y)
    return x_vector + y_vector


def scalar_multiply(scalar: float, vector: ArrayLike) -> FloatVector:
    """Multiply every vector component by a finite scalar."""

    scalar_value = float(scalar)
    if not np.isfinite(scalar_value):
        raise ValueError("scalar must be finite")

    return scalar_value * as_float_vector(vector)


def dot_product(x: ArrayLike, y: ArrayLike) -> float:
    """Compute a dot product from its element-wise mathematical definition.

    This deliberately uses ``sum(x_i * y_i)`` rather than ``np.dot`` so the
    implementation mirrors the formula introduced in the lesson.
    """

    x_vector, y_vector = _matching_vectors(x, y)
    return float(np.sum(x_vector * y_vector))


def l1_norm(vector: ArrayLike) -> float:
    """Return the L1 norm: sum of absolute component values."""

    validated = as_float_vector(vector)
    return float(np.sum(np.abs(validated)))


def l2_norm(vector: ArrayLike) -> float:
    """Return the Euclidean (L2) norm."""

    validated = as_float_vector(vector)
    return float(np.sqrt(np.sum(validated**2)))


def cosine_similarity(x: ArrayLike, y: ArrayLike) -> float:
    """Return cosine similarity for two non-zero vectors.

    Raises:
        ValueError: If either vector is a zero vector, because its direction is
            undefined and the cosine denominator would be zero.
    """

    x_vector, y_vector = _matching_vectors(x, y)
    x_norm = l2_norm(x_vector)
    y_norm = l2_norm(y_vector)

    if x_norm == 0.0 or y_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    similarity = dot_product(x_vector, y_vector) / (x_norm * y_norm)

    # Floating-point roundoff can produce values such as 1.0000000000000002.
    return float(np.clip(similarity, -1.0, 1.0))


def linear_score(features: ArrayLike, weights: ArrayLike, bias: float = 0.0) -> float:
    """Compute ``features @ weights + bias`` with explicit validation."""

    feature_vector, weight_vector = _matching_vectors(features, weights)
    bias_value = float(bias)
    if not np.isfinite(bias_value):
        raise ValueError("bias must be finite")

    return dot_product(feature_vector, weight_vector) + bias_value


def main() -> None:
    """Run a small example from the lesson."""

    feature_names = ("area_m2", "bedrooms", "age_years")
    features = np.array([80.0, 2.0, 10.0])
    weights = np.array([1.5, 20.0, -0.8])
    bias = 5.0

    print("Feature contract:")
    for name, value in zip(feature_names, features, strict=True):
        print(f"  {name}: {value}")

    print(f"features.shape = {features.shape}")
    print(f"weights.shape  = {weights.shape}")
    print(f"dot product    = {dot_product(features, weights):.2f}")
    print(f"linear score   = {linear_score(features, weights, bias):.2f}")
    print(f"L1 norm        = {l1_norm(features):.2f}")
    print(f"L2 norm        = {l2_norm(features):.2f}")


if __name__ == "__main__":
    main()
