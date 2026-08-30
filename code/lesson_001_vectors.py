"""Lesson 001: vectors and data representation.

Run:
    python code/lesson_001_vectors.py

The implementation intentionally uses basic NumPy operations so each formula
can be matched to the mathematics in the lesson.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


def as_vector(values: Sequence[float] | NDArray[np.number]) -> FloatVector:
    """Convert values to a finite one-dimensional float vector."""
    vector = np.asarray(values, dtype=np.float64)

    if vector.ndim != 1:
        raise ValueError(
            f"expected a one-dimensional vector, got shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite numbers")

    return vector


def require_same_shape(x: FloatVector, y: FloatVector) -> None:
    """Raise a clear error when two vectors cannot be paired element-wise."""
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} != {y.shape}")


def add(
    x_values: Sequence[float] | NDArray[np.number],
    y_values: Sequence[float] | NDArray[np.number],
) -> FloatVector:
    """Return x + y for two vectors with the same shape."""
    x = as_vector(x_values)
    y = as_vector(y_values)
    require_same_shape(x, y)
    return x + y


def scale(
    scalar: float,
    values: Sequence[float] | NDArray[np.number],
) -> FloatVector:
    """Multiply every vector element by a finite scalar."""
    if not np.isfinite(scalar):
        raise ValueError("scalar must be finite")
    return float(scalar) * as_vector(values)


def dot(
    x_values: Sequence[float] | NDArray[np.number],
    y_values: Sequence[float] | NDArray[np.number],
) -> float:
    """Compute a dot product directly from its summation definition."""
    x = as_vector(x_values)
    y = as_vector(y_values)
    require_same_shape(x, y)

    # Equivalent to sum(x_i * y_i). The explicit sum makes the definition
    # visible instead of delegating the lesson to np.dot.
    return float(np.sum(x * y))


def l1_norm(values: Sequence[float] | NDArray[np.number]) -> float:
    """Return the L1 norm: sum(abs(x_i))."""
    vector = as_vector(values)
    return float(np.sum(np.abs(vector)))


def l2_norm(values: Sequence[float] | NDArray[np.number]) -> float:
    """Return the L2 norm: sqrt(sum(x_i ** 2))."""
    vector = as_vector(values)
    return float(np.sqrt(np.sum(vector * vector)))


def euclidean_distance(
    x_values: Sequence[float] | NDArray[np.number],
    y_values: Sequence[float] | NDArray[np.number],
) -> float:
    """Return ||x - y||_2 for two vectors with the same shape."""
    x = as_vector(x_values)
    y = as_vector(y_values)
    require_same_shape(x, y)
    return l2_norm(x - y)


def cosine_similarity(
    x_values: Sequence[float] | NDArray[np.number],
    y_values: Sequence[float] | NDArray[np.number],
) -> float:
    """Return cosine similarity for two non-zero vectors.

    Cosine similarity is undefined when either input is a zero vector. Raising
    an error keeps "undefined" distinct from a valid similarity of zero.
    """
    x = as_vector(x_values)
    y = as_vector(y_values)
    require_same_shape(x, y)

    denominator = l2_norm(x) * l2_norm(y)
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    # Floating-point roundoff can produce values infinitesimally outside the
    # theoretical interval [-1, 1]. Clip only that numerical noise.
    similarity = dot(x, y) / denominator
    return float(np.clip(similarity, -1.0, 1.0))


def linear_score(
    weights_values: Sequence[float] | NDArray[np.number],
    features_values: Sequence[float] | NDArray[np.number],
    bias: float = 0.0,
) -> float:
    """Compute z = w^T x + b for one sample."""
    if not np.isfinite(bias):
        raise ValueError("bias must be finite")
    return dot(weights_values, features_values) + float(bias)


def batch_linear_scores(
    feature_matrix: Sequence[Sequence[float]] | NDArray[np.number],
    weights_values: Sequence[float] | NDArray[np.number],
    bias: float = 0.0,
) -> FloatVector:
    """Compute X @ w + b for a batch of samples.

    X has shape (n_samples, n_features), w has shape (n_features,), and the
    result has shape (n_samples,).
    """
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    weights = as_vector(weights_values)

    if matrix.ndim != 2:
        raise ValueError(f"expected a two-dimensional matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature matrix must contain only finite numbers")
    if matrix.shape[1] != weights.shape[0]:
        raise ValueError(
            "feature dimension mismatch: "
            f"matrix has {matrix.shape[1]} columns, weights have "
            f"{weights.shape[0]} elements"
        )
    if not np.isfinite(bias):
        raise ValueError("bias must be finite")

    return matrix @ weights + float(bias)


def run_self_checks() -> None:
    """Run small checks that mirror the hand calculations in the lesson."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, -1.0, 2.0])

    np.testing.assert_allclose(add(x, y), np.array([5.0, 1.0, 5.0]))
    np.testing.assert_allclose(scale(2.0, x), np.array([2.0, 4.0, 6.0]))
    assert np.isclose(dot(x, y), 8.0)
    assert np.isclose(l1_norm(np.array([3.0, -4.0])), 7.0)
    assert np.isclose(l2_norm(np.array([3.0, -4.0])), 5.0)
    assert np.isclose(
        euclidean_distance(np.array([1.0, 2.0]), np.array([4.0, 6.0])),
        5.0,
    )
    assert np.isclose(
        cosine_similarity(np.array([1.0, 0.0]), np.array([5.0, 0.0])),
        1.0,
    )
    assert np.isclose(
        cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 2.0])),
        0.0,
    )
    assert np.isclose(
        cosine_similarity(np.array([1.0, 0.0]), np.array([-3.0, 0.0])),
        -1.0,
    )

    features = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    weights = np.array([0.5, -1.0, 2.0])
    scores = batch_linear_scores(features, weights, bias=0.25)
    np.testing.assert_allclose(scores, np.array([4.75, 9.25]))

    try:
        cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    except ValueError as error:
        assert "zero vector" in str(error)
    else:
        raise AssertionError("zero-vector cosine similarity should fail")


def main() -> None:
    run_self_checks()

    house = np.array([80.0, 2.0, 0.6])
    weights = np.array([0.8, 12.0, -5.0])
    score = linear_score(weights, house, bias=10.0)

    print("All lesson 001 self-checks passed.")
    print(f"house vector: {house}")
    print(f"house vector shape: {house.shape}")
    print(f"example linear score: {score:.2f}")


if __name__ == "__main__":
    main()
