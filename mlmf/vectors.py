"""Vector operations used by lesson 001.

The implementations are deliberately small and explicit. Validation is included
because shape, non-finite values, and zero vectors are common sources of silent
errors in machine-learning code.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatVector: TypeAlias = NDArray[np.float64]
FloatMatrix: TypeAlias = NDArray[np.float64]


class VectorValidationError(ValueError):
    """Raised when an input does not satisfy the vector contract."""


def as_vector(values: ArrayLike, *, name: str = "vector") -> FloatVector:
    """Convert input to a finite, non-empty, one-dimensional float vector.

    Args:
        values: Values accepted by ``numpy.asarray``.
        name: Human-readable name used in error messages.

    Returns:
        A one-dimensional ``float64`` NumPy array.

    Raises:
        VectorValidationError: If the input is empty, not one-dimensional, or
            contains NaN or infinity.
    """

    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if vector.ndim != 1:
        raise VectorValidationError(
            f"{name} must be one-dimensional, got shape {vector.shape}"
        )
    if vector.size == 0:
        raise VectorValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise VectorValidationError(
            f"{name} must contain only finite values"
        )

    return vector


def _validated_pair(
    left: ArrayLike,
    right: ArrayLike,
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[FloatVector, FloatVector]:
    left_vector = as_vector(left, name=left_name)
    right_vector = as_vector(right, name=right_name)

    if left_vector.shape != right_vector.shape:
        raise VectorValidationError(
            f"shape mismatch: {left_name} has shape {left_vector.shape}, "
            f"but {right_name} has shape {right_vector.shape}"
        )

    return left_vector, right_vector


def vector_add(left: ArrayLike, right: ArrayLike) -> FloatVector:
    """Return the element-wise sum of two vectors with equal shape."""

    left_vector, right_vector = _validated_pair(left, right)
    return left_vector + right_vector


def scalar_multiply(scalar: float, vector: ArrayLike) -> FloatVector:
    """Multiply every vector element by a finite scalar."""

    try:
        scalar_value = float(scalar)
    except (TypeError, ValueError) as exc:
        raise VectorValidationError("scalar must be numeric") from exc

    if not math.isfinite(scalar_value):
        raise VectorValidationError("scalar must be finite")

    return scalar_value * as_vector(vector)


def dot_product(left: ArrayLike, right: ArrayLike) -> float:
    """Compute a dot product explicitly, without calling ``numpy.dot``.

    ``math.fsum`` reduces avoidable floating-point accumulation error while
    preserving the direct mathematical definition.
    """

    left_vector, right_vector = _validated_pair(left, right)
    products = (
        float(left_vector[index] * right_vector[index])
        for index in range(left_vector.size)
    )
    return float(math.fsum(products))


def l1_norm(vector: ArrayLike) -> float:
    """Return the L1 norm: the sum of absolute element values."""

    validated = as_vector(vector)
    return float(math.fsum(float(abs(value)) for value in validated))


def l2_norm(vector: ArrayLike) -> float:
    """Return the Euclidean length of a vector."""

    validated = as_vector(vector)
    return float(math.sqrt(dot_product(validated, validated)))


def euclidean_distance(left: ArrayLike, right: ArrayLike) -> float:
    """Return the Euclidean distance between two equal-shaped vectors."""

    left_vector, right_vector = _validated_pair(left, right)
    return l2_norm(left_vector - right_vector)


def cosine_similarity(
    left: ArrayLike,
    right: ArrayLike,
    *,
    zero_tolerance: float = 1e-12,
) -> float:
    """Return cosine similarity for two non-zero vectors.

    Raises:
        VectorValidationError: If either vector has a norm at or below
            ``zero_tolerance``.
    """

    if not math.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise VectorValidationError(
            "zero_tolerance must be a finite, non-negative number"
        )

    left_vector, right_vector = _validated_pair(left, right)
    left_norm = l2_norm(left_vector)
    right_norm = l2_norm(right_vector)

    if left_norm <= zero_tolerance:
        raise VectorValidationError(
            "cosine similarity is undefined for a zero left vector"
        )
    if right_norm <= zero_tolerance:
        raise VectorValidationError(
            "cosine similarity is undefined for a zero right vector"
        )

    similarity = dot_product(left_vector, right_vector) / (
        left_norm * right_norm
    )

    # Floating-point rounding can produce values such as 1.0000000000000002.
    return float(np.clip(similarity, -1.0, 1.0))


def linear_score(
    features: ArrayLike,
    weights: ArrayLike,
    bias: float = 0.0,
) -> float:
    """Compute ``features · weights + bias`` for one sample."""

    try:
        bias_value = float(bias)
    except (TypeError, ValueError) as exc:
        raise VectorValidationError("bias must be numeric") from exc

    if not math.isfinite(bias_value):
        raise VectorValidationError("bias must be finite")

    feature_vector, weight_vector = _validated_pair(
        features,
        weights,
        left_name="features",
        right_name="weights",
    )
    return dot_product(feature_vector, weight_vector) + bias_value


def batch_linear_scores(
    features: ArrayLike,
    weights: ArrayLike,
    bias: float = 0.0,
) -> FloatVector:
    """Compute a linear score for every row in a feature matrix.

    Args:
        features: Matrix with shape ``(n_samples, n_features)``.
        weights: Vector with shape ``(n_features,)``.
        bias: Finite scalar added to every sample score.
    """

    try:
        feature_matrix = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VectorValidationError(
            "features must contain numeric values"
        ) from exc

    if feature_matrix.ndim != 2:
        raise VectorValidationError(
            "features must be a two-dimensional matrix, "
            f"got shape {feature_matrix.shape}"
        )
    if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
        raise VectorValidationError("features must not have an empty axis")
    if not np.all(np.isfinite(feature_matrix)):
        raise VectorValidationError(
            "features must contain only finite values"
        )

    weight_vector = as_vector(weights, name="weights")
    if feature_matrix.shape[1] != weight_vector.size:
        raise VectorValidationError(
            "shape mismatch: features has "
            f"{feature_matrix.shape[1]} columns, but weights has "
            f"length {weight_vector.size}"
        )

    try:
        bias_value = float(bias)
    except (TypeError, ValueError) as exc:
        raise VectorValidationError("bias must be numeric") from exc
    if not math.isfinite(bias_value):
        raise VectorValidationError("bias must be finite")

    scores = feature_matrix @ weight_vector + bias_value
    return np.asarray(scores, dtype=np.float64)
