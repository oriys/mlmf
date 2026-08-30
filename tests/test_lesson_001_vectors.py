"""Tests for Lesson 001 vector operations."""

import numpy as np
import pytest

from mlmf import (
    VectorValidationError,
    as_vector,
    batch_linear_scores,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l1_norm,
    l2_norm,
    linear_score,
    scalar_multiply,
    vector_add,
)


def test_as_vector_converts_numeric_sequence() -> None:
    result = as_vector([1, 2, 3])

    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))
    assert result.dtype == np.float64
    assert result.shape == (3,)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "must not be empty"),
        ([[1.0, 2.0]], "must be one-dimensional"),
        ([1.0, np.nan], "only finite"),
        ([1.0, np.inf], "only finite"),
    ],
)
def test_as_vector_rejects_invalid_inputs(
    values: object, message: str
) -> None:
    with pytest.raises(VectorValidationError, match=message):
        as_vector(values)


def test_as_vector_rejects_non_numeric_values() -> None:
    with pytest.raises(VectorValidationError, match="numeric values"):
        as_vector(["area", "bedrooms"])


def test_vector_add() -> None:
    result = vector_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

    np.testing.assert_allclose(result, [5.0, 7.0, 9.0])


def test_vector_add_rejects_different_shapes() -> None:
    with pytest.raises(VectorValidationError, match="shape mismatch"):
        vector_add([1.0, 2.0], [1.0, 2.0, 3.0])


def test_scalar_multiply() -> None:
    result = scalar_multiply(-2.0, [1.0, -3.0, 4.0])

    np.testing.assert_allclose(result, [-2.0, 6.0, -8.0])


@pytest.mark.parametrize("scalar", [np.inf, -np.inf, np.nan])
def test_scalar_multiply_rejects_non_finite_scalar(scalar: float) -> None:
    with pytest.raises(VectorValidationError, match="scalar must be finite"):
        scalar_multiply(scalar, [1.0, 2.0])


def test_dot_product_matches_numpy() -> None:
    x = np.array([2.0, -1.0, 3.0])
    y = np.array([4.0, 2.0, -2.0])

    assert dot_product(x, y) == pytest.approx(float(np.dot(x, y)))
    assert dot_product(x, y) == pytest.approx(0.0)


def test_norms() -> None:
    vector = [-3.0, 4.0]

    assert l1_norm(vector) == pytest.approx(7.0)
    assert l2_norm(vector) == pytest.approx(5.0)


def test_euclidean_distance() -> None:
    assert euclidean_distance([1.0, 2.0], [4.0, 6.0]) == pytest.approx(5.0)


def test_cosine_similarity_for_parallel_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)


def test_cosine_similarity_for_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 5.0]) == pytest.approx(0.0)


def test_cosine_similarity_for_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [-2.0, -4.0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(VectorValidationError, match="zero left vector"):
        cosine_similarity([0.0, 0.0], [1.0, 2.0])


def test_cosine_similarity_rejects_invalid_tolerance() -> None:
    with pytest.raises(VectorValidationError, match="non-negative"):
        cosine_similarity([1.0], [1.0], zero_tolerance=-1.0)


def test_linear_score() -> None:
    result = linear_score(
        features=[80.0, 2.0, 6.0],
        weights=[0.7, 10.0, -2.0],
        bias=5.0,
    )

    assert result == pytest.approx(69.0)


def test_linear_score_rejects_feature_weight_mismatch() -> None:
    with pytest.raises(VectorValidationError, match="shape mismatch"):
        linear_score([1.0, 2.0, 3.0], [0.5, 0.25])


def test_linear_score_rejects_non_finite_bias() -> None:
    with pytest.raises(VectorValidationError, match="bias must be finite"):
        linear_score([1.0], [2.0], bias=np.nan)


def test_batch_linear_scores_match_individual_scores() -> None:
    features = np.array(
        [
            [80.0, 2.0, 6.0],
            [75.0, 3.0, 5.0],
            [120.0, 4.0, 12.0],
        ]
    )
    weights = np.array([0.7, 10.0, -2.0])
    bias = 5.0

    result = batch_linear_scores(features, weights, bias)
    expected = np.array(
        [linear_score(row, weights, bias) for row in features]
    )

    np.testing.assert_allclose(result, expected)
    assert result.shape == (3,)


@pytest.mark.parametrize(
    ("features", "weights", "message"),
    [
        ([1.0, 2.0], [1.0, 2.0], "two-dimensional"),
        (np.empty((0, 2)), [1.0, 2.0], "empty axis"),
        ([[1.0, np.nan]], [1.0, 2.0], "only finite"),
        ([[1.0, 2.0]], [1.0], "shape mismatch"),
    ],
)
def test_batch_linear_scores_rejects_invalid_inputs(
    features: object,
    weights: object,
    message: str,
) -> None:
    with pytest.raises(VectorValidationError, match=message):
        batch_linear_scores(features, weights)
