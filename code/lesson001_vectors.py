"""Lesson 001: vectors and data representation.

Run from the repository root:

    python code/lesson001_vectors.py

The examples deliberately use small values so every result can also be checked
by hand.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def linear_predict(
    features: FloatArray,
    weights: FloatArray,
    bias: float,
) -> FloatArray:
    """Return one linear prediction for every row in ``features``.

    Args:
        features: Two-dimensional array with shape ``(n_samples, n_features)``.
        weights: One-dimensional array with shape ``(n_features,)``.
        bias: Scalar added to every prediction.

    Returns:
        One-dimensional array with shape ``(n_samples,)``.

    Raises:
        ValueError: If an input has an unexpected number of dimensions or the
            feature counts do not match.
    """
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if features.ndim != 2:
        raise ValueError(
            f"features must be two-dimensional, got shape {features.shape}"
        )
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be one-dimensional, got shape {weights.shape}"
        )
    if features.shape[1] != weights.shape[0]:
        raise ValueError(
            "feature count does not match weight count: "
            f"features.shape={features.shape}, weights.shape={weights.shape}"
        )
    if not np.isscalar(bias):
        raise ValueError(f"bias must be a scalar, got {type(bias).__name__}")

    predictions = features @ weights + float(bias)

    expected_shape = (features.shape[0],)
    if predictions.shape != expected_shape:
        raise AssertionError(
            f"unexpected output shape: expected {expected_shape}, "
            f"got {predictions.shape}"
        )

    return predictions


def demonstrate_vector_operations() -> None:
    """Compare vector addition, element-wise multiplication, and dot product."""
    x = np.array([2.0, -1.0, 3.0])
    y = np.array([4.0, 5.0, -2.0])

    vector_sum = x + y
    elementwise_product = x * y
    dot_product = x @ y

    assert vector_sum.shape == (3,)
    assert elementwise_product.shape == (3,)
    assert np.ndim(dot_product) == 0

    print("Vector operations")
    print(f"x + y       = {vector_sum}")
    print(f"x * y       = {elementwise_product}")
    print(f"x @ y       = {dot_product}")
    print()


def demonstrate_batch_prediction() -> None:
    """Run a small batch through a linear model."""
    features = np.array(
        [
            [80.0, 2.0, 10.0, 5.0],
            [120.0, 3.0, 5.0, 8.0],
            [60.0, 1.0, 20.0, 3.0],
        ]
    )
    weights = np.array([0.5, 20.0, -1.0, -3.0])
    bias = 10.0

    predictions = linear_predict(features, weights, bias)

    assert features.shape == (3, 4)
    assert weights.shape == (4,)
    assert predictions.shape == (3,)
    assert np.isclose(predictions[0], 65.0)

    print("Batch linear prediction")
    print(f"features shape    = {features.shape}")
    print(f"weights shape     = {weights.shape}")
    print(f"predictions shape = {predictions.shape}")
    print(f"predictions       = {predictions}")
    print()


def demonstrate_vectorization() -> None:
    """Verify that a loop and a vectorized expression produce equal results."""
    features = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    weights = np.array([2.0, -1.0])
    bias = 3.0

    loop_predictions = np.array(
        [row @ weights + bias for row in features],
        dtype=np.float64,
    )
    vectorized_predictions = linear_predict(features, weights, bias)

    np.testing.assert_allclose(loop_predictions, vectorized_predictions)

    print("Loop and vectorized implementations")
    print(f"loop       = {loop_predictions}")
    print(f"vectorized = {vectorized_predictions}")
    print()


def demonstrate_array_orientations() -> None:
    """Show the difference between 1-D, row, and column arrays in NumPy."""
    one_dimensional = np.array([1.0, 2.0, 3.0])
    row_vector = np.array([[1.0, 2.0, 3.0]])
    column_vector = np.array([[1.0], [2.0], [3.0]])

    print("Array orientations")
    print(
        f"1-D:    {one_dimensional.shape} -> transpose "
        f"{one_dimensional.T.shape}"
    )
    print(f"row:    {row_vector.shape} -> transpose {row_vector.T.shape}")
    print(
        f"column: {column_vector.shape} -> transpose "
        f"{column_vector.T.shape}"
    )
    print()


def demonstrate_broadcasting_risk() -> None:
    """Show how valid broadcasting can still produce an unintended shape."""
    column = np.ones((3, 1))
    one_dimensional = np.ones((3,))
    result = column + one_dimensional

    assert result.shape == (3, 3)

    print("Broadcasting risk")
    print(f"{column.shape} + {one_dimensional.shape} -> {result.shape}")
    print("The operation is legal, but the result may be semantically wrong.")
    print()


def demonstrate_invalid_input() -> None:
    """Verify that invalid feature and weight dimensions fail early."""
    features = np.ones((100, 20))
    wrong_weights = np.ones((100,))

    try:
        linear_predict(features, wrong_weights, 0.0)
    except ValueError as error:
        print("Expected validation error")
        print(error)
        print()
    else:
        raise AssertionError("invalid dimensions should have raised ValueError")


def main() -> None:
    np.set_printoptions(precision=3, suppress=True)

    demonstrate_vector_operations()
    demonstrate_batch_prediction()
    demonstrate_vectorization()
    demonstrate_array_orientations()
    demonstrate_broadcasting_risk()
    demonstrate_invalid_input()


if __name__ == "__main__":
    main()
