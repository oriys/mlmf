"""Runnable examples for lesson 001: vectors and data representation.

Run from the repository root:

    python code/lesson_001_vectors.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Make the repository package importable when this file is executed directly.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mlmf import (  # noqa: E402
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


def demonstrate_basic_operations() -> None:
    """Show the direct correspondence between formulas and code."""

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    print("Basic vector operations")
    print("-----------------------")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {vector_add(x, y)}")
    print(f"2x = {scalar_multiply(2.0, x)}")
    print(f"x · y = {dot_product(x, y):.6f}")
    print(f"||x||_1 = {l1_norm(x):.6f}")
    print(f"||x||_2 = {l2_norm(x):.6f}")
    print(f"distance(x, y) = {euclidean_distance(x, y):.6f}")
    print(
        "cosine_similarity(x, y) = "
        f"{cosine_similarity(x, y):.6f}"
    )
    print()


def demonstrate_linear_model() -> None:
    """Represent houses as vectors and compute linear model scores."""

    feature_names = ("area_m2", "bedrooms", "distance_to_center_km")
    houses = np.array(
        [
            [80.0, 2.0, 6.0],
            [75.0, 3.0, 5.0],
            [120.0, 4.0, 12.0],
        ]
    )
    weights = np.array([0.7, 10.0, -2.0])
    bias = 5.0

    first_score = linear_score(houses[0], weights, bias)
    all_scores = batch_linear_scores(houses, weights, bias)

    print("A tiny linear-model example")
    print("---------------------------")
    print(f"feature order = {feature_names}")
    print(f"X.shape = {houses.shape}")
    print(f"w.shape = {weights.shape}")
    print(f"first sample score = {first_score:.6f}")
    print(f"batch scores = {all_scores}")
    print()

    # A single-sample score and the matching row in a batch must agree.
    assert np.isclose(first_score, all_scores[0])


def demonstrate_shape_semantics() -> None:
    """Show why one-dimensional, row, and column arrays differ."""

    vector = np.array([1.0, 2.0, 3.0])
    row = vector.reshape(1, 3)
    column = vector.reshape(3, 1)

    print("Shape semantics")
    print("---------------")
    print(f"vector.shape = {vector.shape}")
    print(f"row.shape = {row.shape}")
    print(f"column.shape = {column.shape}")
    print(f"row @ column shape = {(row @ column).shape}")
    print(f"column @ row shape = {(column @ row).shape}")
    print()


def main() -> None:
    demonstrate_basic_operations()
    demonstrate_linear_model()
    demonstrate_shape_semantics()


if __name__ == "__main__":
    main()
