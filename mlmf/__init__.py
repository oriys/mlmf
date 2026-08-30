"""Reusable implementations for the Machine Learning Math Foundations course."""

from .vectors import (
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

__all__ = [
    "VectorValidationError",
    "as_vector",
    "batch_linear_scores",
    "cosine_similarity",
    "dot_product",
    "euclidean_distance",
    "l1_norm",
    "l2_norm",
    "linear_score",
    "scalar_multiply",
    "vector_add",
]
