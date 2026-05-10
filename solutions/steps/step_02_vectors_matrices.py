from __future__ import annotations


def _require_same_length(left: list[float], right: list[float]) -> None:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length.")


def dot(left: list[float], right: list[float]) -> float:
    _require_same_length(left, right)
    return sum(a * b for a, b in zip(left, right, strict=True))


def vector_add(left: list[float], right: list[float]) -> list[float]:
    _require_same_length(left, right)
    return [a + b for a, b in zip(left, right, strict=True)]


def scalar_multiply(scalar: float, values: list[float]) -> list[float]:
    return [scalar * value for value in values]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        raise ValueError("matrix must not be empty.")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("all rows must have the same length.")

    return [[row[column_index] for row in matrix] for column_index in range(width)]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    if not left or not right:
        raise ValueError("matrices must not be empty.")
    if len(left[0]) != len(right):
        raise ValueError("left columns must match right rows.")

    right_columns = transpose(right)
    return [[dot(row, column) for column in right_columns] for row in left]
