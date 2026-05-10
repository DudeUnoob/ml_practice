"""Step 04: a tiny scalar autograd engine.

This is the first version of backpropagation. Keep it scalar so the chain rule is
visible.

Run:

    python3 tools/coach.py check 04
"""

from __future__ import annotations

from collections.abc import Callable


class Value:
    """A scalar number that remembers how it was created."""

    def __init__(
        self,
        data: float,
        children: tuple["Value", ...] = (),
        op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: float | "Value") -> "Value":
        raise NotImplementedError("TODO: create an output Value and local backward.")

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __mul__(self, other: float | "Value") -> "Value":
        raise NotImplementedError("TODO: implement multiplication and local backward.")

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        raise NotImplementedError("TODO: implement negation.")

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-_ensure_value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return _ensure_value(other) + (-self)

    def __pow__(self, power: float) -> "Value":
        raise NotImplementedError("TODO: implement power for Python int/float powers.")

    def tanh(self) -> "Value":
        raise NotImplementedError("TODO: implement tanh and its local derivative.")

    def backward(self) -> None:
        raise NotImplementedError("TODO: topologically sort the graph and call _backward.")


def _ensure_value(value: float | Value) -> Value:
    if isinstance(value, Value):
        return value

    return Value(value)
