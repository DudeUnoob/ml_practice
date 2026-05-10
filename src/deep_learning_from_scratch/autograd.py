"""A tiny scalar reverse-mode autodiff engine.

The implementation mirrors the core idea used by modern frameworks:

1. Each operation creates a new value and remembers its parents.
2. Calling ``backward`` topologically sorts the graph.
3. Local derivatives are chained from the output back to every input.

It is scalar on purpose. Scalars make the chain rule visible before tensors add
shape bookkeeping.
"""

from __future__ import annotations

from collections.abc import Callable
from math import exp, log
from typing import Self


def _ensure_value(value: float | "Value") -> "Value":
    if isinstance(value, Value):
        return value

    return Value(value)


class Value:
    """A scalar value that records how it was computed."""

    def __init__(
        self,
        data: float,
        children: tuple["Value", ...] = (),
        op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._op = op
        self._prev = set(children)
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        label = f", label={self.label!r}" if self.label else ""
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f}{label})"

    def __add__(self, other: float | "Value") -> "Value":
        other = _ensure_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-_ensure_value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return _ensure_value(other) + (-self)

    def __neg__(self) -> "Value":
        out = Value(-self.data, (self,), "neg")

        def _backward() -> None:
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: float | "Value") -> "Value":
        other = _ensure_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __truediv__(self, other: float | "Value") -> "Value":
        return self * _ensure_value(other) ** -1

    def __rtruediv__(self, other: float | "Value") -> "Value":
        return _ensure_value(other) * self**-1

    def __pow__(self, power: float) -> "Value":
        if not isinstance(power, int | float):
            raise TypeError("Power must be a Python int or float.")

        out = Value(self.data**power, (self,), f"**{power}")

        def _backward() -> None:
            self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        value = (exp(2 * self.data) - 1) / (exp(2 * self.data) + 1)
        out = Value(value, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - value**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        value = exp(self.data)
        out = Value(value, (self,), "exp")

        def _backward() -> None:
            self.grad += value * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0:
            raise ValueError("log is only defined for positive values.")

        out = Value(log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Backpropagate from this value to all ancestors."""

        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topology(value: Value) -> None:
            if value in visited:
                return

            visited.add(value)
            for child in value._prev:
                build_topology(child)
            topo.append(value)

        build_topology(self)

        for value in topo:
            value.grad = 0.0

        self.grad = 1.0
        for value in reversed(topo):
            value._backward()

    def with_label(self, label: str) -> Self:
        self.label = label
        return self
