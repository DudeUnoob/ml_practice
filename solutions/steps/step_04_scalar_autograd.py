from __future__ import annotations

from collections.abc import Callable
from math import exp


class Value:
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
        other = _ensure_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

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

    def __neg__(self) -> "Value":
        out = Value(-self.data, (self,), "neg")

        def _backward() -> None:
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-_ensure_value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return _ensure_value(other) + (-self)

    def __pow__(self, power: float) -> "Value":
        if not isinstance(power, int | float):
            raise TypeError("power must be an int or float.")

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

    def backward(self) -> None:
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


def _ensure_value(value: float | Value) -> Value:
    if isinstance(value, Value):
        return value

    return Value(value)
