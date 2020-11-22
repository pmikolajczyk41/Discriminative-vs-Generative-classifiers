from typing import Callable

from X import X
from algebra import Vector


class Hypothesis:
    @classmethod
    def from_func(cls, fun: Callable[[X], Vector]) -> 'Hypothesis':
        h = Hypothesis()
        h.predict = fun
        return h

    def predict(self, x: X) -> Vector:
        raise NotImplementedError


class Model:
    def train(self, x: X, y: Vector) -> Hypothesis:
        raise NotImplementedError
