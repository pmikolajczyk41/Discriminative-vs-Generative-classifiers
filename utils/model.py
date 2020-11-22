from typing import NoReturn

from utils.X import X
from utils.algebra import Vector, Scalar, vector


class Model:
    def train(self, x: X, y: Vector) -> NoReturn:
        raise NotImplementedError

    def predict(self, x: X) -> Vector:
        return vector(map(lambda xi: self._predict_one(xi), x.by_sample()))

    def _predict_one(self, x: Vector) -> Scalar:
        raise NotImplementedError
