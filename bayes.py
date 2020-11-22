from itertools import product
from math import log
from pathlib import Path
from typing import Tuple

from X import X
from algebra import Vector, Scalar, vector, Matrix
from model import Model, Hypothesis


class NaiveBayes(Model):
    def __init__(self, nfeatures: int, domain_size: int):
        self._domain_size, self._nfeatures = domain_size, nfeatures
        self._py = 0.
        self._pi = [[[0.] * self._domain_size for _ in range(self._nfeatures)],
                    [[0.] * self._domain_size for _ in range(self._nfeatures)]]

    def train(self, x: X, y: Vector) -> Hypothesis:
        assert x.nfeatures() == self._nfeatures

        m = len(y)
        ones = sum(y)
        zeros = m - ones

        self._py = (1. + ones) / (m + 2.)

        counters = [[[0] * self._domain_size for _ in range(self._nfeatures)],
                    [[0] * self._domain_size for _ in range(self._nfeatures)]]

        for xi, yi in zip(x.by_sample(), y):
            for j, xij in enumerate(xi):
                counters[yi][j][xij] += 1

        for y_val, feature, k in product([0, 1], range(self._nfeatures), range(self._domain_size)):
            denominator = self._domain_size + y_val * ones + (1 - y_val) * zeros
            self._pi[y_val][feature][k] = (1 + counters[y_val][feature][k]) / denominator

        return self._construct_hypothesis()

    def _likelihood(self, x: Vector, y_val: Scalar):
        assert y_val in [0, 1]
        return sum(map(lambda k: log(self._pi[y_val][k][x[k]]), range(len(x))))

    def _construct_hypothesis(self) -> Hypothesis:
        def predict_one(x: Vector) -> Scalar:
            assert len(x) == self._nfeatures
            p0 = log(1. - self._py) + self._likelihood(x, 0)
            p1 = log(self._py) + self._likelihood(x, 1)
            return 0 if p0 >= p1 else 1

        def predict(x: X) -> Vector:
            return vector(map(lambda xi: predict_one(xi), x.by_sample()))

        return Hypothesis.from_func(predict)
