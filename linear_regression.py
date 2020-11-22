from X import X
from algebra import Vector, mult_mv
from model import Hypothesis, Model


class LinearHypothesis(Hypothesis):
    def __init__(self, theta: Vector):
        self._theta = theta

    def predict(self, x: X) -> Vector:
        assert x.nfeatures() + 1 == len(self._theta)
        return mult_mv(x.append_ones().by_sample(), self._theta)


class LinearRegression(Model):
    def train(self, x: X, y: Vector) -> LinearHypothesis:
        pass
