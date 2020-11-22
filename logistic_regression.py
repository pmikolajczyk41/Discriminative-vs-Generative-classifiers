from itertools import starmap
from math import exp
from random import gauss
from typing import NoReturn

from X import X
from algebra import Vector, mult_mv, sum_vv, mult_vs, diff_vv, vector
from model import Model
from stop_conditions import StopConditions

STDEV = 1.
STOP_CONDITION = StopConditions(0.000001, None, 800)
GRADIENT_STEP = 0.0005


def compute_error(pred: Vector, y: Vector) -> float:
    assert len(pred) == len(y)
    pred = vector(map(lambda p: 0 if p <= 0.5 else 1, pred))
    same = sum(starmap(lambda a, b: a == b, zip(pred, y)))
    return 1. - (float(same) / len(y))


class LogisticRegression(Model):
    def __init__(self):
        self._theta = None

    def train(self, x: X, y: Vector) -> NoReturn:
        x = x.append_ones()

        theta = vector(map(lambda _: gauss(0., STDEV), range(x.nfeatures())))

        stop_condition, stop = STOP_CONDITION, False
        while not stop:
            prediction = self._compute_prob(x, theta)
            diff = diff_vv(y, prediction)
            gradient = mult_mv(x.by_feature(), diff)

            theta = sum_vv(theta, mult_vs(gradient, GRADIENT_STEP))

            error = compute_error(prediction, y)
            stop_condition, stop = stop_condition.update(gradient, error)
            print(error)

    def _compute_prob(self, x: X, theta: Vector) -> Vector:
        multiplied = mult_mv(x.by_sample(), theta)
        return vector(map(lambda m: 1. / (1. + exp(-m)), multiplied))

    def predict(self, x: X) -> Vector:
        assert self._theta is not None, "Model not trained yet"
        probs = self._compute_prob(x, self._theta)
        return vector(map(lambda p: 0 if p <= 0.5 else 1, probs))

