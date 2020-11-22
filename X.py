from itertools import repeat

from algebra import Matrix, vector


class X:
    def __init__(self, data: Matrix):
        k = len(data[0])
        assert all(map(lambda s: len(s) == k, data))

        self._data = data
        self._m = len(data)
        self._k = k

    def nsamples(self) -> int:
        return self._m

    def nfeatures(self) -> int:
        return self._k

    def by_sample(self) -> Matrix:
        return self._data

    def by_feature(self) -> Matrix:
        return vector(zip(*self._data))

    def append_ones(self) -> 'X':
        ones = vector(repeat(1., self._m))
        extended = self.by_feature() + (ones,)
        return X(vector(zip(*extended)))
