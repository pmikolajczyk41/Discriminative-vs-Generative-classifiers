from pathlib import Path
from random import shuffle
from typing import Tuple

from utils.algebra import Matrix, Vector


def load(filepath: Path) -> Tuple[Matrix, Vector]:
    assert filepath.exists() and filepath.is_file()

    with filepath.open() as file:
        lines = file.readlines()

    samples = map(lambda l: l.split(), lines)
    samples = (tuple(map(lambda v: int(v), s)) for s in samples)
    samples = ((s[:-1], s[-1]) for s in samples)

    return tuple(zip(*samples))


def split2(x: Matrix, y: Vector, p: float) -> Tuple[Tuple[Matrix, Vector], Tuple[Matrix, Vector]]:
    assert len(x) == len(y) and 0. < p < 1.
    domain = list(zip(x, y))
    shuffle(domain)

    d0 = list(filter(lambda xy: xy[1] == 0, domain))
    d1 = list(filter(lambda xy: xy[1] == 1, domain))

    prefix_size_0, prefix_size_1 = int(p * len(d0)), int(p * len(d1))
    prefix0, suffix0 = d0[:prefix_size_0], d0[prefix_size_0:]
    prefix1, suffix1 = d1[:prefix_size_1], d1[prefix_size_1:]

    prefix, suffix = prefix0 + prefix1, suffix0 + suffix1
    shuffle(prefix)
    shuffle(suffix)

    return tuple(zip(*prefix)), tuple(zip(*suffix))
