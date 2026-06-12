"""Yut sampling utilities."""

from __future__ import annotations

import random
from enum import Enum
from typing import Iterable


class YutResult(str, Enum):
    """Supported yut results."""

    DO = "DO"
    GAE = "GAE"
    GEOL = "GEOL"
    YUT = "YUT"
    MO = "MO"
    BACK_DO = "BACK_DO"


YUT_ORDER: tuple[YutResult, ...] = (
    YutResult.DO,
    YutResult.GAE,
    YutResult.GEOL,
    YutResult.YUT,
    YutResult.MO,
    YutResult.BACK_DO,
)

YUT_STEPS: dict[YutResult, int] = {
    YutResult.DO: 1,
    YutResult.GAE: 2,
    YutResult.GEOL: 3,
    YutResult.YUT: 4,
    YutResult.MO: 5,
    YutResult.BACK_DO: -1,
}

YUT_PROBABILITIES: dict[YutResult, float] = {
    YutResult.DO: 0.1152,
    YutResult.GAE: 0.3456,
    YutResult.GEOL: 0.3456,
    YutResult.YUT: 0.1296,
    YutResult.MO: 0.0256,
    YutResult.BACK_DO: 0.0384,
}

BONUS_RESULTS: frozenset[YutResult] = frozenset({YutResult.YUT, YutResult.MO})


def steps_for(result: YutResult) -> int:
    return YUT_STEPS[result]


def is_bonus_result(result: YutResult) -> bool:
    return result in BONUS_RESULTS


class YutSampler:
    """Seedable sampler for project yut probabilities."""

    def __init__(self, seed: int | None = None, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random(seed)

    def sample(self) -> YutResult:
        value = self._rng.random()
        cumulative = 0.0
        for result in YUT_ORDER:
            cumulative += YUT_PROBABILITIES[result]
            if value < cumulative:
                return result
        return YUT_ORDER[-1]

    def sample_many(self, count: int) -> list[YutResult]:
        if count < 0:
            raise ValueError("count must be non-negative")
        return [self.sample() for _ in range(count)]


def probability_items() -> Iterable[tuple[YutResult, float]]:
    return ((result, YUT_PROBABILITIES[result]) for result in YUT_ORDER)
