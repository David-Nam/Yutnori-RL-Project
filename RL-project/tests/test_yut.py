import pytest

from yutnori.core import (
    YUT_PROBABILITIES,
    YUT_STEPS,
    YutResult,
    YutSampler,
    is_bonus_result,
    steps_for,
)


def test_yut_probability_distribution_matches_project_rule():
    assert YUT_PROBABILITIES[YutResult.BACK_DO] == pytest.approx(0.0384)
    assert YUT_PROBABILITIES[YutResult.DO] == pytest.approx(0.1152)
    assert YUT_PROBABILITIES[YutResult.GAE] == pytest.approx(0.3456)
    assert YUT_PROBABILITIES[YutResult.GEOL] == pytest.approx(0.3456)
    assert YUT_PROBABILITIES[YutResult.YUT] == pytest.approx(0.1296)
    assert YUT_PROBABILITIES[YutResult.MO] == pytest.approx(0.0256)
    assert (
        YUT_PROBABILITIES[YutResult.BACK_DO]
        + YUT_PROBABILITIES[YutResult.DO]
    ) == pytest.approx(0.1536)
    assert sum(YUT_PROBABILITIES.values()) == pytest.approx(1.0)


def test_yut_steps_and_bonus_results():
    assert YUT_STEPS == {
        YutResult.DO: 1,
        YutResult.GAE: 2,
        YutResult.GEOL: 3,
        YutResult.YUT: 4,
        YutResult.MO: 5,
        YutResult.BACK_DO: -1,
    }
    assert steps_for(YutResult.BACK_DO) == -1
    assert steps_for(YutResult.GEOL) == 3
    assert not is_bonus_result(YutResult.BACK_DO)
    assert not is_bonus_result(YutResult.DO)
    assert not is_bonus_result(YutResult.GAE)
    assert not is_bonus_result(YutResult.GEOL)
    assert is_bonus_result(YutResult.YUT)
    assert is_bonus_result(YutResult.MO)


def test_yut_sampler_is_seed_reproducible():
    first = YutSampler(seed=7).sample_many(30)
    second = YutSampler(seed=7).sample_many(30)
    different = YutSampler(seed=8).sample_many(30)

    assert first == second
    assert first != different
