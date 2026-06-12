"""Training helpers for learned Yutnori agents."""

from yutnori.training.common_evaluation import (
    COMMON_EVALUATION_PROTOCOL,
    COMMON_RULE_OPPONENT,
    CommonEvaluationError,
    CommonEvaluationSplit,
    CommonPolicyEvaluationResult,
    evaluate_common_rule_agent,
    evaluate_common_rule_policy,
    seed_list_sha256,
    wilson_interval,
)
from yutnori.training.env_factory import (
    OPPONENT_NAMES,
    VEC_ENV_DUMMY,
    VEC_ENV_SUBPROC,
    VEC_ENV_TYPES,
    make_opponent,
    make_yutnori_env,
    make_yutnori_vec_env,
)
from yutnori.training.model_config import (
    resolve_model_observation_mode,
    resolve_model_reward_mode,
    resolve_model_ruleset,
)
from yutnori.training.ppo_evaluation import (
    PolicyEvaluationResult,
    evaluate_maskable_policy,
)
from yutnori.training.reward_shaping import (
    RF_SHAPING_CAPTURE_WEIGHT,
    RF_SHAPING_FINISH_WEIGHT,
    RF_SHAPING_SHORTCUT_BONUS,
    project_rf_event_shaping_reward,
    project_rf_events_shaping_reward,
)

__all__ = [
    "COMMON_EVALUATION_PROTOCOL",
    "COMMON_RULE_OPPONENT",
    "CommonEvaluationError",
    "CommonEvaluationSplit",
    "CommonPolicyEvaluationResult",
    "OPPONENT_NAMES",
    "PolicyEvaluationResult",
    "RF_SHAPING_CAPTURE_WEIGHT",
    "RF_SHAPING_FINISH_WEIGHT",
    "RF_SHAPING_SHORTCUT_BONUS",
    "VEC_ENV_DUMMY",
    "VEC_ENV_SUBPROC",
    "VEC_ENV_TYPES",
    "evaluate_common_rule_agent",
    "evaluate_common_rule_policy",
    "evaluate_maskable_policy",
    "make_opponent",
    "make_yutnori_env",
    "make_yutnori_vec_env",
    "project_rf_event_shaping_reward",
    "project_rf_events_shaping_reward",
    "resolve_model_observation_mode",
    "resolve_model_reward_mode",
    "resolve_model_ruleset",
    "seed_list_sha256",
    "wilson_interval",
]
