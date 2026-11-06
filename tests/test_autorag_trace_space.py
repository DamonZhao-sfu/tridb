"""The source optimizer must propose exactly the same physical TRACE space."""
import pytest
from tools.autorag_trace_space import decode, encode, project
from trimem.optimizer import session_space_agentenv_v2 as space


@pytest.mark.parametrize('plan', space.BASE_PLANS + ('P1+Rank',))
def test_upstream_config_roundtrip(plan):
    raw = {'plan': plan, 'seed_count': 32, 'depth': 3, 'fanout': 16,
           'top_k': 8, 'outcome_policy': 'success_failure_capped',
           'semfilter_threshold': 0.8}
    expected = space.validate(raw)
    upstream = encode(expected)
    assert project().validate_trial(upstream) == []
    assert decode(upstream) == expected


def test_invalid_and_inactive_thresholds_are_rejected():
    p = project()
    trial = encode(space.validate({'plan': 'VGR'}))
    assert p.validate_trial(trial.model_copy(update={'graph_top_k': 7}))
    assert p.validate_trial(trial.model_copy(update={'reranker': 'threshold_0.8'}))
    with pytest.raises(ValueError):
        decode(trial.model_copy(update={'graph_query_mode': 'unknown'}))


def test_prompt_explains_adapter_and_fixed_actor():
    text = project().to_agent_prompt()
    assert 'seed_count' in text and 'graph_query_mode' in text
    assert 'not an embedding model' in text
    assert 'mean tokens' in text
