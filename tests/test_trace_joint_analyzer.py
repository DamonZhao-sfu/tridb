from types import SimpleNamespace

from trimem.optimizer.joint_plan_analyzer import advise
from trimem.optimizer.plan_schema import JointSpace


def test_analyzer_schema_and_fail_closed_parse():
    class Client:
        def chat(self, system, user, temperature):
            assert "environment score" in system
            assert "vector_impl" not in user
            return SimpleNamespace(
                text='{"diagnosis":"cost", "proposals":[{"seed_count":32},{"vector_impl":"multi"}]}',
                error=None,
                input_tokens=50,
                output_tokens=20,
            )

    proposals, receipt = advise(JointSpace.live(), [], Client())
    assert len(proposals) == 1 and proposals[0]["seed_count"] == 32
    assert receipt["rejected"] == 1 and receipt["input_tokens"] == 50

    class Broken:
        def chat(self, *args, **kwargs):
            raise TimeoutError("test timeout")

    assert advise(JointSpace.live(), [], Broken())[0] == []


def test_analyzer_accepts_fenced_json_reply():
    """Regression 2026-11-03: Qwen wraps the JSON in ```json fences; the old
    json.loads(text) failed every time and all LLM probes became Sobol fallback."""
    from types import SimpleNamespace

    from trimem.optimizer.joint_plan_analyzer import advise, parse_json_reply
    from trimem.optimizer.plan_schema import JointSpace

    space = JointSpace.live()
    good = space.default(filter_order="none", seed_count=32)
    body = '{"diagnosis": "d", "proposals": [{"config": %s, "rationale": "r"}]}' % __import__("json").dumps(good)
    for text in ("```json\n" + body + "\n```", "Here you go:\n" + body + "\nThanks", body):
        assert parse_json_reply(text)["diagnosis"] == "d"

        class Client:
            def chat(self, system, user, temperature=0):
                return SimpleNamespace(text=text, error=None, input_tokens=1, output_tokens=2)

        proposals, receipt = advise(space, [], Client(), max_proposals=5, initialization=True)
        assert receipt["status"] == "ok" and proposals == [space.canonicalize(good)]
        assert receipt["raw_text"] == text


def test_analyzer_backend_factory_and_deadline():
    import pytest

    from trimem.experiments.trace_joint_plan_search import analyzer_client, analyzer_deadline
    from trimem.optimizer.analyzer_session import CodexCLI

    c = analyzer_client({"backend": "codex", "model": "gpt-5.6-luna", "effort": "medium", "timeout_s": 900})
    assert isinstance(c, CodexCLI) and c.model == "gpt-5.6-luna"
    assert analyzer_deadline({"backend": "codex", "timeout_s": 900}) == 960
    assert analyzer_deadline({"backend": "qwen"}) == 95
    with pytest.raises(ValueError):
        analyzer_client({"backend": "other", "model": "x"})
