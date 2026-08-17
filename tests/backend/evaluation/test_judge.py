import pytest

from backend.evaluation.metrics.judge import call_judge, parse_judge_response


def test_parse_judge_response_clean_json():
    score, reasoning = parse_judge_response('{"score": 0.85, "reasoning": "Mostly grounded."}')
    assert score == 0.85
    assert reasoning == "Mostly grounded."


def test_parse_judge_response_strips_surrounding_text():
    text = 'Here is my evaluation:\n{"score": 0.4, "reasoning": "Partially supported."}\nThanks!'
    score, reasoning = parse_judge_response(text)
    assert score == 0.4
    assert reasoning == "Partially supported."


def test_parse_judge_response_clamps_out_of_range_scores():
    score, _ = parse_judge_response('{"score": 1.7, "reasoning": "x"}')
    assert score == 1.0
    score, _ = parse_judge_response('{"score": -0.3, "reasoning": "x"}')
    assert score == 0.0


def test_parse_judge_response_raises_on_no_json():
    with pytest.raises(ValueError):
        parse_judge_response("I refuse to answer in JSON today.")


class _FakeLocalClient:
    active_provider = "local"


class _FakeCloudClientGood:
    active_provider = "openai"

    class _Resp:
        text = '{"score": 0.9, "reasoning": "Looks solid."}'

    def complete(self, prompt, system, temperature):
        return self._Resp()


class _FakeCloudClientBroken:
    active_provider = "openai"

    def complete(self, prompt, system, temperature):
        raise RuntimeError("network error")


def test_call_judge_returns_none_for_local_provider():
    assert call_judge(_FakeLocalClient(), "some prompt") is None


def test_call_judge_returns_score_for_working_cloud_client():
    result = call_judge(_FakeCloudClientGood(), "some prompt")
    assert result == (0.9, "Looks solid.")


def test_call_judge_returns_none_when_call_fails():
    assert call_judge(_FakeCloudClientBroken(), "some prompt") is None
