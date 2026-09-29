from __future__ import annotations

import pytest

from nextaction.data import Example, examples_from
from nextaction.metrics import paired_difference, parse, score, summarize, transcript_ceiling


def ex(action="pull-up-account", values=("crystal minh",), convo=1, context=None):
    return Example(convo, 0, context or [("customer", "I'm Crystal Minh")], action, list(values))


# ---- parsing ----------------------------------------------------------------


def test_parses_plain_json():
    p = parse('{"action": "pull-up-account", "values": ["Crystal  Minh"]}')
    assert p.valid and p.action == "pull-up-account" and p.values == ("crystal minh",)


def test_tolerates_code_fence_and_trailing_text():
    assert parse('```json\n{"action": "search-faq", "values": []}\n```\nDone.').valid


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "pull-up-account",
        '{"action": "x"}',
        '{"action": 3, "values": []}',
        '{"action": "x", "values": "a"}',
        "{not json}",
    ],
)
def test_malformed_output_is_invalid(bad):
    assert not parse(bad).valid


# ---- scoring ---------------------------------------------------------------------


def test_right_action_wrong_value_is_not_joint():
    s = score(ex(), '{"action": "pull-up-account", "values": ["crystal min"]}')
    assert s.action and not s.values and not s.joint


def test_value_order_matters():
    gold = ex("validate-purchase", ("cminh730", "c@email.com", "123"))
    assert not score(
        gold, '{"action": "validate-purchase", "values": ["123", "cminh730", "c@email.com"]}'
    ).values


def test_invalid_output_scores_zero_everywhere():
    s = score(ex(), "I would pull up the account.")
    assert (s.valid, s.action, s.values, s.joint) == (False, False, False, False)


# ---- intervals -----------------------------------------------------------------------


def test_summary_mean_and_interval_bracket_the_truth():
    scores = [
        score(ex(convo=i), '{"action": "pull-up-account", "values": ["crystal minh"]}')
        for i in range(50)
    ]
    scores += [score(ex(convo=100 + i), "nope") for i in range(50)]
    s = summarize(scores, reps=500)
    assert s["joint"]["mean"] == 0.5
    lo, hi = s["joint"]["ci95"]
    assert lo < 0.5 < hi


def test_paired_difference_detects_a_real_improvement():
    right = '{"action": "pull-up-account", "values": ["crystal minh"]}'
    base = [score(ex(convo=i), "nope") for i in range(60)]
    tuned = [score(ex(convo=i), right) for i in range(60)]
    d = paired_difference(base, tuned, reps=500)
    assert d["diff"] == 1.0 and d["excludes_zero"]


def test_paired_difference_refuses_misaligned_examples():
    with pytest.raises(ValueError):
        paired_difference([score(ex(convo=1), "")], [score(ex(convo=2), "")])


# ---- data --------------------------------------------------------------------------


def test_transcript_ceiling_counts_only_recoverable_values():
    in_text = ex(values=("crystal minh",))
    not_in_text = ex("select-faq", ("timing_4",))
    assert transcript_ceiling([in_text, not_in_text]) == 0.5


def test_examples_exclude_the_action_being_predicted():
    convo = {
        "convo_id": 7,
        "original": [
            ["customer", "hi"],
            ["customer", "Crystal Minh"],
            ["action", "Account pulled up for Crystal Minh."],
        ],
        "delexed": [
            {"speaker": "customer"},
            {"speaker": "customer"},
            {
                "speaker": "action",
                "targets": ["x", "take_action", "pull-up-account", ["crystal minh"], -1],
            },
        ],
    }
    [e] = examples_from(convo)
    assert e.context == [("customer", "hi"), ("customer", "Crystal Minh")]
    assert (e.action, e.values) == ("pull-up-account", ["crystal minh"])
