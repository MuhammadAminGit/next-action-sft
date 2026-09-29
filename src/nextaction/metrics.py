"""Scoring predictions, with uncertainty that respects how the data was collected.

Three things are measured per example:

- **action**: did the model pick the button the human agent pressed?
- **values**: are the values the same, in order, after lowercasing and collapsing spaces?
- **joint**: both. This is the number that matters; a right action with a wrong order
  number is a wrong action.

Output that is not valid JSON of the expected shape scores zero on everything and is
counted separately, so a model cannot look better by producing fewer, safer answers.

Confidence intervals come from a **cluster bootstrap over conversations**. Several
examples come from each conversation and they are not independent (a model that
misreads a customer's name will often get every later action in that call wrong), so
resampling individual examples would understate the uncertainty.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from nextaction.data import Example


@dataclass(frozen=True)
class Parsed:
    valid: bool
    action: str | None
    values: tuple[str, ...]


def normalize(value: str) -> str:
    return " ".join(str(value).lower().split())


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse(output: str) -> Parsed:
    """Read the model's answer. Tolerates a code fence or trailing text; nothing more."""
    match = _JSON_OBJECT.search(output)
    if not match:
        return Parsed(False, None, ())
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Parsed(False, None, ())
    action, values = obj.get("action"), obj.get("values")
    if not isinstance(action, str) or not isinstance(values, list):
        return Parsed(False, None, ())
    return Parsed(True, action.strip(), tuple(normalize(v) for v in values))


@dataclass(frozen=True)
class Score:
    convo_id: int
    turn: int
    valid: bool
    action: bool
    values: bool

    @property
    def joint(self) -> bool:
        return self.action and self.values


def score(example: Example, output: str) -> Score:
    p = parse(output)
    action_ok = p.valid and p.action == example.action
    values_ok = p.valid and p.values == tuple(normalize(v) for v in example.values)
    return Score(example.convo_id, example.turn, p.valid, action_ok, values_ok)


METRICS = ("joint", "action", "values", "valid")


def summarize(scores: Sequence[Score], *, reps: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Point estimate and 95% cluster-bootstrap interval for each metric."""
    by_convo: dict[int, list[Score]] = defaultdict(list)
    for s in scores:
        by_convo[s.convo_id].append(s)
    convos = list(by_convo)
    rng = random.Random(seed)

    out = {}
    for m in METRICS:
        point = sum(getattr(s, m) for s in scores) / len(scores)
        draws = []
        for _ in range(reps):
            picked = [rng.choice(convos) for _ in convos]
            hits = sum(getattr(s, m) for c in picked for s in by_convo[c])
            n = sum(len(by_convo[c]) for c in picked)
            draws.append(hits / n)
        draws.sort()
        out[m] = {
            "mean": point,
            "ci95": (draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]),
        }
    out["n_examples"] = len(scores)
    out["n_conversations"] = len(convos)
    return out


def paired_difference(
    a: Sequence[Score],
    b: Sequence[Score],
    metric: str = "joint",
    *,
    reps: int = 2000,
    seed: int = 0,
) -> dict:
    """How much better b is than a on the same examples, with a cluster-bootstrap interval.

    Pairing matters: both systems answered the same examples, so the per-conversation
    difference is what varies, and its interval is much tighter than comparing two
    independent intervals by eye.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison needs the same examples in the same order")
    diff: dict[int, list[float]] = defaultdict(list)
    for sa, sb in zip(a, b, strict=True):
        if (sa.convo_id, sa.turn) != (sb.convo_id, sb.turn):
            raise ValueError("examples are not aligned")
        diff[sa.convo_id].append(float(getattr(sb, metric)) - float(getattr(sa, metric)))
    convos = list(diff)
    rng = random.Random(seed)
    point = sum(sum(v) for v in diff.values()) / len(a)
    draws = []
    for _ in range(reps):
        picked = [rng.choice(convos) for _ in convos]
        total = sum(sum(diff[c]) for c in picked)
        n = sum(len(diff[c]) for c in picked)
        draws.append(total / n)
    draws.sort()
    lo, hi = draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]
    return {"metric": metric, "diff": point, "ci95": (lo, hi), "excludes_zero": lo > 0 or hi < 0}


def transcript_ceiling(examples: Sequence[Example]) -> float:
    """Share of examples whose gold values all appear, verbatim, in the conversation so far.

    A rough measure of how much of the value task is extraction. It is not an upper bound
    on accuracy: values that are not in the conversation can still be learned (which FAQ
    article answers which question), and the substring test is lenient for very short
    values. The fine-tuned model gets about a third of the "not in the conversation"
    examples right.
    """
    ok = 0
    for ex in examples:
        text = normalize(" ".join(t for _, t in ex.context))
        if all(normalize(v) in text for v in ex.values):
            ok += 1
    return ok / len(examples)
