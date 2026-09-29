"""Turning ABCD conversations into prediction examples.

ABCD (Chen et al., NAACL 2021) is 10,042 human-to-human customer service conversations,
collected by pairing crowd workers as customer and trained agent in real time. The agent
works a dashboard of action buttons and must follow a written company policy. It is real
human data but role-played, with fictional customers; it is not production call traffic.

One example is created for every action the agent took. The input is everything said
and done before that action; the target is the action and the values the agent entered.
This is ABCD's Action State Tracking task, reformulated as text-to-JSON so a language
model can do it directly.

Each conversation is stored twice, `original` and `delexed`, aligned turn for turn. The
text comes from `original` (real names and numbers, which the model has to copy) and the
labels from `delexed` (where the targets live).
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nextaction.prompt import ACTION_SPECS, messages

RAW = Path("data/raw/abcd_v1.1.json.gz")


@dataclass(frozen=True)
class Example:
    convo_id: int
    turn: int  # index of the action turn in the conversation
    context: list[tuple[str, str]]
    action: str
    values: list[str]

    @property
    def answer(self) -> str:
        return target_json(self.action, self.values)


def target_json(action: str, values: list[str]) -> str:
    return json.dumps({"action": action, "values": values}, ensure_ascii=False)


def load_raw(path: Path = RAW) -> dict[str, list[dict[str, Any]]]:
    with gzip.open(path) as f:
        return json.load(f)


def examples_from(conversation: dict[str, Any]) -> list[Example]:
    original, delexed = conversation["original"], conversation["delexed"]
    if len(original) != len(delexed):
        raise ValueError(f"conversation {conversation['convo_id']}: original/delexed misaligned")
    out = []
    for i, turn in enumerate(delexed):
        if turn["speaker"] != "action":
            continue
        _, _, action, values, _ = turn["targets"]
        if action not in ACTION_SPECS:
            raise ValueError(f"unknown action {action!r}")
        out.append(
            Example(
                convo_id=conversation["convo_id"],
                turn=i,
                context=[(spk, txt) for spk, txt in original[:i]],
                action=action,
                values=list(values),
            )
        )
    return out


def build(split: list[dict[str, Any]]) -> list[Example]:
    return [ex for convo in split for ex in examples_from(convo)]


def sample(examples: list[Example], n: int | None, seed: int) -> list[Example]:
    if n is None or n >= len(examples):
        return list(examples)
    return random.Random(seed).sample(examples, n)


def write_sft(examples: list[Example], path: Path) -> None:
    """mlx-lm chat format, with the compact prompt the fine-tuned model is trained on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            msgs = messages(ex.context, ex.answer, prompt="compact")
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")


def write_eval(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def read_eval(path: Path) -> list[Example]:
    rows = [json.loads(line) for line in path.open()]
    return [
        Example(
            r["convo_id"], r["turn"], [tuple(t) for t in r["context"]], r["action"], r["values"]
        )
        for r in rows
    ]
