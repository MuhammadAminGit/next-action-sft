"""The questions a careful reader asks after seeing the headline.

    python -m nextaction.analysis

1. **Memorisation.** ABCD reuses a small pool of fictional customers, so many test values
   (names especially) also occur in training. How accurate is each system on values it
   could have memorised, against values it has never seen?
   It also checks the test customers that look new, which turn out not to be.
2. **Extraction.** How accurate is each system when every gold value is in the
   conversation, against when some are not?
3. **Label quality.** For actions whose system message repeats what the agent entered
   ("Account has been pulled up for Crystal Minh."), how often does the gold label differ
   from the logged value, and why: nothing logged, a misspelt log, or a different value?

Writes results/analysis.json and prints the tables.
"""

from __future__ import annotations

import collections
import difflib
import json
import os
from pathlib import Path

from nextaction.data import build, load_raw
from nextaction.metrics import normalize, parse

SYSTEMS = ("base-0shot-compact", "base-0shot", "base-5shot", "lora")


def predictions(name: str) -> dict[tuple[int, int], dict]:
    with Path(f"results/predictions/test-{name}.jsonl").open() as f:
        return {(r["convo_id"], r["turn"]): r for r in map(json.loads, f)}


def subset_accuracy(groups: dict[str, list[tuple[int, int]]], preds: dict) -> dict:
    out = {}
    for group, keys in groups.items():
        out[group] = {"n": len(keys)}
        for name in SYSTEMS:
            rows = [preds[name][k] for k in keys]
            out[group][name] = {
                "joint": sum(r["joint"] for r in rows) / len(rows),
                "values": sum(r["values"] for r in rows) / len(rows),
            }
    return out


def main() -> None:
    raw = load_raw()
    with Path("data/sft/train.jsonl").open() as f:
        trained_on = [json.loads(line) for line in f]
    seen_values = {
        normalize(v)
        for row in trained_on
        for v in json.loads(row["messages"][-1]["content"])["values"]
    }
    test = build(raw["test"])
    preds = {name: predictions(name) for name in SYSTEMS}
    key = lambda ex: (ex.convo_id, ex.turn)

    # 1. memorisation
    memo: dict[str, list] = collections.defaultdict(list)
    for ex in test:
        vals = [normalize(v) for v in ex.values]
        group = (
            "no values"
            if not vals
            else (
                "all values seen in training"
                if all(v in seen_values for v in vals)
                else "some values never seen in training"
            )
        )
        memo[group].append(key(ex))

    # Customers repeat across splits. The test pull-up-account values that never occur in
    # the training sample turn out not to be new customers at all.
    known_names = {
        normalize(v)
        for row in trained_on
        for v in json.loads(row["messages"][-1]["content"])["values"]
        if v and not any(ch.isdigit() for ch in v)
    }
    unseen_accounts = collections.Counter()
    unseen_examples: dict[str, list[str]] = collections.defaultdict(list)
    for ex in test:
        if ex.action != "pull-up-account":
            continue
        v = normalize(ex.values[0])
        if v in seen_values:
            continue
        if any(ch.isdigit() for ch in v):
            kind = "account id"
        elif max(difflib.SequenceMatcher(None, v, n).ratio() for n in known_names) >= 0.75:
            kind = "misspelling of a customer in training"
        else:
            kind = "other"
        unseen_accounts[kind] += 1
        unseen_examples[kind].append(v)

    # 2. extraction
    extract: dict[str, list] = collections.defaultdict(list)
    for ex in test:
        text = normalize(" ".join(t for _, t in ex.context))
        present = all(normalize(v) in text for v in ex.values)
        extract["all values in the conversation" if present else "some values not in it"].append(
            key(ex)
        )

    # 3. label quality. For actions whose system message repeats the value the agent entered
    # ("Account has been pulled up for Crystal Minh."), pull that value out of the message
    # template and compare it, whole, with the gold label.
    by_convo = {c["convo_id"]: c for c in raw["test"]}
    echo_train = collections.defaultdict(lambda: [0, 0])
    train_messages = collections.defaultdict(list)
    for convo in raw["train"]:
        for turn, t in zip(convo["original"], convo["delexed"], strict=True):
            if t["speaker"] == "action" and len(t["targets"][3]) == 1 and t["targets"][3][0]:
                stats = echo_train[t["targets"][2]]
                stats[0] += normalize(t["targets"][3][0]) in normalize(turn[1])
                stats[1] += 1
                train_messages[t["targets"][2]].append(normalize(turn[1]))
    echo_actions = sorted(a for a, (hit, n) in echo_train.items() if n >= 50 and hit / n >= 0.8)
    templates = {a: message_template(train_messages[a]) for a in echo_actions}

    kinds = collections.Counter()
    by_kind_examples: dict[str, list] = collections.defaultdict(list)
    checked = 0
    lora_matched_the_log = 0
    for ex in test:
        if ex.action not in echo_actions or len(ex.values) != 1 or not ex.values[0]:
            continue
        checked += 1
        logged = logged_value(by_convo[ex.convo_id]["original"][ex.turn][1], templates[ex.action])
        gold = clean(ex.values[0])
        if logged == gold:
            continue
        if logged == "":
            kind = "agent logged nothing"
        elif difflib.SequenceMatcher(None, logged, gold).ratio() >= 0.8:
            kind = "near match: a typo or truncation on one side"
        else:
            kind = "logged a different value"
        kinds[kind] += 1
        if len(by_kind_examples[kind]) < 5:
            by_kind_examples[kind].append({"action": ex.action, "gold": gold, "logged": logged})
        if kind == "logged a different value":
            p = parse(preds["lora"][key(ex)]["output"])
            if (
                p.valid
                and p.action == ex.action
                and len(p.values) == 1
                and clean(p.values[0]) == logged
            ):
                lora_matched_the_log += 1

    by_key = {key(ex): ex.action for ex in test}
    never_seen_actions = collections.Counter(
        by_key[k] for k in memo["some values never seen in training"]
    )

    out = {
        "memorisation": subset_accuracy(memo, preds),
        "never_seen_group_actions": dict(never_seen_actions.most_common()),
        "unseen_pull_up_account_values": {
            "total": sum(unseen_accounts.values()),
            "pull_up_account_test_examples": sum(ex.action == "pull-up-account" for ex in test),
            "by_kind": dict(unseen_accounts),
            "values": dict(unseen_examples),
        },
        "extraction": subset_accuracy(extract, preds),
        "label_quality": {
            "echo_actions": echo_actions,
            "templates": {a: list(tp) for a, tp in templates.items()},
            "checked": checked,
            "gold_differs_from_logged_value": sum(kinds.values()),
            "by_kind": dict(kinds),
            "examples": dict(by_kind_examples),
            "lora_matched_the_log_but_scored_wrong": lora_matched_the_log,
        },
    }
    Path("results/analysis.json").write_text(json.dumps(out, indent=2))

    for title, section in (("Memorisation", "memorisation"), ("Extraction", "extraction")):
        print(f"\n{title}  (joint accuracy, %)")
        for group, row in out[section].items():
            cells = "  ".join(f"{s}={100 * row[s]['joint']:.1f}" for s in SYSTEMS)
            print(f"  {group:38} n={row['n']:<5} {cells}")
    u = out["unseen_pull_up_account_values"]
    print(
        f"\nTest pull-up-account values never seen in training: {u['total']} of "
        f"{u['pull_up_account_test_examples']}: {u['by_kind']}"
    )
    lq = out["label_quality"]
    print(
        f"\nLabel quality: of {lq['checked']} checked, gold differs from the logged value in "
        f"{lq['gold_differs_from_logged_value']}: {lq['by_kind']}"
    )
    print(
        f"  fine-tuned model matched a differing log exactly, scored wrong: "
        f"{lq['lora_matched_the_log_but_scored_wrong']}"
    )


def clean(value: str) -> str:
    """Normalise a value for comparison: case, whitespace, and edge punctuation."""
    return normalize(value).strip(" .,$")


def message_template(messages: list[str]) -> tuple[str, str]:
    """The fixed text before and after the value, from many messages of one action."""
    prefix = os.path.commonprefix(messages)
    suffix = os.path.commonprefix([m[::-1] for m in messages])[::-1]
    return prefix, suffix


def logged_value(message: str, template: tuple[str, str]) -> str | None:
    """The value inside a system message, "" if the agent entered nothing, None if it
    does not fit the template at all."""
    prefix, suffix = template
    m = normalize(message)
    if not (m.startswith(prefix.rstrip()) and m.endswith(suffix.lstrip())):
        return None
    if len(m) < len(prefix) + len(suffix):
        return ""  # prefix and suffix overlap: nothing between them
    return clean(m[len(prefix) : len(m) - len(suffix)])


if __name__ == "__main__":
    main()
