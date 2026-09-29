"""Build the training and evaluation files, and describe the data honestly first.

    python -m nextaction.prepare --train-n 8000

Writes, under data/:

    sft/train.jsonl   chat-format training examples, sampled from ABCD's train split
    sft/valid.jsonl   a small slice of ABCD's dev split, for the loss curve only
    eval/dev.jsonl    ABCD's full dev split
    eval/dev500.jsonl 500 dev examples that chose the training checkpoint
    eval/test.jsonl   ABCD's full test split, touched once, for the reported results

Nothing is ever selected by looking at test scores.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from nextaction.data import build, load_raw, sample, write_eval, write_sft
from nextaction.metrics import normalize, transcript_ceiling

# The official options for actions whose values come from a fixed set in ABCD's ontology.
OFFICIAL = {
    "membership": {"guest", "bronze", "silver", "gold"},
    "shipping-status": {"delivered", "in transit", "order received", "out for delivery"},
    "notify-team": {"manager", "website team", "purchasing department"},
}


def label_noise(examples) -> dict[str, float]:
    """Share of gold values outside the official options, for actions that have them.

    Human agents typed these into a field. "sliver" for silver and "order recieved" are
    in the gold labels, and no model should be expected to reproduce a typo.
    """
    bad, total = collections.Counter(), collections.Counter()
    for ex in examples:
        if ex.action in OFFICIAL:
            total[ex.action] += 1
            if normalize(ex.values[0]) not in OFFICIAL[ex.action]:
                bad[ex.action] += 1
    return {a: bad[a] / total[a] for a in total}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-n", type=int, default=8000)
    p.add_argument("--valid-n", type=int, default=300)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    raw = load_raw()
    train, dev, test = build(raw["train"]), build(raw["dev"]), build(raw["test"])

    write_sft(sample(train, args.train_n, args.seed), Path("data/sft/train.jsonl"))
    write_sft(sample(dev, args.valid_n, args.seed), Path("data/sft/valid.jsonl"))
    write_eval(dev, Path("data/eval/dev.jsonl"))
    # The fixed dev sample that chose the checkpoint. Seed 2026 is the one it was drawn with.
    write_eval(sample(dev, 500, seed=2026), Path("data/eval/dev500.jsonl"))
    write_eval(test, Path("data/eval/test.jsonl"))

    counts = collections.Counter(ex.action for ex in train)
    majority = counts.most_common(1)[0][0]
    stats = {
        "conversations": {s: len(raw[s]) for s in ("train", "dev", "test")},
        "examples": {"train": len(train), "dev": len(dev), "test": len(test)},
        "sft_train_used": min(args.train_n, len(train)),
        "majority_action": majority,
        "majority_action_rate_test": sum(ex.action == majority for ex in test) / len(test),
        "transcript_ceiling_test": transcript_ceiling(test),
        "label_noise_train": label_noise(train),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
