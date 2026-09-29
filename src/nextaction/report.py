"""Turn saved predictions into the results tables.

The README's tables are copied from this output, and tests/test_readme_numbers.py checks
every one of them against the result files.

    python -m nextaction.report --split test --baseline base-0shot base-5shot --tuned lora
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from nextaction.metrics import Score, paired_difference, parse

METRICS = ("joint", "action", "values", "valid")


def load_scores(split: str, name: str) -> tuple[list[Score], list[dict]]:
    with Path(f"results/predictions/{split}-{name}.jsonl").open() as f:
        rows = [json.loads(line) for line in f]
    scores = [Score(r["convo_id"], r["turn"], r["valid"], r["action"], r["values"]) for r in rows]
    return scores, rows


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main_table(split: str, names: list[str]) -> str:
    lines = ["| System | Joint | Action | Values | Valid JSON |", "|---|---|---|---|---|"]
    for name in names:
        s = json.loads(Path(f"results/{split}-{name}.json").read_text())
        cells = [
            f"{pct(s[m]['mean'])} ({pct(s[m]['ci95'][0])}–{pct(s[m]['ci95'][1])})" for m in METRICS
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def improvements(split: str, baselines: list[str], tuned: str) -> str:
    t, _ = load_scores(split, tuned)
    lines = [
        "| Comparison | Metric | Change (points) | 95% CI | Excludes zero |",
        "|---|---|---|---|---|",
    ]
    for b in baselines:
        base, _ = load_scores(split, b)
        for m in ("joint", "action", "values"):
            d = paired_difference(base, t, m)
            lo, hi = d["ci95"]
            lines.append(
                f"| {tuned} vs {b} | {m} | {100 * d['diff']:+.1f} | "
                f"{100 * lo:+.1f} to {100 * hi:+.1f} | {'yes' if d['excludes_zero'] else 'no'} |"
            )
    return "\n".join(lines)


def per_action(split: str, names: list[str]) -> str:
    counts: collections.Counter = collections.Counter()
    hits: dict[str, collections.Counter] = {n: collections.Counter() for n in names}
    for n in names:
        _, rows = load_scores(split, n)
        for r in rows:
            if n == names[0]:
                counts[r["gold_action"]] += 1
            hits[n][r["gold_action"]] += r["joint"]
    head = "| Action | n | " + " | ".join(names) + " |"
    lines = [head, "|---|---|" + "---|" * len(names)]
    for action, n in counts.most_common():
        lines.append(
            f"| {action} | {n} | " + " | ".join(pct(hits[x][action] / n) for x in names) + " |"
        )
    return "\n".join(lines)


def top_errors(split: str, name: str, k: int = 10) -> str:
    """The most common ways the system is wrong: which action it chose instead."""
    _, rows = load_scores(split, name)
    wrong = collections.Counter()
    for r in rows:
        if not r["action"]:
            got = parse(r["output"]).action or "(invalid output)"
            wrong[(r["gold_action"], got)] += 1
    lines = ["| Gold action | Predicted instead | Count |", "|---|---|---|"]
    lines += [f"| {g} | {p} | {c} |" for (g, p), c in wrong.most_common(k)]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--baseline", nargs="+", required=True)
    p.add_argument("--tuned", required=True)
    a = p.parse_args()
    names = [*a.baseline, a.tuned]
    stats = json.loads(Path("results/data_stats.json").read_text())

    report = "\n\n".join(
        [
            f"# Results on ABCD {a.split}",
            (
                f"{stats['examples'][a.split]} examples. Floor: always predicting "
                f"`{stats['majority_action']}` gets {pct(stats['majority_action_rate_test'])}% "
                f"action accuracy. {pct(stats['transcript_ceiling_test'])}% of examples have every "
                "gold value present in the conversation. Brackets are 95% "
                "cluster-bootstrap intervals over conversations."
            ),
            main_table(a.split, names),
            "## Improvement from fine-tuning (paired)",
            improvements(a.split, a.baseline, a.tuned),
            "## Joint accuracy by action",
            per_action(a.split, names),
            f"## Most common action errors: {a.tuned}",
            top_errors(a.split, a.tuned),
        ]
    )
    Path(f"results/report-{a.split}.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
