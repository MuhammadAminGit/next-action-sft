"""Every result figure in the README is recomputed here from the saved results.

Each figure is checked in context: a table row as a whole row, a number in a sentence with
the words around it, so the same number appearing elsewhere cannot satisfy the check.
Then every number left in the README, outside code blocks and the checked passages, must
be one of a short list of constants (configuration, not results). A new unchecked figure,
a hand edit, or a rerun that changes a result makes this fail.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nextaction.metrics import Score, paired_difference

ROOT = Path(__file__).resolve().parents[1]


def plain(text: str) -> str:
    """Compare as plain text: no bold markers, and any line wrapping collapsed."""
    return " ".join(text.replace("**", "").split())


def pattern(text: str) -> re.Pattern:
    """Match a checked passage only at number boundaries, so "9.9%" cannot be found inside
    "19.9%" and a passage ending in "85" cannot be found inside "850"."""
    return re.compile(r"(?<![\d.,])" + re.escape(plain(text)) + r"(?!\d)")


RAW_README = (ROOT / "README.md").read_text()
README = plain(re.sub(r"```.*?```", " ", RAW_README, flags=re.DOTALL))


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text())


def rows(system: str) -> list[dict]:
    with (ROOT / "results" / "predictions" / f"test-{system}.jsonl").open() as f:
        return [json.loads(line) for line in f]


def p(x: float) -> str:
    return f"{100 * x:.1f}"


def per_action(system_rows: list[dict]) -> dict[str, tuple[str, int]]:
    hits: dict[str, list[int]] = {}
    for r in system_rows:
        hits.setdefault(r["gold_action"], []).append(r["joint"])
    return {a: (p(sum(h) / len(h)), len(h)) for a, h in hits.items()}


def expected() -> list[tuple[str, str]]:
    stats = load("data_stats.json")
    conv, exs = stats["conversations"], stats["examples"]
    noise = stats["label_noise_train"]
    s = {
        n: load(f"test-{n}.json")
        for n in ("base-0shot-compact", "base-0shot", "base-5shot", "lora")
    }
    base, tuned = rows("base-5shot"), rows("lora")

    def as_scores(rs: list[dict]) -> list[Score]:
        return [Score(r["convo_id"], r["turn"], r["valid"], r["action"], r["values"]) for r in rs]

    gain = paired_difference(as_scores(base), as_scores(tuned), "joint")
    pairs = list(zip(base, tuned, strict=True))
    tuned_only = sum(t["joint"] and not b["joint"] for b, t in pairs)
    base_only = sum(b["joint"] and not t["joint"] for b, t in pairs)
    dev = [load(f"dev500-step{k}.json")["joint"]["mean"] for k in (1000, 2000, 3000, 4000)]
    a = load("analysis.json")
    memo, extract = a["memorisation"], a["extraction"]
    unseen, lq = a["unseen_pull_up_account_values"], a["label_quality"]
    groups = a["never_seen_group_actions"]
    id_checks = groups["verify-identity"] + groups["validate-purchase"]
    assert unseen["by_kind"]["other"] == 1, "README says 'at most one' new customer"
    num = load("numerics.json")
    metrics = ("joint", "action", "values", "valid")
    num_diffs = sorted(num["fast"][m] - num["reference"][m] for m in metrics)
    b5, lo = per_action(base), per_action(tuned)
    never = memo["some values never seen in training"]

    def row(label: str, prompt: str, sys: str) -> str:
        x = s[sys]
        j = f"{p(x['joint']['mean'])} ({p(x['joint']['ci95'][0])}–{p(x['joint']['ci95'][1])})"
        cells = [p(x[m]["mean"]) for m in ("action", "values", "valid")]
        return f"| {label} | {prompt} | {j} | " + " | ".join(cells) + " |"

    def mrow(label: str, g: dict) -> str:
        return f"| {label} | {g['n']:,} | {p(g['base-5shot']['joint'])} | {p(g['lora']['joint'])} |"

    return [
        # data
        (
            "total conversations",
            f"{sum(conv.values()):,} human-to-human customer service conversations",
        ),
        (
            "train row",
            f"| train | {conv['train']:,} | {exs['train']:,} ({stats['sft_train_used']:,} used) |",
        ),
        ("dev row", f"| dev | {conv['dev']:,} | {exs['dev']:,} |"),
        ("test row", f"| test | {conv['test']:,} | {exs['test']:,} |"),
        ("floor", f"is right {p(stats['majority_action_rate_test'])}% of the time on test"),
        (
            "extractable",
            f"In {p(stats['transcript_ceiling_test'])}% of test examples, every gold value appears",
        ),
        (
            "label noise",
            (
                f"{p(noise['notify-team'])}% for `notify-team`, {p(noise['shipping-status'])}% for "
                f"`shipping-status`, {p(noise['membership'])}% for `membership`"
            ),
        ),
        # main results
        ("test size", f"{exs['test']:,} examples from {conv['test']:,} conversations"),
        ("row compact", row("Base, zero-shot", "action names only", "base-0shot-compact")),
        ("row 0-shot", row("Base, zero-shot", "full descriptions", "base-0shot")),
        ("row 5-shot", row("Base, five-shot", "full descriptions", "base-5shot")),
        ("row lora", row("Fine-tuned (LoRA)", "action names only", "lora")),
        (
            "gain",
            (
                f"five-shot, by {100 * gain['diff']:.1f} points of joint accuracy (paired 95% "
                f"interval {100 * gain['ci95'][0]:.1f} to {100 * gain['ci95'][1]:.1f})"
            ),
        ),
        ("discordant", f"wrong {tuned_only:,} times, and the reverse {base_only} times"),
        (
            "descriptions",
            (
                f"{p(s['base-0shot-compact']['joint']['mean'])}% with action names only, "
                f"{p(s['base-0shot']['joint']['mean'])}% with descriptions"
            ),
        ),
        (
            "format",
            (
                f"Valid JSON went from {p(s['base-0shot']['valid']['mean'])}% to "
                f"{p(s['base-5shot']['valid']['mean'])}%, but the base model still chose the right "
                f"action only {p(s['base-5shot']['action']['mean'])}% of the time"
            ),
        ),
        (
            "below floor",
            (
                f"action accuracy, {p(s['base-0shot']['action']['mean'])}%, is below the "
                f"{p(stats['majority_action_rate_test'])}%"
            ),
        ),
        (
            "tuned",
            (
                f"Action accuracy rose to {p(s['lora']['action']['mean'])}%, and value accuracy to "
                f"{p(s['lora']['values']['mean'])}%"
            ),
        ),
        (
            "dev vs test",
            (
                f"{p(dev[-1])}% joint on the dev sample used to pick the checkpoint, "
                f"{p(s['lora']['joint']['mean'])}% on test"
            ),
        ),
        ("dev ranking", "(" + ", ".join(p(d) for d in dev) + ", each about ±4)"),
        # measured after the results
        (
            "unseen customers",
            (f"Of the {unseen['pull_up_account_test_examples']} test `pull-up-account` actions, "
            f"only {unseen['total']} have a value absent from the training sample, and at most "
            f"one is a new customer: {unseen['by_kind']['account id']} are account ids, "
            f"{unseen['by_kind']['misspelling of a customer in training']} are misspellings of "
            "customers in training, and 1 is a name not in the training sample"),
        ),
        (
            "never-seen confound",
            (f"{100 * id_checks / sum(groups.values()):.0f}% of the never-seen group is identity "
            "and purchase checks"),
        ),
        ("memo none", mrow("Action takes no values", memo["no values"])),
        ("memo seen", mrow("Every value seen in training", memo["all values seen in training"])),
        ("memo never", mrow("Some value never seen in training", never)),
        (
            "extract in",
            mrow("Every value in the conversation", extract["all values in the conversation"]),
        ),
        (
            "extract out",
            mrow("Some value not in the conversation", extract["some values not in it"]),
        ),
        (
            "never gap",
            (
                f"never seen in training, by "
                f"{100 * (never['lora']['joint'] - never['base-5shot']['joint']):.1f} points"
            ),
        ),
        (
            "label actions",
            f"For the {len(lq['echo_actions'])} actions whose own system message repeats the value",
        ),
        (
            "label differ",
            (
                f"They differ in {lq['gold_differs_from_logged_value']} of {lq['checked']:,} test "
                f"examples. In {lq['by_kind']['agent logged nothing']} the agent logged nothing"
            ),
        ),
        (
            "label near",
            f"In {lq['by_kind']['near match: a typo or truncation on one side']} they nearly match",
        ),
        (
            "label other",
            f"In {lq['by_kind']['logged a different value']} they are different values",
        ),
        (
            "label error",
            (
                f"In {lq['lora_matched_the_log_but_scored_wrong']} test example the fine-tuned model "
                "predicted exactly the value the agent logged"
            ),
        ),
        # where it is weak
        (
            "five-shot wins",
            (
                f"`shipping-status` ({b5['shipping-status'][0]} against {lo['shipping-status'][0]}, "
                f"{lo['shipping-status'][1]} examples) and `log-out-in` ({b5['log-out-in'][0]} "
                f"against {lo['log-out-in'][0]}, {lo['log-out-in'][1]} examples)"
            ),
        ),
        ("weakest", f"`select-faq` is the weakest at {lo['select-faq'][0]}%"),
        # numerics
        (
            "numerics",
            (
                f"moved base-model scores by {100 * num_diffs[0]:.1f} to {100 * num_diffs[-1]:.1f} "
                f"points against the one-at-a-time reference, with {num['identical_outputs']} of "
                f"{num['examples']} outputs identical"
            ),
        ),
    ]


EXPECTED = expected()

# Numbers in the README that are configuration or fixed facts, not results.
CONSTANTS = {
    "1.5", "2.5", "16", "28", "10.5", "0.68", "8,000", "8", "30", "5", "3", "2021", "4",
    "24", "0.31.3", "200", "1", "7",
    "2.0",  # Apache-2.0
    "95",   # 95% intervals
    "500",  # the dev sample size
}  # fmt: skip


@pytest.mark.parametrize(("what", "text"), EXPECTED, ids=[w for w, _ in EXPECTED])
def test_readme_states_the_measured_result(what, text):
    assert pattern(text).search(README), f"README does not say: {plain(text)!r}"


def test_every_number_in_the_readme_is_checked_or_a_constant():
    rest = README
    for _, text in EXPECTED:
        rest = pattern(text).sub(" ", rest)
    rest = re.sub(r"\(https?://[^)]*\)|`[^`]*`|Qwen2\.5-1\.5B-Instruct|[A-Za-z]+\d+", " ", rest)
    found = re.findall(r"\d[\d,]*(?:\.\d+)*", rest)
    unchecked = [n for n in found if n.rstrip(",.") not in CONSTANTS]
    assert not unchecked, f"numbers in the README that no test checks: {unchecked}"


def test_the_described_weaknesses_are_the_real_ones():
    """The README names the actions where five-shot wins, ties, and the weakest; check them."""
    b5, lo = per_action(rows("base-5shot")), per_action(rows("lora"))
    wins = {a for a in lo if float(b5[a][0]) > float(lo[a][0])}
    ties = {a for a in lo if b5[a][0] == lo[a][0]}
    assert wins == {"shipping-status", "log-out-in"}
    assert len(ties) == 2
    assert min(lo, key=lambda a: float(lo[a][0])) == "select-faq"
