"""How much does batched generation change the answers?

    python -m nextaction.numerics

Runs the base model on 200 dev examples twice: one example at a time with no cache (the
reference), and in the configuration every reported system used (batch 16 with a cached
shared prompt). Writes results/numerics.json. Batched floating-point arithmetic happens
in a different order, so near-ties flip; this measures whether that moves the scores.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from mlx_lm import load

from nextaction.data import read_eval
from nextaction.evaluate import MODEL, generate_all, prompt_tokens, shared_prefix
from nextaction.metrics import paired_difference, score, summarize


def main() -> None:
    examples = random.Random(1).sample(read_eval(Path("data/eval/dev.jsonl")), 200)
    model, tokenizer = load(MODEL)
    prompts = [prompt_tokens(tokenizer, ex, [], "full") for ex in examples]
    prefix = shared_prefix(tokenizer, [], "full")

    ref = generate_all(model, tokenizer, prompts, prefix, 1, use_cache=False)
    fast = generate_all(model, tokenizer, prompts, prefix, 16, use_cache=True)
    s_ref = [score(e, o) for e, o in zip(examples, ref, strict=True)]
    s_fast = [score(e, o) for e, o in zip(examples, fast, strict=True)]

    out = {
        "examples": len(examples),
        "identical_outputs": sum(a == b for a, b in zip(ref, fast, strict=True)),
        "reference": {
            m: summarize(s_ref)[m]["mean"] for m in ("joint", "action", "values", "valid")
        },
        "fast": {m: summarize(s_fast)[m]["mean"] for m in ("joint", "action", "values", "valid")},
        "paired": {m: paired_difference(s_ref, s_fast, m) for m in ("joint", "action", "values")},
    }
    Path("results/numerics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
