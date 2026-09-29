"""Run a model over an evaluation split and score it.

    python -m nextaction.evaluate --split test --name base-0shot --prompt full
    python -m nextaction.evaluate --split test --name base-5shot --prompt full --shots 5
    python -m nextaction.evaluate --split test --name lora --adapter adapters/lora --prompt compact

Decoding is greedy. Batched generation is not bit-identical to one example at a time
(see docs/decisions.md, "Inference numerics"), so every system runs in the same batch
configuration. Baselines use the full prompt and the fine-tuned model the compact one it
was trained with; few-shot adds solved training examples as earlier turns of the chat.
Predictions are kept, one line per example, so every number in the README can be traced
back to the outputs that produced it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.models.cache import make_prompt_cache

from nextaction.data import Example, build, load_raw, read_eval
from nextaction.metrics import score, summarize
from nextaction.prompt import messages

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def demonstrations(k: int, seed: int, prompt: str) -> list[dict[str, str]]:
    """k solved training examples, each a different action, as earlier chat turns."""
    if k == 0:
        return []
    pool = build(load_raw()["train"])
    random.Random(seed).shuffle(pool)
    chosen, seen = [], set()
    for ex in pool:
        if ex.action not in seen:
            chosen.append(ex)
            seen.add(ex.action)
        if len(chosen) == k:
            break
    turns = []
    for ex in chosen:
        _, user, assistant = messages(ex.context, ex.answer, prompt)
        turns += [user, assistant]
    return turns


def prompt_tokens(tokenizer, ex: Example, demos: list[dict[str, str]], prompt: str) -> list[int]:
    system, user = messages(ex.context, prompt=prompt)
    return tokenizer.apply_chat_template([system, *demos, user], add_generation_prompt=True)


def shared_prefix(tokenizer, demos: list[dict[str, str]], prompt: str) -> list[int]:
    """The tokens every prompt starts with: the system prompt and any demonstrations."""
    system, _ = messages([], prompt=prompt)
    return tokenizer.apply_chat_template([system, *demos], add_generation_prompt=False)


def prefix_cache(model, prefix: list[int]) -> list:
    """Run the shared prefix through the model once and keep its attention cache.

    Prompt processing is almost all of the evaluation cost here, and the system prompt
    alone is about half of every prompt (with five demonstrations, most of it). Every
    example then only processes its own conversation. It changes a few outputs through
    floating-point effects, as batching itself does; `nextaction.numerics` measures that.
    """
    cache = make_prompt_cache(model)
    model(mx.array(prefix)[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def generate_all(model, tokenizer, prompts, prefix, batch_size, *, use_cache: bool) -> list[str]:
    """Greedy answers for every prompt, optionally reusing the shared prefix's cache."""
    if use_cache:
        if any(p[: len(prefix)] != prefix for p in prompts):
            raise ValueError("a prompt does not start with the shared prefix; cannot cache")
        cache = prefix_cache(model, prefix)
        inputs = [p[len(prefix) :] for p in prompts]
    else:
        inputs = prompts

    # Longest first, so each batch is similar in length and the first batch finds any
    # out-of-memory problem immediately rather than an hour in.
    order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]))
    outputs: list[str] = [""] * len(inputs)
    started = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        caches = [cache] * len(idx) if use_cache else None
        resp = batch_generate(
            model, tokenizer, [inputs[i] for i in idx], prompt_caches=caches, max_tokens=96
        )
        for i, text in zip(idx, resp.texts, strict=True):
            outputs[i] = text
        done = min(b + batch_size, len(order))
        print(f"  {done}/{len(order)}  {done / (time.time() - started):.1f} ex/s", flush=True)
    return outputs


def run(
    split: str,
    name: str,
    adapter: str | None,
    shots: int,
    limit: int | None,
    batch_size: int,
    prompt: str,
) -> dict:
    examples = read_eval(Path(f"data/eval/{split}.jsonl"))
    if limit:
        examples = examples[:limit]

    model, tokenizer = load(MODEL, adapter_path=adapter)
    demos = demonstrations(shots, seed=7, prompt=prompt)
    prompts = [prompt_tokens(tokenizer, ex, demos, prompt) for ex in examples]
    prefix = shared_prefix(tokenizer, demos, prompt)

    started = time.time()
    outputs = generate_all(model, tokenizer, prompts, prefix, batch_size, use_cache=True)
    elapsed = time.time() - started

    scores = [score(ex, out) for ex, out in zip(examples, outputs, strict=True)]
    out_dir = Path("results/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{split}-{name}.jsonl").open("w") as f:
        for ex, out, s in zip(examples, outputs, scores, strict=True):
            row = {
                "convo_id": ex.convo_id,
                "turn": ex.turn,
                "gold_action": ex.action,
                "gold_values": ex.values,
                "output": out,
                **asdict(s),
                "joint": s.joint,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "name": name,
        "split": split,
        "model": MODEL,
        "adapter": adapter,
        "shots": shots,
        "prompt": prompt,
        "seconds": round(elapsed, 1),
        "mean_prompt_tokens": sum(map(len, prompts)) / len(prompts),
        **summarize(scores),
    }
    (Path("results") / f"{split}-{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def check_cache(
    split: str, adapter: str | None, shots: int, n: int, batch_size: int, prompt: str
) -> None:
    examples = read_eval(Path(f"data/eval/{split}.jsonl"))[:n]
    model, tokenizer = load(MODEL, adapter_path=adapter)
    demos = demonstrations(shots, seed=7, prompt=prompt)
    prompts = [prompt_tokens(tokenizer, ex, demos, prompt) for ex in examples]
    prefix = shared_prefix(tokenizer, demos, prompt)
    t0 = time.time()
    plain = generate_all(model, tokenizer, prompts, prefix, batch_size, use_cache=False)
    t1 = time.time()
    cached = generate_all(model, tokenizer, prompts, prefix, batch_size, use_cache=True)
    t2 = time.time()
    same = sum(a == b for a, b in zip(plain, cached, strict=True))
    print(f"identical outputs: {same}/{n}   uncached {t1 - t0:.1f}s   cached {t2 - t1:.1f}s")
    for a, b in zip(plain, cached, strict=True):
        if a != b:
            print(f"  DIFF\n    uncached: {a!r}\n    cached:   {b!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("dev", "dev500", "test"), default="dev")
    p.add_argument("--name", required=True)
    p.add_argument("--adapter")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument(
        "--prompt",
        choices=("full", "compact"),
        default="full",
        help="full for the untrained baselines; compact for the fine-tuned model",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--check-cache",
        action="store_true",
        help="compare cached and uncached outputs on --limit examples, then exit",
    )
    a = p.parse_args()
    if a.check_cache:
        check_cache(a.split, a.adapter, a.shots, a.limit or 32, a.batch_size, a.prompt)
        return
    s = run(a.split, a.name, a.adapter, a.shots, a.limit, a.batch_size, a.prompt)
    for m in ("joint", "action", "values", "valid"):
        lo, hi = s[m]["ci95"]
        print(f"{m:7} {s[m]['mean']:.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
