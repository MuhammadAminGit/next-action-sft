.PHONY: setup test data prepare baseline train select numerics analysis eval report all

setup:
	uv venv --python 3.12 && uv pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

data:
	mkdir -p data/raw
	curl -sL -H "Accept: application/vnd.github.raw" -o data/raw/abcd_v1.1.json.gz \
	  https://api.github.com/repos/asappresearch/abcd/contents/data/abcd_v1.1.json.gz

prepare:
	.venv/bin/python -m nextaction.prepare --train-n 8000

# The base model three ways: action names only, full descriptions, and five examples.
baseline:
	.venv/bin/python -m nextaction.evaluate --split test --name base-0shot-compact --prompt compact
	.venv/bin/python -m nextaction.evaluate --split test --name base-0shot --prompt full
	.venv/bin/python -m nextaction.evaluate --split test --name base-5shot --prompt full --shots 5

train:
	.venv/bin/mlx_lm.lora --config configs/lora.yaml

# Pick the checkpoint on dev500, never on test. The rule is in docs/decisions.md.
select:
	.venv/bin/python -m nextaction.checkpoint 1000 2000 3000 4000
	for s in 1000 2000 3000 4000; do \
	  .venv/bin/python -m nextaction.evaluate --split dev500 --name step$$s --adapter adapters/step$$s --prompt compact; \
	done

# Batched vs one-at-a-time generation, on 200 dev examples.
numerics:
	.venv/bin/python -m nextaction.numerics

analysis:
	.venv/bin/python -m nextaction.analysis

eval:
	.venv/bin/python -m nextaction.evaluate --split test --name lora --adapter adapters/lora --prompt compact

report:
	.venv/bin/python -m nextaction.report --split test --baseline base-0shot-compact base-0shot base-5shot --tuned lora

all: data prepare baseline train select eval report analysis
