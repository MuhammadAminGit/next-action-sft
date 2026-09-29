"""Make a saved training checkpoint loadable as an adapter.

mlx-lm writes checkpoints as bare weight files (0001000_adapters.safetensors) next to one
adapter_config.json. Loading needs a directory holding both, so each checkpoint gets one.

    python -m nextaction.checkpoint 1000 2000 3000 4000
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

RUN = Path("adapters/lora")


def materialize(step: int) -> Path:
    src = RUN / f"{step:07d}_adapters.safetensors"
    if not src.exists():
        raise FileNotFoundError(src)
    out = Path(f"adapters/step{step}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN / "adapter_config.json", out / "adapter_config.json")
    shutil.copy(src, out / "adapters.safetensors")
    return out


if __name__ == "__main__":
    for step in map(int, sys.argv[1:]):
        print(materialize(step))
