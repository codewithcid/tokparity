"""tokenizers.js backend (`@huggingface/tokenizers`, also used inside transformers.js). Needs Node.js.

`module` is the package's ESM entry: by default the npm package installed by `tokparity setup-tokenizersjs`,
or a local tokenizers.js checkout's built `dist/tokenizers.mjs` to test a branch.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tokparity.backends._process import LineProcess

HARNESS = Path(__file__).resolve().parent.parent / "harness" / "tokenizersjs" / "encode.mjs"


def install_dir() -> Path:
    from tokparity.paths import cache_dir

    return cache_dir() / "tokenizersjs"


def default_module() -> Path:
    return install_dir() / "node_modules" / "@huggingface" / "tokenizers" / "dist" / "tokenizers.mjs"


def setup(version: str = "latest") -> Path:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("`npm` not found on PATH")
    target = install_dir()
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run([npm, "install", "--prefix", str(target), f"@huggingface/tokenizers@{version}"], check=True)
    return default_module()


class TokenizersJS:
    def __init__(self, model_dir: str | Path, module: str | Path | None = None):
        self.module = Path(module or default_module())
        if not self.module.exists():
            raise FileNotFoundError(f"{self.module} not found; run `tokparity setup-tokenizersjs`")
        self.name = "tokenizers.js"
        self.model_dir = Path(model_dir)
        self._node = shutil.which("node")
        if self._node is None:
            raise RuntimeError("`node` not found on PATH")
        self._proc = LineProcess(
            [self._node, str(HARNESS), str(self.module), str(self.model_dir)],
            f"tokenizers.js on {self.model_dir}",
        )

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return self._proc.encode_batch(texts)

    def piece(self, token_id: int) -> str:
        return f"<{token_id}>"
