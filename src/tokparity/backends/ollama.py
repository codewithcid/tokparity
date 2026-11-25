"""Ollama backend: the Go tokenizer of Ollama's MLX engine (`mlxrunner/tokenizer`).

`tokparity build-ollama --ollama-src PATH` compiles the small harness in `harness/ollama` against a local
Ollama checkout. The harness reads tokenizer.json (plus the companion config files) like Ollama does.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tokparity.backends._process import LineProcess

HARNESS_DIR = Path(__file__).resolve().parent.parent / "harness" / "ollama"


def default_binary() -> Path:
    from tokparity.paths import cache_dir

    return cache_dir() / ("ollama-encode.exe" if sys.platform == "win32" else "ollama-encode")


def build(ollama_src: str | Path, out: Path | None = None) -> Path:
    """Compile the harness against the Ollama checkout at `ollama_src` (needs Go)."""
    go = shutil.which("go")
    if go is None:
        raise RuntimeError("`go` not found on PATH")
    out = Path(out or default_binary()).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    src = Path(ollama_src).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(HARNESS_DIR / "main.go", work / "main.go")
        (work / "go.mod").write_text(
            "module tokparity/ollama\n\ngo 1.24\n\nrequire github.com/ollama/ollama v0.0.0\n\n"
            f"replace github.com/ollama/ollama => {src.as_posix()}\n",
            encoding="utf-8",
        )
        env_flags = ["-mod=mod"]
        subprocess.run([go, "mod", "tidy"], cwd=work, check=False, capture_output=True)
        subprocess.run([go, "build", *env_flags, "-o", str(out), "."], cwd=work, check=True)
    return out


class Ollama:
    def __init__(self, model_dir: str | Path, binary: str | Path | None = None):
        self.name = "ollama (mlxrunner/tokenizer)"
        self.model_dir = Path(model_dir)
        self.binary = Path(binary or default_binary())
        if not self.binary.exists():
            raise FileNotFoundError(
                f"{self.binary} not found; run `tokparity build-ollama --ollama-src PATH`"
            )
        self._proc = LineProcess([str(self.binary), str(self.model_dir)], f"Ollama tokenizer on {self.model_dir}")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return self._proc.encode_batch(texts)

    def piece(self, token_id: int) -> str:
        return f"<{token_id}>"
