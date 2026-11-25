"""A tokenizer running in another process, spoken to with one JSON line per batch."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Sequence


class LineProcess:
    def __init__(self, cmd: list[str], what: str):
        self._what = what
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            encoding="utf-8",
            errors="replace",
        )

    def _error(self) -> RuntimeError:
        self._proc.wait(timeout=10)
        self._stderr.seek(0)
        lines = self._stderr.read().decode("utf-8", "replace").strip().splitlines()
        return RuntimeError(f"{self._what}: {lines[0] if lines else 'process exited'}")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        try:
            self._proc.stdin.write(json.dumps(list(texts), ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except OSError:
            raise self._error() from None
        if not line:
            raise self._error()
        return json.loads(line)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
        self._stderr.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
