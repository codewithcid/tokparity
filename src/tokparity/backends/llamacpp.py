"""llama.cpp backend, on a vocab-only GGUF.

Two ways to call llama.cpp:
- `lib`: ctypes on the shared library (fast). The `llama_model_params` layout below matches llama.cpp
  release b12728; other releases may change it, so `cli` is the fallback.
- `cli`: one `llama-tokenize` process per text (slow, but independent of the struct layout).
"""

from __future__ import annotations

import ast
import ctypes as C
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


class _ModelParams(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("load_mode", C.c_int),
        ("lazy_mode", C.c_int),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
        ("load_mtp", C.c_bool),
    ]


_LOG_CB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
_quiet = _LOG_CB(lambda level, text, user_data: None)
_libs: dict[str, C.CDLL] = {}


def _lib_name() -> str:
    if sys.platform == "win32":
        return "llama.dll"
    return "libllama.dylib" if sys.platform == "darwin" else "libllama.so"


def _load_lib(lib_dir: Path) -> C.CDLL:
    key = str(lib_dir)
    if key in _libs:
        return _libs[key]
    if sys.platform == "win32":
        os.add_dll_directory(str(lib_dir))
    lib = C.CDLL(str(lib_dir / _lib_name()))
    lib.llama_model_default_params.restype = _ModelParams
    lib.llama_model_load_from_file.argtypes = [C.c_char_p, _ModelParams]
    lib.llama_model_load_from_file.restype = C.c_void_p
    lib.llama_model_get_vocab.argtypes = [C.c_void_p]
    lib.llama_model_get_vocab.restype = C.c_void_p
    lib.llama_model_free.argtypes = [C.c_void_p]
    lib.llama_tokenize.argtypes = [
        C.c_void_p,
        C.c_char_p,
        C.c_int32,
        C.POINTER(C.c_int32),
        C.c_int32,
        C.c_bool,
        C.c_bool,
    ]
    lib.llama_tokenize.restype = C.c_int32
    lib.llama_token_to_piece.argtypes = [C.c_void_p, C.c_int32, C.c_char_p, C.c_int32, C.c_int32, C.c_bool]
    lib.llama_token_to_piece.restype = C.c_int32
    lib.llama_log_set.argtypes = [_LOG_CB, C.c_void_p]
    lib.llama_log_set(_quiet, None)
    lib.llama_backend_init()
    _libs[key] = lib
    return lib


class LlamaCpp:
    """`gguf` is a (vocab-only) GGUF file; `lib_dir` holds llama.cpp's shared library and binaries."""

    def __init__(self, gguf: str | Path, lib_dir: str | Path, mode: str = "lib"):
        self.name = f"llama.cpp ({mode})"
        self.gguf = Path(gguf)
        self.lib_dir = Path(lib_dir).resolve()
        self.mode = mode
        self._model = None
        if mode == "lib":
            self._lib = _load_lib(self.lib_dir)
            params = self._lib.llama_model_default_params()
            params.vocab_only = True
            self._model = self._lib.llama_model_load_from_file(str(self.gguf).encode(), params)
            if not self._model:
                raise RuntimeError(f"llama.cpp could not load {self.gguf}")
            self._vocab = self._lib.llama_model_get_vocab(self._model)
            self._buf = (C.c_int32 * 8192)()
        elif mode == "cli":
            exe = "llama-tokenize.exe" if sys.platform == "win32" else "llama-tokenize"
            self._exe = self.lib_dir / exe
            if not self._exe.exists():
                raise FileNotFoundError(self._exe)
        else:
            raise ValueError(f"unknown llama.cpp mode {mode!r}")

    def _encode_lib(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        n = self._lib.llama_tokenize(self._vocab, data, len(data), self._buf, len(self._buf), False, False)
        if n < 0:
            self._buf = (C.c_int32 * -n)()
            n = self._lib.llama_tokenize(
                self._vocab, data, len(data), self._buf, len(self._buf), False, False
            )
        return list(self._buf[:n])

    def _encode_cli(self, text: str) -> list[int]:
        with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
            f.write(text.encode("utf-8"))
        try:
            out = subprocess.run(
                [
                    str(self._exe),
                    "-m",
                    str(self.gguf),
                    "-f",
                    f.name,
                    "--ids",
                    "--no-bos",
                    "--no-parse-special",
                    "--log-disable",
                ],
                capture_output=True,
                check=True,
            ).stdout.decode("utf-8", "replace")
        finally:
            os.unlink(f.name)
        return ast.literal_eval(out.strip().splitlines()[-1])

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encode = self._encode_lib if self.mode == "lib" else self._encode_cli
        return [encode(t) for t in texts]

    def piece(self, token_id: int) -> str:
        if self.mode != "lib":
            return f"<{token_id}>"
        buf = C.create_string_buffer(256)
        n = self._lib.llama_token_to_piece(self._vocab, token_id, buf, 256, 0, True)
        return buf.raw[: max(n, 0)].decode("utf-8", "replace")

    def __del__(self):
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None
