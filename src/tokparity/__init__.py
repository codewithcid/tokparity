"""tokparity: differential testing of tokenizer implementations against Hugging Face `tokenizers`."""

from tokparity.engine import Mismatch, compare
from tokparity.minimize import minimize

__all__ = ["Mismatch", "compare", "minimize"]
__version__ = "0.1.0"
