"""mml: detection and correction of meaning loss in machine translation."""

import os as _os

# transformers otherwise downloads a second (safetensors) copy of every .bin-only model in a background thread;
# on the dev laptop that silently doubles NLLB's 2.4 GB. Set before any transformers import.
_os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

__version__ = "0.1.0"
