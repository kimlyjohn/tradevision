"""
_bootstrap.py — Executed at import time to ensure the project root is on sys.path.

This module exists solely so that app.py can do a normal top-level import
(``import _bootstrap``) before importing streamlit and other project modules,
keeping all imports at module level while still patching sys.path first.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
