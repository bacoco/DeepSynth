"""Project-wide interpreter customisation.

When the repository is used without installation (e.g. running scripts
straight from the source checkout) the ``src`` directory is not part of the
module search path.  Adding it here mirrors the standard ``src`` layout used
by Python packaging tools and keeps imports such as ``deepsynth.data`` working
for ad-hoc commands and tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parent / "src"
if _SRC_PATH.is_dir():
    sys.path.insert(0, str(_SRC_PATH))
