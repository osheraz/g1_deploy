"""Expose the recovery package to system Python workers without model imports."""
from pathlib import Path
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "steerable_recovery/paths.py").is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from steerable_recovery.paths import ROOT
