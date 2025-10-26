#!/usr/bin/env python3
"""Script de lancement principal pour le traitement parallèle des datasets."""

import sys
from typing import Callable


def _load_entrypoint() -> Callable[[], int]:
    """Import the parallel processing entrypoint with a helpful error message."""

    try:
        from deepsynth.parallel_processing.run_parallel_datasets import main
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "Unable to import 'deepsynth.parallel_processing'. "
            "Set PYTHONPATH=src or install the project in editable mode."
        ) from exc

    return main


if __name__ == "__main__":
    sys.exit(_load_entrypoint()())