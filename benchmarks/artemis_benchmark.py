#!/usr/bin/env python3
"""
Artemis benchmark for Genetic Melody Reconstruction.

The project keeps the GeneticAlgorithm implementation in notebooks/Genetic.ipynb.
This harness deliberately executes only the cells that define:
  1. the WAV frequency-extraction function
  2. the GeneticAlgorithm class

It does NOT execute the notebook's plotting, comparison, or audio-generation cells.

Usage:
    python benchmarks/artemis_benchmark.py --smoke
    python benchmarks/artemis_benchmark.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "Genetic.ipynb"
TARGET_PATH = REPO_ROOT / "data" / "target_melodies" / "fur_elise_target.wav"

# These match the main Für Elise experiment in the repository notebook.
NOTE_DURATION = 0.30
POP_SIZE = 1000
GENERATIONS = 100
MUTATION_RATE = 0.02
FITNESS_METHOD = "hamming"
SELECTION_METHOD = "tournament"
CROSSOVER_METHOD = "single_point"
NUM_ELITES = 2

# Fixed seed makes repeated Artemis runs comparable.
RANDOM_SEED = 0


def _load_notebook_code() -> dict[str, object]:
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")
    if not TARGET_PATH.is_file():
        raise FileNotFoundError(f"Target WAV not found: {TARGET_PATH}")

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Cell 4 defines extract_frequencies_from_wav.
    # Cell 7 defines threshold and GeneticAlgorithm.
    code_cells = notebook.get("cells", [])
    required_cells = (4, 7)

    namespace: dict[str, object] = {"__name__": "__artemis_benchmark__", "np": np, "wavfile": wavfile}

    for index in required_cells:
        try:
            cell = code_cells[index]
        except IndexError as exc:
            raise RuntimeError(
                f"Expected notebook cell {index}, but the notebook structure changed."
            ) from exc

        if cell.get("cell_type") != "code":
            raise RuntimeError(f"Notebook cell {index} is not a code cell.")

        source = "".join(cell.get("source", []))
        exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

    if "extract_frequencies_from_wav" not in namespace:
        raise RuntimeError("Could not load extract_frequencies_from_wav from the notebook.")
    if "GeneticAlgorithm" not in namespace:
        raise RuntimeError("Could not load GeneticAlgorithm from the notebook.")

    return namespace


def _extract_target(namespace: dict[str, object]) -> np.ndarray:
    extractor = namespace["extract_frequencies_from_wav"]
    if not callable(extractor):
        raise TypeError("extract_frequencies_from_wav is not callable.")

    target = extractor(
        str(TARGET_PATH),
        note_duration=NOTE_DURATION,
    )
    target = np.asarray(target, dtype=np.int64)

    if target.ndim != 1 or target.size == 0:
        raise RuntimeError("Target melody extraction returned an invalid array.")

    return target


def _run_ga(namespace: dict[str, object], target: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    np.random.seed(seed)

    ga_class = namespace["GeneticAlgorithm"]
    if not callable(ga_class):
        raise TypeError("GeneticAlgorithm is not callable.")

    ga = ga_class(
        target_frequencies=target,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        min_freq=int(np.min(target)),
        max_freq=int(np.max(target)),
    )

    start = time.perf_counter()
    best = ga.run(
        fitness_method=FITNESS_METHOD,
        selection_method=SELECTION_METHOD,
        crossover_method=CROSSOVER_METHOD,
        num_elites=NUM_ELITES,
    )
    elapsed = time.perf_counter() - start

    best = np.asarray(best, dtype=np.int64)
    if best.shape != target.shape:
        raise RuntimeError(
            f"Best chromosome shape {best.shape} does not match target shape {target.shape}."
        )

    final_mae = float(np.mean(np.abs(best - target)))

    if not np.isfinite(elapsed) or not np.isfinite(final_mae):
        raise RuntimeError("Benchmark produced a non-finite result.")

    return best, final_mae


def smoke_test() -> None:
    namespace = _load_notebook_code()
    target = _extract_target(namespace)

    # Keep setup validation cheap. The real benchmark uses the notebook's
    # full 1000-population / 100-generation configuration.
    global POP_SIZE, GENERATIONS
    original_pop_size = POP_SIZE
    original_generations = GENERATIONS

    try:
        POP_SIZE = 20
        GENERATIONS = 3
        best, final_mae = _run_ga(namespace, target, RANDOM_SEED)
    finally:
        POP_SIZE = original_pop_size
        GENERATIONS = original_generations

    print(f"smoke_target_length={target.size}")
    print(f"smoke_best_shape={best.shape}")
    print(f"smoke_final_mae={final_mae:.6f}")
    print("smoke_status=PASS")


def benchmark() -> None:
    namespace = _load_notebook_code()
    target = _extract_target(namespace)

    best, final_mae = _run_ga(namespace, target, RANDOM_SEED)

    # Artemis records the command's wall-clock runtime as its built-in
    # performance metric. The values below are additional quality/context
    # signals visible in the benchmark log.
    print(f"target_file={TARGET_PATH.relative_to(REPO_ROOT)}")
    print(f"target_length={target.size}")
    print(f"population={POP_SIZE}")
    print(f"generations={GENERATIONS}")
    print(f"mutation_rate={MUTATION_RATE}")
    print(f"fitness_method={FITNESS_METHOD}")
    print(f"selection_method={SELECTION_METHOD}")
    print(f"crossover_method={CROSSOVER_METHOD}")
    print(f"num_elites={NUM_ELITES}")
    print(f"seed={RANDOM_SEED}")
    print(f"final_mae={final_mae:.6f}")
    print("benchmark_status=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    if args.smoke:
        smoke_test()
    else:
        benchmark()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
