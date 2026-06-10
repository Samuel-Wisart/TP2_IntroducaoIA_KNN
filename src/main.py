"""Main entry point for the project skeleton."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "nba_treino.csv"
TEST_FILE = ROOT / "nba_teste.csv"
RESULTS_DIR = ROOT / "results"


def show_project_overview():
    print("TP2 - Introducao a IA")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Test file:  {TEST_FILE}")
    print(f"Results dir: {RESULTS_DIR}")
    print("The custom KNN and KMeans implementations live in src/.")


if __name__ == "__main__":
    show_project_overview()
