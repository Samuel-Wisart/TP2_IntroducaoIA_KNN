from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import (
    run_kmeans_experiment,
    run_kmeans_sklearn_comparison,
    run_knn_experiment,
    run_knn_sklearn_comparison,
)

TRAIN_FILE = ROOT / "nba_treino.csv"
TEST_FILE = ROOT / "nba_teste.csv"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    knn_k_values = [2, 10, 50, 7]
    kmeans_k_values = [2, 3]

    run_knn_experiment(TRAIN_FILE, TEST_FILE, knn_k_values, RESULTS_DIR / "knn")
    run_knn_sklearn_comparison(TRAIN_FILE, TEST_FILE, knn_k_values, RESULTS_DIR / "knn_sklearn")
    run_kmeans_experiment(TRAIN_FILE, TEST_FILE, kmeans_k_values, RESULTS_DIR / "kmeans")
    run_kmeans_sklearn_comparison(TRAIN_FILE, TEST_FILE, kmeans_k_values, RESULTS_DIR / "kmeans_sklearn")

    print("Experiments finished. Check the results/ folder.")


if __name__ == "__main__":
    main()
