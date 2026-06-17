from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import run_kmeans_experiment, run_kmeans_sklearn_comparison


TRAIN_FILE = ROOT / "nba_treino.csv"
TEST_FILE = ROOT / "nba_teste.csv"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    k_values = [2, 3]
    run_kmeans_experiment(TRAIN_FILE, TEST_FILE, k_values, RESULTS_DIR / "kmeans")
    run_kmeans_sklearn_comparison(TRAIN_FILE, TEST_FILE, k_values, RESULTS_DIR / "kmeans_sklearn")
    print("k-Means experiment finished.")


if __name__ == "__main__":
    main()
