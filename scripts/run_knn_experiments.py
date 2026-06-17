from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import run_knn_experiment, run_knn_sklearn_comparison


TRAIN_FILE = ROOT / "nba_treino.csv"
TEST_FILE = ROOT / "nba_teste.csv"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    k_values = [2, 10, 50, 7]
    run_knn_experiment(TRAIN_FILE, TEST_FILE, k_values, RESULTS_DIR / "knn")
    run_knn_sklearn_comparison(TRAIN_FILE, TEST_FILE, k_values, RESULTS_DIR / "knn_sklearn")
    print("kNN experiment finished.")


if __name__ == "__main__":
    main()
