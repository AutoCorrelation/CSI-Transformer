from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    results_path = (Path(__file__).resolve().parent / "../data/results.npz").resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    data = np.load(str(results_path))
    rates_perfect = data["rates_perfect"]
    rates_pred = data["rates_pred"]

    plt.figure(figsize=(7, 4))
    plt.plot(rates_perfect, label="Perfect CSI")
    plt.plot(rates_pred, label="Predicted CSI")
    plt.xlabel("Time index")
    plt.ylabel("Achievable sum-rate (bps/Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = (Path(__file__).resolve().parent / "../data/sum_rate.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
