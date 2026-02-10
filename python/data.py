import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


def load_mat_dataset(mat_path):
    path = Path(mat_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    try:
        mat = sio.loadmat(str(path))
        H = mat["H"]  # [T, K, Nsc, M]
        params = mat.get("params", {})
        return H, params
    except NotImplementedError:
        if h5py is None:
            raise RuntimeError("MATLAB v7.3 file detected. Install h5py to read it.")
        with h5py.File(str(path), "r") as f:
            H = np.array(f["H"])
        # MATLAB saves arrays in column-major; transpose to [T, K, Nsc, M]
        H = np.transpose(H, (3, 2, 1, 0))
        return H, {}


def average_subcarriers(H):
    if isinstance(H.dtype, np.dtype) and H.dtype.fields is not None:
        if "real" in H.dtype.fields and "imag" in H.dtype.fields:
            H = H["real"] + 1j * H["imag"]
    return np.mean(H, axis=2)  # [T, K, M]


def split_complex(x):
    return np.concatenate([x.real, x.imag], axis=-1)


def combine_complex(x):
    half = x.shape[-1] // 2
    return x[..., :half] + 1j * x[..., half:]


class CSIPredictionDataset(Dataset):
    def __init__(self, series, lookback=8, horizon=1, mean=None, std=None):
        self.lookback = lookback
        self.horizon = horizon
        self.series = series

        self.x = []
        self.y = []
        for t in range(lookback, series.shape[0] - horizon + 1):
            self.x.append(series[t - lookback:t])
            self.y.append(series[t + horizon - 1])

        self.x = np.stack(self.x, axis=0)
        self.y = np.stack(self.y, axis=0)

        if mean is None or std is None:
            mean = self.x.mean(axis=(0, 1), keepdims=True)
            std = self.x.std(axis=(0, 1), keepdims=True) + 1e-6
        self.mean = mean
        self.std = std

        self.x = (self.x - self.mean) / self.std
        self.y = (self.y - self.mean.squeeze(0)) / self.std.squeeze(0)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]).float(),
            torch.from_numpy(self.y[idx]).float(),
        )


def prepare_datasets(H_avg, lookback=8, horizon=1, train_ratio=0.7, val_ratio=0.15):
    T = H_avg.shape[0]
    H_vec = H_avg.reshape(T, -1)  # [T, K*M]
    H_feat = split_complex(H_vec)

    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)

    train_series = H_feat[:n_train]
    val_series = H_feat[n_train - lookback:n_train + n_val]
    test_series = H_feat[n_train + n_val - lookback:]

    train_ds = CSIPredictionDataset(train_series, lookback, horizon)
    val_ds = CSIPredictionDataset(val_series, lookback, horizon, train_ds.mean, train_ds.std)
    test_ds = CSIPredictionDataset(test_series, lookback, horizon, train_ds.mean, train_ds.std)

    return train_ds, val_ds, test_ds, train_ds.mean, train_ds.std
