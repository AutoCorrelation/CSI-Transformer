import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import load_mat_dataset, average_subcarriers, prepare_datasets, combine_complex
from models import CSITransformer
from metrics import sum_rate


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x, x)
        loss = torch.mean((y_hat - y) ** 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x, x)
            loss = torch.mean((y_hat - y) ** 2)
            total_loss += loss.item() * x.size(0)
            preds.append(y_hat.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    return total_loss / len(loader.dataset), preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="../data/csi_dataset.mat")
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--snr_db", type=float, default=10.0)
    args = parser.parse_args()

    H, params = load_mat_dataset(args.mat)
    H_avg = average_subcarriers(H)  # [T, K, M]

    train_ds, val_ds, test_ds, mean, std = prepare_datasets(
        H_avg, lookback=args.lookback, horizon=args.horizon
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_ds.x.shape[-1]
    model = CSITransformer(input_dim=input_dim).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = eval_one_epoch(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - train {tr_loss:.4e} - val {val_loss:.4e}")

    test_loss, preds, targets = eval_one_epoch(model, test_loader, device)
    print(f"Test MSE: {test_loss:.4e}")

    # De-normalize predictions and targets
    mean = mean.squeeze(0)
    std = std.squeeze(0)
    preds = preds * std + mean
    targets = targets * std + mean

    # Convert to complex and reshape to [T, K, M]
    preds_c = combine_complex(preds)
    targets_c = combine_complex(targets)

    K = H_avg.shape[1]
    M = H_avg.shape[2]

    preds_c = preds_c.reshape(-1, K, M)
    targets_c = targets_c.reshape(-1, K, M)

    # Sum-rate evaluation
    rates_perfect = sum_rate(targets_c, targets_c, snr_db=args.snr_db)
    rates_pred = sum_rate(targets_c, preds_c, snr_db=args.snr_db)

    out = {
        "train_loss": np.array(history["train_loss"]),
        "val_loss": np.array(history["val_loss"]),
        "test_loss": np.array(test_loss),
        "rates_perfect": rates_perfect,
        "rates_pred": rates_pred,
        "snr_db": np.array(args.snr_db),
    }
    out_path = (Path(__file__).resolve().parent / "../data/results.npz").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **out)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
