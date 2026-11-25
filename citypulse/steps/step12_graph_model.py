# =============================================================================
# STEP NAME  : step12 — Simple Graph + GRU Model (PyTorch)
# WHAT       : Combines a Graph Convolution layer (using our road adjacency
#              matrix) with a GRU in PyTorch, trained with masked MAE loss.
# WHY        : Roads form a GRAPH, not isolated lines! When a traffic jam
#              starts on one avenue, it spills over to nearby connected roads.
#              Graph convolution lets every sensor read its neighbours' speeds
#              before the GRU predicts the future.
# INPUT      : data/processed/deep_train.npz, deep_val.npz, deep_test.npz
#              data/processed/adj_matrix.npy
# OUTPUT     : models/graph_gru_model.pt
#              reports/step12_graph_gru_results.csv
#              reports/step12_graph_gru_loss_curve.png
#              Appended entry in reports/step_summaries.md
# =============================================================================

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_SEED, USE_SYNTHETIC

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 65)
print("  STEP 12 — Simple Graph + GRU Model (PyTorch)")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# SECTION A — Load 4D tensors and normalized road adjacency matrix
# ===========================================================================
print("\n[A] Loading tensors and road adjacency matrix ...")
tr = np.load(os.path.join(DATA_PROCESSED_DIR, "deep_train.npz"))
va = np.load(os.path.join(DATA_PROCESSED_DIR, "deep_val.npz"))
te = np.load(os.path.join(DATA_PROCESSED_DIR, "deep_test.npz"))

adj = np.load(os.path.join(DATA_PROCESSED_DIR, "adj_matrix.npy")).astype(np.float32)

# Add self-loops (each sensor also keeps its own info) and row-normalize
# A_norm = D^(-1) * (A + I)
adj_self = adj + np.eye(adj.shape[0], dtype=np.float32)
row_sum  = adj_self.sum(axis=1, keepdims=True)
adj_norm = torch.tensor(adj_self / row_sum, device=device)   # (50, 50)

print(f"  Train X shape: {tr['X'].shape}  |  Adjacency: {adj_norm.shape}")

train_loader = DataLoader(TensorDataset(torch.tensor(tr["X"]), torch.tensor(tr["Y"])),
                          batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(va["X"]), torch.tensor(va["Y"])),
                          batch_size=64, shuffle=False)

# ===========================================================================
# SECTION B — Define Graph Convolution + GRU Model
# ===========================================================================
# Why roads form a graph and jams spread:
# - Each road sensor is a NODE in the graph.
# - Connected roads are EDGES weighted by road distance (adj_norm).
# - Multiplying features by adj_norm mixes each sensor's data with its
#   road neighbours' data — just like watching traffic approach from upstream!

class SimpleGraphGRU(nn.Module):
    def __init__(self, adj_mat, in_features=5, hidden_size=32, out_horizons=4):
        super().__init__()
        self.adj = adj_mat
        # Linear transform after graph mixing (1-hop spatial convolution)
        self.gcn_linear = nn.Linear(in_features, hidden_size)
        self.relu       = nn.ReLU()
        # GRU reads the time sequence for each sensor
        self.gru        = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc_out     = nn.Linear(hidden_size, out_horizons)

    def forward(self, x):
        # x shape: (batch B, time T=12, sensors N=50, features F=5)
        B, T, N, F = x.shape

        # 1. Graph convolution: mix neighbours across N for every time step
        #    (B, T, N, F) -> matrix multiply with adj (N, N)
        x_gcn = torch.einsum("nm,btmf->btnf", self.adj, x)
        x_gcn = self.relu(self.gcn_linear(x_gcn))   # (B, T, N, hidden)

        # 2. Reshape so GRU processes each sensor's time sequence
        x_seq = x_gcn.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.gru(x_seq)                    # (B*N, T, hidden)
        pred   = self.fc_out(out[:, -1, :])         # (B*N, 4)

        # 3. Reshape back to (B, 4 horizons, N sensors)
        return pred.reshape(B, N, -1).permute(0, 2, 1)


def masked_mae_loss(pred, target):
    """Ignore any NaN targets safely when computing MAE."""
    mask = ~torch.isnan(target)
    return torch.abs(pred[mask] - target[mask]).mean()


model     = SimpleGraphGRU(adj_norm, in_features=tr["X"].shape[3],
                           hidden_size=32, out_horizons=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# ===========================================================================
# SECTION C — Train with early stopping (fast on CPU: max 12 epochs)
#             Note: for bigger real datasets, run 30-50 epochs on Google Colab GPU.
# ===========================================================================
print("\n[C] Training Graph+GRU ...")
EPOCHS       = 8
PATIENCE     = 3
best_val     = float("inf")
wait         = 0
train_losses = []
val_losses   = []
best_path    = os.path.join(MODELS_DIR, "graph_gru_model.pt")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_sum, tr_n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = masked_mae_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        tr_sum += loss.item() * len(xb)
        tr_n   += len(xb)
    tr_mae = tr_sum / tr_n

    model.eval()
    va_sum, va_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            va_sum += masked_mae_loss(model(xb), yb).item() * len(xb)
            va_n   += len(xb)
    va_mae = va_sum / va_n

    train_losses.append(tr_mae)
    val_losses.append(va_mae)
    print(f"  Epoch {epoch:2d}/{EPOCHS}  |  Train MAE: {tr_mae:.5f}  |  Val MAE: {va_mae:.5f}")

    if va_mae < best_val:
        best_val = va_mae
        wait     = 0
        torch.save(model.state_dict(), best_path)
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

# ===========================================================================
# SECTION D — Evaluate on TEST set
# ===========================================================================
print("\n[D] Evaluating Graph+GRU on test set ...")
model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    preds_te = model(torch.tensor(te["X"]).to(device)).cpu().numpy()   # (S, 4, 50)

Y_te = te["Y"]
results = []
for idx, mins in enumerate([15, 30, 45, 60]):
    y_true_flat = Y_te[:, idx, :].ravel()
    y_pred_flat = preds_te[:, idx, :].ravel()
    mae  = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = root_mean_squared_error(y_true_flat, y_pred_flat)
    results.append({
        "model"   : "Graph_GRU",
        "horizon" : f"+{mins}m",
        "mae"     : round(float(mae), 5),
        "rmse"    : round(float(rmse), 5),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

out_csv = os.path.join(REPORTS_DIR, "step12_graph_gru_results.csv")
results_df.to_csv(out_csv, index=False)

# Plot loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train MAE")
ax.plot(range(1, len(val_losses) + 1),   val_losses,   marker="s", label="Validation MAE")
ax.set_xlabel("Epoch")
ax.set_ylabel("Masked MAE Loss")
ax.set_title("Step 12 — Graph+GRU Train vs Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, "step12_graph_gru_loss_curve.png"), dpi=120, bbox_inches="tight")
plt.close()

mae_15 = results_df.loc[results_df["horizon"] == "+15m", "mae"].values[0]
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 12: Graph + GRU Model (PyTorch)\n\n")
    f.write("**What we did:** Combined graph convolution (using the road adjacency matrix) "
            "with a GRU and masked MAE loss.\n\n")
    f.write(f"**What we got:** Best validation MAE = **{best_val:.4f}**. "
            f"Test +15m MAE = **{mae_15:.4f}**.\n\n")
    f.write(f"**Key number:** Graph+GRU +15m MAE = **{mae_15:.4f}**.\n\n")
    f.write("**Plot:** `reports/step12_graph_gru_loss_curve.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 12 DONE.  Graph+GRU +15m Test MAE = {mae_15:.4f}")
print("=" * 65)
