# =============================================================================
# STEP NAME  : step11 — Simple GRU Sequence Model (PyTorch)
# WHAT       : Trains a small GRU neural network on 12 past time steps (3 hours)
#              to predict the next 4 time steps (+15, +30, +45, +60 min).
# WHY        : A GRU reads sequences step-by-step and remembers trends over
#              time. We compare it honestly against LightGBM in Step 13.
#              NOTE: Kept small (12 epochs) to run fast on a laptop CPU;
#              increase EPOCHS=40 on Google Colab GPU for heavier training.
# INPUT      : data/processed/deep_train.npz, deep_val.npz, deep_test.npz
# OUTPUT     : models/gru_model.pt
#              reports/step11_gru_results.csv
#              reports/step11_gru_loss_curve.png
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
print("  STEP 11 — Simple PyTorch GRU Model")
if USE_SYNTHETIC:
    print("  *** SYNTHETIC DATA — results are only for testing the pipeline ***")
print("=" * 65)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}  (Tip: on Google Colab GPU this runs even faster!)")

# ===========================================================================
# SECTION A — Load sequence arrays and reshape per sensor
# ===========================================================================
print("\n[A] Loading 4D sequence arrays ...")
tr = np.load(os.path.join(DATA_PROCESSED_DIR, "deep_train.npz"))
va = np.load(os.path.join(DATA_PROCESSED_DIR, "deep_val.npz"))
te = np.load(os.path.join(DATA_PROCESSED_DIR, "deep_test.npz"))

def to_sensor_sequences(X, Y):
    """
    Reshape (samples, 12, N_sensors, F) -> (samples * N_sensors, 12, F)
    and     (samples, 4, N_sensors)     -> (samples * N_sensors, 4)
    So the GRU treats each sensor's 3-hour window as one sequence.
    """
    S, T, N, F = X.shape
    X_seq = np.transpose(X, (0, 2, 1, 3)).reshape(S * N, T, F)
    Y_seq = np.transpose(Y, (0, 2, 1)).reshape(S * N, Y.shape[1])
    return X_seq, Y_seq

X_tr, Y_tr = to_sensor_sequences(tr["X"], tr["Y"])
X_va, Y_va = to_sensor_sequences(va["X"], va["Y"])
X_te, Y_te = to_sensor_sequences(te["X"], te["Y"])

print(f"  GRU train sequences: {X_tr.shape}  |  val: {X_va.shape}  |  test: {X_te.shape}")

train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
                          batch_size=1024, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(Y_va)),
                          batch_size=1024, shuffle=False)

# ===========================================================================
# SECTION B — Define a simple 1-layer GRU model
# ===========================================================================
class SimpleGRU(nn.Module):
    def __init__(self, in_features=5, hidden_size=32, out_horizons=4):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, out_horizons)

    def forward(self, x):
        # x shape: (batch, 12 steps, 5 features)
        out, _ = self.gru(x)       # out shape: (batch, 12, hidden_size)
        last   = out[:, -1, :]     # take the final time step summary
        return self.fc(last)       # predict 4 future horizons

model     = SimpleGRU(in_features=X_tr.shape[2], hidden_size=32, out_horizons=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
loss_fn   = nn.L1Loss()   # L1Loss = MAE loss

# ===========================================================================
# SECTION C — Train with early stopping (fast on CPU: max 12 epochs)
# ===========================================================================
print("\n[C] Training GRU ...")
EPOCHS       = 8
PATIENCE     = 3
best_val     = float("inf")
wait         = 0
train_losses = []
val_losses   = []
best_path    = os.path.join(MODELS_DIR, "gru_model.pt")

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    tr_sum, tr_n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        tr_sum += loss.item() * len(xb)
        tr_n   += len(xb)
    tr_mae = tr_sum / tr_n

    # --- Validate ---
    model.eval()
    va_sum, va_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            va_sum += loss_fn(pred, yb).item() * len(xb)
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
# SECTION D — Evaluate best model on TEST set
# ===========================================================================
print("\n[D] Evaluating best GRU on test set ...")
model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    preds_te = model(torch.tensor(X_te).to(device)).cpu().numpy()

results = []
for idx, mins in enumerate([15, 30, 45, 60]):
    mae  = mean_absolute_error(Y_te[:, idx], preds_te[:, idx])
    rmse = root_mean_squared_error(Y_te[:, idx], preds_te[:, idx])
    results.append({
        "model"   : "GRU",
        "horizon" : f"+{mins}m",
        "mae"     : round(float(mae), 5),
        "rmse"    : round(float(rmse), 5),
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

out_csv = os.path.join(REPORTS_DIR, "step11_gru_results.csv")
results_df.to_csv(out_csv, index=False)

# Plot train vs val loss
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train MAE")
ax.plot(range(1, len(val_losses) + 1),   val_losses,   marker="s", label="Validation MAE")
ax.set_xlabel("Epoch")
ax.set_ylabel("MAE Loss")
ax.set_title("Step 11 — GRU Train vs Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(REPORTS_DIR, "step11_gru_loss_curve.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()

# Append summary
mae_15 = results_df.loc[results_df["horizon"] == "+15m", "mae"].values[0]
summary_path = os.path.join(REPORTS_DIR, "step_summaries.md")
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("## Step 11: Simple GRU Model (PyTorch)\n\n")
    f.write("**What we did:** Trained a 1-layer PyTorch GRU on 12 past time steps "
            "to forecast +15, +30, +45, and +60 min ahead.\n\n")
    f.write(f"**What we got:** Best validation MAE = **{best_val:.4f}**. "
            f"Test +15m MAE = **{mae_15:.4f}**.\n\n")
    f.write(f"**Key number:** GRU +15m MAE = **{mae_15:.4f}**.\n\n")
    f.write("**Plot:** `reports/step11_gru_loss_curve.png`\n\n---\n\n")

print("\n" + "=" * 65)
print(f"  STEP 11 DONE.  GRU +15m Test MAE = {mae_15:.4f}")
print("=" * 65)
