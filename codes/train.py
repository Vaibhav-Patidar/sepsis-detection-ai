from google.colab import drive
drive.mount('/content/drive')

import os, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

DATA_DIR   = "/content/drive/MyDrive/sepsis_data"
OUTPUT_DIR = "/content/outputs"
SEED       = 42
BATCH_SIZE = 512
NUM_EPOCHS = 80
LR         = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

print("\nLoading PSV files...")
all_dfs   = []
psv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.psv')]
print(f"Found {len(psv_files)} patient files.")

for fname in psv_files:
    df = pd.read_csv(os.path.join(DATA_DIR, fname), sep='|')
    df['patient_id'] = fname.replace('.psv', '')
    all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows: {len(data)}  |  Sepsis rate: {data['SepsisLabel'].mean()*100:.2f}%")

VITAL_COLS = [c for c in ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2']
              if c in data.columns]
LAB_COLS   = [c for c in ['BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN',
                           'Alkalinephos','Calcium','Chloride','Creatinine',
                           'Bilirubin_direct','Glucose','Lactate','Magnesium',
                           'Phosphate','Potassium','Bilirubin_total','TroponinI',
                           'Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
              if c in data.columns]
DEMO_COLS  = [c for c in ['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']
              if c in data.columns]
BASE_COLS  = VITAL_COLS + LAB_COLS + DEMO_COLS

print("Engineering features...")
data = data.sort_values(['patient_id','ICULOS']).reset_index(drop=True)

roll_dfs = []
for pid, grp in data.groupby('patient_id'):
    grp = grp.copy()
    for col in VITAL_COLS:
        grp[f'{col}_rmean6'] = grp[col].rolling(6,  min_periods=1).mean()
        grp[f'{col}_rmean3'] = grp[col].rolling(3,  min_periods=1).mean()
        grp[f'{col}_rstd']   = grp[col].rolling(6,  min_periods=1).std()
        grp[f'{col}_delta']  = grp[col].diff()
        grp[f'{col}_delta2'] = grp[col].diff(2)
    roll_dfs.append(grp)

data      = pd.concat(roll_dfs, ignore_index=True)
ROLL_COLS = [c for c in data.columns
             if any(c.endswith(s) for s in ('_rmean6','_rmean3','_rstd','_delta','_delta2'))]
ALL_FEATURES = BASE_COLS + ROLL_COLS

X = data[ALL_FEATURES].copy()
y = data['SepsisLabel'].values.astype(np.float32)

print("Cleaning data...")
medians = X.median().fillna(0)
X = X.fillna(medians).fillna(0).replace([np.inf, -np.inf], 0)
X = np.clip(X.values.astype(np.float32), -1e6, 1e6)

assert not np.isnan(X).any() and not np.isinf(X).any()
print(f"Features: {X.shape}  |  Positives: {y.sum():.0f} ({y.mean()*100:.2f}%)")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

scaler  = StandardScaler()
X_train = np.clip(scaler.fit_transform(X_train).astype(np.float32), -10, 10)
X_val   = np.clip(scaler.transform(X_val).astype(np.float32),        -10, 10)

joblib.dump(scaler,     os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(medians,    os.path.join(OUTPUT_DIR, "medians.pkl"))
json.dump(ALL_FEATURES, open(os.path.join(OUTPUT_DIR, "feature_cols.json"), "w"))


raw_ratio  = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
pos_weight = raw_ratio 
print(f"Imbalance ratio: {raw_ratio:.1f}x  |  pos_weight: {pos_weight:.1f}x")

sample_weights = np.where(y_train == 1, pos_weight, 1.0)
sampler = WeightedRandomSampler(torch.FloatTensor(sample_weights),
                                len(sample_weights), replacement=True)

train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

class SepsisNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model     = SepsisNet(X_train.shape[1]).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

history      = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
best_f1      = 0.0
best_t_final = 0.3 
best_weights = None

print("\nTraining...")
for epoch in range(NUM_EPOCHS):
    model.train()
    tr_loss, tr_total = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss  += loss.item() * xb.size(0)
        tr_total += xb.size(0)
    scheduler.step()

    model.eval()
    vl_loss, vl_total = 0.0, 0
    val_probs, val_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out  = model(xb)
            loss = criterion(out, yb)
            vl_loss  += loss.item() * xb.size(0)
            vl_total += xb.size(0)
            val_probs.extend(torch.sigmoid(out).cpu().numpy().tolist())
            val_true.extend(yb.cpu().numpy().tolist())

    vp = np.array(val_probs, dtype=np.float64)
    vt = np.array(val_true,  dtype=np.float64)

    if np.isnan(vp).any():
        continue

    val_auc = roc_auc_score(vt, vp)

    ths    = np.arange(0.10, 0.61, 0.01)
    f1s    = [f1_score(vt, (vp >= t).astype(int), zero_division=0) for t in ths]
    best_t = float(ths[np.argmax(f1s)])
    val_f1 = float(max(f1s))

    history["train_loss"].append(tr_loss / tr_total)
    history["val_loss"].append(vl_loss / vl_total)
    history["val_auc"].append(val_auc)
    history["val_f1"].append(val_f1)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss/tr_total:.4f}  "
              f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}  "
              f"threshold={best_t:.2f}")

    if val_f1 > best_f1:
        best_f1      = val_f1
        best_t_final = best_t
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_weights, os.path.join(OUTPUT_DIR, "best_model.pth"))

print(f"\nBest Val F1: {best_f1:.4f}  at threshold: {best_t_final:.2f}")
json.dump({"threshold": best_t_final},
          open(os.path.join(OUTPUT_DIR, "threshold.json"), "w"))

model.load_state_dict(best_weights)
model.eval()
val_probs, val_true = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        out = model(xb.to(DEVICE))
        val_probs.extend(torch.sigmoid(out).cpu().numpy().tolist())
        val_true.extend(yb.numpy().tolist())

vp = np.array(val_probs, dtype=np.float64)
vt = np.array(val_true,  dtype=np.float64)
vd = (vp >= best_t_final).astype(int)

overall_acc = float((vd == vt).mean())
precision   = float(precision_score(vt, vd, zero_division=0))
recall      = float(recall_score(vt, vd, zero_division=0))
f1          = float(f1_score(vt, vd, zero_division=0))
roc_auc     = float(roc_auc_score(vt, vp))
per_class   = {
    "No Sepsis": float(((vd==0)&(vt==0)).sum() / max((vt==0).sum(), 1)),
    "Sepsis":    float(((vd==1)&(vt==1)).sum() / max((vt==1).sum(), 1)),
}
t0     = time.time()
with torch.no_grad():
    _ = model(torch.FloatTensor(X_val[:100]).to(DEVICE))
inf_ms = (time.time() - t0) / 100 * 1000

print("\n" + "="*55)
print("         TABLE 1 — ALL METRICS (Val Set)")
print("="*55)
print(f"  1. Overall Accuracy     : {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"  2. Precision            : {precision:.4f}")
print(f"  3. Recall / Sensitivity : {recall:.4f}")
print(f"  4. F1-Score             : {f1:.4f}")
print(f"  5. Per-Class Accuracy   :")
for cls, acc in per_class.items():
    print(f"       {cls:<14}: {acc:.4f}")
print(f"  6. ROC-AUC              : {roc_auc:.4f}")
print(f"  7. Confusion Matrix     : see confusion_matrix.png")
print(f"  8. Inference Time/Speed : {inf_ms:.3f} ms/sample")
print("="*55)
print(classification_report(vt, vd, target_names=["No Sepsis","Sepsis"]))

metrics = {
    "overall_accuracy":   round(overall_acc, 4),
    "precision":          round(precision, 4),
    "recall":             round(recall, 4),
    "f1_score":           round(f1, 4),
    "per_class_accuracy": {k: round(v, 4) for k, v in per_class.items()},
    "roc_auc":            round(roc_auc, 4),
    "inference_time_ms":  round(inf_ms, 4),
    "best_threshold":     best_t_final,
}
json.dump(metrics, open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"), indent=2)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history["val_auc"], color="green", label="Val AUC")
axes[1].set_title("ROC-AUC"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(history["val_f1"], color="orange", label="Val F1")
axes[2].set_title("F1-Score"); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150)
plt.show()

cm = confusion_matrix(vt, vd)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Sepsis","Sepsis"],
            yticklabels=["No Sepsis","Sepsis"])
plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

fpr, tpr, _ = roc_curve(vt, vp)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_auc.png"), dpi=150)
plt.show()

print(f"\nDone. Outputs saved to {OUTPUT_DIR}/")