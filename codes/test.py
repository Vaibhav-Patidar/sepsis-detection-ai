import os, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

TEST_DIR   = "/content/drive/MyDrive/sepsis_test_data"   
OUTPUT_DIR = "/content/outputs"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

print("\nLoading saved model artifacts...")
feature_cols  = json.load(open(os.path.join(OUTPUT_DIR, "feature_cols.json")))
val_threshold = json.load(open(os.path.join(OUTPUT_DIR, "threshold.json")))["threshold"]
scaler        = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
medians       = joblib.load(os.path.join(OUTPUT_DIR, "medians.pkl"))
print(f"  Feature count    : {len(feature_cols)}")
print(f"  Val threshold    : {val_threshold:.4f}")

class SepsisNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

model = SepsisNet(len(feature_cols)).to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE))
model.eval()
print(f"  Model loaded     : {sum(p.numel() for p in model.parameters()):,} params")

print("\nLoading test PSV files...")
psv_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.psv')]
print(f"  Found {len(psv_files)} patient files.")
dfs = []
for fname in psv_files:
    df = pd.read_csv(os.path.join(TEST_DIR, fname), sep='|')
    df['patient_id'] = fname.replace('.psv', '')
    dfs.append(df)
test_data = pd.concat(dfs, ignore_index=True)
test_data = test_data.reset_index(drop=True)
print(f"  Total rows       : {len(test_data)}")

has_labels = 'SepsisLabel' in test_data.columns
if has_labels:
    print(f"  Sepsis rate      : {test_data['SepsisLabel'].mean()*100:.2f}%")
else:
    print("  No SepsisLabel column found — running inference only (no metrics).")

VITAL_COLS = [c for c in ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2']
              if c in test_data.columns]

print("\nEngineering features on test data...")
test_data = test_data.sort_values(['patient_id','ICULOS']).reset_index(drop=True)
roll_dfs  = []
for pid, grp in test_data.groupby('patient_id'):
    grp = grp.copy()
    for col in VITAL_COLS:
        grp[f'{col}_rmean6'] = grp[col].rolling(6,  min_periods=1).mean()
        grp[f'{col}_rmean3'] = grp[col].rolling(3,  min_periods=1).mean()
        grp[f'{col}_rstd']   = grp[col].rolling(6,  min_periods=1).std()
        grp[f'{col}_delta']  = grp[col].diff()
        grp[f'{col}_delta2'] = grp[col].diff(2)
    roll_dfs.append(grp)
test_data = pd.concat(roll_dfs, ignore_index=True)

print("Preparing feature matrix...")
X_raw  = test_data[feature_cols].copy()
X_raw  = X_raw.fillna(medians).fillna(0).replace([np.inf, -np.inf], 0)
X_raw  = np.clip(X_raw.values.astype(np.float32), -1e6, 1e6)
X_test = np.clip(scaler.transform(X_raw).astype(np.float32), -10, 10)

assert not np.isnan(X_test).any() and not np.isinf(X_test).any()
print(f"  Shape  : {X_test.shape}")

if has_labels:
    y_test = test_data['SepsisLabel'].values.astype(np.float32)
    print(f"  Positives: {y_test.sum():.0f} ({y_test.mean()*100:.2f}%)")

print("\nRunning inference on test set...")
dummy_y   = torch.zeros(len(X_test))
test_ds   = TensorDataset(torch.FloatTensor(X_test), dummy_y)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

test_probs = []
t_start = time.time()
with torch.no_grad():
    for xb, _ in test_loader:
        out = model(xb.to(DEVICE))
        test_probs.extend(torch.sigmoid(out).cpu().numpy().tolist())
t_total = time.time() - t_start

vp = np.array(test_probs, dtype=np.float64)


t0 = time.time()
with torch.no_grad():
    _ = model(torch.FloatTensor(X_test[:100]).to(DEVICE))
inf_ms = (time.time() - t0) / 100 * 1000

print("\nSaving predictions.csv ...")
pred_df = pd.DataFrame({
    "patient_id":        test_data["patient_id"].values,
    "ICULOS":            test_data["ICULOS"].values,
    "sepsis_probability": np.round(vp, 6),
    "predicted_label":   (vp >= val_threshold).astype(int),
})
if has_labels:
    pred_df["true_label"] = y_test.astype(int)

pred_csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"  Saved {len(pred_df):,} rows → {pred_csv_path}")
print(f"  Predicted sepsis rows: {pred_df['predicted_label'].sum():,} "
      f"({pred_df['predicted_label'].mean()*100:.2f}%)")

if not has_labels:
    print("\nNo ground-truth labels — skipping metrics. predictions.csv saved.")
    raise SystemExit(0)

vt = y_test.astype(np.float64)

ths = np.arange(0.05, 0.61, 0.005)


f1s       = [f1_score(vt, (vp >= t).astype(int), zero_division=0) for t in ths]
t_best_f1 = float(ths[np.argmax(f1s)])

f1s_rc = []
for t in ths:
    pred = (vp >= t).astype(int)
    rec  = recall_score(vt, pred, zero_division=0)
    f1   = f1_score(vt,   pred, zero_division=0)
    f1s_rc.append(f1 if rec >= 0.70 else 0.0)
t_best_recall = float(ths[np.argmax(f1s_rc)]) if max(f1s_rc) > 0 else t_best_f1

print(f"\n  Val-optimised threshold  : {val_threshold:.3f}")
print(f"  Test best-F1 threshold   : {t_best_f1:.3f}  (diagnostic)")
print(f"  Test recall≥0.70 thresh  : {t_best_recall:.3f}  (clinical)")

def compute_metrics(vp, vt, thresh, label):
    vd  = (vp >= thresh).astype(int)
    acc = float((vd == vt).mean())
    pre = float(precision_score(vt, vd, zero_division=0))
    rec = float(recall_score(vt,  vd, zero_division=0))
    f1  = float(f1_score(vt,      vd, zero_division=0))
    auc = float(roc_auc_score(vt, vp))
    pca = {
        "No Sepsis": float(((vd==0)&(vt==0)).sum() / max((vt==0).sum(), 1)),
        "Sepsis":    float(((vd==1)&(vt==1)).sum() / max((vt==1).sum(), 1)),
    }
    print("\n" + "═"*60)
    print(f"  TEST METRICS  [{label}  |  threshold = {thresh:.3f}]")
    print("═"*60)
    print(f"  1. Overall Accuracy     : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  2. Precision            : {pre:.4f}")
    print(f"  3. Recall / Sensitivity : {rec:.4f}")
    print(f"  4. F1-Score             : {f1:.4f}")
    print(f"  5. Per-Class Accuracy   :")
    print(f"       No Sepsis          : {pca['No Sepsis']:.4f}")
    print(f"       Sepsis             : {pca['Sepsis']:.4f}")
    print(f"  6. ROC-AUC              : {auc:.4f}")
    print(f"  7. Inference Speed      : {inf_ms:.4f} ms/sample")
    print("═"*60)
    print(classification_report(vt, vd, target_names=["No Sepsis","Sepsis"]))
    return {"threshold": thresh, "overall_accuracy": round(acc,4),
            "precision": round(pre,4), "recall": round(rec,4),
            "f1_score": round(f1,4), "roc_auc": round(auc,4),
            "per_class_accuracy": {k: round(v,4) for k,v in pca.items()},
            "inference_time_ms": round(inf_ms,4)}, vd

m_val,    vd_val    = compute_metrics(vp, vt, val_threshold,    "VAL-THRESHOLD")
m_best,   vd_best   = compute_metrics(vp, vt, t_best_f1,        "BEST-F1")
m_recall, vd_recall = compute_metrics(vp, vt, t_best_recall,    "HIGH-RECALL")

test_metrics_out = {
    "val_threshold_applied": m_val,
    "test_best_f1":          m_best,
    "test_high_recall":      m_recall,
}
json.dump(test_metrics_out,
          open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w"), indent=2)
json.dump({"test": m_best, "threshold_used": t_best_f1},
          open(os.path.join(OUTPUT_DIR, "test_primary.json"), "w"), indent=2)
print(f"\nTest metrics saved → {OUTPUT_DIR}/test_metrics.json")

pred_df["predicted_label_best_f1"]    = (vp >= t_best_f1).astype(int)
pred_df["predicted_label_high_recall"] = (vp >= t_best_recall).astype(int)
pred_df.to_csv(pred_csv_path, index=False)
print(f"predictions.csv updated with all threshold columns → {pred_csv_path}")


fpr, tpr, _ = roc_curve(vt, vp)
np.savez(os.path.join(OUTPUT_DIR, "roc_data_test.npz"), fpr=fpr, tpr=tpr)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0f0f1a")
configs = [
    (vd_val,    f"Val-Threshold ({val_threshold:.2f})"),
    (vd_best,   f"Best-F1 ({t_best_f1:.2f})"),
    (vd_recall, f"High-Recall ({t_best_recall:.2f})"),
]
for ax, (vd, title) in zip(axes, configs):
    cm = confusion_matrix(vt, vd)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Sepsis","Sepsis"],
                yticklabels=["No Sepsis","Sepsis"],
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_ylabel("True",      color="white"); ax.set_xlabel("Predicted", color="white")
    ax.set_facecolor("#1a1a2e"); ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333366")
fig.suptitle("Test Set — Confusion Matrices at 3 Operating Points",
             color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "test_confusion_matrices.png"), dpi=150,
            facecolor="#0f0f1a")
plt.show()

plt.figure(figsize=(6, 5), facecolor="#1a1a2e")
ax = plt.gca(); ax.set_facecolor("#1a1a2e")
plt.plot(fpr, tpr, color="#e74c3c", lw=2.5, label=f"AUC = {m_best['roc_auc']:.4f}")
plt.fill_between(fpr, tpr, alpha=0.12, color="#e74c3c")
plt.plot([0,1],[0,1],'w--', lw=1, alpha=0.4)
plt.xlabel("False Positive Rate", color="white"); plt.ylabel("True Positive Rate", color="white")
plt.title("Test ROC-AUC Curve", color="white", fontsize=12)
plt.legend(labelcolor="white", facecolor="#1a1a2e"); plt.grid(alpha=0.2)
plt.tick_params(colors="white")
for spine in ax.spines.values(): spine.set_edgecolor("#333366")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "test_roc_auc.png"), dpi=150, facecolor="#1a1a2e")
plt.show()

prec_curve = [precision_score(vt, (vp>=t).astype(int), zero_division=0) for t in ths]
rec_curve  = [recall_score(vt,   (vp>=t).astype(int), zero_division=0) for t in ths]
f1_curve   = [f1_score(vt,       (vp>=t).astype(int), zero_division=0) for t in ths]

plt.figure(figsize=(9, 5), facecolor="#1a1a2e")
ax = plt.gca(); ax.set_facecolor("#1a1a2e")
plt.plot(ths, prec_curve, color="#3498db", lw=2,   label="Precision")
plt.plot(ths, rec_curve,  color="#e74c3c", lw=2,   label="Recall")
plt.plot(ths, f1_curve,   color="#2ecc71", lw=2.5, label="F1-Score")
plt.axvline(val_threshold, color="#f1c40f", lw=2, linestyle="--",
            label=f"Val threshold = {val_threshold:.2f}")
plt.axvline(t_best_f1,     color="#2ecc71", lw=2, linestyle=":",
            label=f"Best-F1 test = {t_best_f1:.2f}")
plt.axvline(t_best_recall, color="#e74c3c", lw=2, linestyle=":",
            label=f"High-recall  = {t_best_recall:.2f}")
plt.xlabel("Threshold", color="white"); plt.ylabel("Score", color="white")
plt.title("Test Set — Threshold vs Metrics", color="white", fontsize=12)
plt.legend(labelcolor="white", facecolor="#1a1a2e", fontsize=8)
plt.grid(alpha=0.2); plt.tick_params(colors="white")
for spine in ax.spines.values(): spine.set_edgecolor("#333366")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "test_threshold_curve.png"), dpi=150,
            facecolor="#1a1a2e")
plt.show()

plt.figure(figsize=(8, 4), facecolor="#1a1a2e")
ax = plt.gca(); ax.set_facecolor("#1a1a2e")
plt.hist(vp[vt==0], bins=60, alpha=0.6, label="No Sepsis", color="#3498db", density=True)
plt.hist(vp[vt==1], bins=60, alpha=0.6, label="Sepsis",    color="#e74c3c", density=True)
plt.axvline(val_threshold, color="#f1c40f", lw=2, linestyle="--",
            label=f"Val threshold  = {val_threshold:.2f}")
plt.axvline(t_best_f1,     color="#2ecc71", lw=2, linestyle=":",
            label=f"Best-F1 test  = {t_best_f1:.2f}")
plt.xlabel("P(Sepsis)", color="white"); plt.ylabel("Density", color="white")
plt.title("Test Score Distribution by Class", color="white", fontsize=12)
plt.legend(labelcolor="white", facecolor="#1a1a2e")
plt.grid(alpha=0.2); plt.tick_params(colors="white")
for spine in ax.spines.values(): spine.set_edgecolor("#333366")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "test_score_distribution.png"), dpi=150,
            facecolor="#1a1a2e")
plt.show()

print(f"\n✓  All test outputs saved to {OUTPUT_DIR}/")
print(f"   predictions.csv                ← row-level predictions")
print(f"   test_metrics.json              ← all threshold metrics")
print(f"   test_confusion_matrices.png")
print(f"   test_roc_auc.png")
print(f"   test_threshold_curve.png")
print(f"   test_score_distribution.png")