
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import gradio as gr
warnings.filterwarnings("ignore")

OUTPUT_DIR = "/content/outputs"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_cols  = json.load(open(os.path.join(OUTPUT_DIR, "feature_cols.json")))
val_threshold = json.load(open(os.path.join(OUTPUT_DIR, "threshold.json")))["threshold"]
scaler        = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
medians       = joblib.load(os.path.join(OUTPUT_DIR, "medians.pkl"))
VITAL_COLS    = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]

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
    def forward(self, x): return self.net(x).squeeze(1)

model = SepsisNet(len(feature_cols)).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE))
model.eval()

def engineer_features(df):
    df = df.copy().sort_values("ICULOS").reset_index(drop=True)
    for col in VITAL_COLS:
        if col in df.columns:
            df[f"{col}_rmean6"] = df[col].rolling(6, min_periods=1).mean()
            df[f"{col}_rmean3"] = df[col].rolling(3, min_periods=1).mean()
            df[f"{col}_rstd"]   = df[col].rolling(6, min_periods=1).std()
            df[f"{col}_delta"]  = df[col].diff()
            df[f"{col}_delta2"] = df[col].diff(2)
    return df

def run_inference(df):
    X = df[feature_cols].copy()
    X = X.fillna(medians).fillna(0).replace([np.inf, -np.inf], 0)
    X = np.clip(X.values.astype(np.float32), -1e6, 1e6)
    X = np.clip(scaler.transform(X).astype(np.float32), -10, 10)
    with torch.no_grad():
        return torch.sigmoid(model(torch.FloatTensor(X).to(DEVICE))).cpu().numpy()

def load_psv(file_obj):
    if file_obj is None: return gr.update(choices=[], value=None), "No file uploaded.", None
    try: df = pd.read_csv(file_obj.name, sep="|")
    except Exception as e: return gr.update(choices=[], value=None), f"Error: {e}", None
    if "patient_id" not in df.columns:
        df["patient_id"] = os.path.basename(file_obj.name).replace(".psv","")
    patients = sorted(df["patient_id"].unique().tolist())
    has_labels = "SepsisLabel" in df.columns
    summary = f"✓ Loaded **{len(df)}** rows · **{len(patients)}** patient(s) · {'Labels ✓' if has_labels else 'No labels'}"
    choices = [str(p) for p in patients]
    return gr.update(choices=choices, value=choices[0]), summary, df.to_json()

def predict_patient(patient_id, df_json, use_thr):
    if df_json is None or patient_id is None: return None, "Upload a file first.", ""
    df_full = pd.read_json(df_json)
    if "patient_id" not in df_full.columns: df_full["patient_id"] = "patient"
    pdf = df_full[df_full["patient_id"].astype(str) == str(patient_id)].copy()
    if len(pdf) == 0: return None, "Patient not found.", ""
    pdf = engineer_features(pdf)
    for col in feature_cols:
        if col not in pdf.columns: pdf[col] = 0.0
    probs = run_inference(pdf)
    hours = pdf["ICULOS"].values if "ICULOS" in pdf.columns else np.arange(len(pdf))
    has_labels = "SepsisLabel" in pdf.columns
    true_labels = pdf["SepsisLabel"].values if has_labels else None
    pred_labels = (probs >= val_threshold).astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.patch.set_facecolor("#0d0d1a")
    for ax in axes:
        ax.set_facecolor("#12122a"); ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2a4a")

    ax = axes[0]
    for h, p in zip(hours, pred_labels):
        if p == 1: ax.axvspan(h-0.5, h+0.5, alpha=0.18, color="#e74c3c", zorder=0)
    ax.plot(hours, probs, color="#e74c3c", lw=2.5, label="Sepsis Risk", zorder=3)
    ax.fill_between(hours, probs, alpha=0.15, color="#e74c3c")
    ax.axhline(val_threshold, color="#f1c40f", lw=1.8, linestyle="--", label=f"Threshold = {val_threshold:.2f}")
    if has_labels and true_labels is not None:
        onset_idx = np.where(true_labels == 1)[0]
        if len(onset_idx) > 0:
            ax.axvline(hours[onset_idx[0]], color="#ff6b6b", lw=2.5, linestyle="-.", label=f"True Onset (hr {hours[onset_idx[0]]})")
    ax.set_ylim(0, 1.05); ax.set_ylabel("P(Sepsis)", fontsize=12)
    ax.set_title(f"Patient {patient_id} — Sepsis Risk Timeline", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9); ax.grid(alpha=0.15)

    ax2 = axes[1]
    for v, c in zip([x for x in ["HR","SBP","Resp","Temp"] if x in pdf.columns], ["#3498db","#2ecc71","#e67e22","#9b59b6"]):
        vals = pdf[v].values; vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        ax2.plot(hours, (vals-vmin)/(vmax-vmin+1e-9), lw=1.8, color=c, label=v)
    ax2.set_ylabel("Vitals (norm.)", fontsize=10); ax2.set_title("Key Vital Signs", fontsize=11)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, ncol=4); ax2.grid(alpha=0.15)

    ax3 = axes[2]
    ax3.bar(hours, np.ones(len(hours)), color=["#e74c3c" if p==1 else "#2ecc71" for p in pred_labels], width=0.85, alpha=0.85)
    if has_labels and true_labels is not None:
        for h, pred, true in zip(hours, pred_labels, true_labels):
            if true==1 and pred==0: ax3.bar(h, 1, color="#f39c12", width=0.85, alpha=0.9)
    ax3.set_yticks([]); ax3.set_xlabel("ICU Hour", fontsize=11)
    ax3.set_title("Prediction per Hour  (🔴 Sepsis · 🟢 No Sepsis · 🟠 Missed)", fontsize=10); ax3.grid(alpha=0.1)
    plt.tight_layout(pad=2.0)
    fig_path = "/content/outputs/demo_plot.png"
    plt.savefig(fig_path, dpi=150, facecolor="#0d0d1a"); plt.close()

    max_prob = float(probs.max()); n_flagged = int(pred_labels.sum())
    risk = "🔴 HIGH RISK" if max_prob > 0.6 else ("🟡 MODERATE" if max_prob > val_threshold else "🟢 LOW RISK")
    summary_md = f"### {risk} — Patient `{patient_id}`\n\n| Metric | Value |\n|---|---|\n| ICU Hours | {len(hours)} |\n| Peak Risk | **{max_prob:.1%}** |\n| Hours Flagged | {n_flagged}/{len(hours)} |\n| Threshold | {val_threshold:.3f} |\n"
    alert = f"⚠️ **First flagged at hour {hours[np.where(pred_labels==1)[0][0]]}**" if n_flagged > 0 else ""
    return fig_path, summary_md, alert

DARK_CSS = "body, .gradio-container { background: #0d0d1a !important; color: #e0e0ff !important; }"
with gr.Blocks(css=DARK_CSS, title="Sepsis Early Warning System") as demo:
    gr.Markdown("# 🏥 Sepsis Early Warning System\n### Upload a PSV file → select patient → view risk timeline")
    df_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(label="📂 Upload PSV File", file_types=[".psv",".csv"], type="filepath")
            file_status = gr.Markdown("*No file loaded.*")
        with gr.Column(scale=1):
            patient_dd = gr.Dropdown(label="👤 Patient ID", choices=[], value=None, interactive=True)
            predict_btn = gr.Button("🔍 Analyse Patient", variant="primary")
    with gr.Row():
        with gr.Column(scale=3): plot_out = gr.Image(label="Risk Timeline", type="filepath")
        with gr.Column(scale=1):
            alert_out = gr.Markdown("")
            summary_out = gr.Markdown("*Run a prediction to see summary.*")
    file_input.change(load_psv, [file_input], [patient_dd, file_status, df_state])
    predict_btn.click(predict_patient, [patient_dd, df_state, gr.Checkbox(value=True, visible=False)], [plot_out, summary_out, alert_out])
    patient_dd.change(predict_patient, [patient_dd, df_state, gr.Checkbox(value=True, visible=False)], [plot_out, summary_out, alert_out])

demo.launch(share=True, debug=False)
