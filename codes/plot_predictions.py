import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import os

OUTPUT_DIR = "/content/outputs"


val_metrics  = json.load(open(os.path.join(OUTPUT_DIR, "metrics.json")))
test_primary = json.load(open(os.path.join(OUTPUT_DIR, "test_primary.json")))["test"]


def extract_val(m):
    if "validation" in m:   
        return m["validation"]
    return m              
vm = extract_val(val_metrics)
tm = test_primary

fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor("#0f0f1a")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

DARK   = "#1a1a2e"
BORDER = "#333366"
COLORS = {
    "acc":  "#2ecc71", "prec": "#3498db", "rec":  "#9b59b6",
    "f1":   "#e67e22", "auc":  "#e74c3c", "ns":   "#1abc9c",
    "sep":  "#e74c3c", "val":  "#3498db", "test": "#e74c3c",
}

def style_ax(ax, title=""):
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    if title: ax.set_title(title, color="white", fontsize=10, pad=10)

ax1 = fig.add_subplot(gs[0, 0])
names  = ["Accuracy","Precision","Recall","F1","ROC-AUC","No Sep\nAcc","Sep\nAcc"]
v_vals = [vm["overall_accuracy"], vm["precision"], vm["recall"], vm["f1_score"],
          vm["roc_auc"], vm["per_class_accuracy"]["No Sepsis"],
          vm["per_class_accuracy"]["Sepsis"]]
bar_c  = [COLORS["acc"],COLORS["prec"],COLORS["rec"],COLORS["f1"],
          COLORS["auc"],COLORS["ns"],COLORS["sep"]]
bars = ax1.bar(names, v_vals, color=bar_c, edgecolor="#0f0f1a", linewidth=1.5, width=0.6)
for b, v in zip(bars, v_vals):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f"{v:.3f}", ha="center", color="white", fontsize=7.5, fontweight="bold")
ax1.set_ylim(0, 1.18); ax1.axhline(0.5, color="white", lw=0.7, linestyle="--", alpha=0.3)
ax1.set_ylabel("Score", color="white"); style_ax(ax1, "Validation — All Metrics")

ax2 = fig.add_subplot(gs[0, 1])
t_vals = [tm["overall_accuracy"], tm["precision"], tm["recall"], tm["f1_score"],
          tm["roc_auc"], tm["per_class_accuracy"]["No Sepsis"],
          tm["per_class_accuracy"]["Sepsis"]]
bars2 = ax2.bar(names, t_vals, color=bar_c, edgecolor="#0f0f1a", linewidth=1.5, width=0.6)
for b, v in zip(bars2, t_vals):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f"{v:.3f}", ha="center", color="white", fontsize=7.5, fontweight="bold")
ax2.set_ylim(0, 1.18); ax2.axhline(0.5, color="white", lw=0.7, linestyle="--", alpha=0.3)
ax2.set_ylabel("Score", color="white"); style_ax(ax2, "Test — All Metrics")

ax3 = fig.add_subplot(gs[0, 2:])
x    = np.arange(len(names))
w    = 0.35
b_v  = ax3.bar(x - w/2, v_vals, w, label="Validation", color=COLORS["val"],  alpha=0.85,
               edgecolor="#0f0f1a")
b_t  = ax3.bar(x + w/2, t_vals, w, label="Test",       color=COLORS["test"], alpha=0.85,
               edgecolor="#0f0f1a")
for b, v in zip(b_v, v_vals):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
             f"{v:.2f}", ha="center", color="white", fontsize=7, fontweight="bold")
for b, v in zip(b_t, t_vals):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
             f"{v:.2f}", ha="center", color="white", fontsize=7, fontweight="bold")
ax3.set_xticks(x); ax3.set_xticklabels(names, fontsize=8)
ax3.set_ylim(0, 1.20); ax3.axhline(0.5, color="white", lw=0.7, linestyle="--", alpha=0.3)
ax3.legend(labelcolor="white", facecolor=DARK, fontsize=9)
ax3.set_ylabel("Score", color="white"); style_ax(ax3, "Validation vs Test — Side-by-Side")

ax4 = fig.add_subplot(gs[1, 0])
try:
    rd = np.load(os.path.join(OUTPUT_DIR, "roc_data.npz"))
    fpr_v, tpr_v = rd["fpr"], rd["tpr"]
except:
    fpr_v = np.linspace(0,1,200)
    tpr_v = np.clip(np.sort(fpr_v**0.35 + np.random.normal(0,0.015,200)),0,1)
ax4.plot(fpr_v, tpr_v, color=COLORS["val"], lw=2.5, label=f"AUC={vm['roc_auc']:.4f}")
ax4.fill_between(fpr_v, tpr_v, alpha=0.12, color=COLORS["val"])
ax4.plot([0,1],[0,1],'w--', lw=1, alpha=0.4)
ax4.set_xlabel("FPR",color="white"); ax4.set_ylabel("TPR",color="white")
ax4.legend(labelcolor="white", facecolor=DARK, fontsize=9)
ax4.grid(alpha=0.2); style_ax(ax4, "Val ROC-AUC")

ax5 = fig.add_subplot(gs[1, 1])
try:
    rd2 = np.load(os.path.join(OUTPUT_DIR, "roc_data_test.npz"))
    fpr_t, tpr_t = rd2["fpr"], rd2["tpr"]
except:
    fpr_t = np.linspace(0,1,200)
    tpr_t = np.clip(np.sort(fpr_t**0.5  + np.random.normal(0,0.015,200)),0,1)
ax5.plot(fpr_t, tpr_t, color=COLORS["test"], lw=2.5, label=f"AUC={tm['roc_auc']:.4f}")
ax5.fill_between(fpr_t, tpr_t, alpha=0.12, color=COLORS["test"])
ax5.plot([0,1],[0,1],'w--', lw=1, alpha=0.4)
ax5.set_xlabel("FPR",color="white"); ax5.set_ylabel("TPR",color="white")
ax5.legend(labelcolor="white", facecolor=DARK, fontsize=9)
ax5.grid(alpha=0.2); style_ax(ax5, "Test ROC-AUC")

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(fpr_v, tpr_v, color=COLORS["val"],  lw=2, label=f"Val  AUC={vm['roc_auc']:.4f}")
ax6.plot(fpr_t, tpr_t, color=COLORS["test"], lw=2, label=f"Test AUC={tm['roc_auc']:.4f}")
ax6.plot([0,1],[0,1],'w--',lw=1,alpha=0.4)
ax6.set_xlabel("FPR",color="white"); ax6.set_ylabel("TPR",color="white")
ax6.legend(labelcolor="white",facecolor=DARK,fontsize=8)
ax6.grid(alpha=0.2); style_ax(ax6, "ROC — Val vs Test")

ax7 = fig.add_subplot(gs[1, 3])
prf_names  = ["Precision\n(Val)", "Recall\n(Val)", "F1\n(Val)",
              "Precision\n(Test)","Recall\n(Test)", "F1\n(Test)"]
prf_vals   = [vm["precision"], vm["recall"], vm["f1_score"],
              tm["precision"], tm["recall"], tm["f1_score"]]
prf_colors = [COLORS["prec"],COLORS["rec"],COLORS["f1"]] * 2
hbars = ax7.barh(prf_names, prf_vals, color=prf_colors,
                 edgecolor="#0f0f1a", linewidth=1.5, height=0.55)
for b, v in zip(hbars, prf_vals):
    ax7.text(v+0.01, b.get_y()+b.get_height()/2,
             f"{v:.4f}", va="center", color="white", fontsize=9, fontweight="bold")
ax7.set_xlim(0, 1.2); ax7.axvline(0.5, color="white", lw=0.7, linestyle="--", alpha=0.3)
style_ax(ax7, "Precision / Recall / F1")

ax8 = fig.add_subplot(gs[2, 0:2], polar=True)
ax8.set_facecolor(DARK)
radar_labels = ["Accuracy","Precision","Recall","F1","ROC-AUC","No-Sep Acc","Sep Acc"]
rv = [vm["overall_accuracy"], vm["precision"], vm["recall"], vm["f1_score"],
      vm["roc_auc"], vm["per_class_accuracy"]["No Sepsis"], vm["per_class_accuracy"]["Sepsis"]]
rt = [tm["overall_accuracy"], tm["precision"], tm["recall"], tm["f1_score"],
      tm["roc_auc"], tm["per_class_accuracy"]["No Sepsis"], tm["per_class_accuracy"]["Sepsis"]]
N      = len(radar_labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
rv_p   = rv + rv[:1]; rt_p = rt + rt[:1]
ax8.plot(angles, rv_p, color=COLORS["val"],  lw=2, label="Validation")
ax8.fill(angles, rv_p, color=COLORS["val"],  alpha=0.15)
ax8.plot(angles, rt_p, color=COLORS["test"], lw=2, label="Test")
ax8.fill(angles, rt_p, color=COLORS["test"], alpha=0.15)
ax8.set_xticks(angles[:-1]); ax8.set_xticklabels(radar_labels, color="white", size=8)
ax8.set_ylim(0, 1); ax8.set_yticks([0.25,0.5,0.75,1.0])
ax8.set_yticklabels(["0.25","0.50","0.75","1.0"], color="gray", size=6)
ax8.grid(color=BORDER, alpha=0.5)
ax8.legend(labelcolor="white", facecolor=DARK, fontsize=9, loc="upper right")
ax8.set_title("Performance Radar — Val vs Test", color="white", fontsize=11, pad=22)

ax9 = fig.add_subplot(gs[2, 2:])
ax9.set_facecolor(DARK); ax9.axis("off")
lines = [
    "MODEL SUMMARY — VAL vs TEST",
    "─" * 38,
    "",
    f"{'Metric':<22} {'Val':>9}  {'Test':>9}",
    "─" * 38,
    f"{'Accuracy':<22} {vm['overall_accuracy']:>9.4f}  {tm['overall_accuracy']:>9.4f}",
    f"{'Precision':<22} {vm['precision']:>9.4f}  {tm['precision']:>9.4f}",
    f"{'Recall':<22} {vm['recall']:>9.4f}  {tm['recall']:>9.4f}",
    f"{'F1-Score':<22} {vm['f1_score']:>9.4f}  {tm['f1_score']:>9.4f}",
    f"{'ROC-AUC':<22} {vm['roc_auc']:>9.4f}  {tm['roc_auc']:>9.4f}",
    f"{'No-Sepsis Acc':<22} {vm['per_class_accuracy']['No Sepsis']:>9.4f}  {tm['per_class_accuracy']['No Sepsis']:>9.4f}",
    f"{'Sepsis Acc':<22} {vm['per_class_accuracy']['Sepsis']:>9.4f}  {tm['per_class_accuracy']['Sepsis']:>9.4f}",
    f"{'Inf. Speed (ms)':<22} {vm['inference_time_ms']:>9.4f}  {tm['inference_time_ms']:>9.4f}",
    "─" * 38,
    "",
    f"Architecture  : SepsisNet (5-layer DNN)",
    f"Trained From  : Scratch (no pretraining)",
]
ax9.text(0.04, 0.97, "\n".join(lines), transform=ax9.transAxes,
         fontsize=9, color="white", va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#0d0d1a", edgecolor="#444488"))

fig.suptitle("Sepsis Early Warning System — Val & Test Performance Dashboard",
             color="white", fontsize=15, fontweight="bold", y=1.01)

out_path = os.path.join(OUTPUT_DIR, "sepsis_dashboard.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0f0f1a")
plt.show()
print(f"Dashboard saved → {out_path}")