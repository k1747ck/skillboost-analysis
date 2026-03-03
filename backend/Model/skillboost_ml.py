"""
╔══════════════════════════════════════════════════════════════════════╗
║        SkillBoost Analytics — ML Training + Visualisation          ║
║        Academic & Training Performance Prediction System            ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings, os, sys
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import joblib

# ══════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════
PALETTE = {
    "bg":        "#0F1117",
    "card":      "#1A1D27",
    "accent1":   "#4F8EF7",
    "accent2":   "#A78BFA",
    "accent3":   "#34D399",
    "accent4":   "#F59E0B",
    "accent5":   "#F87171",
    "text":      "#E2E8F0",
    "subtext":   "#94A3B8",
    "grid":      "#2D3748",
    "Excellent": "#34D399",
    "Good":      "#4F8EF7",
    "Average":   "#F59E0B",
    "At-Risk":   "#F87171",
    "Poor":      "#EF4444",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["card"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlecolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUTPUT_DIR = "outputs"          # ← change to your preferred folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  ✅  Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE DATA
# ══════════════════════════════════════════════════════════════════════
print("\n📂  Loading data …")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

students = pd.read_csv(os.path.join(BASE_DIR, "students_clean.csv"))
training = pd.read_csv(os.path.join(BASE_DIR, "training.csv"))
df = pd.merge(students, training, on="Student_ID", how="left")

# Fill missing training fields for students who didn't attend
df["Training_Attendance"]  = df["Training_Attendance"].fillna("no")
df["Training_Score"]       = df["Training_Score"].fillna(0)
df["Feedback_Rating"]      = df["Feedback_Rating"].fillna(0)
df["Pre_Training_Score"]   = df["Pre_Training_Score"].fillna(0)
df["Post_Training_Score"]  = df["Post_Training_Score"].fillna(0)
df["Training_Impact"]      = df["Training_Impact"].fillna("no")
df["Improvement"]          = df["Improvement"].fillna(0)

# Normalise strings
cat_cols = ["Activity_Participation", "Training_Attendance",
            "Training_Impact", "Performance_Label"]
for c in cat_cols:
    df[c] = df[c].astype(str).str.strip().str.lower()

print(f"  Dataset shape after merge: {df.shape}")
print(f"  Performance labels: {df['Performance_Label'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════
# 2. ENCODE & PREPARE
# ══════════════════════════════════════════════════════════════════════
encoders = {}
for c in ["Activity_Participation", "Training_Attendance",
          "Training_Impact", "Performance_Label"]:
    le = LabelEncoder()
    df[c + "_enc"] = le.fit_transform(df[c])
    encoders[c] = le

FEATURES = [
    "Attendance_Percentage", "Mid_Term_Marks", "Assignment_Score",
    "Class_Participation",   "Activity_Participation_enc",
    "Aggregate_Academic_Score",
    "Training_Attendance_enc", "Training_Score", "Feedback_Rating",
    "Pre_Training_Score",     "Post_Training_Score", "Improvement",
]
TARGET = "Performance_Label_enc"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════════════
# 3. TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════
print("\n🤖  Training models …")

models = {
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
for name, mdl in models.items():
    mdl.fit(X_train_sc, y_train)
    y_pred = mdl.predict(X_test_sc)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    cv   = cross_val_score(mdl, scaler.transform(X), y,
                           cv=StratifiedKFold(5), scoring="accuracy").mean()
    results[name] = {"model": mdl, "acc": acc, "f1": f1, "cv": cv, "y_pred": y_pred}
    print(f"  {name:25s}  Acc={acc:.4f}  F1={f1:.4f}  CV={cv:.4f}")

# Pick best model
best_name = max(results, key=lambda k: results[k]["f1"])
best      = results[best_name]
print(f"\n  🏆  Best model → {best_name}  (F1 = {best['f1']:.4f})")

# Save artefacts
joblib.dump(best["model"], os.path.join(OUTPUT_DIR, "student_model.pkl"))
joblib.dump(scaler,        os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(encoders,      os.path.join(OUTPUT_DIR, "encoders.pkl"))
print("  💾  Artefacts saved.")


# ══════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════
label_names = encoders["Performance_Label"].classes_

def label_color(lbl):
    lbl = str(lbl).strip().capitalize()
    return PALETTE.get(lbl, PALETTE["accent1"])

def add_title_bar(fig, title, subtitle=""):
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=20, fontweight="bold", color=PALETTE["text"])
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="top",
                 fontsize=11, color=PALETTE["subtext"])


# ══════════════════════════════════════════════════════════════════════
# CHART 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════
print("\n📊  Generating visualisations …")

fig = plt.figure(figsize=(20, 14), facecolor=PALETTE["bg"])
add_title_bar(fig, "SkillBoost Analytics — Dataset Overview",
              f"Total Students: {df['Student_ID'].nunique()} | Total Records: {len(df)}")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                       top=0.90, bottom=0.06, left=0.06, right=0.97)

order = [l for l in ["excellent","good","average","at-risk","poor"] if l in df["Performance_Label"].unique()]

# 1a — Performance distribution donut
ax = fig.add_subplot(gs[0, 0])
perf_counts = df.groupby("Performance_Label")["Student_ID"].nunique().reindex(order)
colors      = [label_color(l) for l in perf_counts.index]
wedges, texts, autos = ax.pie(
    perf_counts.values, labels=None,
    colors=colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.5, edgecolor=PALETTE["bg"], linewidth=2),
)
for at in autos: at.set_color("white"); at.set_fontsize(9); at.set_fontweight("bold")
patches = [mpatches.Patch(color=c, label=l.capitalize())
           for l, c in zip(perf_counts.index, colors)]
ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, fontsize=8, framealpha=0)
ax.set_title("Performance Distribution", fontsize=13, fontweight="bold", pad=12)

# 1b — Attendance by performance group
ax2 = fig.add_subplot(gs[0, 1])
group_att = df.groupby("Performance_Label")["Attendance_Percentage"].mean().reindex(order)
bars = ax2.bar(
    [l.capitalize() for l in group_att.index],
    group_att.values,
    color=[label_color(l) for l in group_att.index],
    edgecolor=PALETTE["bg"], linewidth=0.8, width=0.55,
)
ax2.set_ylim(0, 110)
ax2.axhline(75, color=PALETTE["subtext"], linestyle="--", linewidth=1, alpha=0.7)
ax2.set_ylabel("Avg Attendance (%)", fontsize=9)
ax2.set_title("Avg Attendance by Performance", fontsize=13, fontweight="bold")
ax2.grid(axis="y", alpha=0.4)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{bar.get_height():.1f}%", ha="center", va="bottom",
             fontsize=8.5, color=PALETTE["text"], fontweight="bold")

# 1c — Mid-term score distribution
ax3 = fig.add_subplot(gs[0, 2])
for lbl in order:
    subset = df[df["Performance_Label"] == lbl]["Mid_Term_Marks"]
    ax3.hist(subset, bins=20, alpha=0.65, color=label_color(lbl),
             label=lbl.capitalize(), edgecolor="none")
ax3.set_xlabel("Mid-Term Marks", fontsize=9)
ax3.set_ylabel("Frequency", fontsize=9)
ax3.set_title("Mid-Term Score Distribution", fontsize=13, fontweight="bold")
ax3.legend(fontsize=8, framealpha=0)
ax3.grid(axis="y", alpha=0.4)

# 1d — Training attendance impact
ax4 = fig.add_subplot(gs[1, 0])
ta = df.groupby(["Training_Attendance", "Performance_Label"]).size().unstack(fill_value=0)
ta = ta.reindex(columns=[c for c in order if c in ta.columns])
x = np.arange(len(ta.index))
width = 0.18
for i, col in enumerate(ta.columns):
    offset = (i - len(ta.columns)/2 + 0.5) * width
    ax4.bar(x + offset, ta[col], width, label=col.capitalize(),
            color=label_color(col), edgecolor=PALETTE["bg"], linewidth=0.5)
ax4.set_xticks(x)
ax4.set_xticklabels([s.capitalize() for s in ta.index], fontsize=9)
ax4.set_ylabel("Count", fontsize=9)
ax4.set_title("Training Attendance vs Performance", fontsize=13, fontweight="bold")
ax4.legend(fontsize=7.5, framealpha=0)
ax4.grid(axis="y", alpha=0.4)

# 1e — Improvement distribution
ax5 = fig.add_subplot(gs[1, 1])
attended = df[df["Training_Attendance"] == "yes"]["Improvement"]
not_att  = df[df["Training_Attendance"] == "no"]["Improvement"]
ax5.hist(attended, bins=30, alpha=0.7, color=PALETTE["accent3"], label="Attended", edgecolor="none")
ax5.hist(not_att,  bins=30, alpha=0.5, color=PALETTE["accent5"], label="Not Attended", edgecolor="none")
ax5.axvline(0, color=PALETTE["text"], linestyle="--", linewidth=1)
ax5.set_xlabel("Improvement (Post − Pre Training)", fontsize=9)
ax5.set_ylabel("Count", fontsize=9)
ax5.set_title("Training Improvement Distribution", fontsize=13, fontweight="bold")
ax5.legend(fontsize=8.5, framealpha=0)
ax5.grid(axis="y", alpha=0.4)

# 1f — Activity participation
ax6 = fig.add_subplot(gs[1, 2])
act = df.groupby(["Activity_Participation", "Performance_Label"]).size().unstack(fill_value=0)
act = act.reindex(columns=[c for c in order if c in act.columns])
x2 = np.arange(len(act.index))
for i, col in enumerate(act.columns):
    offset = (i - len(act.columns)/2 + 0.5) * 0.18
    ax6.bar(x2 + offset, act[col], 0.18, label=col.capitalize(),
            color=label_color(col), edgecolor=PALETTE["bg"], linewidth=0.5)
ax6.set_xticks(x2)
ax6.set_xticklabels([s.capitalize() for s in act.index], fontsize=9)
ax6.set_ylabel("Count", fontsize=9)
ax6.set_title("Activity Participation vs Performance", fontsize=13, fontweight="bold")
ax6.legend(fontsize=7.5, framealpha=0)
ax6.grid(axis="y", alpha=0.4)

save(fig, "01_dataset_overview.png")


# ══════════════════════════════════════════════════════════════════════
# CHART 2 — MODEL PERFORMANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor=PALETTE["bg"])
add_title_bar(fig, "ML Model Performance Comparison",
              "Accuracy · F1 Score · Cross-Validation")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38,
                       top=0.89, bottom=0.06, left=0.06, right=0.97)

ax = fig.add_subplot(gs[0, :2])
model_names = list(results.keys())
accs = [results[n]["acc"] for n in model_names]
f1s  = [results[n]["f1"]  for n in model_names]
cvs  = [results[n]["cv"]  for n in model_names]
x = np.arange(len(model_names)); w = 0.25
bars1 = ax.bar(x - w, accs, w, label="Accuracy",  color=PALETTE["accent1"], edgecolor=PALETTE["bg"])
bars2 = ax.bar(x,     f1s,  w, label="F1 Score",  color=PALETTE["accent2"], edgecolor=PALETTE["bg"])
bars3 = ax.bar(x + w, cvs,  w, label="CV Score",  color=PALETTE["accent3"], edgecolor=PALETTE["bg"])
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Model Comparison — Accuracy / F1 / CV", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, framealpha=0)
ax.grid(axis="y", alpha=0.4)
ax.axhline(1.0, color=PALETTE["grid"], linestyle="--", linewidth=0.8)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8.5, color=PALETTE["text"], fontweight="bold")
best_idx = model_names.index(best_name)
ax.annotate(f"🏆 Best\n{best_name}", xy=(best_idx, best["f1"] + 0.04),
            fontsize=9, color=PALETTE["accent4"], ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["accent4"]),
            xytext=(best_idx + 0.5, best["f1"] + 0.18))

# Classification report heatmap
ax_cr = fig.add_subplot(gs[0, 2])
y_pred = best["y_pred"]
cr = classification_report(y_test, y_pred, target_names=label_names,
                           output_dict=True, zero_division=0)
cr_df = pd.DataFrame(cr).T.iloc[:-3, :3]
sns.heatmap(cr_df, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, linecolor=PALETTE["bg"],
            ax=ax_cr, cbar=False, annot_kws={"size": 10, "weight": "bold"})
ax_cr.set_title(f"Classification Report\n({best_name})", fontsize=12, fontweight="bold")
ax_cr.set_xticklabels(["Precision", "Recall", "F1"], fontsize=9)
ax_cr.set_yticklabels(ax_cr.get_yticklabels(), fontsize=8.5, rotation=0)

# Confusion matrix
ax_cm = fig.add_subplot(gs[1, :2])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, linecolor=PALETTE["bg"],
            ax=ax_cm, cbar=True, annot_kws={"size": 12, "weight": "bold"})
ax_cm.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
ax_cm.set_xlabel("Predicted Label", fontsize=10)
ax_cm.set_ylabel("True Label", fontsize=10)

# Feature importance
if hasattr(best["model"], "feature_importances_"):
    ax_fi = fig.add_subplot(gs[1, 2])
    feat_labels = ["Attendance %", "Mid-Term", "Assignment", "Class Part.",
                   "Activity Part.", "Aggregate Score", "Training Att.",
                   "Training Score", "Feedback Rating", "Pre-Train Score",
                   "Post-Train Score", "Improvement"]
    imp = best["model"].feature_importances_
    sorted_idx = np.argsort(imp)
    colors_fi = [PALETTE["accent1"] if imp[i] >= np.median(imp)
                 else PALETTE["subtext"] for i in sorted_idx]
    ax_fi.barh([feat_labels[i] for i in sorted_idx], imp[sorted_idx],
               color=colors_fi, edgecolor=PALETTE["bg"])
    ax_fi.set_title("Feature Importance", fontsize=13, fontweight="bold")
    ax_fi.set_xlabel("Importance", fontsize=9)
    ax_fi.grid(axis="x", alpha=0.4)

save(fig, "02_model_performance.png")


# ══════════════════════════════════════════════════════════════════════
# CHART 3 — ACADEMIC INSIGHTS
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14), facecolor=PALETTE["bg"])
add_title_bar(fig, "Academic Insights — Score & Engagement Analysis",
              "Mid-Term · Assignment · Attendance · Participation")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38,
                       top=0.90, bottom=0.06, left=0.06, right=0.97)

# Correlation heatmap
ax = fig.add_subplot(gs[0, :2])
num_cols = ["Attendance_Percentage", "Mid_Term_Marks", "Assignment_Score",
            "Class_Participation", "Aggregate_Academic_Score",
            "Training_Score", "Pre_Training_Score", "Post_Training_Score", "Improvement"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, linecolor=PALETTE["bg"], cbar=True,
            annot_kws={"size": 8.5})
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
tick_lbls = ["Attendance", "Mid-Term", "Assignment", "Class Part.", "Aggregate",
             "Train Score", "Pre-Train", "Post-Train", "Improvement"]
ax.set_xticklabels(tick_lbls, rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(tick_lbls, rotation=0, fontsize=8)

# Box-plot: assignment score by label
ax2 = fig.add_subplot(gs[0, 2])
data_box = [df[df["Performance_Label"]==l]["Assignment_Score"].dropna().values for l in order]
bp = ax2.boxplot(data_box, patch_artist=True, notch=True,
                 medianprops=dict(color="white", linewidth=2))
for patch, lbl in zip(bp["boxes"], order):
    patch.set_facecolor(label_color(lbl)); patch.set_alpha(0.8)
ax2.set_xticklabels([l.capitalize() for l in order], fontsize=9)
ax2.set_ylabel("Assignment Score", fontsize=9)
ax2.set_title("Assignment Scores by Performance", fontsize=13, fontweight="bold")
ax2.grid(axis="y", alpha=0.4)

# Scatter: attendance vs aggregate score
ax3 = fig.add_subplot(gs[1, 0])
for lbl in order:
    sub = df[df["Performance_Label"] == lbl]
    ax3.scatter(sub["Attendance_Percentage"], sub["Aggregate_Academic_Score"],
                alpha=0.35, s=18, color=label_color(lbl), label=lbl.capitalize())
ax3.set_xlabel("Attendance (%)", fontsize=9)
ax3.set_ylabel("Aggregate Academic Score", fontsize=9)
ax3.set_title("Attendance vs Aggregate Score", fontsize=13, fontweight="bold")
ax3.legend(fontsize=8, framealpha=0, markerscale=1.5)
ax3.grid(alpha=0.3)

# Avg scores per performance label
ax4 = fig.add_subplot(gs[1, 1])
metrics = ["Attendance_Percentage", "Mid_Term_Marks", "Assignment_Score", "Aggregate_Academic_Score"]
m_labels = ["Attendance", "Mid-Term", "Assignment", "Aggregate"]
grp = df.groupby("Performance_Label")[metrics].mean().reindex(order)
x = np.arange(len(m_labels)); w = 0.16
for i, lbl in enumerate(grp.index):
    offset = (i - len(grp)/2 + 0.5) * w
    ax4.bar(x + offset, grp.loc[lbl], w, label=lbl.capitalize(),
            color=label_color(lbl), edgecolor=PALETTE["bg"], linewidth=0.5)
ax4.set_xticks(x); ax4.set_xticklabels(m_labels, fontsize=9)
ax4.set_ylabel("Average Score", fontsize=9)
ax4.set_title("Average Academic Metrics by Label", fontsize=13, fontweight="bold")
ax4.legend(fontsize=7.5, framealpha=0); ax4.grid(axis="y", alpha=0.4)

# Class participation violin
ax5 = fig.add_subplot(gs[1, 2])
parts_data = [df[df["Performance_Label"]==l]["Class_Participation"].dropna().values for l in order]
vp = ax5.violinplot(parts_data, positions=np.arange(len(order)),
                    showmedians=True, showmeans=False)
for pc, lbl in zip(vp["bodies"], order):
    pc.set_facecolor(label_color(lbl)); pc.set_alpha(0.7)
vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
vp["cbars"].set_color(PALETTE["subtext"]); vp["cmins"].set_color(PALETTE["subtext"])
vp["cmaxes"].set_color(PALETTE["subtext"])
ax5.set_xticks(np.arange(len(order)))
ax5.set_xticklabels([l.capitalize() for l in order], fontsize=9)
ax5.set_ylabel("Class Participation (0–10)", fontsize=9)
ax5.set_title("Participation Spread by Performance", fontsize=13, fontweight="bold")
ax5.grid(axis="y", alpha=0.4)

save(fig, "03_academic_insights.png")


# ══════════════════════════════════════════════════════════════════════
# CHART 4 — TRAINING IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor=PALETTE["bg"])
add_title_bar(fig, "Training Impact Analysis",
              "Pre vs Post Scores · Improvement · Course Performance")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                       top=0.89, bottom=0.06, left=0.06, right=0.97)

# Pre vs Post training by performance group
ax = fig.add_subplot(gs[0, :2])
trained    = df[df["Training_Attendance"] == "yes"].groupby("Performance_Label")
pre_means  = trained["Pre_Training_Score"].mean().reindex(order)
post_means = trained["Post_Training_Score"].mean().reindex(order)
x = np.arange(len(order)); w = 0.30
b1 = ax.bar(x - w/2, pre_means.values,  w, label="Pre-Training",  color=PALETTE["subtext"], edgecolor=PALETTE["bg"])
b2 = ax.bar(x + w/2, post_means.values, w, label="Post-Training", color=PALETTE["accent3"], edgecolor=PALETTE["bg"])
for xi, (pre, post) in enumerate(zip(pre_means, post_means)):
    diff = post - pre
    col  = PALETTE["accent3"] if diff >= 0 else PALETTE["accent5"]
    ax.annotate(f"{diff:+.1f}", xy=(xi + w/2, post + 0.8),
                ha="center", fontsize=9, color=col, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([l.capitalize() for l in order], fontsize=10)
ax.set_ylabel("Average Score", fontsize=9)
ax.set_title("Pre vs Post Training Scores by Performance Label", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, framealpha=0); ax.grid(axis="y", alpha=0.4)

# Feedback rating donut
ax_fb = fig.add_subplot(gs[0, 2])
fb_counts = df[df["Training_Attendance"]=="yes"]["Feedback_Rating"].value_counts().sort_index()
fb_colors = [PALETTE["accent5"], PALETTE["accent4"], PALETTE["accent1"],
             PALETTE["accent2"], PALETTE["accent3"]]
wedges, texts, autos = ax_fb.pie(
    fb_counts.values, labels=[f"⭐ {int(k)}" for k in fb_counts.index],
    colors=fb_colors[:len(fb_counts)], autopct="%1.1f%%", startangle=90,
    pctdistance=0.78, wedgeprops=dict(width=0.5, edgecolor=PALETTE["bg"], linewidth=2),
)
for at in autos: at.set_fontsize(8); at.set_fontweight("bold"); at.set_color("white")
ax_fb.set_title("Feedback Rating Distribution\n(Training Attendees)", fontsize=12, fontweight="bold")

# Improvement violin by label
ax3 = fig.add_subplot(gs[1, 0])
order_valid  = [l for l in order if len(df[(df["Training_Attendance"]=="yes") & (df["Performance_Label"]==l)]) > 0]
imp_data     = [df[(df["Training_Attendance"]=="yes") & (df["Performance_Label"]==l)]["Improvement"].dropna().values
                for l in order_valid]
vp = ax3.violinplot(imp_data, positions=np.arange(len(order_valid)), showmedians=True)
for pc, lbl in zip(vp["bodies"], order_valid):
    pc.set_facecolor(label_color(lbl)); pc.set_alpha(0.7)
vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
vp["cbars"].set_color(PALETTE["subtext"]); vp["cmins"].set_color(PALETTE["subtext"])
vp["cmaxes"].set_color(PALETTE["subtext"])
ax3.axhline(0, color=PALETTE["text"], linestyle="--", linewidth=1, alpha=0.6)
ax3.set_xticks(np.arange(len(order_valid)))
ax3.set_xticklabels([l.capitalize() for l in order_valid], fontsize=9)
ax3.set_ylabel("Improvement (pts)", fontsize=9)
ax3.set_title("Training Improvement by Performance", fontsize=13, fontweight="bold")
ax3.grid(axis="y", alpha=0.4)

# Course-wise avg improvement
if "Course_Name" in df.columns:
    ax4 = fig.add_subplot(gs[1, 1:])
    course_imp = df[df["Training_Attendance"]=="yes"].groupby("Course_Name")["Improvement"].mean().sort_values(ascending=True)
    colors_c   = [PALETTE["accent3"] if v >= 0 else PALETTE["accent5"] for v in course_imp.values]
    bars_c = ax4.barh(course_imp.index, course_imp.values, color=colors_c, edgecolor=PALETTE["bg"])
    ax4.axvline(0, color=PALETTE["text"], linewidth=1, linestyle="--")
    ax4.set_xlabel("Average Improvement (pts)", fontsize=9)
    ax4.set_title("Average Improvement by Course", fontsize=13, fontweight="bold")
    ax4.grid(axis="x", alpha=0.4)
    for bar in bars_c:
        v = bar.get_width()
        ax4.text(v + (0.3 if v >= 0 else -0.3), bar.get_y() + bar.get_height()/2,
                 f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                 fontsize=8.5, color=PALETTE["text"])

save(fig, "04_training_impact.png")


# ══════════════════════════════════════════════════════════════════════
# CHART 5 — SAMPLE STUDENT PREDICTION REPORT
# ══════════════════════════════════════════════════════════════════════
sample = df.iloc[0]
sample_raw = {
    "Attendance_Percentage":    float(sample["Attendance_Percentage"]),
    "Mid_Term_Marks":           float(sample["Mid_Term_Marks"]),
    "Assignment_Score":         float(sample["Assignment_Score"]),
    "Class_Participation":      float(sample["Class_Participation"]),
    "Activity_Participation":   str(sample["Activity_Participation"]),
    "Aggregate_Academic_Score": float(sample["Aggregate_Academic_Score"]),
    "Training_Attendance":      str(sample["Training_Attendance"]),
    "Training_Score":           float(sample["Training_Score"]),
    "Feedback_Rating":          float(sample["Feedback_Rating"]),
    "Pre_Training_Score":       float(sample["Pre_Training_Score"]),
    "Post_Training_Score":      float(sample["Post_Training_Score"]),
    "Improvement":              float(sample["Improvement"]),
}

row = {}
for col in FEATURES:
    raw_col = col.replace("_enc", "")
    if col.endswith("_enc"):
        val = str(sample_raw.get(raw_col, "no")).strip().lower()
        le  = encoders[raw_col]
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    else:
        row[col] = float(sample_raw[col])

X_s   = pd.DataFrame([row], columns=FEATURES)
X_ssc = scaler.transform(X_s)
pred_n  = best["model"].predict(X_ssc)[0]
pred_lb = encoders["Performance_Label"].inverse_transform([pred_n])[0]
proba   = best["model"].predict_proba(X_ssc)[0]
proba_d = {encoders["Performance_Label"].classes_[i]: round(float(p), 4)
           for i, p in enumerate(proba)}

def build_remarks(data, pred_lb):
    remarks = []
    att  = float(data.get("Attendance_Percentage", 0))
    mid  = float(data.get("Mid_Term_Marks", 0))
    asgn = float(data.get("Assignment_Score", 0))
    part = float(data.get("Class_Participation", 0))
    imp  = float(data.get("Improvement", 0))
    ta   = str(data.get("Training_Attendance", "no")).lower()

    if att >= 90:
        remarks.append(("✅", "Excellent attendance", "Consistent presence reflects strong commitment to academics.", PALETTE["accent3"]))
    elif att >= 75:
        remarks.append(("⚠️", "Acceptable attendance", f"At {att:.0f}%, this meets the minimum but a push to 90%+ would significantly boost outcomes.", PALETTE["accent4"]))
    else:
        remarks.append(("❌", "Critical attendance issue", f"Only {att:.0f}% attendance. Missing classes directly impacts scores — needs immediate attention.", PALETTE["accent5"]))

    if mid >= 80:
        remarks.append(("✅", "Strong mid-term performance", f"Scored {mid:.0f}/100 — demonstrating solid understanding of course material.", PALETTE["accent3"]))
    elif mid >= 60:
        remarks.append(("⚠️", "Average mid-term score", f"A score of {mid:.0f}/100 is passing but leaves room to grow. Focus on revision.", PALETTE["accent4"]))
    else:
        remarks.append(("❌", "Weak mid-term score", f"Score of {mid:.0f}/100 signals knowledge gaps. Immediate tutoring is strongly advised.", PALETTE["accent5"]))

    if asgn >= 80:
        remarks.append(("✅", "Consistent assignments", f"Assignment score of {asgn:.0f}/100 shows discipline and regular effort — keep it up!", PALETTE["accent3"]))
    elif asgn >= 60:
        remarks.append(("⚠️", "Moderate assignment score", f"Score is {asgn:.0f}/100 — review instructor feedback and dedicate more time.", PALETTE["accent4"]))
    else:
        remarks.append(("❌", "Poor assignment performance", f"Score of {asgn:.0f}/100. Submissions must improve to build GPA.", PALETTE["accent5"]))

    if part >= 8:
        remarks.append(("✅", "Highly engaged in class", f"Participation score of {part:.0f}/10 — active and contributing learner.", PALETTE["accent3"]))
    elif part >= 5:
        remarks.append(("⚠️", "Moderate participation", f"Rated {part:.0f}/10. Ask or answer at least one question per class.", PALETTE["accent4"]))
    else:
        remarks.append(("❌", "Low class engagement", f"Only {part:.0f}/10. Passive learning leads to weaker retention.", PALETTE["accent5"]))

    if ta == "yes":
        if imp > 10:
            remarks.append(("📈", "Exceptional training impact", f"Training drove a {imp:+.1f} point improvement — skill development is clearly paying off.", PALETTE["accent1"]))
        elif imp > 0:
            remarks.append(("📈", "Positive training impact", f"A {imp:+.1f} point improvement is a good start. More practice will compound these gains.", PALETTE["accent2"]))
        else:
            remarks.append(("📉", "Training had negative impact", f"Score dropped {abs(imp):.1f} pts post-training. Consider a better-matched training program.", PALETTE["accent5"]))
    else:
        remarks.append(("⚪", "No training attended", "Enrolling in a relevant training program could significantly boost skills and performance.", PALETTE["subtext"]))

    suggestions = {
        "excellent": "🎯  Top tier! Consider mentoring peers, joining competitions, or pursuing advanced certifications.",
        "good":      "🎯  Solid performance! Focus on 1–2 weak areas and maintain training momentum.",
        "average":   "🎯  Create a structured weekly study plan, attend all trainings, and seek help on weak subjects.",
        "at-risk":   "🎯  Urgent action needed. Meet your academic advisor, increase attendance, enrol in training now.",
        "poor":      "🎯  Critical intervention required. Speak with faculty and counsellor today.",
    }
    action = suggestions.get(pred_lb.lower(), "🎯  Review all metrics and create a structured improvement plan.")
    return remarks, action

remarks, action_plan = build_remarks(sample_raw, pred_lb)

fig = plt.figure(figsize=(20, 18), facecolor=PALETTE["bg"])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                        top=0.93, bottom=0.04, left=0.06, right=0.97)

# Banner
ax_b = fig.add_subplot(gs[0, :])
ax_b.set_facecolor(label_color(pred_lb)); ax_b.axis("off")
ax_b.text(0.5, 0.70, f"Student Performance Report  —  {sample.get('Student_ID','S0001')}",
          ha="center", va="center", fontsize=21, fontweight="bold",
          color="white", transform=ax_b.transAxes)
ax_b.text(0.5, 0.38, f"Predicted Label:   {pred_lb.upper()}",
          ha="center", va="center", fontsize=30, fontweight="bold",
          color="white", transform=ax_b.transAxes)
chip_x = np.linspace(0.12, 0.88, len(proba_d))
for (lbl, p), cx in zip(sorted(proba_d.items(), key=lambda x:-x[1]), chip_x):
    ax_b.text(cx, 0.10, f"{lbl.upper()}  {p*100:.1f}%",
              ha="center", va="center", fontsize=11, color="white", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                        alpha=0.22, edgecolor="white", linewidth=0.8),
              transform=ax_b.transAxes)

# Academic metrics bar
ax_ac = fig.add_subplot(gs[1, 0])
ac_labels = ["Attendance", "Mid-Term", "Assignment", "Participation\n(×10)", "Aggregate", "Train Score"]
ac_values = [
    sample_raw["Attendance_Percentage"], sample_raw["Mid_Term_Marks"],
    sample_raw["Assignment_Score"],      sample_raw["Class_Participation"] * 10,
    sample_raw["Aggregate_Academic_Score"], sample_raw["Training_Score"],
]
bar_cols = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
            PALETTE["accent4"], PALETTE["accent5"], PALETTE["accent3"]]
bars = ax_ac.barh(ac_labels, ac_values, color=bar_cols, edgecolor=PALETTE["bg"], height=0.55)
ax_ac.set_xlim(0, 120)
ax_ac.axvline(60, color=PALETTE["grid"], linestyle="--", linewidth=1)
ax_ac.axvline(80, color=PALETTE["grid"], linestyle=":",  linewidth=1)
ax_ac.set_title("Academic Metrics (out of 100)", fontsize=12, fontweight="bold")
ax_ac.grid(axis="x", alpha=0.3)
for bar in bars:
    ax_ac.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f"{bar.get_width():.0f}", va="center", fontsize=9.5,
               color=PALETTE["text"], fontweight="bold")

# Radar
ax_r = fig.add_subplot(gs[1, 1], projection="polar")
ax_r.set_facecolor(PALETTE["card"])
N = len(ac_labels)
angles  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
vals_lp = ac_values + [ac_values[0]]; angs_lp = angles + [angles[0]]
accent  = label_color(pred_lb)
ax_r.plot(angs_lp, vals_lp, color=accent, linewidth=2.5)
ax_r.fill(angs_lp, vals_lp, color=accent, alpha=0.25)
for ring in [25, 50, 75, 100]:
    ax_r.plot(angs_lp, [ring]*(N+1), color=PALETTE["grid"], linewidth=0.6, linestyle="--")
ax_r.set_xticks(angles)
ax_r.set_xticklabels(["Att.", "Mid-T", "Asgn", "Part.", "Aggr", "Train"], fontsize=9)
ax_r.set_yticks([25, 50, 75, 100])
ax_r.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color=PALETTE["subtext"])
ax_r.set_ylim(0, 115)
ax_r.set_title("Performance Radar", fontsize=12, fontweight="bold", pad=20)
ax_r.spines["polar"].set_visible(False)

# Training panel
ax_t = fig.add_subplot(gs[1, 2])
ax_t.set_facecolor(PALETTE["card"]); ax_t.axis("off")
ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)
ax_t.set_title("Training Impact", fontsize=12, fontweight="bold")
pre = sample_raw["Pre_Training_Score"]; post = sample_raw["Post_Training_Score"]
imp = sample_raw["Improvement"]
acolor = PALETTE["accent3"] if imp >= 0 else PALETTE["accent5"]
sym    = "▲" if imp >= 0 else "▼"
ax_t.text(0.5, 0.87, "Pre-Training",  ha="center", fontsize=10, color=PALETTE["subtext"])
ax_t.text(0.5, 0.76, f"{pre:.0f}",    ha="center", fontsize=36, fontweight="bold", color=PALETTE["subtext"])
ax_t.text(0.5, 0.60, f"{sym}  {imp:+.1f} pts", ha="center", fontsize=22, fontweight="bold", color=acolor)
ax_t.text(0.5, 0.47, "Post-Training", ha="center", fontsize=10, color=PALETTE["text"])
ax_t.text(0.5, 0.36, f"{post:.0f}",   ha="center", fontsize=36, fontweight="bold", color=acolor)
ta_str = sample_raw["Training_Attendance"].lower()
tb_bg  = "#1A3A2A" if ta_str == "yes" else "#3A1A1A"
tb_txt = "✅  Attended Training" if ta_str == "yes" else "❌  Did Not Attend"
tb_col = PALETTE["accent3"]    if ta_str == "yes" else PALETTE["accent5"]
ax_t.text(0.5, 0.18, tb_txt, ha="center", fontsize=10, color=tb_col, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.45", facecolor=tb_bg, edgecolor="none"),
          transform=ax_t.transAxes)
stars = int(round(sample_raw.get("Feedback_Rating", 0)))
ax_t.text(0.5, 0.06, "★"*stars + "☆"*(5-stars) + f"  {sample_raw.get('Feedback_Rating',0):.1f}/5",
          ha="center", fontsize=12, color=PALETTE["accent4"], transform=ax_t.transAxes)

# Probability donut
ax_d = fig.add_subplot(gs[2, 0])
ax_d.set_facecolor(PALETTE["card"])
d_labels = [l.capitalize() for l in proba_d.keys()]
d_vals   = list(proba_d.values())
d_colors = [label_color(l) for l in proba_d.keys()]
wedges2, _, autos2 = ax_d.pie(
    d_vals, labels=d_labels, colors=d_colors,
    autopct="%1.1f%%", startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.52, edgecolor=PALETTE["bg"], linewidth=2),
)
for at2 in autos2: at2.set_fontsize(8); at2.set_fontweight("bold"); at2.set_color("white")
ax_d.text(0, 0, pred_lb.capitalize(), ha="center", va="center",
          fontsize=12, fontweight="bold", color=label_color(pred_lb))
ax_d.set_title("Class Probabilities", fontsize=12, fontweight="bold")

# Remarks panel
ax_rm = fig.add_subplot(gs[2, 1:])
ax_rm.set_facecolor(PALETTE["card"]); ax_rm.axis("off")
ax_rm.set_title("💬  Personalised Feedback & Recommendations", fontsize=12, fontweight="bold")
ax_rm.set_xlim(0, 1); ax_rm.set_ylim(0, 1)
n_r   = len(remarks) + 1
step  = 0.92 / (n_r + 0.8)
y_top = 0.95
for i, (icon, heading, detail, col) in enumerate(remarks):
    y = y_top - i * step
    ax_rm.text(0.01, y, f"{icon}  {heading}",
               va="top", ha="left", fontsize=10.5, fontweight="bold",
               color=col, transform=ax_rm.transAxes)
    ax_rm.text(0.01, y - step*0.40, detail,
               va="top", ha="left", fontsize=9, color=PALETTE["subtext"],
               transform=ax_rm.transAxes, style="italic")
y_action = y_top - len(remarks) * step - 0.01
ax_rm.text(0.01, y_action, action_plan,
           va="top", ha="left", fontsize=10, fontweight="bold",
           color=PALETTE["accent2"], transform=ax_rm.transAxes,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["accent2"]+"22",
                     edgecolor=PALETTE["accent2"], linewidth=1))

fig.suptitle("SkillBoost Analytics — Prediction Dashboard",
             fontsize=18, fontweight="bold", color=PALETTE["text"], y=0.975)

save(fig, "05_student_prediction_report.png")


# ══════════════════════════════════════════════════════════════════════
# CHART 6 — STUDENT COHORT SUMMARY
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 10), facecolor=PALETTE["bg"])
add_title_bar(fig, "Student Cohort Summary — At-Risk & Top Performers",
              "Identify students needing intervention and those excelling")

gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.38,
                       top=0.88, bottom=0.10, left=0.06, right=0.97)

agg = df.groupby("Student_ID").agg(
    Aggregate=("Aggregate_Academic_Score", "mean"),
    Label=("Performance_Label", "first")
).reset_index().sort_values("Aggregate")

bottom10 = agg.head(10); top10 = agg.tail(10)

ax_bot = fig.add_subplot(gs[0, 0])
ax_bot.barh(bottom10["Student_ID"], bottom10["Aggregate"],
            color=PALETTE["accent5"], edgecolor=PALETTE["bg"])
ax_bot.set_title("⚠️  Bottom 10 Students\n(by Aggregate Score)", fontsize=12, fontweight="bold")
ax_bot.set_xlabel("Aggregate Score", fontsize=9); ax_bot.grid(axis="x", alpha=0.4)

ax_top = fig.add_subplot(gs[0, 1])
ax_top.barh(top10["Student_ID"], top10["Aggregate"],
            color=PALETTE["accent3"], edgecolor=PALETTE["bg"])
ax_top.set_title("🌟  Top 10 Students\n(by Aggregate Score)", fontsize=12, fontweight="bold")
ax_top.set_xlabel("Aggregate Score", fontsize=9); ax_top.grid(axis="x", alpha=0.4)

ax_s = fig.add_subplot(gs[0, 2])
train_perf = df.groupby(["Training_Attendance", "Performance_Label"]).size().unstack(fill_value=0)
train_perf = train_perf.reindex(columns=[c for c in order if c in train_perf.columns])
bottom_vals = np.zeros(len(train_perf))
for col in train_perf.columns:
    ax_s.bar(train_perf.index, train_perf[col], bottom=bottom_vals,
             label=col.capitalize(), color=label_color(col), edgecolor=PALETTE["bg"])
    bottom_vals += train_perf[col].values
ax_s.set_title("Performance Split\nby Training Attendance", fontsize=12, fontweight="bold")
ax_s.set_ylabel("Student Count", fontsize=9)
ax_s.legend(fontsize=8, framealpha=0, loc="upper right")
ax_s.set_xticklabels([s.capitalize() for s in train_perf.index], fontsize=10)
ax_s.grid(axis="y", alpha=0.4)

save(fig, "06_cohort_summary.png")

print("\n\n🎉  All done! 6 charts + 3 model files saved to:", OUTPUT_DIR)
