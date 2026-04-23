#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  eval_visualize_loco.sh --loco_dir /path/to/loco_eval --train_npz /path/to/train.npz [options]

Required:
  --loco_dir PATH       LOCO result directory containing loco_summary.csv and loco_per_slide.csv
  --train_npz PATH      train.npz used by STOnco; needed for best/worst prediction maps

Options:
  --out_dir PATH        Output directory. Default: ${loco_dir}/eval_visualizations
  --repo_dir PATH       STOnco code directory. Default: /apps/users/sky_luozhihui/STOnco/model/STOnco
  --threshold VALUE     Prediction threshold for spatial maps. Default: 0.5
  --plot_formats LIST   Figure formats for LOCO per-slide plots. Default: svg,png
  --skip_prediction_maps
                        Skip best/worst slide spatial prediction maps
  --help                Show this help

Outputs:
  loco_eval_summary.md
  loco_overall_metrics.csv
  loco_cancer_ranked_metrics.csv
  loco_worst30_slides_by_macro_f1.csv
  loco_best30_slides_by_macro_f1.csv
  loco_selected_best_worst_slide_per_cancer.csv
  bar/scatter summary figures
  per_slide_{accuracy,macro_f1,auroc,auprc}/
  prediction_maps_best_worst/ unless --skip_prediction_maps is set
EOF
}

LOCO_DIR=""
TRAIN_NPZ=""
OUT_DIR=""
REPO_DIR="/apps/users/sky_luozhihui/STOnco/model/STOnco"
THRESHOLD="0.5"
PLOT_FORMATS="svg,png"
SKIP_PREDICTION_MAPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loco_dir)
      LOCO_DIR="$2"
      shift 2
      ;;
    --train_npz)
      TRAIN_NPZ="$2"
      shift 2
      ;;
    --out_dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --repo_dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --plot_formats)
      PLOT_FORMATS="$2"
      shift 2
      ;;
    --skip_prediction_maps)
      SKIP_PREDICTION_MAPS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LOCO_DIR" || -z "$TRAIN_NPZ" ]]; then
  echo "Error: --loco_dir and --train_npz are required." >&2
  usage >&2
  exit 2
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${LOCO_DIR%/}/eval_visualizations"
fi
PRED_DIR="$OUT_DIR/prediction_maps_best_worst"

cd "$REPO_DIR"
source /apps/users/sky_luozhihui/miniconda3/etc/profile.d/conda.sh
conda activate stonco

test -d "$LOCO_DIR"
test -f "$LOCO_DIR/loco_summary.csv"
test -f "$LOCO_DIR/loco_per_slide.csv"
test -f "$LOCO_DIR/loco_per_slide_summary.csv"
test -f "$TRAIN_NPZ"
mkdir -p "$OUT_DIR" "$PRED_DIR"

export LOCO_DIR TRAIN_NPZ OUT_DIR PRED_DIR THRESHOLD PLOT_FORMATS SKIP_PREDICTION_MAPS

echo "[Config] LOCO_DIR=$LOCO_DIR"
echo "[Config] TRAIN_NPZ=$TRAIN_NPZ"
echo "[Config] OUT_DIR=$OUT_DIR"
echo "[Config] PLOT_FORMATS=$PLOT_FORMATS"
echo "[Config] SKIP_PREDICTION_MAPS=$SKIP_PREDICTION_MAPS"

echo "[1/4] Plot LOCO per-slide distributions"
for metric in accuracy macro_f1 auroc auprc; do
  python -m stonco.utils.plot_loco_per_slide \
    --loco_csv "$LOCO_DIR/loco_per_slide.csv" \
    --out_dir "$OUT_DIR/per_slide_${metric}" \
    --box_metric "$metric" \
    --heatmap_metrics auroc,auprc,accuracy,macro_f1 \
    --heatmap_agg median \
    --plot_formats "$PLOT_FORMATS"
done

echo "[2/4] Build summary CSV/Markdown and custom summary figures"
python - <<'PY'
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

loco_dir = Path(os.environ["LOCO_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(loco_dir / "loco_summary.csv")
per_slide = pd.read_csv(loco_dir / "loco_per_slide.csv")
per_slide_summary = pd.read_csv(loco_dir / "loco_per_slide_summary.csv")

metric_cols = ["auroc", "auprc", "accuracy", "macro_f1"]
saved_cols = ["saved_auroc", "saved_auprc", "saved_accuracy", "saved_macro_f1"]

overall = {
    "n_cancers": int(summary["cancer_type"].nunique()),
    "n_val_slides": int(per_slide["slide_id"].nunique()),
}
for col in metric_cols:
    overall[f"cancer_best_mean_{col}"] = float(summary[col].mean())
    overall[f"cancer_best_median_{col}"] = float(summary[col].median())
for col in saved_cols:
    name = col.replace("saved_", "")
    overall[f"cancer_saved_mean_{name}"] = float(summary[col].mean())
    overall[f"cancer_saved_median_{name}"] = float(summary[col].median())
for col in metric_cols:
    overall[f"slide_mean_{col}"] = float(per_slide[col].mean())
    overall[f"slide_median_{col}"] = float(per_slide[col].median())

pd.DataFrame([overall]).to_csv(out_dir / "loco_overall_metrics.csv", index=False)

rank_cols = [
    "cancer_type",
    "best_epoch",
    "last_epoch",
    "saved_epoch",
    "auroc",
    "auprc",
    "accuracy",
    "macro_f1",
    "last_auroc",
    "last_auprc",
    "last_accuracy",
    "last_macro_f1",
    "saved_auroc",
    "saved_auprc",
    "saved_accuracy",
    "saved_macro_f1",
    "n_train",
    "n_val",
]
available_rank_cols = [c for c in rank_cols if c in summary.columns]
summary[available_rank_cols].sort_values("saved_macro_f1", ascending=False).to_csv(
    out_dir / "loco_cancer_ranked_metrics.csv",
    index=False,
)

per_slide.sort_values("macro_f1").head(30).to_csv(out_dir / "loco_worst30_slides_by_macro_f1.csv", index=False)
per_slide.sort_values("macro_f1", ascending=False).head(30).to_csv(out_dir / "loco_best30_slides_by_macro_f1.csv", index=False)

sel_rows = []
for cancer, sub in per_slide.groupby("cancer_type"):
    sub_sorted = sub.sort_values("macro_f1")
    sel_rows.append({**sub_sorted.iloc[0].to_dict(), "selection": "worst_macro_f1"})
    sel_rows.append({**sub_sorted.iloc[-1].to_dict(), "selection": "best_macro_f1"})
selected = pd.DataFrame(sel_rows)
selected.to_csv(out_dir / "loco_selected_best_worst_slide_per_cancer.csv", index=False)

def table_text(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)

with open(out_dir / "loco_eval_summary.md", "w", encoding="utf-8") as f:
    f.write("# LOCO Evaluation\n\n")
    f.write(f"- LOCO directory: `{loco_dir}`\n\n")
    f.write("## Overall\n\n")
    for k, v in overall.items():
        if isinstance(v, float):
            f.write(f"- {k}: {v:.6f}\n")
        else:
            f.write(f"- {k}: {v}\n")
    f.write("\n## Per-cancer saved checkpoint metrics\n\n")
    saved_table_cols = [
        c for c in ["cancer_type", "saved_auroc", "saved_auprc", "saved_accuracy", "saved_macro_f1", "best_epoch", "saved_epoch"]
        if c in summary.columns
    ]
    f.write(table_text(summary[saved_table_cols]))
    f.write("\n\n## Worst slides by macro-F1\n\n")
    worst_cols = [c for c in ["cancer_type", "slide_id", "macro_f1", "accuracy", "auroc", "auprc", "pos_rate", "n_nodes"] if c in per_slide.columns]
    f.write(table_text(per_slide.sort_values("macro_f1").head(15)[worst_cols]))
    f.write("\n")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

plot_df = summary.sort_values("saved_macro_f1", ascending=True)
fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df))))
y = np.arange(len(plot_df))
ax.barh(y, plot_df["saved_macro_f1"], color="#4c78a8")
ax.set_yticks(y)
ax.set_yticklabels(plot_df["cancer_type"])
ax.set_xlabel("Saved macro-F1")
ax.set_title("LOCO saved macro-F1 by held-out cancer")
ax.grid(axis="x", linestyle="--", alpha=0.35)
save(fig, "bar_saved_macro_f1_by_cancer")

fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df))))
width = 0.2
y = np.arange(len(plot_df))
for i, col in enumerate(["saved_auroc", "saved_auprc", "saved_accuracy", "saved_macro_f1"]):
    if col in plot_df.columns:
        ax.barh(y + (i - 1.5) * width, plot_df[col], height=width, label=col.replace("saved_", ""))
ax.set_yticks(y)
ax.set_yticklabels(plot_df["cancer_type"])
ax.set_xlabel("Score")
ax.set_title("LOCO saved metrics by held-out cancer")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", linestyle="--", alpha=0.35)
save(fig, "bar_saved_metrics_by_cancer")

if {"pos_rate", "macro_f1", "n_nodes"}.issubset(per_slide.columns):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        per_slide["pos_rate"],
        per_slide["macro_f1"],
        s=np.clip(np.sqrt(per_slide["n_nodes"]) * 2, 10, 200),
        alpha=0.75,
    )
    ax.set_xlabel("Tumor positive rate")
    ax.set_ylabel("Macro-F1")
    ax.set_title("LOCO per-slide macro-F1 vs positive rate")
    ax.grid(True, linestyle="--", alpha=0.35)
    save(fig, "scatter_macro_f1_vs_pos_rate")

if {"auroc", "auprc", "n_nodes"}.issubset(per_slide.columns):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        per_slide["auroc"],
        per_slide["auprc"],
        s=np.clip(np.sqrt(per_slide["n_nodes"]) * 2, 10, 200),
        alpha=0.75,
    )
    ax.set_xlabel("AUROC")
    ax.set_ylabel("AUPRC")
    ax.set_title("LOCO per-slide AUROC vs AUPRC")
    ax.grid(True, linestyle="--", alpha=0.35)
    save(fig, "scatter_auroc_vs_auprc")

print("Saved summary outputs to", out_dir)
PY

if [[ "$SKIP_PREDICTION_MAPS" -eq 1 ]]; then
  echo "[3/4] Skip prediction maps"
else
  echo "[3/4] Generate prediction maps for best/worst slide per held-out cancer"
  python - <<'PY' > "$OUT_DIR/run_prediction_maps.sh"
from pathlib import Path
import os
import pandas as pd
import shlex

loco_dir = Path(os.environ["LOCO_DIR"])
train_npz = os.environ["TRAIN_NPZ"]
out_dir = Path(os.environ["OUT_DIR"])
pred_dir = Path(os.environ["PRED_DIR"])
threshold = os.environ.get("THRESHOLD", "0.5")
selected = pd.read_csv(out_dir / "loco_selected_best_worst_slide_per_cancer.csv")

print("#!/usr/bin/env bash")
print("set -euo pipefail")
print(f"cd {shlex.quote(str(Path.cwd()))}")
for _, row in selected.iterrows():
    cancer = str(row["cancer_type"])
    slide = str(row["slide_id"])
    selection = str(row["selection"])
    out_svg = pred_dir / cancer / f"{slide}_{selection}.svg"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = loco_dir / cancer
    if not artifacts_dir.exists():
        print(f"echo 'Skip missing artifacts dir: {artifacts_dir}' >&2")
        continue
    cmd = [
        "python", "-m", "stonco.utils.visualize_prediction",
        "--train_npz", train_npz,
        "--artifacts_dir", str(artifacts_dir),
        "--slide_id", slide,
        "--out_svg", str(out_svg),
        "--threshold", str(threshold),
    ]
    print(" ".join(shlex.quote(x) for x in cmd))
PY
  chmod +x "$OUT_DIR/run_prediction_maps.sh"
  bash "$OUT_DIR/run_prediction_maps.sh" > "$OUT_DIR/prediction_maps.log" 2>&1
fi

echo "[4/4] Done"
echo "Main output dir: $OUT_DIR"
echo "Summary: $OUT_DIR/loco_eval_summary.md"
echo "Per-cancer plots: $OUT_DIR/per_slide_accuracy, $OUT_DIR/per_slide_macro_f1, $OUT_DIR/per_slide_auroc, $OUT_DIR/per_slide_auprc"
if [[ "$SKIP_PREDICTION_MAPS" -eq 0 ]]; then
  echo "Prediction maps: $PRED_DIR"
fi
