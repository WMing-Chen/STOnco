#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  plot_loco_umap_tsne.sh --loco_dir /path/to/loco_eval --train_npz /path/to/train.npz [options]

Required:
  --loco_dir PATH          LOCO result directory containing one subdirectory per held-out cancer
  --train_npz PATH         train.npz used by STOnco

Options:
  --out_dir PATH           Output directory. Default: ${loco_dir}/loco_umap_tsne
  --repo_dir PATH          STOnco code directory. Default: /apps/users/sky_luozhihui/STOnco/model/STOnco
  --subset all|train|val   Dataset subset passed to embedding export. Default: all
  --embed_source h|z_clf|z64
                           Embedding source. Default: h
  --device cpu|cuda        Device for embedding export. Default: cuda
  --max_points N           Shared row cap for UMAP/t-SNE/LISI. Default: 50000
  --seed N                 Random seed. Default: 42
  --num_threads N          Optional torch CPU thread count for export
  --metric_spaces LIST     Comma-separated LISI spaces. Default: embedding,umap,tsne
  --k_values LIST          Comma-separated LISI k values. Default: 15,30,50
  --color_cols LIST        Comma-separated plot color columns. Default: sample_id,tumor_label,batch_id,cancer_type
  --group_cols LIST        Comma-separated LISI metadata columns. Default: sample_id,batch_id,tumor_label,cancer_type
  --group_roles LIST       Comma-separated role mappings. Default: sample_id:integration,batch_id:integration,tumor_label:conservation,cancer_type:integration
  --highlight_col COL      Optional column to highlight as triangles, e.g. cancer_type, sample_id, or batch_id
  --highlight_values LIST  Optional comma-separated values in --highlight_col to highlight
  --no_highlight_loco      Disable the LOCO default: highlight current held-out cancer_type as triangles
  --cancers LIST           Optional comma-separated cancer fold names to run. Default: all fold dirs
  --include_checkpoints    Also process each fold's epoch_checkpoints/epoch_*
  --checkpoint_epochs LIST Optional comma-separated epochs to process when --include_checkpoints is set. Default: all
  --skip_final             Do not process final fold model.pt; useful with --include_checkpoints
  --force                  Re-run even when mixing_metrics_${embed_source}.csv already exists
  --help                   Show this help

Outputs:
  ${out_dir}/${cancer}/final_${subset}_${embed_source}/
  ${out_dir}/${cancer}/checkpoints/epoch_${N}_${subset}_${embed_source}/ when --include_checkpoints is set
  ${out_dir}/run_manifest.tsv
EOF
}

LOCO_DIR=""
TRAIN_NPZ=""
OUT_DIR=""
REPO_DIR="/apps/users/sky_luozhihui/STOnco/model/STOnco"
SUBSET="all"
EMBED_SOURCE="h"
DEVICE="cuda"
MAX_POINTS="50000"
SEED="42"
NUM_THREADS=""
METRIC_SPACES="embedding,umap,tsne"
K_VALUES="15,30,50"
COLOR_COLS="sample_id,tumor_label,batch_id,cancer_type"
GROUP_COLS="sample_id,batch_id,tumor_label,cancer_type"
GROUP_ROLES="sample_id:integration,batch_id:integration,tumor_label:conservation,cancer_type:integration"
HIGHLIGHT_COL=""
HIGHLIGHT_VALUES=""
HIGHLIGHT_LOCO=1
CANCERS=""
INCLUDE_CHECKPOINTS=0
CHECKPOINT_EPOCHS=""
SKIP_FINAL=0
SKIP_EXISTING=1

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
    --subset)
      SUBSET="$2"
      shift 2
      ;;
    --embed_source)
      EMBED_SOURCE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --max_points)
      MAX_POINTS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --num_threads)
      NUM_THREADS="$2"
      shift 2
      ;;
    --metric_spaces)
      METRIC_SPACES="$2"
      shift 2
      ;;
    --k_values)
      K_VALUES="$2"
      shift 2
      ;;
    --color_cols)
      COLOR_COLS="$2"
      shift 2
      ;;
    --group_cols)
      GROUP_COLS="$2"
      shift 2
      ;;
    --group_roles)
      GROUP_ROLES="$2"
      shift 2
      ;;
    --highlight_col)
      HIGHLIGHT_COL="$2"
      shift 2
      ;;
    --highlight_values)
      HIGHLIGHT_VALUES="$2"
      shift 2
      ;;
    --no_highlight_loco)
      HIGHLIGHT_LOCO=0
      shift
      ;;
    --cancers)
      CANCERS="$2"
      shift 2
      ;;
    --include_checkpoints)
      INCLUDE_CHECKPOINTS=1
      shift
      ;;
    --checkpoint_epochs)
      CHECKPOINT_EPOCHS="$2"
      shift 2
      ;;
    --skip_final)
      SKIP_FINAL=1
      shift
      ;;
    --force)
      SKIP_EXISTING=0
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

case "$SUBSET" in
  all|train|val) ;;
  *)
    echo "Error: --subset must be one of all, train, val; got $SUBSET" >&2
    exit 2
    ;;
esac

case "$EMBED_SOURCE" in
  h|z_clf|z64) ;;
  *)
    echo "Error: --embed_source must be one of h, z_clf, z64; got $EMBED_SOURCE" >&2
    exit 2
    ;;
esac

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${LOCO_DIR%/}/loco_umap_tsne"
fi

test -d "$LOCO_DIR"
test -f "$TRAIN_NPZ"
test -d "$REPO_DIR"
mkdir -p "$OUT_DIR"

cd "$REPO_DIR"
source /apps/users/sky_luozhihui/miniconda3/etc/profile.d/conda.sh
conda activate stonco

comma_to_array() {
  local value="$1"
  local target="$2"
  local old_ifs="$IFS"
  local parsed=()
  local cleaned=()
  local item
  set_array_target() {
    case "$target" in
      METRIC_SPACE_ARGS)
        METRIC_SPACE_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          METRIC_SPACE_ARGS=("${cleaned[@]}")
        fi
        ;;
      K_VALUE_ARGS)
        K_VALUE_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          K_VALUE_ARGS=("${cleaned[@]}")
        fi
        ;;
      COLOR_COL_ARGS)
        COLOR_COL_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          COLOR_COL_ARGS=("${cleaned[@]}")
        fi
        ;;
      GROUP_COL_ARGS)
        GROUP_COL_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          GROUP_COL_ARGS=("${cleaned[@]}")
        fi
        ;;
      GROUP_ROLE_ARGS)
        GROUP_ROLE_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          GROUP_ROLE_ARGS=("${cleaned[@]}")
        fi
        ;;
      HIGHLIGHT_VALUE_ARGS)
        HIGHLIGHT_VALUE_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          HIGHLIGHT_VALUE_ARGS=("${cleaned[@]}")
        fi
        ;;
      CANCER_FILTER_ARGS)
        CANCER_FILTER_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          CANCER_FILTER_ARGS=("${cleaned[@]}")
        fi
        ;;
      CHECKPOINT_EPOCH_ARGS)
        CHECKPOINT_EPOCH_ARGS=()
        if (( ${#cleaned[@]} > 0 )); then
          CHECKPOINT_EPOCH_ARGS=("${cleaned[@]}")
        fi
        ;;
      *)
        echo "Internal error: unknown array target $target" >&2
        exit 2
        ;;
    esac
  }
  if [[ -z "$value" ]]; then
    set_array_target
    return 0
  fi
  IFS=','
  read -r -a parsed <<< "$value"
  IFS="$old_ifs"
  if (( ${#parsed[@]} > 0 )); then
    for item in "${parsed[@]}"; do
      item="${item#"${item%%[![:space:]]*}"}"
      item="${item%"${item##*[![:space:]]}"}"
      [[ -n "$item" ]] && cleaned+=("$item")
    done
  fi
  set_array_target
}

comma_to_array "$METRIC_SPACES" METRIC_SPACE_ARGS
comma_to_array "$K_VALUES" K_VALUE_ARGS
comma_to_array "$COLOR_COLS" COLOR_COL_ARGS
comma_to_array "$GROUP_COLS" GROUP_COL_ARGS
comma_to_array "$GROUP_ROLES" GROUP_ROLE_ARGS
comma_to_array "$HIGHLIGHT_VALUES" HIGHLIGHT_VALUE_ARGS
comma_to_array "$CANCERS" CANCER_FILTER_ARGS
comma_to_array "$CHECKPOINT_EPOCHS" CHECKPOINT_EPOCH_ARGS

in_list_or_empty() {
  local needle="$1"
  shift
  if [[ $# -eq 0 || -z "${1:-}" ]]; then
    return 0
  fi
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

epoch_selected() {
  local epoch_dir="$1"
  local epoch_num
  epoch_num="$(basename "$epoch_dir" | sed -E 's/^epoch_0*//')"
  if (( ${#CHECKPOINT_EPOCH_ARGS[@]} > 0 )); then
    in_list_or_empty "$epoch_num" "${CHECKPOINT_EPOCH_ARGS[@]}"
  else
    in_list_or_empty "$epoch_num"
  fi
}

run_manifest="$OUT_DIR/run_manifest.tsv"
printf "cancer\tstage\tartifacts_dir\tout_dir\tstatus\n" > "$run_manifest"

echo "[Config] LOCO_DIR=$LOCO_DIR"
echo "[Config] TRAIN_NPZ=$TRAIN_NPZ"
echo "[Config] OUT_DIR=$OUT_DIR"
echo "[Config] SUBSET=$SUBSET"
echo "[Config] EMBED_SOURCE=$EMBED_SOURCE"
echo "[Config] DEVICE=$DEVICE"
echo "[Config] INCLUDE_CHECKPOINTS=$INCLUDE_CHECKPOINTS"
echo "[Config] CHECKPOINT_EPOCHS=${CHECKPOINT_EPOCHS:-all}"
if [[ -n "$HIGHLIGHT_COL" || -n "$HIGHLIGHT_VALUES" ]]; then
  echo "[Config] HIGHLIGHT=${HIGHLIGHT_COL:-unset}:${HIGHLIGHT_VALUES:-unset}"
elif [[ "$HIGHLIGHT_LOCO" -eq 1 ]]; then
  echo "[Config] HIGHLIGHT=LOCO held-out cancer_type per fold"
else
  echo "[Config] HIGHLIGHT=disabled"
fi

run_embedding_pipeline() {
  local cancer="$1"
  local stage="$2"
  local artifacts_dir="$3"
  local run_out="$4"

  if [[ ! -f "$artifacts_dir/meta.json" || ! -f "$artifacts_dir/model.pt" ]]; then
    echo "[Skip] Missing meta.json or model.pt: $artifacts_dir"
    printf "%s\t%s\t%s\t%s\tmissing_artifacts\n" "$cancer" "$stage" "$artifacts_dir" "$run_out" >> "$run_manifest"
    return 0
  fi

  local metrics_csv="$run_out/mixing_metrics_${EMBED_SOURCE}.csv"
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$metrics_csv" ]]; then
    echo "[Skip] Existing result: $metrics_csv"
    printf "%s\t%s\t%s\t%s\tskipped_existing\n" "$cancer" "$stage" "$artifacts_dir" "$run_out" >> "$run_manifest"
    return 0
  fi

  mkdir -p "$run_out"
  echo "[Run] $cancer $stage"
  echo "      artifacts: $artifacts_dir"
  echo "      out:       $run_out"

  local highlight_col="$HIGHLIGHT_COL"
  local highlight_values=()
  if (( ${#HIGHLIGHT_VALUE_ARGS[@]} > 0 )); then
    highlight_values=("${HIGHLIGHT_VALUE_ARGS[@]}")
  fi
  if [[ -z "$highlight_col" && "${#highlight_values[@]}" -eq 0 && "$HIGHLIGHT_LOCO" -eq 1 ]]; then
    highlight_col="cancer_type"
    highlight_values=("$cancer")
  fi

  local cmd=(
    python -m stonco.utils.run_embedding_analysis_pipeline
    --artifacts_dir "$artifacts_dir"
    --train_npz "$TRAIN_NPZ"
    --subset "$SUBSET"
    --out_dir "$run_out"
    --embed_source "$EMBED_SOURCE"
    --group_cols "${GROUP_COL_ARGS[@]}"
    --group_roles "${GROUP_ROLE_ARGS[@]}"
    --metric_spaces "${METRIC_SPACE_ARGS[@]}"
    --k_values "${K_VALUE_ARGS[@]}"
    --max_points "$MAX_POINTS"
    --seed "$SEED"
    --color_cols "${COLOR_COL_ARGS[@]}"
    --save_spot_metrics
    --device "$DEVICE"
  )
  if [[ -n "$highlight_col" || "${#highlight_values[@]}" -gt 0 ]]; then
    if [[ -z "$highlight_col" || "${#highlight_values[@]}" -eq 0 ]]; then
      echo "Error: --highlight_col and --highlight_values must be provided together." >&2
      exit 2
    fi
    cmd+=(--highlight_col "$highlight_col" --highlight_values "${highlight_values[@]}")
  fi
  if [[ -n "$NUM_THREADS" ]]; then
    cmd+=(--num_threads "$NUM_THREADS")
  fi

  {
    printf '[Command]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
  } > "$run_out/pipeline.log" 2>&1

  printf "%s\t%s\t%s\t%s\tdone\n" "$cancer" "$stage" "$artifacts_dir" "$run_out" >> "$run_manifest"
}

mapfile -t fold_dirs < <(
  find "$LOCO_DIR" -mindepth 1 -maxdepth 1 -type d \
    ! -name 'eval_visualizations' \
    ! -name 'loco_umap_tsne' \
    ! -name 'embedding_*' \
    | sort
)

if [[ "${#fold_dirs[@]}" -eq 0 ]]; then
  echo "Error: no LOCO fold directories found under $LOCO_DIR" >&2
  exit 1
fi

for fold_dir in "${fold_dirs[@]}"; do
  cancer="$(basename "$fold_dir")"
  if (( ${#CANCER_FILTER_ARGS[@]} > 0 )); then
    if ! in_list_or_empty "$cancer" "${CANCER_FILTER_ARGS[@]}"; then
      continue
    fi
  fi

  if [[ "$SKIP_FINAL" -eq 0 ]]; then
    run_embedding_pipeline \
      "$cancer" \
      "final" \
      "$fold_dir" \
      "$OUT_DIR/$cancer/final_${SUBSET}_${EMBED_SOURCE}"
  fi

  if [[ "$INCLUDE_CHECKPOINTS" -eq 1 ]]; then
    checkpoint_root="$fold_dir/epoch_checkpoints"
    if [[ ! -d "$checkpoint_root" ]]; then
      echo "[Skip] No checkpoint directory for $cancer: $checkpoint_root"
      continue
    fi
    mapfile -t checkpoint_dirs < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' | sort -V)
    for checkpoint_dir in "${checkpoint_dirs[@]}"; do
      if ! epoch_selected "$checkpoint_dir"; then
        continue
      fi
      epoch_name="$(basename "$checkpoint_dir")"
      run_embedding_pipeline \
        "$cancer" \
        "$epoch_name" \
        "$checkpoint_dir" \
        "$OUT_DIR/$cancer/checkpoints/${epoch_name}_${SUBSET}_${EMBED_SOURCE}"
    done
  fi
done

echo "[Done] Manifest: $run_manifest"
