import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from stonco.core.infer import InferenceEngine
from stonco.utils.utils import load_json


def _parse_epochs(text: str) -> list[int]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid checkpoint epochs were provided.")
    return vals


def _parse_list(text: str | None) -> list[str]:
    if text is None:
        return []
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(part)
    return vals


def _eval_probs(probs: np.ndarray, y: np.ndarray) -> dict:
    mask = y >= 0
    if int(mask.sum()) == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
        }

    ytrue = y[mask].astype(int)
    p = probs[mask].astype(float)
    pred = (p > 0.5).astype(int)

    try:
        accuracy = float((pred == ytrue).mean())
    except Exception:
        accuracy = float("nan")

    try:
        auroc = _binary_auroc(ytrue, p)
    except Exception:
        auroc = float("nan")

    try:
        auprc = _binary_average_precision(ytrue, p)
    except Exception:
        auprc = float("nan")

    try:
        macro_f1 = _binary_macro_f1(ytrue, pred)
    except Exception:
        macro_f1 = float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def _binary_macro_f1(ytrue: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for cls in (0, 1):
        tp = float(np.sum((pred == cls) & (ytrue == cls)))
        fp = float(np.sum((pred == cls) & (ytrue != cls)))
        fn = float(np.sum((pred != cls) & (ytrue == cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(float(2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _binary_auroc(ytrue: np.ndarray, scores: np.ndarray) -> float:
    ytrue = np.asarray(ytrue).astype(int)
    scores = np.asarray(scores).astype(float)
    n_pos = int(np.sum(ytrue == 1))
    n_neg = int(np.sum(ytrue == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = ytrue[order]
    s = scores[order]
    distinct = np.where(np.diff(s))[0]
    threshold_idxs = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y == 1)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    return float(np.trapz(tpr, fpr))


def _binary_average_precision(ytrue: np.ndarray, scores: np.ndarray) -> float:
    ytrue = np.asarray(ytrue).astype(int)
    scores = np.asarray(scores).astype(float)
    n_pos = int(np.sum(ytrue == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = ytrue[order]
    s = scores[order]
    distinct = np.where(np.diff(s))[0]
    threshold_idxs = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y == 1)[threshold_idxs].astype(float)
    fps = ((threshold_idxs + 1) - tps).astype(float)
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _load_train_npz(train_npz: str) -> tuple[dict, list[str]]:
    data = np.load(train_npz, allow_pickle=True)
    files = set(data.files)
    required = {"Xs", "ys", "xys", "slide_ids", "gene_names"}
    missing = sorted(required - files)
    if missing:
        raise ValueError(f"train_npz missing keys: {missing}")

    Xs = list(data["Xs"])
    ys = list(data["ys"])
    xys = list(data["xys"])
    slide_ids = list(map(str, list(data["slide_ids"])))
    gene_names = list(data["gene_names"])

    X_imgs = list(data["X_imgs"]) if "X_imgs" in files else None
    img_masks = list(data["img_masks"]) if "img_masks" in files else None

    slides = {}
    for i, sid in enumerate(slide_ids):
        item = {
            "slide_id": sid,
            "X": Xs[i],
            "y": ys[i],
            "xy": xys[i],
        }
        if X_imgs is not None and img_masks is not None:
            item["X_img"] = X_imgs[i]
            item["img_mask"] = img_masks[i]
        slides[sid] = item
    return slides, gene_names


def _predict_probs_for_slide(engine: InferenceEngine, slide: dict, gene_names: list[str]) -> np.ndarray:
    Xp_gene = engine.pp.transform(slide["X"], gene_names)

    if bool(engine.cfg.get("use_image_features", False)):
        if engine.img_pp is None:
            raise RuntimeError("use_image_features=1 but ImagePreprocessor is missing in artifacts_dir.")
        Xp_img = engine.img_pp.transform(slide["X_img"], slide["img_mask"])
        data_g = engine.assemble_pyg(Xp_gene, slide["xy"], Xp_img=Xp_img, img_mask=slide["img_mask"])
    else:
        data_g = engine.assemble_pyg(Xp_gene, slide["xy"])

    return engine.predict_proba(data_g)


def main():
    p = argparse.ArgumentParser(description="Evaluate LOCO checkpoint models for selected epochs.")
    p.add_argument("--run_dir", required=True, help="Run root containing loco_eval/")
    p.add_argument("--train_npz", required=True)
    p.add_argument("--epochs", default="50,100", help="Comma-separated checkpoint epochs, e.g. 50,100")
    p.add_argument("--out_dir", default=None, help="Default: <run_dir>/loco_eval/checkpoint_eval_<epochs>")
    p.add_argument("--device", default=None)
    p.add_argument("--num_threads", type=int, default=None)
    p.add_argument("--cancers", default=None, help="Optional comma-separated cancer fold subset, e.g. BRCA,CRC")
    p.add_argument("--skip_if_complete", action="store_true", help="Skip if output summary already exists.")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    loco_dir = run_dir / "loco_eval"
    if not loco_dir.is_dir():
        raise FileNotFoundError(f"Missing loco_eval directory: {loco_dir}")

    epochs = _parse_epochs(args.epochs)
    cancer_subset = set(_parse_list(args.cancers))
    out_dir = Path(args.out_dir) if args.out_dir else loco_dir / f"checkpoint_eval_{'_'.join(map(str, epochs))}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "checkpoint_loco_summary.csv"
    per_slide_path = out_dir / "checkpoint_loco_per_slide.csv"
    per_slide_sum_path = out_dir / "checkpoint_loco_per_slide_summary.csv"
    status_path = out_dir / "checkpoint_eval_status.tsv"

    if args.skip_if_complete and summary_path.exists() and per_slide_path.exists() and per_slide_sum_path.exists():
        print(f"[Skip] Existing checkpoint evaluation found in {out_dir}")
        return

    slides_by_id, gene_names = _load_train_npz(args.train_npz)

    summary_rows = []
    per_slide_rows = []
    status_rows = []

    fold_dirs = [d for d in sorted(loco_dir.iterdir()) if d.is_dir() and (d / "meta.json").exists()]
    if not fold_dirs:
        raise RuntimeError(f"No LOCO fold directories with meta.json found under {loco_dir}")

    for fold_dir in fold_dirs:
        cancer_type = fold_dir.name
        if cancer_subset and cancer_type not in cancer_subset:
            continue
        fold_meta = load_json(str(fold_dir / "meta.json"))
        val_ids = list(map(str, fold_meta.get("val_ids", []) or []))
        train_ids = list(map(str, fold_meta.get("train_ids", []) or []))

        for epoch in epochs:
            ckpt_dir = fold_dir / "epoch_checkpoints" / f"epoch_{epoch}"
            if not (ckpt_dir / "model.pt").exists() or not (ckpt_dir / "meta.json").exists():
                status_rows.append({
                    "cancer_type": cancer_type,
                    "checkpoint_epoch": int(epoch),
                    "state": "skipped",
                    "message": "missing_checkpoint_dir_or_model",
                })
                continue

            try:
                engine = InferenceEngine(str(ckpt_dir), device=args.device, num_threads=args.num_threads)
            except Exception as e:
                status_rows.append({
                    "cancer_type": cancer_type,
                    "checkpoint_epoch": int(epoch),
                    "state": "failed",
                    "message": f"engine_init_failed: {e}",
                })
                continue

            all_probs = []
            all_y = []
            n_missing = 0
            try:
                for slide_id in val_ids:
                    slide = slides_by_id.get(slide_id)
                    if slide is None:
                        n_missing += 1
                        continue
                    probs = _predict_probs_for_slide(engine, slide, gene_names)
                    y_np = np.asarray(slide["y"])
                    all_probs.append(probs)
                    all_y.append(y_np)

                    m_slide = _eval_probs(probs, y_np)
                    n_nodes = int(y_np.shape[0])
                    n_pos = int((y_np == 1).sum())
                    n_neg = int((y_np == 0).sum())
                    pos_rate = float(n_pos / n_nodes) if n_nodes > 0 else float("nan")
                    per_slide_rows.append({
                        "run_name": run_dir.name,
                        "cancer_type": cancer_type,
                        "checkpoint_epoch": int(epoch),
                        "slide_id": str(slide_id),
                        "n_nodes": n_nodes,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        "pos_rate": pos_rate,
                        "auroc": m_slide["auroc"],
                        "auprc": m_slide["auprc"],
                        "accuracy": m_slide["accuracy"],
                        "macro_f1": m_slide["macro_f1"],
                        "best_epoch": int(epoch),
                        "last_epoch": int(epoch),
                        "saved_checkpoint": "epoch",
                        "saved_epoch": int(epoch),
                    })

                if all_probs:
                    agg_probs = np.concatenate(all_probs, axis=0)
                    agg_y = np.concatenate(all_y, axis=0)
                    m_all = _eval_probs(agg_probs, agg_y)
                else:
                    m_all = _eval_probs(np.array([]), np.array([]))

                summary_rows.append({
                    "run_name": run_dir.name,
                    "cancer_type": cancer_type,
                    "checkpoint_epoch": int(epoch),
                    "saved_checkpoint": "epoch",
                    "completed_full_epochs": False,
                    "best_epoch": int(epoch),
                    "last_epoch": int(epoch),
                    "saved_epoch": int(epoch),
                    "auroc": m_all["auroc"],
                    "auprc": m_all["auprc"],
                    "accuracy": m_all["accuracy"],
                    "macro_f1": m_all["macro_f1"],
                    "last_auroc": m_all["auroc"],
                    "last_auprc": m_all["auprc"],
                    "last_accuracy": m_all["accuracy"],
                    "last_macro_f1": m_all["macro_f1"],
                    "saved_auroc": m_all["auroc"],
                    "saved_auprc": m_all["auprc"],
                    "saved_accuracy": m_all["accuracy"],
                    "saved_macro_f1": m_all["macro_f1"],
                    "n_train": int(len(train_ids)),
                    "n_val": int(len(val_ids)),
                    "n_missing_slides": int(n_missing),
                })
                status_rows.append({
                    "cancer_type": cancer_type,
                    "checkpoint_epoch": int(epoch),
                    "state": "ok",
                    "message": "",
                })
                print(f"[Done] {run_dir.name} {cancer_type} epoch_{epoch}")
            except Exception as e:
                status_rows.append({
                    "cancer_type": cancer_type,
                    "checkpoint_epoch": int(epoch),
                    "state": "failed",
                    "message": str(e),
                })
                print(f"[Failed] {run_dir.name} {cancer_type} epoch_{epoch}: {e}")

    df_summary = pd.DataFrame(summary_rows)
    df_per_slide = pd.DataFrame(per_slide_rows)
    df_status = pd.DataFrame(status_rows)

    if not df_summary.empty:
        df_summary.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(columns=[
            "run_name", "cancer_type", "checkpoint_epoch", "saved_checkpoint",
            "completed_full_epochs", "best_epoch", "last_epoch", "saved_epoch",
            "auroc", "auprc", "accuracy", "macro_f1",
            "last_auroc", "last_auprc", "last_accuracy", "last_macro_f1",
            "saved_auroc", "saved_auprc", "saved_accuracy", "saved_macro_f1",
            "n_train", "n_val", "n_missing_slides",
        ]).to_csv(summary_path, index=False)

    if not df_per_slide.empty:
        df_per_slide.to_csv(per_slide_path, index=False)
        metric_cols = ["auroc", "auprc", "accuracy", "macro_f1"]
        rows = []
        for (epoch, cancer), sub in df_per_slide.groupby(["checkpoint_epoch", "cancer_type"]):
            row = {
                "checkpoint_epoch": int(epoch),
                "cancer_type": cancer,
                "n_val_slides": int(len(sub)),
            }
            for m in metric_cols:
                row[f"mean_{m}_slide"] = float(sub[m].mean()) if m in sub else float("nan")
                row[f"std_{m}_slide"] = float(sub[m].std()) if m in sub else float("nan")
            rows.append(row)
        for epoch, sub in df_per_slide.groupby("checkpoint_epoch"):
            row = {
                "checkpoint_epoch": int(epoch),
                "cancer_type": "ALL",
                "n_val_slides": int(len(sub)),
            }
            for m in metric_cols:
                row[f"mean_{m}_slide"] = float(sub[m].mean()) if m in sub else float("nan")
                row[f"std_{m}_slide"] = float(sub[m].std()) if m in sub else float("nan")
            rows.append(row)
        pd.DataFrame(rows).sort_values(["checkpoint_epoch", "cancer_type"]).to_csv(per_slide_sum_path, index=False)
    else:
        pd.DataFrame(columns=[
            "run_name", "cancer_type", "checkpoint_epoch", "slide_id", "n_nodes",
            "n_pos", "n_neg", "pos_rate", "auroc", "auprc", "accuracy", "macro_f1",
            "best_epoch", "last_epoch", "saved_checkpoint", "saved_epoch",
        ]).to_csv(per_slide_path, index=False)
        pd.DataFrame(columns=[
            "checkpoint_epoch", "cancer_type", "n_val_slides",
            "mean_auroc_slide", "std_auroc_slide", "mean_auprc_slide", "std_auprc_slide",
            "mean_accuracy_slide", "std_accuracy_slide", "mean_macro_f1_slide", "std_macro_f1_slide",
        ]).to_csv(per_slide_sum_path, index=False)

    df_status.to_csv(status_path, index=False)
    print(f"Saved checkpoint summary to {summary_path}")
    print(f"Saved checkpoint per-slide metrics to {per_slide_path}")
    print(f"Saved checkpoint per-slide summary to {per_slide_sum_path}")
    print(f"Saved status to {status_path}")


if __name__ == "__main__":
    main()
