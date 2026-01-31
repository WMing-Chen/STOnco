import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


IMG_FEATURE_DIM = 2048


def _find_unique_file_by_suffix(sample_dir: str, suffix: str, kind: str, sample_id: str) -> str:
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith(suffix.lower())]
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 '*{suffix}' file for {kind} in sample_dir={sample_dir} "
            f"(sample_id={sample_id}), found {len(files)}: {files}"
        )
    return os.path.join(sample_dir, files[0])


def _read_expression(exp_csv: str) -> Tuple[pd.Index, List[str], pd.DataFrame]:
    df = pd.read_csv(exp_csv)
    if df.shape[1] < 2:
        raise ValueError(f"Expression CSV must have at least 2 columns (Barcode + genes): {exp_csv}")
    if df.columns[0].lower() != 'barcode':
        raise ValueError(f"First column must be 'Barcode' in {exp_csv}, got {df.columns[0]}")
    barcodes = df.iloc[:, 0].astype(str)
    gene_cols = [str(c) for c in df.columns[1:]]
    # Use barcodes as index
    expr = df.iloc[:, 1:].copy()
    expr.index = barcodes
    # Ensure numeric
    for c in expr.columns:
        expr[c] = pd.to_numeric(expr[c], errors='coerce').fillna(0.0)
    return expr.index, gene_cols, expr


def _read_coords(coord_csv: str, x_col: str, y_col: str, label_col: str | None) -> Tuple[pd.Index, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(coord_csv)
    if 'Barcode'.lower() not in [c.lower() for c in df.columns]:
        raise ValueError(f"Coordinates CSV must contain 'Barcode' column: {coord_csv}")
    # Normalize column names to access regardless of case
    colmap = {c.lower(): c for c in df.columns}
    bc_col = colmap.get('barcode', None)
    if bc_col is None:
        raise ValueError(f"'Barcode' column not found in {coord_csv}")

    def _resolve(col_name: str) -> str:
        c = colmap.get(col_name.lower())
        if c is None:
            raise ValueError(f"Column '{col_name}' not found in {coord_csv}. Available: {list(df.columns)}")
        return c

    x_name = _resolve(x_col)
    y_name = _resolve(y_col)
    df[x_name] = pd.to_numeric(df[x_name], errors='coerce')
    df[y_name] = pd.to_numeric(df[y_name], errors='coerce')
    df = df.dropna(subset=[x_name, y_name])

    barcodes = df[bc_col].astype(str)
    xy = df[[x_name, y_name]].to_numpy(dtype=float)

    if label_col is None:
        y = None
    else:
        # Strictly require the provided label_col (no fallback)
        lbl_name = _resolve(label_col)
        y_series = df[lbl_name]
        # If numeric, ensure binary; otherwise map only 'tumor'->1, 'normal'->0
        if np.issubdtype(y_series.dtype, np.number):
            y_numeric = pd.to_numeric(y_series, errors='coerce')
            uniq = set(pd.unique(y_numeric.dropna()))
            if not uniq.issubset({0, 1}):
                bad = sorted([v for v in uniq if v not in {0, 1}])
                print(f"Warning: Found non-binary numeric labels in {coord_csv}: {bad}. Coercing to 0/1 by (value != 0).")
            y = (y_numeric.fillna(0) != 0).astype(int).to_numpy()
        else:
            s = y_series.astype(str).str.strip().str.lower()
            mapping = {'tumor': 1, 'normal': 0, 'mal': 1, 'nmal': 0}
            mapped = s.map(mapping)
            unknown = sorted(list(set(s[mapped.isna()].unique()) - {''}))
            if unknown:
                print(f"Warning: Found unexpected label values in {coord_csv}: {unknown}. Mapping only 'tumor'->1 and 'normal'->0; others -> 0.")
            y = mapped.fillna(0).astype(int).to_numpy()
    return barcodes, xy, y


def _read_image_features(image_csv: str) -> Tuple[pd.Index, List[str], pd.DataFrame]:
    df = pd.read_csv(image_csv)
    colmap = {str(c).lower(): c for c in df.columns}
    bc_col = colmap.get('barcode', None)
    if bc_col is None:
        raise ValueError(f"Image features CSV must contain 'Barcode' column: {image_csv}. Available: {list(df.columns)}")

    barcodes = df[bc_col].astype(str)
    if barcodes.duplicated().any():
        dup = barcodes[barcodes.duplicated()].tolist()[:10]
        raise ValueError(f"Duplicate Barcode values found in image features CSV: {image_csv}. Examples: {dup}")

    feat_cols = [c for c in df.columns if c != bc_col]
    if len(feat_cols) != IMG_FEATURE_DIM:
        raise ValueError(
            f"Image features CSV must have exactly {IMG_FEATURE_DIM} feature columns (excluding Barcode): "
            f"{image_csv}. Got {len(feat_cols)}."
        )

    feature_names = [str(c) for c in feat_cols]
    feats = df[feat_cols].copy()
    for c in feats.columns:
        feats[c] = pd.to_numeric(feats[c], errors='coerce')
    if feats.isna().any().any():
        bad_cols = feats.columns[feats.isna().any()].tolist()[:10]
        raise ValueError(f"Found NaN (or non-numeric) in image features CSV: {image_csv}. Bad cols examples: {bad_cols}")

    arr = feats.to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        i, j = int(bad[0, 0]), int(bad[0, 1])
        bc = str(barcodes.iloc[i])
        col = feature_names[j]
        val = arr[i, j]
        raise ValueError(f"Found non-finite value in image features CSV: {image_csv} (Barcode={bc}, col={col}, value={val})")

    feats.index = barcodes
    return feats.index, feature_names, feats


def _union_gene_order(slide_gene_lists: List[List[str]]) -> List[str]:
    seen: Dict[str, None] = {}
    for genes in slide_gene_lists:
        for g in genes:
            if g not in seen:
                seen[g] = None
    return list(seen.keys())


def _build_X_for_union(expr: pd.DataFrame, union_genes: List[str]) -> np.ndarray:
    # Ensure all union genes present; fill missing with 0
    cols = []
    for g in union_genes:
        if g in expr.columns:
            cols.append(expr[g])
        else:
            cols.append(pd.Series(0.0, index=expr.index, name=g))
    X = pd.concat(cols, axis=1).to_numpy(dtype=float)
    return X


def build_train_npz(train_dir: str, out_npz: str, x_col: str, y_col: str, label_col: str) -> None:
    train_dir = os.path.abspath(train_dir)
    subdirs = sorted([p for p in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, p))])
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {train_dir}")

    slide_exprs: List[pd.DataFrame] = []
    slide_barcodes: List[pd.Index] = []
    slide_genes: List[List[str]] = []
    slide_coords_bc: List[pd.Index] = []
    slide_xys: List[np.ndarray] = []
    slide_ys: List[np.ndarray] = []
    slide_ids: List[str] = []
    slide_barcodes_aligned: List[pd.Index] = []
    slide_X_imgs: List[np.ndarray] = []
    slide_img_masks: List[np.ndarray] = []
    img_feature_names_ref: List[str] | None = None

    for sd in subdirs:
        slide_path = os.path.join(train_dir, sd)
        coord_csv = _find_unique_file_by_suffix(slide_path, 'coordinates.csv', kind='coordinates', sample_id=sd)
        exp_csv = _find_unique_file_by_suffix(slide_path, 'exp.csv', kind='expression', sample_id=sd)
        img_csv = _find_unique_file_by_suffix(slide_path, 'image_features.csv', kind='image_features', sample_id=sd)

        bc_exp, genes, expr = _read_expression(exp_csv)
        bc_coord, xy, y = _read_coords(coord_csv, x_col=x_col, y_col=y_col, label_col=label_col)

        # align by intersection of barcodes, keep coordinates.csv barcode order
        bc_coord_arr = np.asarray(bc_coord, dtype=str)
        bc_exp_set = set(map(str, bc_exp))
        keep_mask = np.array([bc in bc_exp_set for bc in bc_coord_arr], dtype=bool)
        barcodes_used = bc_coord_arr[keep_mask]
        if barcodes_used.shape[0] == 0:
            raise RuntimeError(f"No overlapping Barcodes between {coord_csv} and {exp_csv}")
        expr_aligned = expr.reindex(pd.Index(barcodes_used, dtype=str), copy=False)
        xy_aligned = xy[keep_mask]
        y_aligned = y[keep_mask] if y is not None else None

        _, img_feature_names, img_df = _read_image_features(img_csv)
        if img_feature_names_ref is None:
            img_feature_names_ref = img_feature_names
        elif img_feature_names != img_feature_names_ref:
            raise ValueError(
                f"Image feature names/order mismatch across slides. sample_id={sd}. "
                f"Expected first 5={img_feature_names_ref[:5]}, got first 5={img_feature_names[:5]}"
            )
        img_aligned = img_df.reindex(pd.Index(barcodes_used, dtype=str))
        row_has_any_nan = img_aligned.isna().any(axis=1)
        img_mask = (~row_has_any_nan).astype(np.uint8).to_numpy()
        missing_count = int((img_mask == 0).sum())
        if missing_count > 0:
            print(
                f"Warning: sample_id={sd} missing image features for {missing_count} / {len(img_mask)} spots "
                f"(filled with zeros, img_mask=0)"
            )
        X_img = img_aligned.fillna(0.0).to_numpy(dtype=np.float32)

        slide_exprs.append(expr_aligned)
        slide_barcodes.append(expr_aligned.index)
        slide_genes.append(genes)
        slide_coords_bc.append(pd.Index(barcodes_used))
        slide_xys.append(xy_aligned)
        if y_aligned is None:
            raise RuntimeError(f"Training requires labels. '{coord_csv}' must contain column '{label_col}'.")
        slide_ys.append(y_aligned.astype(int))
        slide_ids.append(sd)
        slide_barcodes_aligned.append(pd.Index(barcodes_used))
        slide_X_imgs.append(X_img)
        slide_img_masks.append(img_mask.astype(np.uint8))

    union_genes = _union_gene_order(slide_genes)

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    xys: List[np.ndarray] = []
    barcodes_arrays: List[np.ndarray] = []
    for expr, xy, y, barcodes in zip(slide_exprs, slide_xys, slide_ys, slide_barcodes_aligned):
        X = _build_X_for_union(expr, union_genes)
        Xs.append(X.astype(np.float32))
        xys.append(xy.astype(np.float32))
        ys.append(y.astype(np.int64))
        barcodes_arrays.append(np.array(barcodes.astype(str), dtype='U64'))

    if img_feature_names_ref is None:
        raise RuntimeError('No image features loaded. This should not happen.')

    np.savez_compressed(
        out_npz,
        Xs=np.array(Xs, dtype=object),
        ys=np.array(ys, dtype=object),
        xys=np.array(xys, dtype=object),
        slide_ids=np.array(slide_ids, dtype=object),
        gene_names=np.array(union_genes, dtype=object),
        barcodes=np.array(barcodes_arrays, dtype=object),
        X_imgs=np.array(slide_X_imgs, dtype=object),
        img_masks=np.array(slide_img_masks, dtype=object),
        img_feature_names=np.array(img_feature_names_ref, dtype=object),
    )
    print(f"Saved training NPZ to: {out_npz}")


def build_single_npz(exp_csv: str, coord_csv: str, out_npz: str, x_col: str, y_col: str, label_col: str | None = None, sample_id: str | None = None) -> None:
    bc_exp, genes, expr = _read_expression(exp_csv)
    bc_coord, xy, y = _read_coords(coord_csv, x_col=x_col, y_col=y_col, label_col=label_col)

    # align by intersection of barcodes, keep coordinates.csv barcode order
    bc_coord_arr = np.asarray(bc_coord, dtype=str)
    bc_exp_set = set(map(str, bc_exp))
    keep_mask = np.array([bc in bc_exp_set for bc in bc_coord_arr], dtype=bool)
    barcodes_used = bc_coord_arr[keep_mask]
    if barcodes_used.shape[0] == 0:
        raise RuntimeError(f"No overlapping Barcodes between {coord_csv} and {exp_csv}")

    expr_aligned = expr.reindex(pd.Index(barcodes_used, dtype=str), copy=False)
    xy_aligned = xy[keep_mask]
    y_aligned = y[keep_mask] if y is not None else None
    barcodes_aligned = np.array(barcodes_used.astype(str), dtype='U64')
    
    # Infer sample_id from file path if not provided
    if sample_id is None:
        sample_id = Path(coord_csv).parent.name

    img_csv = _find_unique_file_by_suffix(Path(coord_csv).parent.as_posix(), 'image_features.csv', kind='image_features', sample_id=sample_id)
    _, img_feature_names, img_df = _read_image_features(img_csv)
    img_aligned = img_df.reindex(pd.Index(barcodes_used, dtype=str))
    row_has_any_nan = img_aligned.isna().any(axis=1)
    img_mask = (~row_has_any_nan).astype(np.uint8).to_numpy()
    missing_count = int((img_mask == 0).sum())
    if missing_count > 0:
        print(
            f"Warning: sample_id={sample_id} missing image features for {missing_count} / {len(img_mask)} spots "
            f"(filled with zeros, img_mask=0)"
        )
    X_img = img_aligned.fillna(0.0).to_numpy(dtype=np.float32)

    # Keep original gene order as in CSV
    X = expr_aligned.to_numpy(dtype=float)
    save_dict = {
        'X': X.astype(np.float32),
        'xy': xy_aligned.astype(np.float32),
        'gene_names': np.array(genes, dtype=object),
        'barcodes': barcodes_aligned,
        'sample_id': sample_id,
        'X_img': X_img.astype(np.float32),
        'img_mask': img_mask.astype(np.uint8),
        'img_feature_names': np.array(img_feature_names, dtype=object),
    }
    if y_aligned is not None:
        save_dict['y'] = y_aligned.astype(np.int64)
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"Saved single-sample NPZ to: {out_npz} (sample_id: {sample_id}, spots: {len(barcodes_aligned)})")


def build_val_npz(val_dir: str, out_dir: str, x_col: str, y_col: str, label_col: str) -> None:
    """Build individual NPZ files for each validation slide."""
    val_dir = os.path.abspath(val_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    subdirs = sorted([p for p in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, p))])
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {val_dir}")
    
    print(f"Processing {len(subdirs)} validation slides...")
    for sd in subdirs:
        slide_path = os.path.join(val_dir, sd)
        try:
            coord_csv = _find_unique_file_by_suffix(slide_path, 'coordinates.csv', kind='coordinates', sample_id=sd)
            exp_csv = _find_unique_file_by_suffix(slide_path, 'exp.csv', kind='expression', sample_id=sd)
            _find_unique_file_by_suffix(slide_path, 'image_features.csv', kind='image_features', sample_id=sd)
        except Exception as e:
            print(f"Warning: {e}. Skipping.")
            continue
        out_npz = os.path.join(out_dir, f"{sd}.npz")
        
        try:
            build_single_npz(exp_csv, coord_csv, out_npz, x_col=x_col, y_col=y_col, 
                            label_col=label_col, sample_id=sd)
        except Exception as e:
            print(f"Error processing {sd}: {e}")
            continue
    
    print(f"Validation NPZ files saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare NPZ data for Spotonco GNN from CSV inputs')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('build-train-npz', help='Scan a folder of slides to create train_data.npz')
    p_train.add_argument('--train_dir', required=True, help='Directory containing slide subfolders')
    p_train.add_argument('--out_npz', default='train_data.npz', help='Output NPZ path')
    p_train.add_argument('--xy_cols', nargs=2, default=['row', 'col'], metavar=('X_COL', 'Y_COL'), help="Column names in coordinates.csv to use as x/y (default: row col)")
    p_train.add_argument('--label_col', default='true_label', help="Label column in coordinates.csv (default: true_label, values 0/1)")

    p_single = sub.add_parser('build-single-npz', help='Create a single-sample NPZ from exp.csv + coordinates.csv')
    p_single.add_argument('--exp_csv', required=True, help='Path to *_exp.csv (first column Barcode)')
    p_single.add_argument('--coord_csv', required=True, help='Path to *_coordinates.csv (must contain Barcode + x/y columns)')
    p_single.add_argument('--out_npz', default='sample.npz', help='Output NPZ path')
    p_single.add_argument('--xy_cols', nargs=2, default=['row', 'col'], metavar=('X_COL', 'Y_COL'), help="Column names in coordinates.csv to use as x/y (default: row col)")
    p_single.add_argument('--label_col', default=None, help="Label column in coordinates.csv (optional)")
    p_single.add_argument('--sample_id', default=None, help="Sample ID (default: infer from parent directory)")
    
    p_val = sub.add_parser('build-val-npz', help='Build individual NPZ files for validation slides')
    p_val.add_argument('--val_dir', required=True, help='Directory containing validation slide subfolders')
    p_val.add_argument('--out_dir', required=True, help='Output directory for NPZ files')
    p_val.add_argument('--xy_cols', nargs=2, default=['row', 'col'], metavar=('X_COL', 'Y_COL'), help="Column names in coordinates.csv to use as x/y (default: row col)")
    p_val.add_argument('--label_col', default='true_label', help="Label column in coordinates.csv (default: true_label)")

    args = parser.parse_args()

    if args.cmd == 'build-train-npz':
        x_col, y_col = args.xy_cols
        build_train_npz(args.train_dir, args.out_npz, x_col=x_col, y_col=y_col, label_col=args.label_col)
    elif args.cmd == 'build-single-npz':
        x_col, y_col = args.xy_cols
        build_single_npz(args.exp_csv, args.coord_csv, args.out_npz, x_col=x_col, y_col=y_col, 
                        label_col=args.label_col, sample_id=args.sample_id)
    elif args.cmd == 'build-val-npz':
        x_col, y_col = args.xy_cols
        build_val_npz(args.val_dir, args.out_dir, x_col=x_col, y_col=y_col, label_col=args.label_col)
    else:
        raise RuntimeError('Unknown command')


if __name__ == '__main__':
    main()
