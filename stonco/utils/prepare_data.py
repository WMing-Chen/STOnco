import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


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

    for sd in subdirs:
        slide_path = os.path.join(train_dir, sd)
        # auto-detect files: prefer *coordinates.csv and *exp.csv
        coord_files = [f for f in os.listdir(slide_path) if f.lower().endswith('coordinates.csv')]
        exp_files = [f for f in os.listdir(slide_path) if f.lower().endswith('exp.csv')]
        if not coord_files or not exp_files:
            raise RuntimeError(f"Missing coordinates/exp CSV in {slide_path}. Found: {os.listdir(slide_path)}")
        coord_csv = os.path.join(slide_path, coord_files[0])
        exp_csv = os.path.join(slide_path, exp_files[0])

        bc_exp, genes, expr = _read_expression(exp_csv)
        bc_coord, xy, y = _read_coords(coord_csv, x_col=x_col, y_col=y_col, label_col=label_col)

        # align by intersection of barcodes, keep coordinate order
        bc_inter = pd.Index(bc_coord, dtype=str).intersection(pd.Index(bc_exp, dtype=str))
        if len(bc_inter) == 0:
            raise RuntimeError(f"No overlapping Barcodes between {coord_csv} and {exp_csv}")
        # Reindex expr to coord order
        expr_aligned = expr.reindex(pd.Index(bc_inter), copy=False)
        xy_aligned = xy[np.isin(bc_coord, bc_inter)]
        y_aligned = y[np.isin(bc_coord, bc_inter)] if y is not None else None

        slide_exprs.append(expr_aligned)
        slide_barcodes.append(expr_aligned.index)
        slide_genes.append(genes)
        slide_coords_bc.append(pd.Index(bc_inter))
        slide_xys.append(xy_aligned)
        if y_aligned is None:
            raise RuntimeError(f"Training requires labels. '{coord_csv}' must contain column '{label_col}'.")
        slide_ys.append(y_aligned.astype(int))
        slide_ids.append(sd)
        slide_barcodes_aligned.append(pd.Index(bc_inter))

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

    np.savez_compressed(
        out_npz,
        Xs=np.array(Xs, dtype=object),
        ys=np.array(ys, dtype=object),
        xys=np.array(xys, dtype=object),
        slide_ids=np.array(slide_ids, dtype=object),
        gene_names=np.array(union_genes, dtype=object),
        barcodes=np.array(barcodes_arrays, dtype=object),
    )
    print(f"Saved training NPZ to: {out_npz}")


def build_single_npz(exp_csv: str, coord_csv: str, out_npz: str, x_col: str, y_col: str, label_col: str | None = None, sample_id: str | None = None) -> None:
    bc_exp, genes, expr = _read_expression(exp_csv)
    bc_coord, xy, y = _read_coords(coord_csv, x_col=x_col, y_col=y_col, label_col=label_col)

    # align by intersection, keep coordinate order
    bc_inter = pd.Index(bc_coord, dtype=str).intersection(pd.Index(bc_exp, dtype=str))
    if len(bc_inter) == 0:
        raise RuntimeError(f"No overlapping Barcodes between {coord_csv} and {exp_csv}")

    expr_aligned = expr.reindex(pd.Index(bc_inter), copy=False)
    xy_aligned = xy[np.isin(bc_coord, bc_inter)]
    y_aligned = y[np.isin(bc_coord, bc_inter)] if y is not None else None
    barcodes_aligned = np.array(bc_inter.astype(str), dtype='U64')
    
    # Infer sample_id from file path if not provided
    if sample_id is None:
        sample_id = Path(coord_csv).parent.name

    # Keep original gene order as in CSV
    X = expr_aligned.to_numpy(dtype=float)
    save_dict = {
        'X': X.astype(np.float32),
        'xy': xy_aligned.astype(np.float32),
        'gene_names': np.array(genes, dtype=object),
        'barcodes': barcodes_aligned,
        'sample_id': sample_id,
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
        # auto-detect files: prefer *coordinates.csv and *exp.csv
        coord_files = [f for f in os.listdir(slide_path) if f.lower().endswith('coordinates.csv')]
        exp_files = [f for f in os.listdir(slide_path) if f.lower().endswith('exp.csv')]
        if not coord_files or not exp_files:
            print(f"Warning: Missing coordinates/exp CSV in {slide_path}. Skipping.")
            continue
        
        coord_csv = os.path.join(slide_path, coord_files[0])
        exp_csv = os.path.join(slide_path, exp_files[0])
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