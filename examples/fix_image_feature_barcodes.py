#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _norm_full(barcode: str) -> str:
    return barcode.split('_', 1)[1] if '_' in barcode else barcode


def _base_key(barcode: str) -> str:
    s = _norm_full(barcode)
    return s.rsplit('-', 1)[0] if '-' in s else s


def _find_col_idx(header: list[str], name: str) -> int | None:
    name_l = name.strip().lower()
    for i, c in enumerate(header):
        if str(c).strip().lower() == name_l:
            return i
    return None


def _is_finite_number(s: str) -> bool:
    try:
        x = float(s)
    except Exception:
        return False
    return x == x and x not in (float('inf'), float('-inf'))


def _read_barcodes_from_coords(coord_csv: Path, x_col: str, y_col: str) -> list[str]:
    out: list[str] = []
    with coord_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return out

        idx_bc = _find_col_idx(header, 'Barcode')
        idx_x = _find_col_idx(header, x_col)
        idx_y = _find_col_idx(header, y_col)
        if idx_bc is None:
            raise ValueError(f"Missing 'Barcode' column in {coord_csv}")
        if idx_x is None or idx_y is None:
            raise ValueError(f"Missing x/y columns ({x_col}, {y_col}) in {coord_csv}")

        for row in reader:
            if not row:
                continue
            if idx_bc >= len(row) or idx_x >= len(row) or idx_y >= len(row):
                continue
            if not _is_finite_number(row[idx_x]) or not _is_finite_number(row[idx_y]):
                continue
            out.append(str(row[idx_bc]))
    return out


def _read_barcodes_from_exp(exp_csv: Path) -> list[str]:
    out: list[str] = []
    with exp_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return out
        idx_bc = _find_col_idx(header, 'Barcode')
        if idx_bc is None:
            idx_bc = 0
        for row in reader:
            if not row or idx_bc >= len(row):
                continue
            out.append(str(row[idx_bc]))
    return out


def _find_unique_file_by_suffix(sample_dir: Path, suffix: str) -> Path | None:
    matches = [p for p in sample_dir.iterdir() if p.is_file() and p.name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        return None
    return matches[0]


def _make_backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + '.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        return bak
    for i in range(1, 1000):
        cand = path.with_suffix(path.suffix + f'.bak{i}')
        if not cand.exists():
            shutil.copy2(path, cand)
            return cand
    raise RuntimeError(f'Failed to create backup for {path}: too many existing .bak files')


@dataclass(frozen=True)
class FixPlan:
    bc_idx: int
    selected_row_idx: dict[str, int]
    target_total: int
    overlap_before: int
    overlap_after: int
    target_found_rows: int


def _build_maps(target_barcodes: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    norm_full_map: dict[str, str] = {}
    base_map: dict[str, str] = {}
    for bc in target_barcodes:
        nf = _norm_full(bc)
        prev = norm_full_map.get(nf)
        if prev is not None and prev != bc:
            raise ValueError(f'Ambiguous mapping by stripped prefix: {nf} -> {prev} / {bc}')
        norm_full_map[nf] = bc

        b = _base_key(bc)
        prev_b = base_map.get(b)
        if prev_b is not None and prev_b != bc:
            raise ValueError(f'Ambiguous mapping by base barcode: {b} -> {prev_b} / {bc}')
        base_map[b] = bc
    return norm_full_map, base_map


def _map_to_target(barcode: str, target_set: set[str], norm_full_map: dict[str, str], base_map: dict[str, str]) -> str | None:
    if barcode in target_set:
        return barcode
    nf = _norm_full(barcode)
    if nf in norm_full_map:
        return norm_full_map[nf]
    b = nf.rsplit('-', 1)[0] if '-' in nf else nf
    if b in base_map:
        return base_map[b]
    return None


def _plan_fix(image_csv: Path, target_barcodes: list[str]) -> FixPlan | None:
    target_set = set(map(str, target_barcodes))
    if not target_set:
        return None

    norm_full_map, base_map = _build_maps(target_barcodes)

    selected_row_idx: dict[str, tuple[int, bool]] = {}
    img_set: set[str] = set()

    with image_csv.open('r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
        bc_idx = _find_col_idx(header, 'Barcode')
        if bc_idx is None:
            raise ValueError(f"Missing 'Barcode' column in {image_csv}")

        for row_idx, row in enumerate(reader):
            if not row or bc_idx >= len(row):
                continue
            bc = str(row[bc_idx])
            img_set.add(bc)
            mapped = _map_to_target(bc, target_set, norm_full_map, base_map)
            if mapped is None or mapped not in target_set:
                continue
            is_exact = (bc == mapped)
            prev = selected_row_idx.get(mapped)
            if prev is None:
                selected_row_idx[mapped] = (row_idx, is_exact)
            else:
                prev_idx, prev_exact = prev
                if is_exact and not prev_exact:
                    selected_row_idx[mapped] = (row_idx, True)

    overlap_before = len(target_set & img_set)
    selected = {k: v[0] for k, v in selected_row_idx.items()}
    overlap_after = len(selected)

    return FixPlan(
        bc_idx=int(bc_idx),
        selected_row_idx=selected,
        target_total=len(target_set),
        overlap_before=int(overlap_before),
        overlap_after=int(overlap_after),
        target_found_rows=int(overlap_after),
    )


def _apply_fix(image_csv: Path, plan: FixPlan, target_barcodes: list[str], dry_run: bool, backup: bool) -> None:
    target_set = set(map(str, target_barcodes))
    norm_full_map, base_map = _build_maps(target_barcodes)

    if dry_run:
        return

    if backup:
        _make_backup(image_csv)

    tmp_path = image_csv.with_suffix(image_csv.suffix + '.tmp')
    with image_csv.open('r', encoding='utf-8', errors='replace', newline='') as fin, tmp_path.open(
        'w', encoding='utf-8', newline=''
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        bc_idx = plan.bc_idx
        selected_idx = plan.selected_row_idx

        for row_idx, row in enumerate(reader):
            if not row or bc_idx >= len(row):
                continue
            bc = str(row[bc_idx])
            mapped = _map_to_target(bc, target_set, norm_full_map, base_map)
            if mapped is not None and mapped in target_set:
                keep_idx = selected_idx.get(mapped)
                if keep_idx is None or keep_idx != row_idx:
                    continue
                row[bc_idx] = mapped
            writer.writerow(row)

    os.replace(tmp_path, image_csv)


def _collect_sample_dirs(data_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for sub in ['ST_train_datasets', 'ST_validation_datasets']:
        p = data_root / sub
        if not p.is_dir():
            continue
        for sd in sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.name):
            dirs.append(sd)
    if dirs:
        return dirs
    return sorted([x for x in data_root.iterdir() if x.is_dir()], key=lambda x: x.name)


def main() -> int:
    ap = argparse.ArgumentParser(description='Fix image_features.csv Barcode to align with coordinates/exp Barcodes.')
    ap.add_argument('--data_root', required=True, help='Dataset root (contains ST_train_datasets / ST_validation_datasets).')
    ap.add_argument('--xy_cols', nargs=2, default=['row', 'col'], metavar=('X_COL', 'Y_COL'), help='x/y columns in coordinates.csv.')
    ap.add_argument('--min_coverage', type=float, default=0.99, help='Only apply fix if mapped coverage >= this ratio.')
    ap.add_argument('--dry_run', action='store_true', help='Only report, do not modify files.')
    ap.add_argument('--no_backup', action='store_true', help='Do not create *.bak backups.')
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    x_col, y_col = args.xy_cols
    min_coverage = float(args.min_coverage)
    dry_run = bool(args.dry_run)
    backup = not bool(args.no_backup)

    sample_dirs = _collect_sample_dirs(data_root)
    if not sample_dirs:
        raise SystemExit(f'No sample directories found under: {data_root}')

    changed = 0
    skipped = 0
    failed = 0

    for sd in sample_dirs:
        sample_id = sd.name
        coord_csv = _find_unique_file_by_suffix(sd, 'coordinates.csv')
        exp_csv = _find_unique_file_by_suffix(sd, 'exp.csv')
        img_csv = _find_unique_file_by_suffix(sd, 'image_features.csv')

        if coord_csv is None or exp_csv is None or img_csv is None:
            skipped += 1
            continue

        try:
            coord_barcodes = _read_barcodes_from_coords(coord_csv, x_col=x_col, y_col=y_col)
            exp_barcodes = _read_barcodes_from_exp(exp_csv)
            exp_set = set(exp_barcodes)
            target_barcodes = [bc for bc in coord_barcodes if bc in exp_set]

            plan = _plan_fix(img_csv, target_barcodes)
            if plan is None:
                skipped += 1
                continue

            if plan.target_total == 0:
                skipped += 1
                continue

            if plan.overlap_before == plan.target_total:
                skipped += 1
                continue

            coverage = (plan.overlap_after / plan.target_total) if plan.target_total > 0 else 0.0
            if coverage < min_coverage:
                print(
                    f'[Skip] {sample_id}: coverage too low ({plan.overlap_after}/{plan.target_total}={coverage:.3f}), '
                    f'overlap_before={plan.overlap_before}'
                )
                skipped += 1
                continue

            if plan.overlap_after <= plan.overlap_before:
                print(
                    f'[Skip] {sample_id}: no improvement (before={plan.overlap_before}, after={plan.overlap_after})'
                )
                skipped += 1
                continue

            _apply_fix(img_csv, plan, target_barcodes, dry_run=dry_run, backup=backup)
            action = 'DryRun' if dry_run else 'Fixed'
            print(
                f'[{action}] {sample_id}: overlap {plan.overlap_before}->{plan.overlap_after} / {plan.target_total} '
                f'(coverage={coverage:.3f})'
            )
            changed += 1
        except Exception as e:
            print(f'[Fail] {sample_id}: {e}')
            failed += 1

    print(f'Done. changed={changed}, skipped={skipped}, failed={failed}, dry_run={dry_run}')
    return 0 if failed == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
