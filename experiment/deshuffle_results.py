#!/usr/bin/env python3
"""
Reorder COLUMNS of an input CSV so they align with the ORIGINAL order derived from
(original.csv vs shuffled.csv), then rename the mapped columns to their original indices.

Rules:
- Unnamed columns (empty header or 'Unnamed: ...') are kept at the BEGINNING in their original order,
  and their headers are set to empty string "" (no 'Unnamed' prefix).
- Mapped columns are reordered into ORIGINAL order and RENAMED to '0','1',...,'n-1'.
- Any extra columns (not unnamed and not mapped) are appended at the END with their original names.

Usage:
    python reorder_columns.py original.csv shuffled.csv input.csv output.csv
"""

import argparse
from collections import defaultdict, deque

import numpy as np
import pandas as pd


def load_clean_csv(path: str) -> pd.DataFrame:
    """Load CSV and drop a phantom first index column if present."""
    df = pd.read_csv(path)
    if df.shape[1] == 0:
        return df

    first_name = str(df.columns[0])
    first_series = df.iloc[:, 0]

    # Detect phantom index column
    name_looks_index = (
        first_name.strip() == ""
        or first_name.lower().startswith("unnamed")
        or first_name.strip().lower() in {"index", "#", "row", "rowid", "row_id"}
    )
    s_num = pd.to_numeric(first_series, errors="coerce")
    sequential0 = np.array_equal(s_num.to_numpy(), np.arange(len(df)))
    sequential1 = np.array_equal(s_num.to_numpy(), np.arange(1, len(df) + 1))

    if name_looks_index and (sequential0 or sequential1):
        df = df.drop(columns=[df.columns[0]])

    return df


def rows_to_tuples(df: pd.DataFrame):
    """Convert DataFrame rows to tuples, replacing NaN with None for safe comparison."""
    return [tuple(None if pd.isna(x) else x for x in row) for row in df.to_numpy()]


def build_index_map(original: pd.DataFrame, shuffled: pd.DataFrame):
    """Build mapping from shuffled row index -> original row index (handles duplicates)."""
    orig_rows = rows_to_tuples(original)
    shuf_rows = rows_to_tuples(shuffled)

    row_to_indices = defaultdict(deque)
    for i, r in enumerate(orig_rows):
        row_to_indices[r].append(i)

    index_map = []
    for r in shuf_rows:
        if not row_to_indices[r]:
            raise ValueError(f"Row in shuffled not found in original: {r}")
        index_map.append(row_to_indices[r].popleft())

    return index_map


def ensure_permutation(index_map, n):
    """Validate that index_map is a permutation and return its inverse (orig_idx -> shuf_idx)."""
    if len(index_map) != n:
        raise ValueError(f"index_map length {len(index_map)} != expected size {n}")
    uniq = set(index_map)
    if uniq != set(range(n)):
        raise ValueError("index_map is not a valid permutation of 0..n-1")
    inv = [None] * n  # inv[original_idx] = shuffled_idx
    for shuffled_idx, original_idx in enumerate(index_map):
        inv[original_idx] = shuffled_idx
    return inv


def parse_int_or_none(s):
    try:
        return int(s)
    except Exception:
        return None


def reorder_columns_by_map(input_csv: str, index_map, output_csv: str):
    """
    Column reordering:
    - Unnamed columns first (keep, with empty header "")
    - Mapped columns next (reordered to original order, renamed to '0'..'n-1')
    - Extra columns last (original names)
    """
    df_in = pd.read_csv(input_csv)
    n = len(index_map)

    # Inverse permutation: original_idx -> shuffled_idx
    inv = ensure_permutation(index_map, n)

    # Unnamed columns (empty header or 'Unnamed: ...')
    unnamed_cols = [
        col
        for col in df_in.columns
        if str(col).strip() == "" or str(col).lower().startswith("unnamed")
    ]

    # Map available shuffled-index columns to labels (accept '0','1',... or integer names)
    col_int_to_label = {}
    for col in df_in.columns:
        key = parse_int_or_none(col)
        if key is not None and 0 <= key < n and key not in col_int_to_label:
            col_int_to_label[key] = col

    # Verify all mapped columns exist
    missing = [inv[i] for i in range(n) if inv[i] not in col_int_to_label]
    if missing:
        raise ValueError(
            f"Input CSV missing {len(missing)} mapped columns "
            f"(expected column names matching shuffled indices 0..{n-1}). "
            f"Examples: {missing[:10]}"
        )

    # Mapped column order: for each original index k, take column whose name == inv[k]
    mapped_order = [col_int_to_label[inv[k]] for k in range(n)]

    mapped_set = set(mapped_order)
    unnamed_set = set(unnamed_cols)
    extras = [c for c in df_in.columns if c not in mapped_set and c not in unnamed_set]

    # Final ordering of labels
    final_order = unnamed_cols + mapped_order + extras
    df_out = df_in.loc[:, final_order]

    # ---- Rename columns:
    # Unnamed columns -> "" (empty header), Mapped columns -> '0','1',...,'n-1', Extras -> unchanged
    unnamed_new = [""] * len(unnamed_cols)
    mapped_new = [str(k) for k in range(n)]
    extras_new = extras  # keep original names for extras

    df_out.columns = unnamed_new + mapped_new + extras_new

    # Save WITHOUT index to avoid creating an 'Unnamed: 0' file column
    df_out.to_csv(output_csv, index=False)
    print(f"Saved column-reordered CSV to {output_csv}")
    print(
        f"- Unnamed columns kept at beginning ({len(unnamed_cols)}), headers set to empty"
    )
    print(f"- Reordered and RENAMED {len(mapped_order)} mapped columns to 0..{n-1}")
    print(f"- Extra columns kept at end ({len(extras)})")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reorder COLUMNS of an input CSV to match ORIGINAL order derived from "
            "original vs shuffled rows; keep unnamed columns first with empty header and "
            "rename mapped columns to 0..n-1."
        )
    )
    parser.add_argument("original", help="Original CSV (reference row order)")
    parser.add_argument("shuffled", help="Shuffled CSV (same rows, different order)")
    parser.add_argument("input", help="Input CSV whose COLUMNS will be reordered")
    parser.add_argument("output", help="Path to save the column-reordered CSV")

    args = parser.parse_args()

    original = load_clean_csv(args.original)
    shuffled = load_clean_csv(args.shuffled)

    if len(original) != len(shuffled):
        raise ValueError(
            f"Row count mismatch: original={len(original)} vs shuffled={len(shuffled)}"
        )

    index_map = build_index_map(original, shuffled)
    reorder_columns_by_map(args.input, index_map, args.output)


if __name__ == "__main__":
    main()
