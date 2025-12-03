#!/usr/bin/env python3
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collate unique label responses by block (1-60) into a wide CSV where "
            "columns are block numbers and rows are unique labels per block."
        )
    )
    parser.add_argument("input_csv", help="Path to input CSV (with Model,label,raw,block columns)")
    parser.add_argument("output_csv", help="Path to output CSV")
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column to collate (default: 'label')",
    )
    parser.add_argument(
        "--block-column",
        default="block",
        help="Name of the block column (default: 'block')",
    )

    args = parser.parse_args()

    # Prepare storage: for each block 1..60, keep list (for order) and set (for uniqueness)
    blocks_labels = {b: [] for b in range(1, 61)}
    blocks_seen = {b: set() for b in range(1, 61)}

    # Read input CSV
    with open(args.input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Basic sanity check on columns
        if args.label_column not in reader.fieldnames:
            raise ValueError(
                f"Label column '{args.label_column}' not found in CSV header: {reader.fieldnames}"
            )
        if args.block_column not in reader.fieldnames:
            raise ValueError(
                f"Block column '{args.block_column}' not found in CSV header: {reader.fieldnames}"
            )

        for row in reader:
            label = (row.get(args.label_column) or "").strip()
            block_val = row.get(args.block_column)

            if not label:
                continue  # skip empty labels

            if block_val is None or block_val == "":
                continue  # skip rows without a block

            try:
                block = int(block_val)
            except ValueError:
                continue  # skip non-integer block values

            if not (1 <= block <= 60):
                continue  # skip blocks outside 1..60

            if label not in blocks_seen[block]:
                blocks_seen[block].add(label)
                blocks_labels[block].append(label)

    # Determine max number of labels in any block
    max_len = max((len(v) for v in blocks_labels.values()), default=0)

    # Build output rows
    # Row 0: column headers "1","2",...,"60"
    header = [str(i) for i in range(1, 61)]
    rows = [header]

    # Subsequent rows: labels per block, padded with empty strings as needed
    for i in range(max_len):
        row = []
        for block in range(1, 61):
            labels_for_block = blocks_labels[block]
            if i < len(labels_for_block):
                row.append(labels_for_block[i])
            else:
                row.append("")
        rows.append(row)

    # Write output CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
