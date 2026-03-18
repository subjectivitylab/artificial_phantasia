# Phantasia Qualtrics Conversion Script

Convert a wide-format response CSV (one stimulus per column) into a Qualtrics
Advanced Format `.txt` file containing multiple-choice survey questions with
embedded images.

## What This Repository Contains

- `csv2qualtrics.py`: Main converter script.
- `phantasia_responses.csv`: Example/source CSV.
- `phantasia_missing_responses.csv`: Example/source CSV.
- `rebuttal_missing_responses.csv`: Example/source CSV.
- `final_missing_responses.csv`: Example/source CSV.

## How It Works

The converter reads each CSV column as one stimulus group:

- Row 1: Block ID (for question ID tagging).
- Row 2: Image URL used in every question for that column.
- Row 3 onward: Labels/responses; each non-empty cell becomes one Qualtrics
  `[[Question:MC]]` item.

For every label, the script builds a prompt like:

`How well does this image represent: "<label>"?`

with the image and these fixed response choices:

1. Not at all
2. A little
3. Moderately
4. A lot
5. Completely

## Requirements

- Python 3.9+ (3.10+ recommended)
- `pandas`

Install dependency:

```bash
python -m pip install pandas
```

## Usage

```bash
python csv2qualtrics.py <input.csv> <output.txt> [--q_start N]
```

Arguments:

- `csv_in`: Input CSV in the required wide format.
- `txt_out`: Output Qualtrics Advanced Format text file.
- `--q_start`: Optional starting question counter (default: `1`).

Example:

```bash
python csv2qualtrics.py phantasia_missing_responses.csv phantasia_missing_responses.txt
```

Example with custom starting number:

```bash
python csv2qualtrics.py rebuttal_missing_responses.csv rebuttal_missing_responses.txt --q_start 1001
```

## Expected Input Format

Each column should follow this structure:

| Row | Meaning |
|---|---|
| 1 | Block identifier (e.g., `1`, `2`, `A`) |
| 2 | Image URL |
| 3+ | Free-text labels (one per row, blank rows allowed) |

Notes:

- The script reads with `header=None`, so all rows are treated as data.
- Leading/trailing spaces are stripped from block IDs, image URLs, and labels.
- Empty labels are skipped.

## Output Format Details

The output starts with:

```text
[[AdvancedFormat]]
[[Block:MainBlock]]
```

Then one `[[Question:MC]]` block per label.

Question IDs are generated as:

`b<block>_q<counter>_<label>`

Example ID:

`b12_q57_anchor`

Important behavior:

- The block value is included in the question ID.
- Questions are currently written under `MainBlock` (per-column block splitting
  is not emitted in output).

## Importing into Qualtrics

1. Run the converter to produce the `.txt` file.
2. In Qualtrics, create/open a survey.
3. Use the import option for **Advanced Format** text.
4. Paste or upload the generated file content.
5. Verify random questions for image rendering and choice order.

## Customization

Edit `csv2qualtrics.py` to customize behavior:

- `CHOICES` list: response options and order.
- `build_question(...)`: prompt text, image HTML, dimensions (`width/height`).
- `csv_to_qualtrics(...)`: block handling and ID format.

## Troubleshooting

- `ModuleNotFoundError: No module named 'pandas'`
  - Install dependency: `python -m pip install pandas`
- Images not visible in Qualtrics
  - Verify URLs are publicly accessible to survey respondents.
- Unexpected question IDs
  - IDs include raw label text; punctuation/spaces in labels will appear in IDs.
- Too many/few generated questions
  - Check for hidden non-empty cells in rows 3+ of your CSV.
