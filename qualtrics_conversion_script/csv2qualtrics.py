"""
csv2qualtrics.py
----------------
Convert a wide CSV (one stimulus per *column*) into a Qualtrics-compatible
Advanced-format .txt file.

Input  layout (one example column):
  ┌──────────────┐
  │Row 0 │ Block │  ← numeric (or string) block identifier
  │Row 1 │ URL   │  ← image URL shown in every question in this column
  │Row 2 │ label │
  │Row 3 │ label │
  │ …    │ …     │
  └──────────────┘

Every label becomes its own MC-single-answer question:

   How well does this image represent a <label>? <br><br><br>
   <img …><br>&nbsp;
   Not at all | A little | A moderate amount | Completely

Usage
-----
$ python csv2qualtrics.py input.csv output.txt
"""
import argparse
import pandas as pd
from pathlib import Path
from textwrap import dedent

CHOICES = [
    "Not at all",
    "A little",
    "Moderately",
    "A lot",
    "Completely",
]

def build_question(label: str, img_url: str, qid: str) -> str:
    """Return one fully-formatted Qualtrics MC question block."""
    question_text = (
        f"How well does this image represent: \"{label}\"?&nbsp;<br>\n"
        "<br>\n<br>\n"
        f'<img src="{img_url}" alt="If you can\'t see this image, please leave the survey." '
        'width="350" height="350"><br>\n&nbsp;'
    )
    return dedent(f"""\
        [[Question:MC]]
        [[ID:{qid}_{label}]]
        {question_text}
        [[Choices]]
        {'\n'.join(CHOICES)}
        """)

def csv_to_qualtrics(df: pd.DataFrame, q_start: int = 1) -> str:
    """Convert the entire DataFrame into one Advanced-format TXT string."""
    lines = ["[[AdvancedFormat]]","[[Block:MainBlock]]"]
    current_block = None
    q_counter = q_start

    for col in df.columns:
        block = str(df.iloc[0, col]).strip()
        img_url = str(df.iloc[1, col]).strip()

        # start a new block when the block ID changes
        if block != current_block:
            #lines.append(f"[[Block:Block {block}]]")
            current_block = block

        # each non-blank label (rows 2+) becomes a question
        for label in df.iloc[2:, col].dropna():
            label = str(label).strip()
            if label:
                lines.append(build_question(label, img_url, f"b{block}_q{q_counter}"))
                q_counter += 1

    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to Qualtrics TXT")
    parser.add_argument("csv_in", type=Path, help="input CSV file")
    parser.add_argument("txt_out", type=Path, help="output TXT file")
    parser.add_argument("--q_start", type=int, default=1, help="starting question number")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_in, header=None, dtype=str, keep_default_na=False)
    qualtrics_txt = csv_to_qualtrics(df, q_start=args.q_start)
    args.txt_out.write_text(qualtrics_txt, encoding="utf-8")
    print(f"Wrote {args.txt_out} with {qualtrics_txt.count('[[Question:MC]]')} questions.")

if __name__ == "__main__":
    main()
