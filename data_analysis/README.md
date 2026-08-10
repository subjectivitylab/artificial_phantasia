# phantasia_data_analysis

This repository contains the implementation of the analyses associated with the paper Artificial Phantasia.

## Authoritative Scope

The repository is the primary source of truth for methods, scripts, and outputs.
If any detail in the paper conflicts with this codebase, follow this codebase.

Tests in `incorrect_tests/` are historical and incorrect; corrected inferential tests are in:

- `mann-whitney-u/`
- `wilcoxon-signed-rank/`

## Repository Structure

- `input_csvs/`: raw inputs (human responses, LLM responses, ranking annotations, metadata).
- `output_csvs/`: notebook-produced intermediate and final analysis tables used by downstream tests.
- `mann-whitney-u/`: corrected non-parametric inferential comparison scripts and rendered reports.
- `wilcoxon-signed-rank/`: corrected paired signed-rank comparison script and rendered report.
- `incorrect_tests/`: archived, incorrect historical statistical pipeline and rendered artifacts.
- `statistical_results/`: outputs from the archived historical test pipeline.
- `data_analysis.ipynb`: main Python analysis and data-preparation pipeline.
- `collate_blocks.py`: utility script for collation of ungraded labels by block.
- `*_environment.yml` and `EXACT_*_env.yml`: reproducible Python and R environments (unpinned/fully pinned variants).

## Environment Setup

Recommended (editable dependency ranges):

```bash
conda env create -f python_environment.yml
conda env create -f r_environment.yml
```

Pinned variants (for exact reproducibility):

- `EXACT_python_env.yml`
- `EXACT_r_env.yml`

## Reproduction Order

1. Run the Python notebook pipeline to generate/update `output_csvs/`.
2. Run corrected R inferential scripts in `mann-whitney-u/` and `wilcoxon-signed-rank/`.
3. Use `incorrect_tests/` only for historical reference.

Example commands:

```bash
# from repository root
jupyter nbconvert --to notebook --execute --inplace data_analysis.ipynb

# utility script (optional)
python collate_blocks.py <input.csv> <output.csv> --label-column label --block-column block

# corrected R tests (run from each script directory so ../output_csvs paths resolve)
cd mann-whitney-u
Rscript -e "rmarkdown::render('humans_vs_models.rmd')"
Rscript -e "rmarkdown::render('reasoning_comparison.rmd')"
Rscript -e "rmarkdown::render('image_no_image_analysis.rmd')"
Rscript -e "rmarkdown::render('qwen_norm_vl_analysis.rmd')"
Rscript -e "rmarkdown::render('temperature_analysis.rmd')"

cd ../wilcoxon-signed-rank
Rscript -e "rmarkdown::render('sc_vs_mc.rmd')"
```

## Script Catalog

### Core Pipeline Scripts

| Script | Language | Purpose | Inputs | Primary Outputs |
|---|---|---|---|---|
| `data_analysis.ipynb` | Python | Main end-to-end data pipeline (tidying ranking data, building grading lookup, computing human/LLM scores, difficulty metrics, VVIQ parsing/correlation, family/model aggregations, grade-distribution exports for inferential tests). | `input_csvs/*.csv` | `output_csvs/*.csv` (full list below) |
| `collate_blocks.py` | Python | Collates unique ungraded labels by block (`1..60`) into a wide CSV (columns are blocks; rows are unique labels per block). | arbitrary CSV containing at least `label` and `block` columns | user-specified output CSV |

Supporting rendered notebook artifacts:

- `data_analysis.md`
- `data_analysis.html`
- `data_analysis_files/` (figures from notebook)

### Corrected Inferential Test Scripts (Authoritative)

| Script | Test Type | Hypothesis / Comparison | Inputs | Outputs |
|---|---|---|---|---|
| `mann-whitney-u/humans_vs_models.rmd` | Mann-Whitney U (`wilcox.test`, unpaired) + effect size | Human score distribution vs collapsed LLM-family distributions | `../output_csvs/h_grade_distribution.csv`, `../output_csvs/llm_collapsed_grade_dist.csv` | `human_vs_model_significance_summary.csv`, `humans_vs_models.html` |
| `mann-whitney-u/reasoning_comparison.rmd` | Mann-Whitney U (`wilcox.test`, unpaired) + effect size | Human distribution vs OpenAI reasoning-level comparison distributions | `../output_csvs/h_grade_distribution.csv`, `../output_csvs/openai_reasoning_comparison_grade_dist.csv` | `human_vs_reasoning_model_significance_summary.csv`, `reasoning_comparison.html` |
| `mann-whitney-u/image_no_image_analysis.rmd` | Mann-Whitney U (`wilcox.test`, unpaired) + effect size | All matched with-images vs without-images model distributions across collapsed, reasoning, and context-specific outputs | `../output_csvs/image_comparison_grade_dist.csv`, `../output_csvs/openai_reasoning_comparison_grade_dist.csv`, `../output_csvs/single_vs_multiple_context_grade_dist.csv` | `image_no_image_mann_whitney_summary.csv`, `image_no_image_analysis.html` |
| `mann-whitney-u/qwen_norm_vl_analysis.rmd` | Mann-Whitney U (`wilcox.test`, unpaired) + effect size | Qwen text-only vs Qwen-VL distribution comparison | `../output_csvs/qwen_comparison_grade_dist.csv` | `qwen_norm_vl_mann_whitney_summary.csv`, `qwen_norm_vl_analysis.html` |
| `mann-whitney-u/temperature_analysis.rmd` | Mann-Whitney U (`wilcox.test`, unpaired) | Pairwise Gemini 3 Pro temperature setting comparisons | `../output_csvs/gemini_3_pro_temp_grade_dist.csv` | `temperature_analysis.html` |
| `wilcoxon-signed-rank/sc_vs_mc.rmd` | Paired Wilcoxon signed-rank (`wilcox.test`, paired) + effect size | Single-context (`_sc`) vs multi-context (`_mc`) per shared model | `../output_csvs/single_vs_multiple_context_grade_dist.csv` | `sc_vs_mc.html` |

### Historical/Incorrect Script (Non-Authoritative)

| Script | Status | Implemented Tests | Inputs | Outputs |
|---|---|---|---|---|
| `incorrect_tests/statistical_analysis.Rmd` | Historical and marked incorrect in `incorrect_tests/README.md` | Pairwise two-sample proportion tests via `prop.test` (chi-squared statistic, CI), including comparisons to original Finke benchmarks | `../output_csvs/*.csv` tables from core pipeline | `../statistical_results/proportion_test_results.csv`, `../statistical_results/proportion_test_detailed_summary.csv`, rendered `statistical_analysis.html/.md/.pdf` |

## Statistical Tests Used

### Current/Authoritative Tests

1. Mann-Whitney U (Wilcoxon rank-sum, unpaired):
   - Implemented in `humans_vs_models.rmd`, `reasoning_comparison.rmd`, `image_no_image_analysis.rmd`, `qwen_norm_vl_analysis.rmd`, and `temperature_analysis.rmd`.
2. Wilcoxon signed-rank (paired):
   - Implemented in `sc_vs_mc.rmd`.
3. Multiple-comparison control:
   - Bonferroni and FDR (`p.adjust`).
4. Effect size:
   - Rank-biserial effect sizes via `rstatix::wilcox_effsize` (not used in `temperature_analysis.rmd`).
5. Exploratory correlation in the core notebook:
   - Pearson correlation matrices via `pandas.DataFrame.corr`.

### Historical/Deprecated Tests

1. Two-sample proportion tests (`prop.test` with continuity correction) in `incorrect_tests/statistical_analysis.Rmd`.
2. Archived outputs are in `statistical_results/`.

## Output Files Produced by `data_analysis.ipynb`

All files below are written by notebook cells and saved to `output_csvs/`.

| File | Purpose |
|---|---|
| `tidy_crowdsourced_data.csv` | Long-format crowd ranking data with block, qid, label, and numeric score. |
| `tidy_expert_data.csv` | Long-format expert ranking data with block, qid, label, and numeric score. |
| `means_with_canon.csv` | Per-(block, qid, label) grading lookup including crowd/expert means and canonical label. |
| `vviq_scores.csv` | Parsed numeric VVIQ per respondent (`VVIQ_sum`, mean, std, item-level). |
| `difficulty_per_item.csv` | Item-level difficulty components (clarity, identifiability, unique-response ratio, etc.). |
| `difficulty_score_summary.csv` | Weighted difficulty score and rank per canonical item. |
| `h_graded_results.csv` | Per-human aggregate scores over all blocks. |
| `h_full_results.csv` | Per-human per-item grading detail including matched label statistics. |
| `h_ungraded_results.csv` | Human labels not found in lookup (fallback handling). |
| `h_graded_results_finke.csv` | Human aggregates restricted to Finke blocks. |
| `h_graded_results_novel.csv` | Human aggregates restricted to novel blocks. |
| `llm_graded_results.csv` | Per-model aggregate scores over all blocks. |
| `llm_full_results.csv` | Per-model per-item grading detail including matched label statistics. |
| `llm_ungraded_results.csv` | Model labels not found in lookup (fallback handling). |
| `llm_graded_results_finke.csv` | Model aggregates restricted to Finke blocks. |
| `llm_graded_results_novel.csv` | Model aggregates restricted to novel blocks. |
| `llm_aggregate_results.csv` | Family/group aggregated model results (o3/gpt5/Anthropic/DeepMind/open models/other OpenAI). |
| `openai_reasoning_comparison_results.csv` | Aggregated reasoning-level comparison table for OpenAI model variants. |
| `gemini_3_pro_temperature_comparison_results.csv` | Aggregated temperature-variant results for Gemini 3 Pro. |
| `single_vs_multiple_context_results.csv` | Aggregated single-context vs multi-context results by mapped model variant. |
| `h_grade_distribution.csv` | Human item-level score distribution vector used by inferential R scripts. |
| `llm_grade_distribution.csv` | Raw model-specific item-level score distributions. |
| `llm_collapsed_grade_dist.csv` | Collapsed family-level item-level distributions used in human-vs-model tests. |
| `single_vs_multiple_context_grade_dist.csv` | Paired `_sc`/`_mc` distribution matrix for signed-rank testing. |
| `openai_reasoning_comparison_grade_dist.csv` | Distribution matrix for OpenAI reasoning-level comparisons. |
| `gemini_3_pro_temp_grade_dist.csv` | Distribution matrix for temperature-pair comparisons. |
| `image_comparison_grade_dist.csv` | Distribution matrix for image vs non-image comparisons. |
| `qwen_comparison_grade_dist.csv` | Two-column distribution matrix for Qwen vs Qwen-VL. |
| `aphant.csv` | Subset of human participants with `VVIQ_sum == 16`. |
| `human_correlation_summary.csv` | Correlation matrix across human performance and VVIQ/difficulty metrics. |

## Inferential Output Files

Generated by corrected R scripts:

- `mann-whitney-u/human_vs_model_significance_summary.csv`
- `mann-whitney-u/human_vs_reasoning_model_significance_summary.csv`
- `statistical_results/image_no_image_mann_whitney_summary.csv`
- `statistical_results/qwen_norm_vl_mann_whitney_summary.csv`

Generated by historical/incorrect pipeline:

- `statistical_results/proportion_test_results.csv`
- `statistical_results/proportion_test_detailed_summary.csv`

## Rendered Reports

These are generated artifacts for inspection and are not source scripts:

- `mann-whitney-u/*.html`
- `wilcoxon-signed-rank/sc_vs_mc.html`
- `incorrect_tests/statistical_analysis.html`
- `incorrect_tests/statistical_analysis.md`
- `incorrect_tests/statistical_analysis.pdf`
- `data_analysis.html`
- `data_analysis.md`
