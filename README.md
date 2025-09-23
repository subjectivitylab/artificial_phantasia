# Artificial Phantasia: Evidence for Propositional Reasoning-Based Mental Imagery in Large Language Models
Experiments on mental imagery in Large Language Models

## Usage
Create two conda environments: one R environment using the ```r_env.yml``` file, and one Python environment using the ```python_env.yml``` file.

From there create an instruction set (or use the provided one), and go into the experimental code. You can shuffle the instruction set (which we recommend if you use single context) using the ```single_shuffle.ipynb``` jupyter notebook.

Next make sure you have api keys loaded for the family of model you wish to use. Models must be added to ```imagine_llms_library.py``` in the ```determine_family_from_model``` function.

Either create a bash script in the format we have provided, or manually run the library file:

```
usage: imagine_llms_library.py [-h] --models MODELS [MODELS ...] --data_path DATA_PATH --out_path OUT_PATH [--api_path_openai API_PATH_OPENAI] [--api_path_gemini API_PATH_GEMINI] [--api_path_claude API_PATH_CLAUDE] [--context_variant CONTEXT_VARIANT] [--images IMAGES] [--reasoning REASONING] [--reasoning_level {minimal,low,medium,high}]

Run Imagine LLMs Library

options:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
                        List of model names to use
  --data_path DATA_PATH
                        Path to the input data CSV file
  --out_path OUT_PATH   Path to save the output CSV file
  --api_path_openai API_PATH_OPENAI
                        Path to OpenAI API key file
  --api_path_gemini API_PATH_GEMINI
                        Path to Gemini API key file
  --api_path_claude API_PATH_CLAUDE
                        Path to Claude API key file
  --context_variant CONTEXT_VARIANT
                        Context variant for instruction processing
  --images IMAGES       Whether to generate images instead of text
  --reasoning REASONING
                        Whether to use reasoning in OpenAI responses
  --reasoning_level {minimal,low,medium,high}
                        Reasoning level for OpenAI responses
```

Shuffled results will be outputed to the provided path. To deshuffle them use your original instruction set, shuffed instruction set, results, and a new file path in ```deshuffle_results.py```.

```
usage: deshuffle_results.py [-h] original shuffled input output

Reorder COLUMNS of an input CSV to match ORIGINAL order derived from original vs shuffled rows; keep unnamed columns first with empty header and rename mapped columns to 0..n-1.

positional arguments:
  original    Original CSV (reference row order)
  shuffled    Shuffled CSV (same rows, different order)
  input       Input CSV whose COLUMNS will be reordered
  output      Path to save the column-reordered CSV

options:
  -h, --help  show this help message and exit
```

Transfer results to the data analysis pipeline to continue.

We do not provided slot-in data analysis at this time due to the subjective nature of the responses. In order to analyze the data you must create a qualtrics survey of any unique answers using the Qualtrics conversion script ```csv2qualtrics.py```.

```
python(ImagineLLMs) [morgan@Morgan phantasia_qualtrics_conversion_script]$ python csv2qualtrics.py --help
usage: csv2qualtrics.py [-h] [--q_start Q_START] csv_in txt_out

Convert CSV to Qualtrics TXT

positional arguments:
  csv_in             input CSV file
  txt_out            output TXT file

options:
  -h, --help         show this help message and exit
  --q_start Q_START  starting question number
```

You can follow the procedure of our data analysis by viewing ```data_analysis.html``` or the raw Jupyter Notebook file ```data_analysis.ipynb```.

## Dataset

In ```data_analysis``` we have provided our dataset of LLM data (```llm_phantasia_data.csv```, which is also available in raw form in ```experiment```), our responses to the experiment from humans (```human_phantasia_data.csv```), our crowdsourced rating responses (```expert_response_ranking_data_primary.csv```, ```expert_response_ranking_secondary.csv```, ```human_response_ranking_data_primary.csv```, ```human_response_ranking_data_secondary.csv```), and our instructions and metadata related to them (```meta_instruction_data.csv```, the instructions are also included in ```experiment```).

Our data processing pipeline automatically grades the ```llm_phantasia_data.csv``` and ```human_phantasia_data.csv``` files using the provided response ranking data (in the specific files mentioned). This process is extensible, through concatenating more response ranking data into the crowdsourced data DataFrames, but not automatic (this would have to be manually done).

The data processing Jupyter Notebook outputs a series of `.csv` files with a number of different charateristics. Notably, the individual gradings collapsed by person (```h_graded_results.csv```), the entire set of gradings for humans (each individual grade, `h_full_results.csv`), the same for LLMs (`llm_graded_data.csv`, `llm_full_results.csv`), VVIQ data from the participants (`vviq_scores.csv`), aggregated LLM data (`llm_aggregate_results.csv`), the tidied crowdsourcing results (`tidy_crowdsourced_data.csv`, `tidy_expert_data.csv`), the grading breakdowns for each response (`means_with_canon.csv`), ungraded responses (`h_ungraded_results.csv`, `llm_ungraded_results.csv`), difficulty data (`difficulty_per_item.csv`, `difficulty_score_summary.csv`), and several sub-datasets for interesting comparisons (`reasoning_effort` tuning: `openai_reasoning_comparison_results.csv`, Single vs. Multiple-Context: `single_vs_multiple_context_results.csv`, the Aphantasic human subject: `aphant.csv`).

This data is read by the R Markdown file to perform statistical tests and output useful charts. The results of the statistical analyses can be found in `proportion_test_detailed_summary.csv` and `proportion_test_results.csv`.

## Further Information

For further information please refer to our paper: _Artificial Phantasia: Evidence for Propositional Reasoning-Based Mental Imagery in Large Language Models_
