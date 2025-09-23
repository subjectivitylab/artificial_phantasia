#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250916"

python imagine_llms_library.py --models gpt-5-2025-08-07 --data_path llm_ins.csv --out_path "gpt5_${DATE}_(med_reasoning).csv" --api_path_openai openai_api_key --api_path_gemini gemini_api_key --reasoning True --reasoning_level medium --context_variant 1 >> "gpt5_${DATE}_(med_reasoning).txt"

python deshuffle_results.py llm_ins_preshuffle.csv llm_ins.csv "gpt5_${DATE}_(med_reasoning).csv" "gpt5_${DATE}_(med_reasoning)_deshuffled.csv"

python deshuffle_results.py llm_ins_preshuffle.csv llm_ins.csv "gpt5_${DATE}_(med_reasoning)_usage.csv" "gpt5_${DATE}_(med_reasoning)_usage_deshuffled.csv"