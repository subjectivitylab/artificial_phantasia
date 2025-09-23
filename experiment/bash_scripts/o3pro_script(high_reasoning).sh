#!/bin/bash

# Run in conda ImagineLLMs environment

DATE="20250916"

python imagine_llms_library.py --models o3-pro-2025-06-10 --data_path llm_ins.csv --out_path "o3pro_${DATE}_(high_reasoning).csv" --api_path_openai openai_api_key --reasoning True --reasoning_level high --context_variant 1 >> "o3pro_${DATE}_(high_reasoning).txt"

python deshuffle_results.py llm_ins_preshuffle.csv llm_ins.csv "o3pro_${DATE}_(high_reasoning).csv" "o3pro_${DATE}_(high_reasoning)_deshuffled.csv"

python deshuffle_results.py llm_ins_preshuffle.csv llm_ins.csv "o3pro_${DATE}_(high_reasoning)_usage.csv" "o3pro_${DATE}_(high_reasoning)_usage_deshuffled.csv"