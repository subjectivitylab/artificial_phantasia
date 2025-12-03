#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250919"

python imagine_llms_library.py --models gemini-3-pro-preview --data_path llm_ins.csv --out_path "gemini-3_pro_r-high_t-0p55-${DATE}.csv" --api_path_gemini gemini_api_key --temperature 0.55 >> "gemini-3_pro_r-high_t-0p55-${DATE}.txt"