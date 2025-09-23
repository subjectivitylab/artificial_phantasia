#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250721"

python imagine_llms_library.py --models o3-2025-04-16 --data_path llm_ins.csv --out_path "o3_${DATE}.csv" --api_path_openai openai_api_key --api_path_gemini gemini_api_key --reasoning True >> "o3_${DATE}.txt"