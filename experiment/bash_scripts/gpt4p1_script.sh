#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250721"

python imagine_llms_library.py --models gpt-4.1-2025-04-14 --data_path llm_ins.csv --out_path "gpt4p1_${DATE}.csv" --api_path_openai openai_api_key --api_path_gemini gemini_api_key >> "gpt4p1_${DATE}.txt"