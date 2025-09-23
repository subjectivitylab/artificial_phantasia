#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250725"

python imagine_llms_library.py --models chatgpt-4o-latest --data_path llm_ins.csv --out_path "chatgpt4o_${DATE}.csv" --api_path_openai openai_api_key --api_path_gemini gemini_api_key >> "chatgpt4o_${DATE}.txt"