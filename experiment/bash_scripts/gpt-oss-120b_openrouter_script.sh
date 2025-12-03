#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250920"

python imagine_llms_library.py --models openai/gpt-oss-120b --data_path llm_ins.csv --out_path "gpt-oss-120b_${DATE}.csv" --api_path_openrouter gpt_oss_api_key >> "gpt-oss-120b_${DATE}.txt"