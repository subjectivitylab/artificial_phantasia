#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250920"

python imagine_llms_library.py --models qwen/qwen3-vl-235b-a22b-thinking --data_path llm_ins.csv --out_path "qwen3-vl-235b-a22b-thinking_${DATE}.csv" --api_path_openrouter qwen_api_key >> "qwen3-vl-235b-a22b-thinking_${DATE}.txt"