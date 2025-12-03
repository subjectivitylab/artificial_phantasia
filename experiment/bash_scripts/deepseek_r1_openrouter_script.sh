#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250920"

python imagine_llms_library.py --models deepseek/deepseek-r1-0528 --data_path llm_ins.csv --out_path "deepseek-r1-0528_${DATE}.csv" --api_path_deepseek deepseek_api_key >> "deepseek-r1-0528_${DATE}.txt"