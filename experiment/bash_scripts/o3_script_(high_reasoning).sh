#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250915"

python imagine_llms_library.py --models o3-2025-04-16 --data_path llm_ins.csv --out_path "o3_${DATE}_(high_reasoning).csv" --api_path_openai openai_api_key --context_variant 1 --reasoning True --reasoning_level high >> "o3_${DATE}_(high_reasoning).txt"