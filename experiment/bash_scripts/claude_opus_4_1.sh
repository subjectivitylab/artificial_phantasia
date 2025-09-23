#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250911"

python imagine_llms_library.py --models claude-opus-4-1-20250805 --data_path llm_ins.csv --out_path "claude_opus_4_1_${DATE}.csv" --api_path_claude claude_api_key --context_variant 1 >> "claude_opus_4_1_${DATE}.txt"