#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250911"

python imagine_llms_library.py --models claude-sonnet-4-20250514 --data_path llm_ins.csv --out_path "claude_sonnet_4_${DATE}.csv" --api_path_claude claude_api_key --context_variant 1 >> "claude_sonnet_4_${DATE}.txt"