#!/bin/bash

# Run in conda Artificial_Phantasia environment

python imagine_llms_library.py --models o4-mini-2025-04-16 --data_path llm_ins.csv --out_path o4mini_20250721.csv --api_path_openai openai_api_key --api_path_gemini gemini_api_key --reasoning True >> 04mini_20250721.txt
