#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250721"

python imagine_llms_library.py --models gemini-2.0-flash --data_path llm_ins.csv --out_path "gemini-2p0_flash_${DATE}.csv" --api_path_openai openai_api_key --api_path_gemini gemini_api_key --reasoning True >> "gemini-2p0_flash_${DATE}.txt"