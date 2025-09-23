#!/bin/bash

# Run in conda Artificial_Phantasia environment

python imagine_llms_library.py --models o3-2025-04-16 --data_path llm_ins.csv --out_path o3_20250725_2_images.csv --api_path_openai openai_api_key --api_path_gemini gemini_api_key --context_variant 0 --images True --reasoning True >> o3_20250725_2_images.txt