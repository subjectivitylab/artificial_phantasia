#!/bin/bash

# Run in conda Artificial_Phantasia environment

DATE="20250725"

python imagine_llms_library.py --models gemini-2.0-flash-preview-image-generation --data_path llm_ins.csv --out_path "gemini-2p0_flash_${DATE}_images.csv" --api_path_openai openai_api_key --api_path_gemini gemini_api_key --context_variant 1 --reasoning True --images True >> "gemini-2p0_flash_${DATE}_images.txt"