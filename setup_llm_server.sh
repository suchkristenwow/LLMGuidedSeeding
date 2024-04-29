#!/bin/bash

# Step 1: Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Step 2: Build the repository
make LLAMA_CUBLAS=1

# Step 3: Create the models/cog_explore directory if it doesn't exist
mkdir -p models/cog_explore

# Step 4: Download the specified files
cd models/cog_explore
wget https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/resolve/main/llama-2-70b-chat.Q4_K_M.gguf
wget https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
wget https://huggingface.co/TheBloke/WizardLM-70B-V1.0-GGUF/resolve/main/wizardlm-70b-v1.0.Q4_K_M.gguf
wget https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K.gguf
wget https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-mmproj-f16.gguf

echo "All tasks completed."

