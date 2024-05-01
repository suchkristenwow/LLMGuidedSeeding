#!/bin/bash

# Check for the skip cloning option
clone=true  # Default to cloning
if [[ "$1" == "--skip-clone" ]]; then
  clone=false
fi

if [ "$clone" = true ]; then
  # Clone the GLIP repository
  git clone https://github.com/arpg/GLIP
fi
cd GLIP

# Install PyTorch and related libraries
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121  torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install additional Python libraries
pip install scikit-learn
pip install nltk
pip install inflect
pip install transformers
pip install pycocotools

# Install other dependencies
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo

# Install GLIP
python setup.py build develop --user

# Create a directory for the model
mkdir MODEL

# Download the GLIP model
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth -O MODEL/glip_large_model.pth

# Additional dependencies for the server
pip install flask opencv-python-headless
