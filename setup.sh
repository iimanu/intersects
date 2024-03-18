#!/bin/bash

eval "$(conda.bat shell.bash hook)"
conda update -y -n base -c defaults conda
conda create -y -n bbox_env
conda activate bbox_env

conda install -y numpy matplotlib pillow tqdm