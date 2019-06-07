#!/usr/bin/env bash

export CUDA_ROOT=/usr/local/cuda
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64/"
export PATH="/usr/local/cuda/bin:$PATH"
export PATH="/usr/local/cuda-9.0/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

optirun python -m src.controller