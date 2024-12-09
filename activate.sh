#!/bin/bash

# Create a new shell with the virtual environment activated
export VIRTUAL_ENV="/home/d/Desktop/AAT/adaptive-attention-tiling-v2/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="/home/d/Desktop/AAT/adaptive-attention-tiling-v2/src:$PYTHONPATH"
unset PYTHON_HOME

echo "Virtual environment activated. Python path: $(which python)"
exec $SHELL
