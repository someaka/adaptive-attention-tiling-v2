#!/bin/bash
export VIRTUAL_ENV="/home/d/Desktop/AAT/adaptive-attention-tiling-v2/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHON_HOME

# Print some debug info
echo "Active Python: $(which python)"
echo "Python version: $(python --version)"
echo "Virtual env: $VIRTUAL_ENV"

exec bash
