#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use the venv python to run pytest
"${SCRIPT_DIR}/.venv/bin/python" -m pytest "$@"
