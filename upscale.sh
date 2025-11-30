#!/bin/bash
# Convenient wrapper script for the bulk image upscaler

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "$SCRIPT_DIR/upscale.py" "$@"
