#!/bin/bash
set -e  # Exit on any error

# Default values
SOURCE_ROOT="path/to/data/train/raw"
TARGET_ROOT="path/to/data/train/preprocessed"
PROCESSES=8

python lib/data_preprocessing.py \
    --source_root "$SOURCE_ROOT" \
    --target_root "$TARGET_ROOT" \
    --processes "$PROCESSES"
