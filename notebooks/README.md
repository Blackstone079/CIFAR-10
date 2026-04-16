# Notebooks

This folder is for thin launch notebooks and lightweight exploration notebooks.

## Intended use
- mount Google Drive
- clone or pull the repo
- run `train.py` with a chosen config
- do light result inspection

## Not intended use
- large training logic
- duplicated model definitions
- duplicated logging logic

The source of truth for training should remain in the Python files at repo level, not inside notebooks.
