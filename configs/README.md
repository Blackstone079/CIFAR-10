# Configs

Each JSON file defines one runnable experiment configuration.

## Naming convention
- `debug_*`: fast smoke tests
- `debug_persistence_*`: save-path stress tests
- `resnet20_*`, `resnet8_*`, etc.: serious benchmark / search runs

## Required fields
- `run_name`
- `model_name`
- `batch_size`
- `test_batch_size`
- `epochs`
- `lr`
- `momentum`
- `weight_decay`
- `scheduler`
- `milestones`
- `gamma`
- `eval_every`
- `data_root`
- `run_root`
- `drive_run_root`
- `num_workers`
- `pin_memory`
- `augmentation`
- `seed`

## Rule
`run_root` must always be local (for example `/content/CIFAR10_runs/full`).
`drive_run_root` is the mirror destination on Google Drive.
