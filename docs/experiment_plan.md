# Experiment Plan

## Current anchor
- ResNet-20
- standard CIFAR augmentation
- batch size 64
- SGD + momentum
- MultiStepLR
- 200 epochs

## What we learned
- Plain ResNet-8 is too weak under the same recipe.
- Shrinking depth alone did not buy enough wall-clock improvement.
- This suggests the next search direction should be small-model capacity tuning, not more depth reduction.

## Next architecture plan
1. Keep the same CIFAR ResNet-8 skeleton:
   - same stem
   - same 3 stages
   - same one block per stage
2. Introduce width scaling:
   - ResNet8-w16 (current plain model)
   - ResNet8-w24
   - ResNet8-w28

## Measurement plan
For each candidate:
- parameter count
- inference latency
- best test accuracy
- final test accuracy
- training runtime

## Decision rule
A candidate is interesting only if it is non-dominated:
- not larger and less accurate
- not slower and less accurate

## Escalation rule
If the best width-scaled small model is still below the acceptable accuracy band, then the next move is distillation from the ResNet-20 teacher.
