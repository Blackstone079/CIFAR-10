# Experiment Plan

## Current anchor
- ResNet-20
- standard CIFAR augmentation
- batch size 64
- SGD + momentum
- MultiStepLR
- 200 epochs

## What we learned
- Plain narrow ResNet-8 is too weak under the same recipe.
- Training runtime is not the optimization target.
- Weight count is the optimization target.

## Next experiment
- ResNet8-w24
- same recipe as the ResNet-20 anchor
- only change: base width from 16 to 24

## Decision rule
- If ResNet8-w24 reaches >= 91%: it survives and becomes the leading small-model candidate.
- If ResNet8-w24 is close but below 91%: try distillation on w24 before increasing width again.
- If ResNet8-w24 is clearly below 91%: try ResNet8-w28.

## What not to change yet
- augmentation
- scheduler
- batch size
- epoch count
- optimizer
