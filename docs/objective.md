# Objective

## Task
CIFAR-10 image classification.

## Optimization problem
Minimize trainable parameter count subject to:

- best test accuracy >= 91%

## Primary metric
- trainable parameter count

## Secondary tie-breakers
- inference latency
- training runtime

## Experimental rule
- Change one serious variable at a time.
- Keep the training recipe fixed unless the training recipe itself is the thing being tested.
- Record negative results, not just positive ones.

## Current status
- ResNet-20 is the current high-accuracy anchor.
- Plain ResNet-8 is too small under the same recipe.
- Next step is the smallest plausible width increase: ResNet8-w24.
