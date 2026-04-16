# Objective

## Task
CIFAR-10 image classification.

## Working goal
Find the most efficient model that still stays in the acceptable accuracy band.

## Current interpretation of efficiency
Primary metrics:
1. inference latency
2. parameter count

Secondary metrics:
1. training runtime
2. checkpoint / run robustness
3. memory footprint

## Current accuracy rule
Use the best verified ResNet-20 full-run result as the anchor.
A candidate survives only if it stays close enough to that anchor to remain practically competitive.

This rule can be tightened or relaxed later, but every experiment should be judged against an explicit threshold rather than vague intuition.

## Experimental discipline
- Change one serious variable at a time.
- Keep the training recipe fixed unless the recipe itself is the object of the experiment.
- Record negative results, not just successful ones.
- Do not treat training runtime as the primary deployment-efficiency metric unless that is explicitly the target.
