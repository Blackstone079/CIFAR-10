# Experiment Log

## 000 - Initial working CIFAR-10 pipeline
- Custom 4-stage residual CNN built from scratch for learning.
- No augmentation.
- Initial training pipeline written and tested.
- Logging/checkpointing later added for Colab robustness.

## 001 - custom4stage_noaug_bs64_recovered
- Purpose: first complete recovered baseline from the custom 4-stage learning-first residual model.
- Best recovered test accuracy: 0.8661 at epoch 190
- Final test accuracy: 0.8633 at epoch 200
- Interpretation: valid recovered baseline, but not a canonical CIFAR benchmark.

## 002 - resnet20_aug_bs64
- Purpose: establish the proper CIFAR-style benchmark anchor.
- Model: ResNet-20
- Recipe: standard CIFAR augmentation, batch size 64, SGD + momentum, MultiStepLR, 200 epochs
- Best visible evaluated test accuracy: 0.9216
- Final test accuracy: 0.9200
- Interpretation: current accuracy anchor.

## 003 - resnet8_aug_bs64
- Purpose: test a much smaller plain CIFAR ResNet under the same recipe.
- Model: ResNet-8 with base width 16
- Best test accuracy: 0.8671
- Final test accuracy: 0.8636
- Interpretation: plain narrow ResNet-8 is too weak under the current recipe.

## 004 - objective update
- Supervisor clarified that the main optimization target is not training time.
- New objective: minimize trainable weights subject to best test accuracy >= 91%.

## 005 - next planned candidate
- Next candidate: ResNet8-w24
- Reason: smallest plausible width increase that may recover enough accuracy while respecting the new weight-minimization objective.
- Escalation rule:
  - if close but below 91%: distill w24
  - if clearly below 91%: try w28
