# Experiment Log

## 000 - Initial working CIFAR-10 pipeline
- Custom 4-stage residual CNN built from scratch for learning.
- No augmentation.
- Initial training pipeline written and tested.
- Logging/checkpointing later added for Colab robustness.

## 001 - custom4stage_noaug_bs64_recovered
- Purpose: first complete recovered baseline from the custom 4-stage learning-first residual model.
- Model: `models/resnet_cifar_custom4stage.py`
- Training recipe: no augmentation, batch size 64, SGD + momentum, MultiStepLR.
- Best recovered test accuracy: **0.8661** at epoch **190**
- Final test accuracy: **0.8633** at epoch **200**
- Status: metrics recovered from notebook output; no weights/checkpoints saved because the Google Drive `train.py` version was outdated.
- Interpretation: valid baseline learning curve, but not a resumable run and not a canonical CIFAR benchmark.

## 002 - resnet20_aug_bs64
- Purpose: establish the proper CIFAR benchmark anchor and teacher run.
- Model: `models/resnet20_cifar.py`
- Training recipe: augmentation, batch size 64, SGD + momentum, MultiStepLR.
- Best test accuracy: **0.9252** at epoch **165**
- Final test accuracy: **0.9221** at epoch **200**
- Interpretation: current teacher / reference run.

## 003 - objective update
- Primary objective: minimize trainable weights subject to meeting the required test accuracy target.
- Current target in practice: **>= 0.92** test accuracy.

## 004 - resnet8w24_kd_aug_bs64
- Purpose: test whether KD can lift a smaller student over the accuracy threshold.
- Model: `resnet8` with `base_width=24`
- Teacher: `resnet20_aug_bs64/best.pt`
- KD settings: **alpha=0.5**, **T=4.0**
- Training recipe: augmentation, batch size 64, SGD + momentum, MultiStepLR.
- Best test accuracy: **0.9104** at epoch **165**
- Final test accuracy: **0.9058** at epoch **200**
- Interpretation: successful proof that KD works in-repo and gets the student above 91%, but still below the 92% bar.

## 005 - next planned candidate
- Candidate: `resnet14w16_kd_aug_bs64`
- Rationale: parameter-efficiency probe at roughly the same scale as `resnet8w24`, but with more depth instead of more width.
