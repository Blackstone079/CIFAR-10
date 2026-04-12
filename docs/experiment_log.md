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
