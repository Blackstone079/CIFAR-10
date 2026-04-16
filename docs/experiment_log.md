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
- Purpose: establish the proper CIFAR-style benchmark anchor.
- Model: `models/resnet20_cifar.py`
- Training recipe: standard CIFAR augmentation, batch size 64, SGD + momentum, MultiStepLR.
- Best visible evaluated test accuracy: **0.9216**
- Final test accuracy: **0.9200**
- Status: benchmark result recovered from notebook output; artifact persistence failed in the older Drive-writing workflow.
- Interpretation: this is the current accuracy anchor.

## 003 - resnet8_aug_bs64
- Purpose: test whether a much smaller plain ResNet can preserve competitive accuracy while improving efficiency.
- Model: `models/resnet8_cifar.py`
- Training recipe: same as the ResNet-20 anchor.
- Best test accuracy: **0.8671**
- Final test accuracy: **0.8636**
- Status: run artifacts saved correctly under the local-write plus Drive-mirror workflow.
- Interpretation: plain narrow ResNet-8 is too weak under the current recipe; next move is to test a width-scaled ResNet-8 family rather than shrinking depth further.
