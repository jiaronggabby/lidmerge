# LidMerge

LidMerge measures how a patient-level malignancy score changes as one to four
photographs are added from the same patient group. A shared torchvision image
encoder is trained with mean image logits and patient-level binary
cross-entropy. Mean, top-2 mean, and maximum aggregation use the same saved
image logits. Training uses four-epoch K=1,2,3,4 cycles and batches of four
patient groups. ConvNeXt-Tiny is the primary encoder; EfficientNetV2-M and
Swin-T provide architecture comparisons. XGBoost, random forest and logistic
regression provide additional score-aggregation comparisons.

Clinical photographs and patient records are not included; provide approved
data separately and keep it outside this directory.

## Installation

Use Python 3.10–3.12 with CUDA-enabled PyTorch. The tested reference
versions are `torch==2.5.1` and `torchvision==0.20.1`.

```bash
python -m pip install -r requirements.txt
```

The input CSV must contain `patient_group`, binary `label`, unique `image_id`,
`canonical_relative_path` (relative to `--image-root`), and `absolute_path`.
The study contract expects 2,720 images, 955 groups, and
252 groups with at least four images.

## Complete workflow

```bash
python run_experiment.py all --manifest /path/to/manifest.csv \
  --image-root /path/to/images --protocol-root ./outputs/protocol \
  --output-root ./outputs/runs --analysis-root ./outputs/learned
```

The workflow fits the image classifiers, selects training duration and
thresholds within the inner folds, evaluates the outer folds, and summarizes
the direct and learned aggregation methods. It includes 90 image-classifier
fits across five outer folds and three training seeds.

ImageNet weights can be downloaded by adding `--allow-weight-download`.
Use `CUDA_VISIBLE_DEVICES` to select the training GPU. The configuration in
`lidpair/contract.json` specifies the model settings, cross-validation folds,
aggregation candidates and 2,000 patient-level bootstrap samples.

## Individual commands

```bash
python run_experiment.py prepare --manifest /path/to/manifest.csv \
  --protocol-root ./outputs/protocol

python run_experiment.py train --protocol-root ./outputs/protocol \
  --image-root /path/to/images --output-root ./outputs/runs \
  --stage inner --outer-fold 0 --inner-fold 0 --seed 42 \
  --backbone convnext_tiny --pretrained

python run_experiment.py eval --predictions ./outputs/runs/.../predictions.csv \
  --budget 4 --variant mean
```

The evaluator requires one budget and variant when present and refuses files
that contain multiple seeds or duplicate patient groups.

```bash
python -m unittest discover -s tests -v
```
