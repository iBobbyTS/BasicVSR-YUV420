# BasicVSR-YUV420

This repository is derived from the original Kaggle notebook at https://www.kaggle.com/code/thenujannagarathnam/basicvsr and reorganized into a standard Python deep learning project.

The original notebook metadata declared Python 3.7.12. It did not pin a PyTorch version, so this project uses PyTorch 1.13.1+cu117 as a Python 3.7 compatible baseline.

## Project Layout

```text
.
|-- data
|-- models
|-- outputs
|-- scripts
|   |-- download_reds.py
|   |-- evaluate.py
|   |-- infer.py
|   `-- train.py
|-- src
|   `-- basicvsr_yuv420
|       |-- checkpoints.py
|       |-- engine.py
|       |-- inference.py
|       |-- losses.py
|       |-- metrics.py
|       |-- utils.py
|       |-- data
|       |   `-- reds.py
|       `-- models
|           |-- basicvsr.py
|           |-- common.py
|           `-- spynet.py
|-- LICENSE
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## Environment Setup

Create and activate the recommended conda environment:

```bash
conda create -y -n basicvsr -c conda-forge python=3.7.12
conda activate basicvsr
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install -e . --no-deps
```

The current `requirements.txt` already points pip at the CUDA 11.7 PyTorch wheel index.

## Dataset Download

The repository includes a REDS downloader based on the official Hugging Face mirror.

List available presets and subsets:

```bash
python scripts/download_reds.py --list
```

Download the REDS training and validation subsets used by this project:

```bash
python scripts/download_reds.py --preset basicvsr_train_val --output-dir data/REDS_downloads --extract --skip-existing
```

Download the REDS test LR x4 subset for inference:

```bash
python scripts/download_reds.py --subsets test_sharp_bicubic --output-dir data/REDS_downloads --extract --skip-existing
```

Download only the training HR and LR x4 subsets:

```bash
python scripts/download_reds.py --subsets train_sharp train_sharp_bicubic --output-dir data/REDS_downloads --extract
```

## Expected REDS Paths

After extraction, the commands below expect the following directories:

```text
data/REDS_downloads/train_sharp/train/train_sharp
data/REDS_downloads/train_sharp_bicubic/train/train_sharp_bicubic/X4
data/REDS_downloads/val_sharp/val/val_sharp
data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4
data/REDS_downloads/test_sharp_bicubic/test/test_sharp_bicubic/X4
```

## Weights

The current project directly uses the pretrained SpyNet optical flow weights:

```text
models/checkpoints/spynet_sintel_final-3d2a1287.pth
```

This file initializes the SpyNet submodule inside BasicVSR. It is not a full BasicVSR training checkpoint.

## Train

Run training with the current local REDS directory layout:

```bash
python scripts/train.py --train-lr-dir data/REDS_downloads/train_sharp_bicubic/train/train_sharp_bicubic/X4 --train-hr-dir data/REDS_downloads/train_sharp/train/train_sharp --val-lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --val-hr-dir data/REDS_downloads/val_sharp/val/val_sharp --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --output-dir outputs/train_run
```

Important defaults:

- Python-compatible notebook defaults are already applied in `train.py`
- mixed precision is enabled by default
- the default REDS validation clip split is enabled by default
- the scheduler behavior intentionally differs from the original notebook and runs across the whole training job

Training outputs are written to the directory passed through `--output-dir`. After one epoch, `latest.pt` will be saved there. `best.pt` is also saved when validation improves.

## Evaluate

Evaluate a trained checkpoint on the validation set:

```bash
python scripts/evaluate.py --lr-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4 --hr-dir data/REDS_downloads/val_sharp/val/val_sharp --checkpoint outputs/train_run/latest.pt --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth
```

If you want to evaluate only the default REDS validation clips from a shared training directory instead of a dedicated validation directory, add `--use-default-reds-split`.

## Inference With REDS Test Data

Inference requires a trained BasicVSR checkpoint. If you have already trained the model, run:

```bash
python scripts/infer.py --input-dir data/REDS_downloads/test_sharp_bicubic/test/test_sharp_bicubic/X4/000 --checkpoint outputs/train_run/latest.pt --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --output-dir outputs/inference_test_000
```

If you only want to smoke test the inference pipeline with the checkpoint already present in this repository, you can use:

```bash
python scripts/infer.py --input-dir data/REDS_downloads/val_sharp_bicubic/val/val_sharp_bicubic/X4/000 --checkpoint outputs/smoke_epoch1_subset5/latest.pt --spynet-weights models/checkpoints/spynet_sintel_final-3d2a1287.pth --output-dir outputs/inference_val_000
```

## Notes

- `requirements.txt` does not install `torchvision` or `torchaudio` because the current codebase does not use them
- the repository currently includes SpyNet weights but does not include a full pretrained BasicVSR checkpoint
- package metadata requires Python `>=3.7,<3.11`
