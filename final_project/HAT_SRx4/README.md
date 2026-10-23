# Visual Recognition using Deep Learning Project

## Project Overview

This project focuses on **x4 image super-resolution** using the **HAT (Hybrid Attention Transformer)** model.  
The main experiment fine-tunes a pretrained `HAT_SRx4.pth` model on the Kaggle video game super-resolution dataset.

The overall training strategy is based on:

- Global cosine learning rate schedule from `2e-4` to `1e-5`
- Total training target: `60000` iterations
- Training split into three stages: `1–10000`, `10000–35000`, and `35000–60000`
- Pretrained model: `HAT_SRx4.pth`
- Clean data filtering threshold: `PSNR >= 19.0`
- Batch size: `2`
- EMA decay: `0.9995`
- Inference with `8x TTA`

Most other settings are similar to the original competition baseline code with around PSNR 33.5.

---

## Result

| Iteration | Private Score | Public Score |
|---:|---:|---:|
| 10000 | 32.978 | 32.984 |
| 35000 | 33.423 | 33.401 |
| 60000 | 33.607 | 33.613 |

---

## Model Architecture Parameters

The official HAT SRx4 full model uses the following basic architecture settings.

| Parameter | Value | Description |
|---|---:|---|
| `type` | `HAT` | Hybrid Attention Transformer |
| `upscale` | 4 | x4 super-resolution |
| `in_chans` | 3 | RGB input |
| `img_size` | 64 | LR patch reference size in the base config |
| `window_size` | 16 | Window attention size |
| `embed_dim` | 180 | Feature channel dimension |
| `depths` | `[6,6,6,6,6,6]` | 6 stages, each with 6 blocks |
| `num_heads` | `[6,6,6,6,6,6]` | 6 attention heads in each stage |
| `mlp_ratio` | 2 | MLP hidden dimension = 2 × embed dimension |
| `compress_ratio` | 3 | Channel attention compression ratio |
| `squeeze_factor` | 30 | Channel squeeze parameter |
| `conv_scale` | 0.01 | Convolution branch scaling |
| `overlap_ratio` | 0.5 | Overlapping cross-attention ratio |
| `upsampler` | `pixelshuffle` | Upsampling method |
| `resi_connection` | `1conv` | Residual connection with one convolution |
| `img_range` | 1.0 | Input range is 0 to 1 |

---

## Difference Between HAT and HAT-S

The most important differences between the full HAT model and the HAT-S model are listed below.

| Parameter | `HAT_SRx4.pth` Full HAT | `HAT-S_SRx4.pth` HAT-S |
|---|---:|---:|
| `embed_dim` | 180 | 144 |
| `compress_ratio` | 3 | 24 |
| `squeeze_factor` | 30 | 24 |
| Params | about 20.8M | about 9.6M |
| Multi-Adds | about 102.4G | about 54.9G |

---

## DataLoader and Augmentation Parameters

| Parameter | Value |
|---|---:|
| Dataset type | `PairedImageDataset` |
| GT patch size | `GT_SIZE = 256` |
| LR patch size | 64 × 64, because scale = 4 |
| Batch size | `BATCH_SIZE = 2` |
| Shuffle | True |
| Workers per GPU | 2 |
| Dataset enlarge ratio | 1 |
| Prefetch mode | `cuda` |
| Pin memory | True |
| Horizontal flip | True |
| Rotation augmentation | True |
| Validation during train | False |

---

## Training Parameters

| Parameter | Value |
|---|---:|
| Total target iteration | `60000` |
| Iterations per session | `10000` |
| Auto resume | True |
| Warmup | `-1`, no warmup |
| Loss | L1 Loss |
| Loss weight | 1.0 |
| Reduction | mean |
| EMA decay | `0.9995` |
| Save checkpoint frequency | `1000` |
| Print frequency | `100` |
| TensorBoard logger | False |
| Progress bar | True |

---

## Optimizer and Scheduler Parameters

| Category | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | `2e-4` |
| Betas | `[0.9, 0.99]` |
| Weight decay | `0.0` |
| Scheduler | CosineAnnealingRestartLR |
| Cosine period | `[60000]` |
| Eta min | `1e-5` |
| Restart weight | `[1.0]` |

---

## Training Flow

### 1. Environment Setup

Install HAT, BasicSR, timm, OpenCV, scikit-image, and other required packages.  
Clone the XPixelGroup/HAT repository.

### 2. Load Dataset

Automatically search the Kaggle input directory for:

- `train/hr`
- `train/lr`
- `test/lr`
- `sample_submission.csv`

### 3. Build Clean Training Set

For each LR/HR pair:

1. Upscale the LR image to the HR size using bicubic interpolation.
2. Calculate the bicubic PSNR.
3. Keep only the image pairs with `PSNR >= 19.0`.
4. Build the cleaned training folders:
   - `clean_train/hr`
   - `clean_train/lr`

### 4. Load Pretrained HAT SRx4 Model

Use `HAT_SRx4.pth` as the pretrained checkpoint.  
The model loads the `params_ema` weights with strict loading.

### 5. Build Training Configuration

The training configuration is based on the official HAT SRx4 fine-tuning YAML file, but the dataset is changed to the Kaggle video game super-resolution dataset.

The main training settings are:

- GT patch size: `256`
- Batch size: `2`
- Augmentation: horizontal flip and rotation

### 6. Train the Model

The model is trained with:

- Optimizer: Adam
- Learning rate: `2e-4`
- Loss function: L1 loss
- Scheduler: cosine learning rate scheduler
- Learning rate range: `2e-4` to `1e-5`
- Total target iterations: `60000`

The training is split into multiple sessions, and each session automatically resumes from the previous checkpoint.

### 7. EMA Weight Update

During training, EMA is applied with:

```text
EMA decay = 0.9995
```

During inference, the model gives priority to the `params_ema` weights for more stable results.

### 8. Save Checkpoints and Resume Package

A checkpoint is saved every `1000` iterations.

After training, the following files are generated:

- `net_g_latest.pth`
- `net_g_{iter}.pth`
- `resume_package`

The resume package is used for continuing training in later sessions.

### 9. Inference and Submission Generation

The latest checkpoint is used for inference on the test set.

Before inference, reflect padding is applied so that the input size is a multiple of `window_size = 16`.  
After inference, the output is cropped back to the correct x4 output size.

### 10. 8x TTA

For each test image, 8 test-time augmentation modes are used.  
The 8 outputs are reversed back to the original orientation and averaged.

The final output is clamped to the range `0–1` and converted to `uint8`.

### 11. Generate Kaggle Submission

The final RGB output is converted to BGR.  
Then the image is encoded using:

- RLE
- zlib
- base64

Finally, the submission file is exported as:

```text
submission.csv
```

---

## Summary

This project fine-tunes a pretrained full HAT SRx4 model on a cleaned video game super-resolution dataset.  
The final training uses cosine learning rate scheduling, L1 loss, EMA weight averaging, clean data filtering, and 8x TTA inference.  
The best result is obtained at 60000 iterations, reaching a private score of `33.607` and a public score of `33.613`.
