# Visual Recognition using Deep Learning Project

## HAT-S ×4 Super-Resolution Fine-Tuning

This project fine-tunes a **HAT-S ×4 Super-Resolution model** for the Kaggle video game super-resolution task. The main pretrained checkpoint used in this experiment is:

```text
HAT-S_SRx4.pth
```

The overall method is similar to the competition reference code that achieved around **PSNR = 33.5**, with additional data cleaning, fine-tuning, checkpoint selection, and **8x Test-Time Augmentation (TTA)** during inference.

---

## Training Summary

The learning rate schedule and training stages were configured as follows:

| Iteration Range | Learning Rate Setting |
|---:|---|
| 1–30000 | Constant LR = `1e-4` |
| 30001–50000 | Cosine schedule from `1e-4` to `1e-5` |
| 50001–70000 | Constant LR = `1e-4` |

The model trained to **70000 iterations**, but the score at 70000 iterations was worse than the score at 50000 iterations. Therefore, the **50000-iteration checkpoint** was selected as the best checkpoint.

---

## Kaggle Scores

| Iteration | Private Score | Public Score |
|---:|---:|---:|
| 30000 | 33.481 | 33.470 |
| 50000 | **33.629** | **33.631** |
| 70000 | 33.581 | 33.570 |

The 50000-iteration model was also tested without TTA. The private score without TTA was:

```text
Private score without TTA at iter 50000: 33.594
```

Since the private score with 8x TTA was **33.629**, TTA improved the final result.

---

## Model Architecture

The model used in this project is **HAT-S ×4 Super Resolution**.

| Parameter | HAT-S SRx4 Value | Description |
|---|---:|---|
| `type` | `HAT` | Hybrid Attention Transformer architecture |
| `upscale` | 4 | ×4 super-resolution |
| `in_chans` | 3 | RGB input |
| `img_size` | 64 | LR patch reference size |
| `window_size` | 16 | Window attention size |
| `embed_dim` | **144** | Feature channel dimension, smaller than full HAT |
| `depths` | `[6,6,6,6,6,6]` | 6 stages, with 6 blocks in each stage |
| `num_heads` | `[6,6,6,6,6,6]` | 6 attention heads in each stage |
| `mlp_ratio` | 2 | MLP hidden dimension = 2 × embed dimension |
| `compress_ratio` | **24** | Channel attention compression ratio |
| `squeeze_factor` | **24** | Channel squeeze factor |
| `conv_scale` | 0.01 | Convolution branch scaling factor |
| `overlap_ratio` | 0.5 | Overlapping cross-attention ratio |
| `upsampler` | `pixelshuffle` | Upsampling method |
| `resi_connection` | `1conv` | Residual connection with one convolution layer |
| `img_range` | 1.0 | Input image range is 0 to 1 |
| `param_key_g` | `params_ema` | Load EMA parameters from checkpoint |
| `strict_load_g` | `true` | Strictly match checkpoint parameter names |

---

## Difference Between Full HAT and HAT-S

The most important differences between `HAT_SRx4.pth` and `HAT-S_SRx4.pth` are listed below.

| Parameter | `HAT_SRx4.pth` Full HAT | `HAT-S_SRx4.pth` HAT-S |
|---|---:|---:|
| `embed_dim` | 180 | 144 |
| `compress_ratio` | 3 | 24 |
| `squeeze_factor` | 30 | 24 |
| Parameters | About 20.8M | About 9.6M |
| Multi-Adds | About 102.4G | About 54.9G |

HAT-S is smaller and lighter than the full HAT model, making it more suitable for faster fine-tuning and inference.

---

## Training Data Parameters

| Parameter | Value | Description |
|---|---:|---|
| `BATCH_SIZE` | `4` | 4 cropped image pairs per batch |
| `GT_SIZE` | `256` | HR crop size is 256×256 |
| `SCALE` | `4` | ×4 super-resolution scale |
| LR crop size | `64` | Calculated from `256 / 4` |
| `ENABLE_HOLDOUT_VAL` | `False` | No validation split; all clean data is used for training |
| `num_worker_per_gpu` | `2` | Number of DataLoader workers |
| `dataset_enlarge_ratio` | `1` | Dataset is not enlarged |
| `prefetch_mode` | `cuda` | CUDA prefetch is used |
| `pin_memory` | `True` | Speeds up CPU-to-GPU data transfer |

---

## Data Cleaning Parameters

| Parameter | Value | Description |
|---|---:|---|
| `CLEAN_PSNR_THRESHOLD` | `18.0` | Pairs with PSNR lower than 18 are treated as suspicious |
| `USE_CLEAN_SYMLINK_DATA` | `True` | Use symlinks to build the clean dataset |
| `USE_PREVIOUS_CLEAN_REPORT` | `True` | Prefer a previously generated clean report |
| `PREVIOUS_CLEAN_REPORT` | `clean_pair_report.csv` | Path to the previous clean report |
| `FORCE_REBUILD_CLEAN_DATA` | `False` | Do not force rebuilding the clean report |
| `USE_PUBLIC_SUSPECT_TXT` | `True` | Also exclude images listed in the public suspect list, if available |
| `PUBLIC_SUSPECT_PATTERNS` | `suspects*.txt` | Search pattern for suspect image lists |

The data cleaning process is:

```text
LR image → bicubic upsampling to HR size
          → compare with the real HR image using PSNR
          → if PSNR < 18.0, the LR/HR pair is considered suspicious
          → remove the suspicious training pair
```

The cleaned dataset is built at:

```text
/kaggle/working/clean_train/hr
/kaggle/working/clean_train/lr
```

These two folders are then used as the training dataset.

---

## Optimizer, Loss, and Learning Rate Parameters

> The actual learning rate and iteration settings follow the training summary at the top of this README.

| Parameter | Value | Description |
|---|---:|---|
| `LEARNING_RATE` | `1e-4` | Adam learning rate |
| `ADAM_BETAS` | `[0.9, 0.99]` | Adam beta parameters |
| `WEIGHT_DECAY` | `0.0` | No weight decay |
| `EMA_DECAY` | `0.999` | EMA update coefficient |
| `WARMUP_ITER` | `-1` | No warmup |
| Optimizer | `Adam` | Optimizer used for fine-tuning |
| Loss | `L1Loss` | Pixel-level L1 loss |
| Scheduler | `MultiStepRestartLR` | Practically equivalent to constant LR in the base setting |
| Scheduler milestone | `[10**12]` | The milestone is almost never reached |
| Scheduler gamma | `1.0` | LR remains unchanged even if the milestone is reached |

---

## Overall Workflow

1. **Prepare the environment**

   Install the required packages, download the HAT repository, and load the pretrained HAT-S SRx4 model.

2. **Clean the training data**

   Upsample `train/lr` images with bicubic interpolation and compare them with `train/hr` images using PSNR. LR/HR pairs with PSNR lower than 18.0 are treated as suspicious and removed.

3. **Build the clean training dataset**

   Use the following folders as the final training dataset:

   ```text
   clean_train/lr
   clean_train/hr
   ```

4. **Fine-tune the model**

   The model receives a 64×64 LR crop as input and predicts a 256×256 HR crop. HAT-S SRx4 is fine-tuned using L1 Loss, Adam optimizer, and LR = `1e-4`.

5. **Save checkpoints**

   A checkpoint is saved every 1000 iterations. Although the model was trained up to 70000 iterations, the 50000-iteration checkpoint was selected because it achieved the best score.

6. **Run inference on `test/lr`**

   Use the selected checkpoint to perform ×4 super-resolution on test LR images.

7. **Apply 8x Test-Time Augmentation**

   Each test image is transformed using flipping and transposition. The model predicts all 8 augmented versions, and the outputs are transformed back and averaged.

8. **Generate `submission.csv`**

   The final HR images are encoded into the Kaggle submission format and saved as `submission.csv`.

---

## Final Selected Setting

The best final setting is:

```text
Model: HAT-S_SRx4.pth
Checkpoint: iter 50000
Batch size: 4
GT patch size: 256
LR crop size: 64
Optimizer: Adam
Loss: L1Loss
EMA decay: 0.999
Data cleaning threshold: PSNR < 18.0 removed
Inference: 8x TTA
Best private score: 33.629
Best public score: 33.631
```
