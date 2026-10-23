# NYCU Computer Vision 2026 HW4

- **Student ID:** `111550136`
- **Name:** `連家堯`

## Introduction

This project implements **PromptIR** for all-in-one blind image restoration.
The model is trained to restore degraded images containing **rain** and **snow** effects.
The notebook includes the full training pipeline, validation process, visualization utilities, and inference code for generating the final submission files.

## Environment Setup

```bash
pip install torch torchvision numpy pillow einops tqdm matplotlib
```

## Usage

Run `main.ipynb` and edit the configuration cell before training or inference.


### Cell 1 — Configuration: set mode, paths, and checkpoints

```python
MODE = "infer"        # "train" for training; "infer" = for test pred.npz
RESUME = False        
TOTAL_EPOCHS = 90    

# ============================================================
# Paths (hard-coded; edit as needed)
# ============================================================
# Dataset root path after the Kaggle Dataset is extracted
DATA_ROOT = "/kaggle/input/datasets/leoooo0/hw4-realse-dataset/hw4_realse_dataset"

# Checkpoint path for resuming training or inference
# - This path is used when RESUME=True or MODE="infer"
# - If no checkpoint exists yet for first training, this path can be missing; keep RESUME=False
CKPT_PATH_INPUT = "/kaggle/input/datasets/leoooo0/thatyou/last (2).pth"

# Kaggle working directory; files generated in this session are saved here and can be downloaded from the Output panel
CKPT_DIR = "/kaggle/working/ckpts"
```

### Cell 2 — Path Check: verify dataset folders

```python
import os
import glob

TRAIN_DEGRADED = os.path.join(DATA_ROOT, "train", "degraded")
TRAIN_CLEAN    = os.path.join(DATA_ROOT, "train", "clean")
TEST_DEGRADED  = os.path.join(DATA_ROOT, "test", "degraded")
os.makedirs(CKPT_DIR, exist_ok=True)

assert os.path.exists(TRAIN_DEGRADED), f"Could not find {TRAIN_DEGRADED}"
assert os.path.exists(TRAIN_CLEAN),    f"Could not find {TRAIN_CLEAN}"
assert os.path.exists(TEST_DEGRADED),  f"Could not find {TEST_DEGRADED}"
print(f"[data] train degraded: {len(os.listdir(TRAIN_DEGRADED))} images")
print(f"[data] train clean   : {len(os.listdir(TRAIN_CLEAN))} images")
print(f"[data] test degraded : {len(os.listdir(TEST_DEGRADED))} images")
print(f"[ckpt] hard-coded CKPT_PATH_INPUT = {CKPT_PATH_INPUT}")
print(f"       {'exists ✓' if os.path.exists(CKPT_PATH_INPUT) else 'does not exist (can be ignored for first training)'}")
```

### Cell 3 — Imports: load packages and set seed

```python
import subprocess
subprocess.run(["pip", "install", "-q", "einops"], check=False)

import math
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from einops import rearrange
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
```

### Cell 4 — Checkpoint: choose resume or inference weights

```python
def resolve_checkpoint_for_resume():
    """For training: find the checkpoint to resume from."""
    if not RESUME:
        print("[ckpt] RESUME=False, training from scratch")
        return None
    # Prefer the checkpoint saved in the same session to avoid loading an older version
    p = os.path.join(CKPT_DIR, "last.pth")
    if os.path.exists(p):
        print(f"[ckpt] found checkpoint in working directory: {p}")
        return p
    if os.path.exists(CKPT_PATH_INPUT):
        print(f"[ckpt] loading checkpoint from input: {CKPT_PATH_INPUT}")
        return CKPT_PATH_INPUT
    print(f"[ckpt] CKPT_PATH_INPUT does not exist: {CKPT_PATH_INPUT}")
    print(f"       training from scratch")
    return None


def resolve_checkpoint_for_infer():
    """For inference: find the final weights to use."""
    # 1) Best checkpoint from the current training session
    for name in ("best.pth", "last.pth"):
        p = os.path.join(CKPT_DIR, name)
        if os.path.exists(p):
            return p
    # 2) Hard-coded input checkpoint
    if os.path.exists(CKPT_PATH_INPUT):
        return CKPT_PATH_INPUT
    raise FileNotFoundError(
        f"No checkpoint found for inference!\n"
        f"  - {CKPT_DIR}/best.pth      (does not exist)\n"
        f"  - {CKPT_DIR}/last.pth      (does not exist)\n"
        f"  - {CKPT_PATH_INPUT}        (does not exist)\n"
        f"Please train first, or upload the trained last.pth as a Dataset and update CKPT_PATH_INPUT"
    )


CKPT_PATH = resolve_checkpoint_for_resume()
```

### Cell 5 — Model Blocks: define PromptIR components

```python
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention (channel-wise self-attention)."""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-DConv Feed-Forward Network."""
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class PromptGenBlock(nn.Module):
    """Generate input-conditioned prompts."""
    def __init__(self, prompt_dim, prompt_len, prompt_size, lin_dim):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        w = F.softmax(self.linear_layer(emb), dim=1)
        prompt = w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param
        prompt = prompt.sum(dim=1)
        prompt = F.interpolate(prompt, (H, W), mode='bilinear', align_corners=False)
        return self.conv3x3(prompt)
```

### Cell 6 — PromptIR: build the restoration model

```python
class PromptIR(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=(4, 6, 6, 8), num_refinement_blocks=4,
                 heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=bias)

        def make_blocks(d, h, n):
            return nn.Sequential(*[
                TransformerBlock(d, h, ffn_expansion_factor, bias) for _ in range(n)
            ])

        # ----- Encoder -----
        self.encoder_level1 = make_blocks(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = make_blocks(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = make_blocks(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = Downsample(dim * 4)

        # ----- Latent -----
        self.latent = make_blocks(dim * 8, heads[3], num_blocks[3])

        # ----- Prompts -----
        self.prompt1 = PromptGenBlock(64, 5, 64, dim * 2)
        self.prompt2 = PromptGenBlock(128, 5, 32, dim * 4)
        self.prompt3 = PromptGenBlock(320, 5, 16, dim * 8)

        self.noise_level3 = TransformerBlock(dim * 8 + 320, heads[2], ffn_expansion_factor, bias)
        self.reduce_noise_level3 = nn.Conv2d(dim * 8 + 320, dim * 8, 1, bias=bias)

        # ----- Decoder -----
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = make_blocks(dim * 4, heads[2], num_blocks[2])

        self.noise_level2 = TransformerBlock(dim * 4 + 128, heads[2], ffn_expansion_factor, bias)
        self.reduce_noise_level2 = nn.Conv2d(dim * 4 + 128, dim * 4, 1, bias=bias)

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = make_blocks(dim * 2, heads[1], num_blocks[1])

        self.noise_level1 = TransformerBlock(dim * 2 + 64, heads[2], ffn_expansion_factor, bias)
        self.reduce_noise_level1 = nn.Conv2d(dim * 2 + 64, dim * 2, 1, bias=bias)

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = make_blocks(dim * 2, heads[0], num_blocks[0])
        self.refinement = make_blocks(dim * 2, heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(dim * 2, out_channels, 3, padding=1, bias=bias)

    def forward(self, inp_img):
        x1 = self.patch_embed(inp_img); e1 = self.encoder_level1(x1)
        x2 = self.down1_2(e1);          e2 = self.encoder_level2(x2)
        x3 = self.down2_3(e2);          e3 = self.encoder_level3(x3)
        x4 = self.down3_4(e3);          latent = self.latent(x4)

        p3 = self.prompt3(latent); latent = torch.cat([latent, p3], 1)
        latent = self.noise_level3(latent); latent = self.reduce_noise_level3(latent)

        d3 = self.up4_3(latent); d3 = torch.cat([d3, e3], 1)
        d3 = self.reduce_chan_level3(d3); d3 = self.decoder_level3(d3)
        p2 = self.prompt2(d3); d3 = torch.cat([d3, p2], 1)
        d3 = self.noise_level2(d3); d3 = self.reduce_noise_level2(d3)

        d2 = self.up3_2(d3); d2 = torch.cat([d2, e2], 1)
        d2 = self.reduce_chan_level2(d2); d2 = self.decoder_level2(d2)
        p1 = self.prompt1(d2); d2 = torch.cat([d2, p1], 1)
        d2 = self.noise_level1(d2); d2 = self.reduce_noise_level1(d2)

        d1 = self.up2_1(d2); d1 = torch.cat([d1, e1], 1)
        d1 = self.decoder_level1(d1); d1 = self.refinement(d1)
        return self.output(d1) + inp_img
```

### Cell 7 — Dataset: load paired images

```python
class RestorationTrainDataset(Dataset):
    """Training: random crop + flip + rotation."""
    def __init__(self, degraded_dir, clean_dir, patch_size=128, indices=None):
        self.patch_size = patch_size
        rains = sorted(glob.glob(os.path.join(degraded_dir, "rain-*.png")))
        snows = sorted(glob.glob(os.path.join(degraded_dir, "snow-*.png")))
        self.pairs = []
        for p in rains + snows:
            name = os.path.basename(p)
            clean_name = (name.replace("rain-", "rain_clean-")
                          if name.startswith("rain-")
                          else name.replace("snow-", "snow_clean-"))
            clean_path = os.path.join(clean_dir, clean_name)
            if os.path.exists(clean_path):
                self.pairs.append((p, clean_path))
        if indices is not None:
            self.pairs = [self.pairs[i] for i in indices]
        print(f"[train ds] {len(self.pairs)} pairs (patch_size={patch_size})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_p, clean_p = self.pairs[idx]
        deg = np.array(Image.open(deg_p).convert("RGB"), dtype=np.uint8)
        clean = np.array(Image.open(clean_p).convert("RGB"), dtype=np.uint8)
        H, W, _ = deg.shape
        ps = self.patch_size
        if H < ps or W < ps:
            pad_h = max(0, ps - H); pad_w = max(0, ps - W)
            deg = np.pad(deg, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            clean = np.pad(clean, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            H, W = deg.shape[:2]
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        deg = deg[top:top + ps, left:left + ps, :]
        clean = clean[top:top + ps, left:left + ps, :]
        if random.random() < 0.5:
            deg = deg[:, ::-1, :].copy(); clean = clean[:, ::-1, :].copy()
        if random.random() < 0.5:
            deg = deg[::-1, :, :].copy(); clean = clean[::-1, :, :].copy()
        k = random.randint(0, 3)
        if k > 0:
            deg = np.rot90(deg, k, axes=(0, 1)).copy()
            clean = np.rot90(clean, k, axes=(0, 1)).copy()
        deg_t = torch.from_numpy(deg).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return deg_t, clean_t


class RestorationValDataset(Dataset):
    """Validation: use full images and return the is_rain label."""
    def __init__(self, degraded_dir, clean_dir, indices):
        rains = sorted(glob.glob(os.path.join(degraded_dir, "rain-*.png")))
        snows = sorted(glob.glob(os.path.join(degraded_dir, "snow-*.png")))
        all_pairs = []
        for p in rains + snows:
            name = os.path.basename(p)
            clean_name = (name.replace("rain-", "rain_clean-")
                          if name.startswith("rain-")
                          else name.replace("snow-", "snow_clean-"))
            clean_path = os.path.join(clean_dir, clean_name)
            if os.path.exists(clean_path):
                all_pairs.append((p, clean_path))
        self.pairs = [all_pairs[i] for i in indices]
        print(f"[val ds] {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_p, clean_p = self.pairs[idx]
        is_rain = 1 if "rain" in os.path.basename(deg_p) else 0
        deg = np.array(Image.open(deg_p).convert("RGB"), dtype=np.uint8)
        clean = np.array(Image.open(clean_p).convert("RGB"), dtype=np.uint8)
        deg_t = torch.from_numpy(deg).permute(2, 0, 1).float() / 255.0
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return deg_t, clean_t, is_rain
```

### Cell 8 — Metrics: define loss and PSNR

```python
class CharbonnierLoss(nn.Module):
    """A smooth version of L1 loss, more robust to outliers."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


def psnr_torch(pred, target, max_val=1.0):
    """PSNR in dB. pred is clamped to [0, 1]."""
    mse = F.mse_loss(pred.clamp(0, 1), target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])
    return (10.0 * torch.log10(max_val * max_val / (mse + 1e-12))).mean().item()
```

### Cell 9 — Validation: set training config and evaluate

```python
CFG = {
    "patch_size": 128,
    "batch_size": 4,
    "num_workers": 2,
    "lr": 2e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "val_ratio": 50,       # Use the last 50 images from each class for validation: rain 50 + snow 50
    "amp": True,
    "grad_clip": 1.0,
    "val_every": 1,        # Validate every epoch
    "log_every": 50,
}


def build_loaders():
    val_idx_rain = list(range(1600 - CFG["val_ratio"], 1600))
    val_idx_snow = list(range(3200 - CFG["val_ratio"], 3200))
    val_indices = val_idx_rain + val_idx_snow
    train_indices = [i for i in range(3200) if i not in set(val_indices)]

    train_ds = RestorationTrainDataset(
        TRAIN_DEGRADED, TRAIN_CLEAN,
        patch_size=CFG["patch_size"], indices=train_indices,
    )
    val_ds = RestorationValDataset(TRAIN_DEGRADED, TRAIN_CLEAN, indices=val_indices)
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=1, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def validate(model, loader):
    """Return three PSNR values in dB: {'all', 'rain', 'snow'}."""
    model.eval()
    rain_psnrs, snow_psnrs = [], []
    for deg, clean, is_rain in loader:
        deg, clean = deg.to(DEVICE), clean.to(DEVICE)
        _, _, h, w = deg.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        deg_p = F.pad(deg, (0, pad_w, 0, pad_h), mode='reflect')
        with autocast(enabled=CFG["amp"]):
            out = model(deg_p)
        out = out[:, :, :h, :w]
        p = psnr_torch(out.float(), clean)
        if int(is_rain.item()) == 1:
            rain_psnrs.append(p)
        else:
            snow_psnrs.append(p)
    model.train()
    all_psnrs = rain_psnrs + snow_psnrs
    return {
        "all":  float(np.mean(all_psnrs))  if all_psnrs  else 0.0,
        "rain": float(np.mean(rain_psnrs)) if rain_psnrs else 0.0,
        "snow": float(np.mean(snow_psnrs)) if snow_psnrs else 0.0,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler,
                    epoch, best_psnr, global_step):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "global_step": global_step,
        "cfg": CFG,
        "total_epochs": TOTAL_EPOCHS,
    }, path)
```

### Cell 10 — Utilities: plot curves and save samples

```python
def plot_training_curves(csv_path, out_path):
    """Plot loss and validation PSNR curves from training_log.csv."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib is not available; skip plotting")
        return
    if not os.path.exists(csv_path):
        print(f"[plot] {csv_path} does not exist; skip plotting")
        return

    epochs, losses, psnr_all, psnr_rain, psnr_snow = [], [], [], [], []
    with open(csv_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            epochs.append(int(parts[0]))
            losses.append(float(parts[1]))
            psnr_all.append(float(parts[2]))
            psnr_rain.append(float(parts[3]))
            psnr_snow.append(float(parts[4]))

    if not epochs:
        print("[plot] CSV has no data; skip plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.plot(epochs, losses, color="tab:red", linewidth=1.6)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training Loss (Charbonnier)")
    ax1.set_title("Training Loss"); ax1.grid(alpha=0.3)

    ax2.plot(epochs, psnr_all, label="All (avg)", linewidth=2.0)
    ax2.plot(epochs, psnr_rain, label="Rain", linewidth=1.3, alpha=0.85)
    ax2.plot(epochs, psnr_snow, label="Snow", linewidth=1.3, alpha=0.85)
    ax2.axhline(26, color="gray", linestyle="--", linewidth=0.8, label="weak baseline ~26")
    ax2.axhline(30, color="gray", linestyle=":",  linewidth=0.8, label="strong baseline ~30")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation PSNR (dB)")
    ax2.set_title("Validation PSNR"); ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] training curve saved -> {out_path}")


@torch.no_grad()
def save_sample_visualizations(model, val_loader, out_dir, n_rain=3, n_snow=3):
    """Save triplet images: [degraded | restored | clean]."""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    rain_saved, snow_saved = 0, 0
    for deg, clean, is_rain in val_loader:
        kind = "rain" if int(is_rain.item()) == 1 else "snow"
        if kind == "rain" and rain_saved >= n_rain: continue
        if kind == "snow" and snow_saved >= n_snow: continue

        deg_dev = deg.to(DEVICE)
        _, _, h, w = deg_dev.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        deg_p = F.pad(deg_dev, (0, pad_w, 0, pad_h), mode='reflect')
        with autocast(enabled=CFG["amp"]):
            out = model(deg_p).float()
        out = out[:, :, :h, :w].clamp(0, 1)
        this_psnr = psnr_torch(out, clean.to(DEVICE))

        deg_np = (deg[0].numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        out_np = (out[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        clean_np = (clean[0].numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        gap = np.ones((deg_np.shape[0], 8, 3), dtype=np.uint8) * 255
        combined = np.concatenate([deg_np, gap, out_np, gap, clean_np], axis=1)

        idx = rain_saved if kind == "rain" else snow_saved
        save_path = os.path.join(out_dir, f"{kind}_{idx}_psnr{this_psnr:.2f}.png")
        Image.fromarray(combined).save(save_path)

        if kind == "rain": rain_saved += 1
        else:              snow_saved += 1
        if rain_saved >= n_rain and snow_saved >= n_snow:
            break

    model.train()
    print(f"[viz] saved {rain_saved+snow_saved} comparison images (left: degraded, middle: restored, right: clean) -> {out_dir}")
```

### Cell 11 — Training: train and save checkpoints

```python
def train():
    print(f"\n===== Training Settings =====")
    print(f"Target total epochs: {TOTAL_EPOCHS}")
    print(f"Resume           : {RESUME}")
    print(f"=====================\n")

    model = PromptIR(
        dim=48, num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4, heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66, bias=False,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] PromptIR: {n_params:.2f}M params")

    train_loader, val_loader = build_loaders()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"], betas=(0.9, 0.999),
    )
    total_steps = len(train_loader) * TOTAL_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=CFG["min_lr"],
    )
    criterion = CharbonnierLoss()
    scaler = GradScaler(enabled=CFG["amp"])

    # ---- Resume ----
    start_epoch = 0; best_psnr = 0.0; global_step = 0
    if CKPT_PATH is not None:
        try:
            sd = torch.load(CKPT_PATH, map_location=DEVICE)
            model.load_state_dict(sd["model"])
            if "optimizer" in sd: optimizer.load_state_dict(sd["optimizer"])
            if "scheduler" in sd: scheduler.load_state_dict(sd["scheduler"])
            if "scaler"    in sd: scaler.load_state_dict(sd["scaler"])
            start_epoch = sd.get("epoch", -1) + 1
            best_psnr   = sd.get("best_psnr", 0.0)
            global_step = sd.get("global_step", 0)
            print(f"[resume] epoch={start_epoch}, best_psnr={best_psnr:.3f} dB, "
                  f"global_step={global_step}")
        except Exception as e:
            print(f"[resume] FAILED: {e}, train from scratch")
            start_epoch = 0; best_psnr = 0.0; global_step = 0

    if start_epoch >= TOTAL_EPOCHS:
        print(f"[done] already completed {start_epoch} / {TOTAL_EPOCHS} epochs")
        print(f"       Increase TOTAL_EPOCHS if you want to train longer")
        return

    print(f"\nThis run will execute: epoch {start_epoch+1} -> {TOTAL_EPOCHS}\n")

    start_time = time.time()
    last_ckpt = os.path.join(CKPT_DIR, "last.pth")
    best_ckpt = os.path.join(CKPT_DIR, "best.pth")
    csv_path  = os.path.join(CKPT_DIR, "training_log.csv")
    psnr_history = []

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_psnr_all,val_psnr_rain,val_psnr_snow,lr,time_min\n")

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
        running_loss = 0.0
        for i, (deg, clean) in enumerate(pbar):
            deg = deg.to(DEVICE, non_blocking=True)
            clean = clean.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=CFG["amp"]):
                out = model(deg)
                loss = criterion(out, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            if (i + 1) % CFG["log_every"] == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{running_loss/(i+1):.4f}",
                                 lr=f"{lr_now:.2e}")

        # Save last.pth at the end of every epoch
        save_checkpoint(last_ckpt, model, optimizer, scheduler, scaler,
                        epoch, best_psnr, global_step)

        # Validation
        if (epoch + 1) % CFG["val_every"] == 0 or (epoch + 1) == TOTAL_EPOCHS:
            val = validate(model, val_loader)
            val_psnr = val["all"]
            avg_loss = running_loss / max(1, len(train_loader))
            elapsed_min = (time.time() - start_time) / 60
            lr_now = optimizer.param_groups[0]["lr"]

            is_best = val_psnr > best_psnr
            delta = val_psnr - best_psnr
            marker = "  ✨ NEW BEST" if is_best else f"  (best={best_psnr:.3f})"
            sign = "+" if delta >= 0 else ""
            print(
                f"[Epoch {epoch+1:3d}/{TOTAL_EPOCHS}] "
                f"loss={avg_loss:.4f} | "
                f"PSNR all={val_psnr:.3f} ({sign}{delta:.3f}) "
                f"rain={val['rain']:.3f} snow={val['snow']:.3f} | "
                f"elapsed={elapsed_min:.1f} min"
                f"{marker}"
            )

            with open(csv_path, "a") as f:
                f.write(f"{epoch+1},{avg_loss:.6f},{val_psnr:.4f},"
                        f"{val['rain']:.4f},{val['snow']:.4f},"
                        f"{lr_now:.6e},{elapsed_min:.2f}\n")

            if is_best:
                best_psnr = val_psnr
                save_checkpoint(last_ckpt, model, optimizer, scheduler, scaler,
                                epoch, best_psnr, global_step)
                save_checkpoint(best_ckpt, model, optimizer, scheduler, scaler,
                                epoch, best_psnr, global_step)
                print(f"           -> saved best.pth (val_PSNR = {best_psnr:.3f} dB)")

            psnr_history.append((epoch + 1, val_psnr, val["rain"], val["snow"]))

    # ---- Training finished ----
    print(f"\n===== Training Finished =====")
    print(f"Saved checkpoints:")
    print(f"  - {last_ckpt}   (for resume)")
    if os.path.exists(best_ckpt):
        print(f"  - {best_ckpt}   (best model)")
    print(f"\nBest val PSNR: {best_psnr:.3f} dB")

    if psnr_history:
        print(f"\n=== Validation PSNR of each epoch in this session ===")
        best_ep, best_val, _, _ = max(psnr_history, key=lambda x: x[1])
        for ep, p_all, p_rain, p_snow in psnr_history:
            marker = "  <- best" if (ep, p_all) == (best_ep, best_val) else ""
            print(f"  Epoch {ep:3d}: all={p_all:.3f}  rain={p_rain:.3f}  snow={p_snow:.3f}{marker}")

    # Outputs for the report
    plot_training_curves(csv_path, os.path.join(CKPT_DIR, "training_curve.png"))
    save_sample_visualizations(model, val_loader,
                               out_dir=os.path.join(CKPT_DIR, "samples"),
                               n_rain=3, n_snow=3)

    print(f"\n===== Report Outputs =====")
    print(f"  Training log CSV      : {csv_path}")
    print(f"  Training curve figure : {os.path.join(CKPT_DIR, 'training_curve.png')}")
    print(f"  Visual comparisons    : {os.path.join(CKPT_DIR, 'samples/')}")
    print(f"  Model weights (best)  : {best_ckpt}")
    print(f"  Model weights (last)  : {last_ckpt}")
    print(f"\n>>> All files above can be downloaded from the Output panel on the right")
```

### Cell 12 — Inference: generate prediction files

```python
@torch.no_grad()
def infer_one(model, img_tensor, use_tta=True):
    _, _, h, w = img_tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    inp = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    if not use_tta:
        with autocast(enabled=CFG["amp"]):
            out = model(inp)
        return out[:, :, :h, :w].clamp(0, 1)

    outs = []
    for flip_h in (False, True):
        for flip_v in (False, True):
            for rot in (0, 1):
                x = inp.clone()
                if flip_h: x = torch.flip(x, dims=[3])
                if flip_v: x = torch.flip(x, dims=[2])
                if rot:    x = torch.rot90(x, k=1, dims=[2, 3])
                with autocast(enabled=CFG["amp"]):
                    y = model(x).float()
                if rot:    y = torch.rot90(y, k=-1, dims=[2, 3])
                if flip_v: y = torch.flip(y, dims=[2])
                if flip_h: y = torch.flip(y, dims=[3])
                outs.append(y[:, :, :h, :w])
    return torch.stack(outs, 0).mean(0).clamp(0, 1)


def run_inference(use_tta=True):
    ckpt_path = resolve_checkpoint_for_infer()
    print(f"[infer] using checkpoint: {ckpt_path}")

    model = PromptIR(
        dim=48, num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4, heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66, bias=False,
    ).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()
    if isinstance(sd, dict) and "best_psnr" in sd:
        print(f"[infer] best validation PSNR recorded in checkpoint = {sd['best_psnr']:.3f} dB")

    test_imgs = sorted(
        glob.glob(os.path.join(TEST_DEGRADED, "*.png")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    print(f"[infer] test images: {len(test_imgs)} images, TTA={use_tta}")

    results = {}
    for p in tqdm(test_imgs, desc="Inference"):
        name = os.path.basename(p)
        img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        t = (torch.from_numpy(img).permute(2, 0, 1).float()
             .unsqueeze(0).to(DEVICE) / 255.0)
        out = infer_one(model, t, use_tta=use_tta)
        out_np = (out.squeeze(0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        results[name] = out_np

    pred_path = "/kaggle/working/pred.npz"
    np.savez(pred_path, **results)
    print(f"[infer] saved {pred_path} (keys={len(results)})")

    sub_zip = "/kaggle/working/submission.zip"
    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="pred.npz")
    print(f"[infer] saved {sub_zip}")

    # Check format
    data = np.load(pred_path)
    keys = list(data.keys())
    sample = data[keys[0]]
    print(f"\n[check] total keys: {len(keys)}")
    print(f"[check] example key: {keys[0]}, shape={sample.shape}, "
          f"dtype={sample.dtype}, range=[{sample.min()}, {sample.max()}]")
    assert sample.dtype == np.uint8 and sample.ndim == 3 and sample.shape[0] == 3
    print("[check] format OK")
```

### Cell 13 — Run: execute by selected mode

```python
if MODE == "train":
    train()
elif MODE == "infer":
    run_inference(use_tta=True)
else:
    raise ValueError(f"Unknown MODE: {MODE} (please set it to 'train' or 'infer')")
```

## Leaderboard

![Leaderboard Result](leaderboard.png)
