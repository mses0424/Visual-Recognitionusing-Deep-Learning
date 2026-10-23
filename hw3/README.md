# NYCU Visual Recognition 2026 — HW3

- **Student ID:** `111550136`
- **Name:** `連家堯`


## Performance Snapshot

![Leaderboard Screenshot](Leaderboard.png)


## Introduction

The model is a **Mask R-CNN with ResNet-50 FPN-v2 backbone**, adapted for
this dataset with three changes: (i) anchor sizes shrunk to `(8, 16, 32, 64,
128)` to match the small nucleus size, (ii) all detection heads trained from
scratch (only the backbone uses ImageNet-1K pretraining), and (iii) heavy
augmentation, weight EMA, multi-scale training, and a two-LR AdamW
optimizer to stabilise training on the small (209-image) dataset.

## Environment Setup

The notebook was developed and tested on a single Kaggle GPU session
(NVIDIA Tesla P100 / T4 class, 16 GB). Python 3.10+, PyTorch 2.x.

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
torch
torchvision
albumentations
pycocotools
tifffile
imagecodecs
psutil
scikit-image
matplotlib
tqdm
pandas
pillow
```

## Usage

### Training

Run the Kaggle notebook `notebook_final.ipynb` from top to bottom — it
trains, runs inference, and produces `submission.zip`.

### Inference

Inference is part of the same notebook (cells 12–14): after training, the
best weights are loaded back into `model` automatically and the test set
is processed to produce `submission.zip`.


Best validation `AP50 = 0.524`.

---

## Code 

### 1. Environment Setup

Imports, sets a CUDA allocator hint to avoid fragmentation on long runs,
and prints a one-line memory summary so we can spot leaks early.

```python
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

!pip install -q --upgrade albumentations 2>/dev/null
!pip install -q pycocotools tifffile imagecodecs psutil 2>/dev/null

import sys
import json
import time
import math
import random
import shutil
import warnings
import gc
import glob
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import psutil
import numpy as np
import pandas as pd
import tifffile
import skimage.io as skio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
print("torch:", torch.__version__, "| torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


def _mem_str():
    """Return a short string with current CPU and GPU memory usage."""
    rss = psutil.Process().memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    s = f"RAM {rss:4.1f}/{vm.total/1e9:.0f}GB (free {vm.available/1e9:.1f}GB)"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        s += f" | GPU alloc {alloc:.1f}GB reserved {reserved:.1f}GB"
    return s

print("Initial:", _mem_str())
```

### 2. Configuration

A single `CFG` class holds every hyperparameter (paths, anchor sizes,
LRs, augmentation toggles, EMA decay, RPN caps, etc.) so any ablation only
requires editing this block.

```python
class CFG:
    seed = 42

    data_root = "/kaggle/input/datasets/leoooo0/hw3-data"
    work_dir = "/kaggle/working"

    num_classes = 5
    class_names = ["bg", "class1", "class2", "class3", "class4"]
    val_ratio = 0.1

    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    anchor_ratios = ((0.5, 1.0, 2.0),) * 5
    trainable_backbone_layers = 4

    batch_size = 2
    num_workers = 1
    pin_memory = False
    num_epochs = 50
    lr_head = 2e-4
    lr_backbone_mult = 0.25
    weight_decay = 1e-4
    warmup_epochs = 3
    grad_clip = 5.0

    train_min_sizes = (640, 800)
    train_max_size = 1333

    use_ema = True
    ema_decay = 0.9998

    test_min_size = 800
    test_max_size = 1024
    score_threshold = 0.05
    max_detections = 500

    use_tta = False
    tta_scales = (896, 1000, 1152)
    tta_hflip = True
    tta_nms_iou = 0.5

    rpn_pre_nms_top_n_train = 1000
    rpn_post_nms_top_n_train = 1000
    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_test = 500


CFG.train_dir = os.path.join(CFG.data_root, "train")
CFG.test_dir = os.path.join(
    CFG.data_root,
    "test_release" if os.path.isdir(os.path.join(CFG.data_root, "test_release")) else "test",
)
CFG.test_json = os.path.join(CFG.data_root, "test_image_name_to_ids.json")
os.makedirs(CFG.work_dir, exist_ok=True)

print("Data root :", CFG.data_root, "->", os.path.isdir(CFG.data_root))
print("Train dir :", CFG.train_dir, "->", os.path.isdir(CFG.train_dir))
print("Test dir  :", CFG.test_dir,  "->", os.path.isdir(CFG.test_dir))
print("Test json :", CFG.test_json, "->", os.path.isfile(CFG.test_json))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(CFG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
```

### 3. Dataset

`load_image` / `load_targets` parse the `.tif` files (each non-zero pixel
in `classN.tif` is one instance ID). The `CellInstanceDataset` returns the
`(image, target)` pair in torchvision's Mask R-CNN format, recomputing
bounding boxes from the post-augmentation masks (rather than transforming
boxes separately) to avoid degenerate boxes.

```python
def load_image(path):
    """Load a .tif image and return HxWx3 uint8 RGB."""
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / img.max() * 255.0).clip(0, 255).astype(np.uint8)
    return img


def load_targets(sample_dir, img_shape):
    H, W = img_shape[:2]
    all_masks, all_labels = [], []
    for cls in range(1, 5):
        p = os.path.join(sample_dir, f"class{cls}.tif")
        if not os.path.exists(p): continue
        m = tifffile.imread(p)
        if m.dtype.kind == "f":
            m = np.rint(m).astype(np.int32)
        if m.shape[:2] != (H, W):
            m = np.asarray(Image.fromarray(m.astype(np.int32)).resize((W, H), Image.NEAREST))
        for inst_id in np.unique(m):
            if inst_id == 0: continue
            bm = (m == inst_id).astype(np.uint8)
            if bm.sum() < 4: continue
            all_masks.append(bm); all_labels.append(cls)

    if not all_masks:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, H, W), dtype=np.uint8))
    masks = np.stack(all_masks, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
    return np.array(boxes, dtype=np.float32), labels, masks


class CellInstanceDataset(Dataset):
    def __init__(self, sample_ids, train_dir, transforms=None, is_train=True):
        self.sample_ids = sample_ids
        self.train_dir = train_dir
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sdir = os.path.join(self.train_dir, sid)
        image = load_image(os.path.join(sdir, "image.tif"))
        boxes, labels, masks = load_targets(sdir, image.shape)

        if self.transforms is not None:
            masks_list = [m for m in masks] if len(masks) else []
            transformed = self.transforms(
                image=image,
                masks=masks_list,
                bboxes=boxes.tolist() if len(boxes) else [],
                labels=labels.tolist() if len(labels) else [],
            )
            image = transformed["image"]
            new_masks = transformed["masks"]
            new_boxes = transformed["bboxes"]
            new_labels = transformed["labels"]

            kept_masks, kept_boxes, kept_labels = [], [], []
            for m, lab in zip(new_masks, new_labels):
                m = np.asarray(m, dtype=np.uint8)
                ys, xs = np.where(m)
                if len(xs) < 2 or len(ys) < 2:
                    continue
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                kept_masks.append(m)
                kept_boxes.append([x1, y1, x2, y2])
                kept_labels.append(lab)

            if kept_masks:
                masks = torch.as_tensor(np.stack(kept_masks), dtype=torch.uint8)
                boxes = torch.as_tensor(kept_boxes, dtype=torch.float32)
                labels = torch.as_tensor(kept_labels, dtype=torch.int64)
            else:
                _, H, W = image.shape
                masks = torch.zeros((0, H, W), dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        else:
            image = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else torch.zeros(0)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "_sample_id": sid,
        }
        return image, target
```

### 4. Augmentation

Albumentations pipeline that jointly transforms image, masks, and boxes.
H&E tiles are flip- and rotation-invariant so we apply geometric transforms
aggressively; CLAHE / colour jitter handle staining variation across slides.

```python
def build_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(0.0, 0.08),
                rotate=(-25, 25),
                p=0.6,
            ),
            A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.3),
            A.OneOf(
                [
                    A.ColorJitter(0.25, 0.25, 0.25, 0.1, p=1.0),
                    A.HueSaturationValue(15, 25, 15, p=1.0),
                    A.RandomBrightnessContrast(0.2, 0.2, p=1.0),
                ],
                p=0.8,
            ),
            A.GaussNoise(p=0.25),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.1,
        ),
    )


def build_val_transforms():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
        ),
    )
```

### 5. Train / Validation Split & DataLoaders

Deterministic 90/10 shuffle (seed = 42), giving ~189 training and ~20
validation samples. The validation set is held out from start to finish
and used only to select the best checkpoint.

```python
def split_train_val(sample_ids, val_ratio, seed):
    rng = random.Random(seed)
    ids = sample_ids.copy()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


train_samples = sorted(os.listdir(CFG.train_dir))
print(f"Training samples: {len(train_samples)}")

train_ids, val_ids = split_train_val(train_samples, CFG.val_ratio, CFG.seed)
print(f"Train: {len(train_ids)}  |  Val: {len(val_ids)}")

train_ds = CellInstanceDataset(train_ids, CFG.train_dir, build_train_transforms(), True)
val_ds   = CellInstanceDataset(val_ids,   CFG.train_dir, build_val_transforms(),   False)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(
    train_ds, batch_size=CFG.batch_size, shuffle=True,
    num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
    collate_fn=collate_fn, drop_last=True,
    persistent_workers=False,
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False,
    num_workers=CFG.num_workers, pin_memory=CFG.pin_memory,
    collate_fn=collate_fn,
    persistent_workers=False,
)

img, tgt = train_ds[0]
print("Image:", img.shape, img.dtype, "| range:", img.min().item(), img.max().item())
print("Boxes:", tgt["boxes"].shape, "| Labels:", tgt["labels"].shape, "| Masks:", tgt["masks"].shape)
```

### 6. Model: Mask R-CNN with Custom Anchors

`maskrcnn_resnet50_fpn_v2` with `weights=None` (no COCO pretraining) and
`weights_backbone=ResNet50_Weights.IMAGENET1K_V2` (ImageNet-1K backbone
only). The default anchor sizes are replaced with `(8, 16, 32, 64, 128)`
to match the small-nucleus regime, and the RPN head is re-initialised so
its conv weights match the new anchor count. Asserts `< 200M` trainable
params per the homework constraint.

```python
from torchvision.models import ResNet50_Weights

def build_model(num_classes=CFG.num_classes, multi_scale_train=True):
    if multi_scale_train:
        min_size = tuple(CFG.train_min_sizes)
        max_size = CFG.train_max_size
    else:
        min_size = CFG.test_min_size
        max_size = CFG.test_max_size

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
        trainable_backbone_layers=CFG.trainable_backbone_layers,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=CFG.rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=CFG.rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=CFG.rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=CFG.rpn_post_nms_top_n_test,
        box_detections_per_img=CFG.max_detections,
        box_nms_thresh=0.5,
        box_score_thresh=0.0,
    )

    anchor_generator = AnchorGenerator(
        sizes=CFG.anchor_sizes,
        aspect_ratios=CFG.anchor_ratios,
    )
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(
        model.backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
    )
    return model


model = build_model(multi_scale_train=True).to(DEVICE)

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {n_total/1e6:.2f} M")
print(f"Trainable parameters: {n_train/1e6:.2f} M")
assert n_train < 200e6, "Model exceeds the 200M parameter budget!"
print("Model min_size tuple:", model.transform.min_size)
print("Pretraining: ImageNet-1K V2 (backbone only); heads trained from scratch.")
```

### 7. COCO Ground Truth & Evaluator

Builds an in-memory COCO-format GT dict from the validation subset and
evaluates the model with `pycocotools` `COCOeval` (segmentation IoU).
Returns `AP @ [0.5:0.95]`, `AP50`, and `AP75`.

```python
def build_coco_gt_from_val(val_dataset):
    """Turn the validation subset into a COCO-format GT dict in memory."""
    images, annotations, categories = [], [], []
    for c in range(1, 5):
        categories.append({"id": c, "name": f"class{c}"})

    ann_id = 1
    for idx in range(len(val_dataset)):
        sid = val_dataset.sample_ids[idx]
        sdir = os.path.join(val_dataset.train_dir, sid)
        img = load_image(os.path.join(sdir, "image.tif"))
        H, W = img.shape[:2]
        images.append({"id": idx + 1, "file_name": sid, "height": H, "width": W})

        boxes, labels, masks = load_targets(sdir, img.shape)
        for b, lab, m in zip(boxes, labels, masks):
            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            x1, y1, x2, y2 = b.tolist()
            annotations.append({
                "id": ann_id,
                "image_id": idx + 1,
                "category_id": int(lab),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "segmentation": rle,
                "iscrowd": 0,
            })
            ann_id += 1

    gt = {"images": images, "annotations": annotations, "categories": categories}
    gt_path = os.path.join(CFG.work_dir, "val_gt.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    coco_gt = COCO(gt_path)
    return coco_gt


@torch.no_grad()
def evaluate_coco(model, val_loader, coco_gt):
    model.eval()
    results = []
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)
        for i, out in enumerate(outputs):
            img_id = batch_idx * val_loader.batch_size + i + 1
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            masks = out["masks"].cpu().numpy()
            for b, s, lab, m in zip(boxes, scores, labels, masks):
                if s < 0.05:
                    continue
                bin_m = (m[0] > 0.5).astype(np.uint8)
                if bin_m.sum() == 0:
                    continue
                rle = mask_utils.encode(np.asfortranarray(bin_m))
                rle["counts"] = rle["counts"].decode("ascii")
                x1, y1, x2, y2 = b
                results.append({
                    "image_id": img_id,
                    "category_id": int(lab),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                    "segmentation": rle,
                })
        del outputs, images, targets

    if not results:
        return {"ap": 0.0, "ap50": 0.0, "ap75": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = {
        "ap":   float(coco_eval.stats[0]),
        "ap50": float(coco_eval.stats[1]),
        "ap75": float(coco_eval.stats[2]),
    }
    del coco_dt, coco_eval, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats
```

### 8. Training Step (one epoch)

Runs a single training epoch with AMP mixed precision, gradient clipping,
and EMA updates. Wraps the training step in `try/except OutOfMemoryError`
so a single oversized batch doesn't kill the run — the batch is skipped
and counted in `n_oom`.

```python
def _oom_retry_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def train_one_epoch(model, loader, optimizer, scheduler, epoch, scaler=None, ema=None):
    model.train()
    loss_meter = defaultdict(float)
    n_batches = 0
    n_oom = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        try:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items() if torch.is_tensor(v)} for t in targets]

            if all(len(t["boxes"]) == 0 for t in targets):
                continue

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                optimizer.step()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            for k, v in loss_dict.items():
                loss_meter[k] += v.item()
            loss_meter["total"] += loss.item()
            n_batches += 1

            head_lr = optimizer.param_groups[1]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                hlr=f"{head_lr:.2e}",
                oom=n_oom,
            )

        except torch.cuda.OutOfMemoryError:
            n_oom += 1
            optimizer.zero_grad(set_to_none=True)
            del images, targets
            if "loss_dict" in locals(): del loss_dict
            if "loss" in locals(): del loss
            _oom_retry_free()
            pbar.set_postfix(oom=n_oom)
            continue

    if n_oom:
        print(f"  (skipped {n_oom} OOM batches this epoch)")
    return {k: v / max(1, n_batches) for k, v in loss_meter.items()}
```

### 9. Optimizer + Scheduler + EMA

Two-group AdamW: backbone gets `lr_head * 0.25 = 5e-5` (ImageNet-pretrained
needs less), heads get `2e-4` (random-init needs more). Schedule is a
3-epoch linear warmup followed by cosine annealing. `ModelEMA` maintains
an exponential moving average of weights with a ramp-up that approaches
`decay = 0.9998` over the first ~2k steps.

```python
import copy
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def build_optim_sched(model, steps_per_epoch):
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    head_lr = CFG.lr_head
    backbone_lr = CFG.lr_head * CFG.lr_backbone_mult

    print(f"Param groups:  backbone={len(backbone_params)} tensors @ "
          f"lr={backbone_lr:.1e}  |  heads={len(head_params)} tensors @ "
          f"lr={head_lr:.1e}")

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params,     "lr": head_lr,     "name": "heads"},
        ],
        weight_decay=CFG.weight_decay,
    )

    warmup_steps = max(1, CFG.warmup_epochs * steps_per_epoch)
    total_steps  = max(warmup_steps + 1, CFG.num_epochs * steps_per_epoch)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.num_updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    def _get_decay(self):
        return self.decay * (1.0 - math.exp(-(self.num_updates + 1) / 2000.0))

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self._get_decay()
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])
```

### 10. Main Training Loop

Trains for `CFG.num_epochs` epochs. Each epoch evaluates *both* the live
weights and the EMA copy and keeps whichever has a higher validation AP50
as the running best (in memory only). At the end of training, the best
weights are loaded back into `model` so inference uses them directly.

```python
import copy

steps_per_epoch = len(train_loader)
optimizer, scheduler = build_optim_sched(model, steps_per_epoch)
scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None
ema = ModelEMA(model, decay=CFG.ema_decay) if CFG.use_ema else None

coco_gt = build_coco_gt_from_val(val_ds)

history = {
    "epoch": [], "train_loss": [],
    "val_ap": [], "val_ap50": [], "val_ap75": [],
    "val_ap_ema": [], "val_ap50_ema": [], "val_ap75_ema": [],
}
best_ap50 = -1.0
best_is_ema = False
best_state_dict = None


for epoch in range(1, CFG.num_epochs + 1):
    t0 = time.time()
    losses = train_one_epoch(
        model, train_loader, optimizer, scheduler,
        epoch=epoch, scaler=scaler, ema=ema,
    )

    m_plain = evaluate_coco(model, val_loader, coco_gt)
    if ema is not None:
        m_ema = evaluate_coco(ema.module, val_loader, coco_gt)
    else:
        m_ema = {"ap": 0.0, "ap50": 0.0, "ap75": 0.0}

    history["epoch"].append(epoch)
    history["train_loss"].append(losses["total"])
    history["val_ap"].append(m_plain["ap"])
    history["val_ap50"].append(m_plain["ap50"])
    history["val_ap75"].append(m_plain["ap75"])
    history["val_ap_ema"].append(m_ema["ap"])
    history["val_ap50_ema"].append(m_ema["ap50"])
    history["val_ap75_ema"].append(m_ema["ap75"])

    cand_plain = m_plain["ap50"]
    cand_ema = m_ema["ap50"]

    if ema is not None and cand_ema > cand_plain and cand_ema > best_ap50:
        best_ap50 = cand_ema
        best_is_ema = True
        best_state_dict = {k: v.detach().cpu().clone()
                           for k, v in ema.module.state_dict().items()}
        print(f"  -> new best (EMA)   AP50={best_ap50:.4f}")
    elif cand_plain > best_ap50:
        best_ap50 = cand_plain
        best_is_ema = False
        best_state_dict = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
        print(f"  -> new best (plain) AP50={best_ap50:.4f}")

    print(
        f"Epoch {epoch:3d} | loss={losses['total']:.4f} | "
        f"AP50={cand_plain:.4f}  AP={m_plain['ap']:.4f} | "
        f"EMA AP50={cand_ema:.4f}  AP={m_ema['ap']:.4f} | "
        f"{time.time()-t0:.0f}s"
    )

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    model.to(DEVICE)

print(f"\nBest val AP50: {best_ap50:.4f}  (is_ema={best_is_ema})")
print("Best weights have been loaded back into `model` for inference.")
```

### 11. Training Curve Visualisation

Plots training loss and validation AP / AP50 / AP75 across all epochs.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["epoch"], history["train_loss"], "-o")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss"); axes[0].set_title("Training loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["epoch"], history["val_ap50"], "-o", label="AP50")
axes[1].plot(history["epoch"], history["val_ap"],   "-o", label="AP")
axes[1].plot(history["epoch"], history["val_ap75"], "-o", label="AP75")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("COCO segm AP"); axes[1].set_title("Validation metrics")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 12. Inference Setup

Switches the model's transform to a fixed inference resolution and loads
the test `image_id` mapping from `test_image_name_to_ids.json`.

```python
model.transform.min_size = (CFG.test_min_size,)
model.transform.max_size = CFG.test_max_size
model.roi_heads.detections_per_img = CFG.max_detections
model.rpn.pre_nms_top_n = lambda: CFG.rpn_pre_nms_top_n_test if model.training else CFG.rpn_pre_nms_top_n_test
model.rpn.post_nms_top_n = lambda: CFG.rpn_post_nms_top_n_test if model.training else CFG.rpn_post_nms_top_n_test

model.eval()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(f"Inference resolution: min={CFG.test_min_size}, max={CFG.test_max_size}, "
      f"max_det={CFG.max_detections}, TTA={CFG.use_tta}")

with open(CFG.test_json, "r") as f:
    test_meta = json.load(f)

name_to_id = {}
for item in test_meta:
    fn = item["file_name"]
    name_to_id[fn] = item["id"]
    stem = os.path.splitext(fn)[0]
    name_to_id[stem] = item["id"]
    name_to_id[stem + ".tif"] = item["id"]

print(f"Test images in JSON: {len({v for v in name_to_id.values()})}")
```

### 13. Run Inference (with optional TTA)

For each test image: forward pass (and optional horizontal-flip TTA),
filter by `score_threshold`, encode each predicted mask as RLE, and
collect the COCO-format submission record. Memory is aggressively freed
every 10 images to keep peak GPU usage stable.

```python
VAL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
VAL_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_test(image_np):
    """HxWx3 uint8 -> normalised float tensor CxHxW."""
    t = torch.as_tensor(image_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    t = (t - VAL_MEAN) / VAL_STD
    return t


@torch.no_grad()
def _forward_once(model, img_tensor):
    """Run one forward pass and move everything to CPU immediately."""
    out = model([img_tensor])[0]
    cpu_out = {
        "boxes":  out["boxes"].detach().cpu(),
        "scores": out["scores"].detach().cpu(),
        "labels": out["labels"].detach().cpu(),
        "masks":  (out["masks"] > 0.5).to(torch.uint8).detach().cpu(),
    }
    del out
    return cpu_out


@torch.no_grad()
def predict_one(model, image_np, use_tta=False):
    H, W = image_np.shape[:2]

    t = preprocess_test(image_np).to(DEVICE)
    out1 = _forward_once(model, t)
    del t
    torch.cuda.empty_cache()

    outs = [out1]

    if use_tta:
        t = preprocess_test(image_np).to(DEVICE)
        t_flip = torch.flip(t, dims=[2])
        del t
        out2 = _forward_once(model, t_flip)
        del t_flip
        torch.cuda.empty_cache()

        boxes = out2["boxes"].clone()
        boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        out2["boxes"] = boxes
        out2["masks"] = torch.flip(out2["masks"], dims=[3])
        outs.append(out2)

    boxes = torch.cat([o["boxes"] for o in outs], dim=0)
    scores = torch.cat([o["scores"] for o in outs], dim=0)
    labels = torch.cat([o["labels"] for o in outs], dim=0)
    masks = torch.cat([o["masks"] for o in outs], dim=0)

    if use_tta and len(boxes) > 0:
        keep_all = []
        for c in labels.unique():
            idx = (labels == c).nonzero(as_tuple=True)[0]
            k = torchvision.ops.nms(boxes[idx].float(), scores[idx], 0.5)
            keep_all.append(idx[k])
        keep = torch.cat(keep_all) if keep_all else torch.tensor([], dtype=torch.long)
        boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]

    return {
        "boxes":  boxes.numpy(),
        "scores": scores.numpy(),
        "labels": labels.numpy(),
        "masks":  masks.numpy(),
    }

_valid_exts = {".tif", ".tiff", ".TIF", ".TIFF"}
test_files = sorted([
    f for f in os.listdir(CFG.test_dir)
    if os.path.splitext(f)[1] in _valid_exts or os.path.splitext(f)[1] == ""
])
print(f"Running inference on {len(test_files)} test images (TTA={CFG.use_tta}) ...")

submission = []
missed = 0

for idx, fn in enumerate(tqdm(test_files)):
    image_id = name_to_id.get(fn) or name_to_id.get(os.path.splitext(fn)[0])
    if image_id is None:
        missed += 1
        continue

    image_np = load_image(os.path.join(CFG.test_dir, fn))
    H, W = image_np.shape[:2]

    try:
        pred = predict_one(model, image_np, use_tta=CFG.use_tta)
    except torch.cuda.OutOfMemoryError:
        print(f"\n  [OOM] skipping {fn} ({H}x{W})")
        torch.cuda.empty_cache()
        gc.collect()
        continue

    for b, s, lab, m in zip(pred["boxes"], pred["scores"], pred["labels"], pred["masks"]):
        if s < CFG.score_threshold:
            continue
        bin_m = m[0] if m.ndim == 3 else m
        if bin_m.sum() == 0:
            continue
        rle = mask_utils.encode(np.asfortranarray(bin_m))
        rle["counts"] = rle["counts"].decode("ascii")
        x1, y1, x2, y2 = b.tolist()
        submission.append({
            "image_id": int(image_id),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(s),
            "category_id": int(lab),
            "segmentation": {
                "size": [int(H), int(W)],
                "counts": rle["counts"],
            },
        })

    del pred, image_np
    if (idx + 1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()

if missed:
    print(f"WARN: {missed} test files had no matching id in test_image_name_to_ids.json")
print(f"Total predictions: {len(submission)}")
```

### 14. Build Submission Zip

Writes both `test-results.json` (correct spelling) and `test-reults.json`
(literal spelling from slide p.5) into a single zip — the grader is known
to look for the typo'd filename, but we include both for safety.

```python
import zipfile

result_filenames = ["test-results.json", "test-reults.json"]

for rf in result_filenames:
    p = os.path.join(CFG.work_dir, rf)
    with open(p, "w") as f:
        json.dump(submission, f)
    print(f"Wrote {p}  ({os.path.getsize(p) / 1e6:.2f} MB)")

unique_ids = {x["image_id"] for x in submission}
print(f"\nPredictions cover {len(unique_ids)} unique image_ids out of "
      f"{len({v for v in name_to_id.values()})} test images")

zip_path = os.path.join(CFG.work_dir, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for rf in result_filenames:
        zf.write(os.path.join(CFG.work_dir, rf), arcname=rf)

print(f"\n*** UPLOAD TO CODABENCH ***")
print(f"{zip_path}  ({os.path.getsize(zip_path) / 1e6:.2f} MB)")
print(f"Contains: {result_filenames}")
```

### 15. Qualitative Check

Visualises predictions on a few test images for a quick sanity check.

```python
def show_prediction(idx=0, score_thr=0.8):
    fn = test_files[idx]
    image_np = load_image(os.path.join(CFG.test_dir, fn))
    pred = predict_one(model, image_np, use_tta=False)
    print(f"模型總共預測了 {len(pred['scores'])} 個框")
    print(f"所有框的信心分數：\n{pred['scores']}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image_np)
    axes[0].set_title(f"{fn}")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    colors = ["red", "green", "blue", "orange"]
    for b, s, lab, m in zip(pred["boxes"], pred["scores"], pred["labels"], pred["masks"]):
        if s < score_thr:
            continue
        bin_m = m[0] > 0.5
        overlay = np.zeros((*bin_m.shape, 4))
        overlay[bin_m] = [*plt.cm.tab10(lab / 10)[:3], 0.4]
        axes[1].imshow(overlay)
        x1, y1, x2, y2 = b
        axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor=colors[(lab - 1) % 4], facecolor="none", lw=1))
    axes[1].set_title("Predictions (score > {:.2f})".format(score_thr))
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

for i in [0, 1, 2]:
    if i < len(test_files):
        show_prediction(i, score_thr=0.3)
```

---


