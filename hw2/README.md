# NYCU Computer Vision 2026 HW2

- **Student ID**: 111550136
- **Name**: 連家堯

---

## Introduction

This project implements a digit detection system based on a Deformable DETR architecture with ResNet-50 as the backbone.
The encoder uses Multi-Scale Deformable Attention across four feature scales (C3, C4, C5, and an extra level).
The decoder uses full global cross-attention so each query attends to the entire feature map, which helps detect small or tilted digits.

**Best validation mAP@0.5:0.95: 0.4240 (epoch 72)**

---

## Environment Setup

**Requirements**
- Python 3.9+
- PyTorch 2.0+
- CUDA (recommended)

**Install dependencies:**

```bash
pip install torch torchvision
pip install scipy pycocotools pillow
```

**Dataset structure:**

```
nycu-hw2-data/
├── train/
├── valid/
├── test/
├── train.json
└── valid.json
```

---

## Usage

### Install dependencies and check GPU

Check that PyTorch and CUDA are available before starting.

```python
!pip install scipy pycocotools -q
import torch
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
```

---

### Imports and configuration

Import all required libraries and set the dataset path.

```python
import json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import resnet50, ResNet50_Weights
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATA_ROOT  = '/kaggle/input/datasets/enweiliu/dataset2/nycu-hw2-data'
WORK_DIR   = '/kaggle/working'
OUTPUT_DIR = os.path.join(WORK_DIR, 'checkpoints')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Data check:')
for n in ['train', 'valid', 'test', 'train.json', 'valid.json']:
    print(f'  {n}: {"OK" if os.path.exists(os.path.join(DATA_ROOT, n)) else "MISSING"}')

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

---

### Image preprocessing utilities

**Letterbox** resizes any image to a square canvas while preserving aspect ratio by padding with black borders.
**rotate_boxes** recomputes bounding boxes after arbitrary-angle rotation by rotating all four corners and taking the enclosing rectangle.

```python
def letterbox(image, size):
    ow, oh = image.size
    scale  = size / max(ow, oh)
    nw     = max(int(ow * scale), 1)
    nh     = max(int(oh * scale), 1)
    image  = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    pad_x  = (size - nw) // 2
    pad_y  = (size - nh) // 2
    canvas.paste(image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, ow, oh

def rotate_boxes(raw_boxes, angle_deg, ow, oh):
    import math as _m
    a = _m.radians(angle_deg)
    cos_a, sin_a = abs(_m.cos(a)), abs(_m.sin(a))
    # Rotated image size
    nw = int(oh * sin_a + ow * cos_a)
    nh = int(oh * cos_a + ow * sin_a)
    cx_img, cy_img = ow / 2, oh / 2
    cx_new, cy_new = nw / 2, nh / 2
    new_boxes = []
    for x, y, w, h in raw_boxes:
        # Rotate all 4 corners of the box
        corners = [
            (x,     y),
            (x + w, y),
            (x,     y + h),
            (x + w, y + h),
        ]
        rot_corners = []
        for cx, cy in corners:
            dx, dy = cx - cx_img, cy - cy_img
            rx = dx * _m.cos(-a) - dy * _m.sin(-a) + cx_new
            ry = dx * _m.sin(-a) + dy * _m.cos(-a) + cy_new
            rot_corners.append((rx, ry))
        xs = [p[0] for p in rot_corners]
        ys = [p[1] for p in rot_corners]
        nx, ny = min(xs), min(ys)
        nw2 = max(xs) - nx
        nh2 = max(ys) - ny
        new_boxes.append([max(0.0, nx), max(0.0, ny),
                          min(nw2, nw - nx), min(nh2, nh - ny)])
    return new_boxes, nw, nh
```

---

### Dataset

`DigitDataset` loads COCO-format annotations and applies the full augmentation pipeline during training:
horizontal flip, 90/180/270° rotation, fine-grained ±30° rotation, color jitter, random scale, random grayscale, and random erasing.
`TestDataset` loads test/validation images without augmentation for inference.

```python
class DigitDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=640, is_train=True):
        self.img_dir  = img_dir
        self.img_size = img_size
        self.is_train = is_train
        with open(ann_file) as f:
            data = json.load(f)
        self.images    = {img['id']: img for img in data['images']}
        self.image_ids = [
            img['id'] for img in data['images']
            if os.path.exists(os.path.join(img_dir, img['file_name']))
        ]
        self.anns = {iid: [] for iid in self.image_ids}
        for ann in data['annotations']:
            if ann['image_id'] in self.anns:
                self.anns[ann['image_id']].append(ann)
        self.color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info   = self.images[img_id]
        image  = Image.open(
            os.path.join(self.img_dir, info['file_name'])
        ).convert('RGB')
        ow, oh = image.size

        raw_boxes, labels = [], []
        for ann in self.anns[img_id]:
            x, y, w, h = ann['bbox']
            if w > 1 and h > 1:
                raw_boxes.append([float(x), float(y), float(w), float(h)])
                labels.append(ann['category_id'])

        if self.is_train:
            # ── Horizontal flip ──────────────────────────────────────────
            if random.random() < 0.5:
                image     = TF.hflip(image)
                raw_boxes = [[ow - x - w, y, w, h] for x, y, w, h in raw_boxes]

            # ── 90/180/270 rotation (handles text orientation) ───────────
            rot90 = random.choice([0, 0, 0, 90, 180, 270])
            if rot90 != 0:
                image = image.rotate(-rot90, expand=True)
                nw2, nh2 = image.size
                new_boxes = []
                for x, y, w, h in raw_boxes:
                    if rot90 == 90:
                        new_boxes.append([oh - y - h, x, h, w])
                    elif rot90 == 180:
                        new_boxes.append([ow - x - w, oh - y - h, w, h])
                    elif rot90 == 270:
                        new_boxes.append([y, ow - x - w, h, w])
                raw_boxes = new_boxes
                ow, oh = nw2, nh2

            # ── Fine-grained rotation [-30, 30] for tilted digits ────────
            # Applied 40% of the time; expand=True so digits aren't cropped
            if random.random() < 0.4:
                angle = random.uniform(-30, 30)
                if abs(angle) > 2:   # skip negligible angles
                    image = image.rotate(angle, expand=True)
                    raw_boxes, ow, oh = rotate_boxes(raw_boxes, angle, ow, oh)

            # ── Color jitter ─────────────────────────────────────────────
            image = self.color_jitter(image)

            # ── Random scale [0.6, 1.4] ──────────────────────────────────
            sc    = random.uniform(0.6, 1.4)
            nw_a  = max(int(ow * sc), 1)
            nh_a  = max(int(oh * sc), 1)
            image = image.resize((nw_a, nh_a), Image.BILINEAR)
            raw_boxes = [[x*sc, y*sc, w*sc, h*sc] for x, y, w, h in raw_boxes]
            ow, oh = nw_a, nh_a

            # ── Random grayscale ─────────────────────────────────────────
            if random.random() < 0.15:
                image = TF.to_grayscale(image, num_output_channels=3)

            # ── Random erasing (simulate occlusion) ──────────────────────
            if random.random() < 0.2:
                img_t_tmp = TF.to_tensor(image)
                img_t_tmp = T.RandomErasing(p=1.0, scale=(0.02, 0.08),
                                             ratio=(0.3, 3.3))(img_t_tmp)
                image = TF.to_pil_image(img_t_tmp)

        img_lb, scale, px, py, _, _ = letterbox(image, self.img_size)
        S = self.img_size

        bt_list = []
        for x, y, w, h in raw_boxes:
            lx  = x * scale + px
            ly  = y * scale + py
            lw  = w * scale
            lh  = h * scale
            cx  = (lx + lw / 2) / S
            cy  = (ly + lh / 2) / S
            nw_ = lw / S
            nh_ = lh / S
            if nw_ > 1e-3 and nh_ > 1e-3:   # skip degenerate boxes
                bt_list.append([cx, cy, nw_, nh_])
            else:
                labels = labels  # keep label list consistent

        # Rebuild labels in sync with bt_list (some may be dropped after rotation)
        valid_labels = []
        valid_bt     = []
        for i, bt in enumerate(bt_list):
            if i < len(labels):
                valid_bt.append(bt)
                valid_labels.append(labels[i])

        bt = torch.tensor(valid_bt,     dtype=torch.float32) if valid_bt     else torch.zeros((0, 4))
        lt = torch.tensor(valid_labels, dtype=torch.long)    if valid_labels else torch.zeros(0, dtype=torch.long)
        img_t = TF.normalize(TF.to_tensor(img_lb), MEAN, STD)
        if len(bt) > 0:
            bt = bt.clamp(1e-4, 1 - 1e-4)
        return img_t, {
            'image_id':  torch.tensor(img_id),
            'orig_size': torch.tensor([oh, ow]),
            'boxes':  bt,
            'labels': lt,
        }


class TestDataset(Dataset):
    def __init__(self, d, s=640):
        self.d = d
        self.s = s
        self.f = sorted(
            f for f in os.listdir(d)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        f   = self.f[i]
        img = Image.open(os.path.join(self.d, f)).convert('RGB')
        img_lb, scale, px, py, ow, oh = letterbox(img, self.s)
        img_t  = TF.normalize(TF.to_tensor(img_lb), MEAN, STD)
        img_id = int(os.path.splitext(f)[0])
        return img_t, img_id, ow, oh, float(scale), float(px), float(py)
```

---

### Model utilities

Helper functions used by the model:
- `inverse_sigmoid`: used for reference point decoding in box prediction
- `get_sine_pos`: generates 2D sine-cosine positional encodings for feature maps
- `MLP`: multi-layer perceptron used as the bounding box prediction head
- `SwiGLU`: gated feed-forward network, used in both encoder and decoder layers instead of standard ReLU FFN

```python
def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return torch.stack(imgs), list(tgts)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def get_sine_pos(feat, temperature=10000):
    """2D sine-cosine positional encoding for (B, C, H, W)."""
    b, c, h, w = feat.shape
    dev = feat.device
    d   = c // 2
    ye  = torch.arange(1, h+1, dtype=torch.float32, device=dev).view(1,h,1).expand(b,h,w)
    xe  = torch.arange(1, w+1, dtype=torch.float32, device=dev).view(1,1,w).expand(b,h,w)
    ye  = ye / (h + 1e-6) * 2 * math.pi
    xe  = xe / (w + 1e-6) * 2 * math.pi
    dt  = temperature ** (2 * (torch.arange(d, dtype=torch.float32, device=dev) // 2) / d)
    px_ = xe[:,:,:,None] / dt
    py_ = ye[:,:,:,None] / dt
    px_ = torch.stack([px_[:,:,:,0::2].sin(), px_[:,:,:,1::2].cos()], 4).flatten(3)
    py_ = torch.stack([py_[:,:,:,0::2].sin(), py_[:,:,:,1::2].cos()], 4).flatten(3)
    return torch.cat([py_, px_], 3).permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, in_d, hid_d, out_d, n_layers):
        super().__init__()
        dims        = [in_d] + [hid_d] * (n_layers - 1) + [out_d]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i+1]) for i in range(n_layers))
        self.n      = n_layers

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = F.relu(l(x)) if i < self.n - 1 else l(x)
        return x


class SwiGLU(nn.Module):
    """Gated FFN: SiLU(W1*x) * W2*x."""
    def __init__(self, d, ff):
        super().__init__()
        hidden    = int(ff * 2 / 3)
        hidden    = (hidden + 7) // 8 * 8
        self.gate = nn.Linear(d, hidden)
        self.val  = nn.Linear(d, hidden)
        self.out  = nn.Linear(hidden, d)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.val(x))
```

---

### Multi-Scale Deformable Attention (MSDeformAttn)

The core component of the encoder. Instead of attending to all N tokens (expensive O(N²)),
each position samples only `n_heads × n_levels × n_points` reference points across all feature levels.
Sampling offsets are learned during training so the model focuses on informative regions.

```python
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_levels = n_levels
        self.n_heads  = n_heads
        self.n_points = n_points
        self.sampling_offsets  = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj        = nn.Linear(d_model, d_model)
        self.output_proj       = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas    = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias,   0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias,   0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias,  0.)

    def forward(self, query, reference_points, input_flatten,
                spatial_shapes, level_start_index):
        B, Lq, _ = query.shape
        B, Lk, _ = input_flatten.shape
        value    = self.value_proj(input_flatten)
        value    = value.view(B, Lk, self.n_heads, self.d_model // self.n_heads)
        offsets  = self.sampling_offsets(query).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn_w   = self.attention_weights(query).view(
            B, Lq, self.n_heads, self.n_levels * self.n_points)
        attn_w   = F.softmax(attn_w, -1).view(
            B, Lq, self.n_heads, self.n_levels, self.n_points)
        norm     = torch.stack([spatial_shapes[..., 1],
                                spatial_shapes[..., 0]], -1).float()
        slocs    = (reference_points[:, :, None, :, None, :]
                    + offsets / norm[None, None, None, :, None, :])
        output   = self._bilinear_sample(value, spatial_shapes, slocs, attn_w)
        return self.output_proj(output)

    def _bilinear_sample(self, value, spatial_shapes, sampling_locations, attn_weights):
        B, Lk, nh, dh = value.shape
        _, Lq, _, nl, np_, _ = sampling_locations.shape
        value_list = value.split(
            [int(H) * int(W) for H, W in spatial_shapes.tolist()], dim=1)
        grids  = 2.0 * sampling_locations - 1.0
        sampled = []
        for lid, (H, W) in enumerate(spatial_shapes.tolist()):
            H, W = int(H), int(W)
            v_l  = (value_list[lid].flatten(2).transpose(1, 2)
                                    .reshape(B * nh, dh, H, W))
            g_l  = (grids[:, :, :, lid, :, :]
                    .permute(0, 2, 1, 3, 4).flatten(0, 1))
            s_l  = F.grid_sample(v_l, g_l, mode='bilinear',
                                  padding_mode='zeros', align_corners=False)
            sampled.append(s_l)
        sampled = torch.stack(sampled, dim=-2).flatten(-2)
        w = (attn_weights.permute(0, 2, 1, 3, 4)
                          .flatten(0, 1).flatten(-2).unsqueeze(1))
        out = (sampled * w).sum(-1)
        out = out.view(B, nh, dh, Lq).permute(0, 3, 1, 2).flatten(-2)
        return out
```

---

### Encoder and Decoder layers

**DeformableEncoderLayer**: Pre-Norm encoder layer using MSDeformAttn for efficient multi-scale self-attention.

**GlobalDecoderLayer**: Pre-Norm decoder layer with standard self-attention between queries and full global cross-attention to encoder memory.
Full (non-deformable) cross-attention is used so queries are not constrained to sampled locations,
which helps detect digits at unusual positions or orientations.

```python
class DeformableEncoderLayer(nn.Module):
    """Pre-Norm Deformable Encoder: O(N*K) self-attention across all feature levels."""
    def __init__(self, d, n_levels, n_heads=8, n_points=4, ff=1024, dr=0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d, n_levels, n_heads, n_points)
        self.norm1     = nn.LayerNorm(d)
        self.dropout1  = nn.Dropout(dr)
        self.ffn       = SwiGLU(d, ff)
        self.norm2     = nn.LayerNorm(d)
        self.dropout2  = nn.Dropout(dr)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2 + pos, reference_points, src2,
                               spatial_shapes, level_start_index)
        src  = src + self.dropout1(src2)
        src  = src + self.dropout2(self.ffn(self.norm2(src)))
        return src

class GlobalDecoderLayer(nn.Module):
    def __init__(self, d, n_heads=8, ff=1024, dr=0.1):
        super().__init__()
        # Self-attention between queries
        self.self_attn  = nn.MultiheadAttention(d, n_heads, dropout=dr, batch_first=True)
        self.norm1      = nn.LayerNorm(d)
        self.dropout1   = nn.Dropout(dr)
        # Full global cross-attention: queries attend to ALL encoder memory positions
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dr, batch_first=True)
        self.norm2      = nn.LayerNorm(d)
        self.dropout2   = nn.Dropout(dr)
        # FFN
        self.ffn        = SwiGLU(d, ff)
        self.norm3      = nn.LayerNorm(d)
        self.dropout3   = nn.Dropout(dr)

    def forward(self, tgt, query_pos, memory, memory_pos):
        # Pre-Norm Self-Attention
        tgt2    = self.norm1(tgt)
        q = k   = tgt2 + query_pos
        tgt2, _ = self.self_attn(q, k, tgt2)
        tgt     = tgt + self.dropout1(tgt2)
        # Pre-Norm Full Cross-Attention (global: each query sees all N positions)
        tgt2    = self.norm2(tgt)
        q       = tgt2 + query_pos
        k       = memory + memory_pos      # key = value + positional context
        tgt2, _ = self.cross_attn(q, k, memory)
        tgt     = tgt + self.dropout2(tgt2)
        # Pre-Norm SwiGLU FFN
        tgt     = tgt + self.dropout3(self.ffn(self.norm3(tgt)))
        return tgt


# ====================== Main Model: HybridDETR ======================
```

---

### HybridDETR model

The full model combining all components:
- ResNet-50 backbone extracts features at 4 scales
- Deformable encoder processes multi-scale features efficiently
- 30 object queries are decoded by the global cross-attention decoder
- Classification head: linear layer over 11 classes (10 digits + no-object)
- Box head: 3-layer MLP predicting offsets from per-query reference points

```python
class HybridDETR(nn.Module):

    def __init__(self, nc=10, nq=100, d=256, nh=8, ne=6, nd=6,
                 ff=1024, dr=0.1, n_levels=4, n_points=4, pt=True):
        super().__init__()
        self.d        = d
        self.n_levels = n_levels

        # ── Backbone: ResNet-50, extract C3 / C4 / C5 ─────────────────────
        bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pt else None)
        self.backbone_c3 = nn.Sequential(*list(bb.children())[:6])   # 512ch,  stride 8
        self.backbone_c4 = nn.Sequential(*list(bb.children())[6:7])  # 1024ch, stride 16
        self.backbone_c5 = nn.Sequential(*list(bb.children())[7:8])  # 2048ch, stride 32
        for mod in [self.backbone_c3, self.backbone_c4, self.backbone_c5]:
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)

        # ── Feature projection to d channels ──────────────────────────────
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512,  d, 1), nn.GroupNorm(32, d)),  # C3
            nn.Sequential(nn.Conv2d(1024, d, 1), nn.GroupNorm(32, d)),  # C4
            nn.Sequential(nn.Conv2d(2048, d, 1), nn.GroupNorm(32, d)),  # C5
            nn.Sequential(                                                # stride 64
                nn.Conv2d(2048, d, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, d)
            ),
        ])
        self.level_embed = nn.Embedding(n_levels, d)

        # ── Encoder: Deformable Self-Attention ────────────────────────────
        self.encoder  = nn.ModuleList([
            DeformableEncoderLayer(d, n_levels, nh, n_points, ff, dr)
            for _ in range(ne)
        ])
        self.enc_norm = nn.LayerNorm(d)

        # ── Decoder: Full Global Cross-Attention ──────────────────────────
        self.decoder  = nn.ModuleList([
            GlobalDecoderLayer(d, nh, ff, dr)
            for _ in range(nd)
        ])
        self.dec_norm = nn.LayerNorm(d)

        # ── Object Queries: content (d) + positional (d) ──────────────────
        self.query_embed = nn.Embedding(nq, d * 2)

        # ── Prediction Heads ──────────────────────────────────────────────
        self.cls_head  = nn.Linear(d, nc + 1)
        self.bbox_head = MLP(d, d, 4, 3)

        self._init_weights()

    def _init_weights(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.constant_(proj[0].bias, 0)
        self.cls_head.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
        nn.init.constant_(self.bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias,   0)
        for layer in [*self.encoder, *self.decoder]:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    @staticmethod
    def _encoder_ref_points(spatial_shapes, device):
        pts = []
        for H, W in spatial_shapes.tolist():
            H, W = int(H), int(W)
            ys = torch.linspace(0.5/H, 1.0 - 0.5/H, H, device=device)
            xs = torch.linspace(0.5/W, 1.0 - 0.5/W, W, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            pts.append(torch.stack([gx.flatten(), gy.flatten()], -1))
        pts = torch.cat(pts, 0)
        return pts.unsqueeze(0).unsqueeze(2).expand(1, -1, len(spatial_shapes), -1)

    def forward(self, x):
        B = x.shape[0]

        # Multi-scale backbone features
        c3 = self.backbone_c3(x)
        c4 = self.backbone_c4(c3)
        c5 = self.backbone_c5(c4)
        feats = [
            self.input_proj[0](c3),
            self.input_proj[1](c4),
            self.input_proj[2](c5),
            self.input_proj[3](c5),
        ]

        # Build flattened multi-scale sequence + positional embeddings
        src_flat, pos_flat = [], []
        spatial_shapes, lvl_starts = [], []
        idx = 0
        for lvl, feat in enumerate(feats):
            b, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            lvl_starts.append(idx)
            idx += h * w
            pos = get_sine_pos(feat)
            pos_flat.append(
                pos.flatten(2).transpose(1, 2)
                + self.level_embed.weight[lvl].view(1, 1, -1)
            )
            src_flat.append(feat.flatten(2).transpose(1, 2))

        src_flat = torch.cat(src_flat, 1)   # (B, sum_HW, d)
        pos_flat = torch.cat(pos_flat, 1)   # (B, sum_HW, d)  -- kept for decoder
        sp_t  = torch.tensor(spatial_shapes, dtype=torch.long, device=x.device)
        ls_t  = torch.tensor(lvl_starts,     dtype=torch.long, device=x.device)

        # Encoder (Deformable)
        enc_ref = self._encoder_ref_points(sp_t, x.device).expand(B, -1, -1, -1)
        memory  = src_flat
        for layer in self.encoder:
            memory = layer(memory, pos_flat, enc_ref, sp_t, ls_t)
        memory = self.enc_norm(memory)      # (B, sum_HW, d)

        # Decoder (Full Global Cross-Attention)
        qe        = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt       = qe[..., :self.d]        # (B, nq, d) content
        query_pos = qe[..., self.d:]        # (B, nq, d) position

        # 2D reference points for box refinement (not used in cross-attn here)
        ref_2d = query_pos[..., :2].sigmoid()   # (B, nq, 2)

        intermediates = []
        for layer in self.decoder:
            # Pass memory + its positional embedding so decoder has full spatial context
            tgt = layer(tgt, query_pos, memory, pos_flat)
            intermediates.append(self.dec_norm(tgt))

        # Box: offset + reference point (iterative refinement)
        def pred_box(feat):
            delta = self.bbox_head(feat)
            xy    = (delta[..., :2] + inverse_sigmoid(ref_2d)).sigmoid()
            wh    = delta[..., 2:].sigmoid()
            return torch.cat([xy, wh], -1)

        out = intermediates[-1]
        return {
            'pred_logits': self.cls_head(out),
            'pred_boxes':  pred_box(out),
            'aux_outputs': [
                {'pred_logits': self.cls_head(f),
                 'pred_boxes':  pred_box(f)}
                for f in intermediates[:-1]
            ],
        }

    def bb_params(self):
        return (list(self.backbone_c3.parameters()) +
                list(self.backbone_c4.parameters()) +
                list(self.backbone_c5.parameters()))

    def other_params(self):
        bb_ids = {id(p) for p in self.bb_params()}
        return [p for p in self.parameters() if id(p) not in bb_ids]

    def freeze_backbone_bn(self):
        for mod in [self.backbone_c3, self.backbone_c4, self.backbone_c5]:
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


# ====================== Loss Functions ======================
```

---

### Loss functions

**HungarianMatcher** solves bipartite matching between predictions and ground truth using a cost matrix
(classification + L1 bbox + GIoU, weights 2.0 / 5.0 / 4.0).

**SetCriterion** computes the total training loss after matching:
`L = 1.0 × L_CE + 5.0 × L_L1 + 6.0 × L_GIoU`
Auxiliary losses from all intermediate decoder layers are added with weight 0.5.

```python
def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], -1)


def compute_giou(b1, b2):
    a1 = ((b1[:,2]-b1[:,0]).clamp(min=0) * (b1[:,3]-b1[:,1]).clamp(min=0))
    a2 = ((b2[:,2]-b2[:,0]).clamp(min=0) * (b2[:,3]-b2[:,1]).clamp(min=0))
    ix1 = torch.max(b1[:,None,0], b2[:,0])
    iy1 = torch.max(b1[:,None,1], b2[:,1])
    ix2 = torch.min(b1[:,None,2], b2[:,2])
    iy2 = torch.min(b1[:,None,3], b2[:,3])
    inter = (ix2-ix1).clamp(min=0) * (iy2-iy1).clamp(min=0)
    union = (a1[:,None] + a2 - inter).clamp(min=1e-7)
    iou   = inter / union
    ex1 = torch.min(b1[:,None,0], b2[:,0])
    ey1 = torch.min(b1[:,None,1], b2[:,1])
    ex2 = torch.max(b1[:,None,2], b2[:,2])
    ey2 = torch.max(b1[:,None,3], b2[:,3])
    enc  = ((ex2-ex1).clamp(min=0) * (ey2-ey1).clamp(min=0)).clamp(min=1e-7)
    giou = iou - (enc - union) / enc
    return giou.clamp(min=-1.0, max=1.0)


class HungarianMatcher(nn.Module):
    def __init__(self, cc=2.0, cb=5.0, cg=4.0):
        super().__init__()
        self.cc = cc; self.cb = cb; self.cg = cg

    @torch.no_grad()
    def forward(self, out, tgts):
        B, nq  = out['pred_logits'].shape[:2]
        op     = out['pred_logits'].flatten(0,1).softmax(-1)
        ob     = out['pred_boxes'].flatten(0,1)
        ti     = torch.cat([t['labels'] for t in tgts])
        tb     = torch.cat([t['boxes']  for t in tgts])
        if len(ti) == 0:
            empty = torch.tensor([], dtype=torch.int64)
            return [(empty, empty)] * B
        C = (
            self.cc * (-op[:, ti])
            + self.cb * torch.cdist(ob, tb, p=1)
            + self.cg * (-compute_giou(cxcywh_to_xyxy(ob), cxcywh_to_xyxy(tb)))
        )
        C = C.view(B, nq, -1).cpu()
        C[~torch.isfinite(C)] = 1e4
        sizes = [len(t['boxes']) for t in tgts]
        idx   = []
        for i, chunk in enumerate(C.split(sizes, -1)):
            if sizes[i] == 0:
                empty = torch.tensor([], dtype=torch.int64)
                idx.append((empty, empty))
            else:
                r, c = linear_sum_assignment(chunk[i])
                idx.append((
                    torch.as_tensor(r, dtype=torch.int64),
                    torch.as_tensor(c, dtype=torch.int64)
                ))
        return idx


class SetCriterion(nn.Module):
    def __init__(self, nc=10, eos=0.02):
        super().__init__()
        self.nc      = nc
        self.matcher = HungarianMatcher()
        self.wd      = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 6.0}
        ew           = torch.ones(nc + 1); ew[-1] = eos
        self.register_buffer('ew', ew)

    def _single_loss(self, out, targets):
        dev = out['pred_logits'].device
        out = {
            'pred_logits': out['pred_logits'].clamp(-10, 10),
            'pred_boxes':  out['pred_boxes'].clamp(1e-4, 1 - 1e-4)
        }
        proc = [{'boxes': t['boxes'].to(dev), 'labels': (t['labels']-1).to(dev)}
                for t in targets]
        idx  = self.matcher(out, proc)
        nb   = max(sum(len(t['labels']) for t in proc), 1)
        B, nq, _ = out['pred_logits'].shape
        tc = torch.full((B, nq), self.nc, dtype=torch.int64, device=dev)
        for i, (r, c) in enumerate(idx):
            if len(r) > 0: tc[i, r] = proc[i]['labels'][c]
        lce = F.cross_entropy(
            out['pred_logits'].flatten(0,1), tc.flatten(),
            weight=self.ew.to(dev), label_smoothing=0.1
        )
        if not torch.isfinite(lce):
            lce = out['pred_logits'].sum() * 0
        bi = torch.cat([torch.full_like(r, i) for i, (r, _) in enumerate(idx)])
        qi = torch.cat([r for r, _ in idx])
        sb = out['pred_boxes'][bi, qi]
        tb = torch.cat([proc[i]['boxes'][c] for i, (_, c) in enumerate(idx)])
        if len(sb) == 0:
            _z = out['pred_boxes'].sum() * 0 + out['pred_logits'].sum() * 0
            lb = lg = _z
        else:
            sb   = sb.clamp(1e-4, 1 - 1e-4)
            tb   = tb.clamp(1e-4, 1 - 1e-4)
            lb   = F.l1_loss(sb, tb, reduction='sum') / nb
            giou = compute_giou(cxcywh_to_xyxy(sb), cxcywh_to_xyxy(tb)).diag()
            lg   = (1 - giou).sum() / nb
            if not torch.isfinite(lb): lb = out['pred_boxes'].sum() * 0
            if not torch.isfinite(lg): lg = out['pred_boxes'].sum() * 0
        losses = {'loss_ce': lce, 'loss_bbox': lb, 'loss_giou': lg}
        losses['total'] = sum(self.wd[k] * losses[k] for k in self.wd)
        return losses

    def forward(self, out, targets):
        losses = self._single_loss(out, targets)
        if 'aux_outputs' in out:
            for aux in out['aux_outputs']:
                aux_l = self._single_loss(
                    {'pred_logits': aux['pred_logits'].clamp(-10, 10),
                     'pred_boxes':  aux['pred_boxes'].clamp(1e-4, 1 - 1e-4)},
                    targets)
                if torch.isfinite(aux_l['total']):
                    losses['total'] = losses['total'] + aux_l['total'] * 0.5
        return losses


def to_dev(t, d):
    return [{k: v.to(d) if isinstance(v, torch.Tensor) else v
             for k, v in x.items()} for x in t]


# ====================== mAP Evaluation ======================
```

---

### Evaluation (mAP)

`evaluate_map` runs inference on the validation set and computes COCO mAP using pycocotools.
`nms` applies class-agnostic Non-Maximum Suppression: all boxes compete together regardless of class,
and any pair with IoU > threshold keeps only the higher-score box.

```python
def evaluate_map(model, val_ann_file, val_img_dir, img_size,
                 batch_size, device, amp, threshold=0.3):
    ds     = TestDataset(val_img_dir, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()
    results = []
    S = img_size
    with torch.no_grad():
        for batch in loader:
            imgs, ids, ws, hs, scales, pxs, pys = batch
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda', enabled=amp):
                out = model(imgs)
            probs = out['pred_logits'].softmax(-1)
            boxes = out['pred_boxes']
            for b in range(imgs.shape[0]):
                ow = float(ws[b]); oh = float(hs[b])
                sc_v = float(scales[b]); px = float(pxs[b]); py = float(pys[b])
                sc, cl = probs[b, :, :-1].max(-1)
                keep   = sc > threshold
                bx_list, sc_list, cl_list = [], [], []
                for s, c, bx in zip(sc[keep], cl[keep], boxes[b][keep]):
                    cx, cy, bw, bh = bx.tolist()
                    x = max(0.0, (cx - bw/2) * S - px) / sc_v
                    y = max(0.0, (cy - bh/2) * S - py) / sc_v
                    w = min(bw * S / sc_v, ow - x)
                    h = min(bh * S / sc_v, oh - y)
                    if w > 1 and h > 1:
                        bx_list.append([x, y, x+w, y+h])
                        sc_list.append(float(s))
                        cl_list.append(int(c))
                if bx_list:
                    bx_t = torch.tensor(bx_list)
                    sc_t = torch.tensor(sc_list)
                    cl_t = torch.tensor(cl_list)
                    for cat in cl_t.unique():
                        mask = cl_t == cat
                        idx  = mask.nonzero(as_tuple=True)[0]
                        kept = nms(bx_t[mask], sc_t[mask], iou_threshold=0.3)
                        for ki in kept:
                            x1b, y1b, x2b, y2b = bx_list[idx[ki]]
                            results.append({
                                'image_id':    int(ids[b]),
                                'bbox':        [round(x1b,2), round(y1b,2),
                                                round(x2b-x1b,2), round(y2b-y1b,2)],
                                'score':       round(sc_list[idx[ki]], 6),
                                'category_id': cl_list[idx[ki]] + 1,
                            })
    if len(results) == 0:
        print('    [mAP] 0 detections -- model not yet converged')
        return 0.0, 0.0
    coco_gt = COCO(val_ann_file)
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def nms(boxes_xyxy, scores, iou_threshold=0.5):
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = (boxes_xyxy[:,0], boxes_xyxy[:,1],
                       boxes_xyxy[:,2], boxes_xyxy[:,3])
    areas = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep  = []
    while len(order):
        i = order[0].item()
        keep.append(i)
        if len(order) == 1: break
        rest  = order[1:]
        ix1   = x1[rest].clamp(min=x1[i].item())
        iy1   = y1[rest].clamp(min=y1[i].item())
        ix2   = x2[rest].clamp(max=x2[i].item())
        iy2   = y2[rest].clamp(max=y2[i].item())
        inter = (ix2-ix1).clamp(min=0) * (iy2-iy1).clamp(min=0)
        iou   = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-7)
        order = rest[iou <= iou_threshold]
    return keep


print('All HybridDETR definitions loaded!')
```

---

### Training loop

Sets up CFG, model, optimizer (AdamW), LR scheduler (warmup + step decay), and AMP scaler.
Trains for up to 80 epochs with early stopping (patience 15).
Best model weights are kept in memory and loaded back after training completes.

```python
# ==================== Training ====================
CFG = dict(
    nc=10,
    nq=30,
    n_levels=4,
    n_points=4,
    d=256, nh=8,
    ne=4,
    nd=4,
    ff=1024,
    dr=0.1,
    img=512,
    bs=8,
    accum=1,
    epochs=80,
    lr=2e-4,
    lr_bb=2e-6,
    wd=1e-4,
    warmup_epochs=5,
    min_lr_ratio=0.01,
    clip=0.1,
    nw=4, amp=True,
    map_threshold=0.3,
    map_every=1,
    early_stop_patience=15,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

train_ds = DigitDataset(
    os.path.join(DATA_ROOT, 'train'),
    os.path.join(DATA_ROOT, 'train.json'),
    CFG['img'], True
)
eval_ds = DigitDataset(
    os.path.join(DATA_ROOT, 'valid'),
    os.path.join(DATA_ROOT, 'valid.json'),
    CFG['img'], False
)
train_loader = DataLoader(
    train_ds, CFG['bs'], shuffle=True,
    num_workers=CFG['nw'], collate_fn=collate_fn,
    pin_memory=True, drop_last=True
)
eval_loader = DataLoader(
    eval_ds, CFG['bs'], shuffle=False,
    num_workers=CFG['nw'], collate_fn=collate_fn,
    pin_memory=True
)
print(f'Train: {len(train_ds)} | Val: {len(eval_ds)} | Steps/ep: {len(train_loader)}')

model = HybridDETR(
    nc=CFG['nc'], nq=CFG['nq'], d=CFG['d'], nh=CFG['nh'],
    ne=CFG['ne'], nd=CFG['nd'], ff=CFG['ff'],
    n_levels=CFG['n_levels'], n_points=CFG['n_points'],
    dr=CFG['dr'], pt=True
).to(device)
criterion = SetCriterion(CFG['nc'])
optimizer = optim.AdamW(
    [
        {'params': model.bb_params(),    'lr': CFG['lr_bb']},
        {'params': model.other_params(), 'lr': CFG['lr']},
    ],
    weight_decay=CFG['wd']
)

def make_step_lr_fn(warmup, total_epochs, min_lr_ratio=0.01, start_decay=45):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        if ep < start_decay:
            return 1.0
        return min_lr_ratio
    return fn

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    [make_step_lr_fn(CFG['warmup_epochs'], CFG['epochs'],
                     CFG['min_lr_ratio'], start_decay=45)] * 2
)
scaler = torch.amp.GradScaler('cuda', enabled=CFG['amp'], init_scale=256.0, growth_interval=2000)

best_map        = 0.0
best_model_state = None   # keep best weights in memory
hist = {
    'train_loss': [], 'val_loss': [],
    'val_ce': [], 'val_bb': [], 'val_gi': [],
    'map_5095': [], 'map_50': [], 'map_epochs': []
}
nan_count = 0

# ══════════════════ Training Loop ══════════════════
for ep in range(CFG['epochs']):
    t0 = time.time()

    # ── Train ──
    model.train()
    criterion.train()
    model.freeze_backbone_bn()

    running_loss  = 0.0
    valid_steps   = 0
    skipped_steps = 0
    t_step = time.time()
    optimizer.zero_grad()

    for step, (imgs, tgts) in enumerate(train_loader):
        imgs = imgs.to(device)
        tgts = to_dev(tgts, device)

        with torch.amp.autocast('cuda', enabled=CFG['amp']):
            losses = criterion(model(imgs), tgts)
            loss   = losses['total'] / CFG['accum']

        if not torch.isfinite(loss):
            skipped_steps += 1
            optimizer.zero_grad()
            if CFG['amp'] and scaler._scale is not None:
                scaler._scale.mul_(0.5).clamp_(min=1.0)
            if skipped_steps % 50 == 1:
                print(f'  ⚠ NaN at step {step+1}, skipped {skipped_steps}')
            continue

        if not loss.requires_grad:
            skipped_steps += 1
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % CFG['accum'] == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += losses['total'].item()
        valid_steps  += 1

        if (step + 1) % 400 == 0:
            print(
                f'  [Ep{ep+1}|{step+1}/{len(train_loader)}] '
                f'total={losses["total"].item():.3f} '
                f'ce={losses["loss_ce"].item():.3f} '
                f'bb={losses["loss_bbox"].item():.3f} '
                f'gi={losses["loss_giou"].item():.3f} '
                f'lr={optimizer.param_groups[1]["lr"]:.2e} '
                f'({time.time()-t_step:.0f}s)'
            )
            t_step = time.time()

    if valid_steps == 0:
        train_loss = float('nan')
        nan_count += 1
        print(f'  ⚠ Entire epoch NaN ({nan_count}/3)')
        scheduler.step()
        hist['train_loss'].append(train_loss)
        for k in ['val_loss', 'val_ce', 'val_bb', 'val_gi']:
            hist[k].append(float('nan'))
        continue
    else:
        nan_count  = 0
        train_loss = running_loss / valid_steps

    # ── Validation Loss ──
    model.eval()
    criterion.eval()
    vt = vc = vb = vg = 0.0
    valid_val = 0
    with torch.no_grad():
        for imgs, tgts in eval_loader:
            imgs = imgs.to(device)
            tgts = to_dev(tgts, device)
            with torch.amp.autocast('cuda', enabled=CFG['amp']):
                ls = criterion(model(imgs), tgts)
            if not torch.isfinite(ls['total']):
                continue
            vt += ls['total'].item()
            vc += ls['loss_ce'].item()
            vb += ls['loss_bbox'].item()
            vg += ls['loss_giou'].item()
            valid_val += 1

    nv       = max(valid_val, 1)
    val_loss = vt / nv

    scheduler.step()

    hist['train_loss'].append(train_loss)
    hist['val_loss'].append(val_loss)
    hist['val_ce'].append(vc / nv)
    hist['val_bb'].append(vb / nv)
    hist['val_gi'].append(vg / nv)

    # ── mAP ──
    map_5095 = map_50 = 0.0
    if (ep + 1) % CFG['map_every'] == 0:
        map_5095, map_50 = evaluate_map(
            model,
            os.path.join(DATA_ROOT, 'valid.json'),
            os.path.join(DATA_ROOT, 'valid'),
            CFG['img'], 8, device, CFG['amp'],
            CFG['map_threshold']
        )
        hist['map_5095'].append(map_5095)
        hist['map_50'].append(map_50)
        hist['map_epochs'].append(ep + 1)

    print(
        f'\n═ Ep {ep+1:>2}/{CFG["epochs"]} '
        f'| Train={train_loss:.3f} Val={val_loss:.3f} '
        f'| mAP@.5:.95={map_5095:.4f} mAP@.5={map_50:.4f} '
        f'| LR={optimizer.param_groups[1]["lr"]:.2e} '
        f'| {time.time()-t0:.0f}s ═\n'
    )

    # Keep best weights in memory
    if map_5095 > best_map:
        best_map         = map_5095
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f'  ★ New best mAP = {best_map:.4f}\n')

    # ── Early Stopping ──
    patience = CFG.get('early_stop_patience', 12)
    if len(hist['map_5095']) >= patience:
        recent_best  = max(hist['map_5095'][-patience:])
        overall_best = max(hist['map_5095'][:-patience]) if len(hist['map_5095']) > patience else -1
        if recent_best <= overall_best:
            print(f'  ★ Early stopping at ep{ep+1}')
            break

print(f'\nDone! Best mAP@0.5:0.95 = {best_map:.4f}')

# Load best weights back into model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print('Best weights loaded into model.')

```

---


---

## Performance Snapshot

| Metric | Value |
|---|---|
| mAP@0.5:0.95 (val) | 0.4240 |
| Best epoch | 72 |
| Kaggle public score | 0.33 |

![Leaderboard](leaderboard.png)
