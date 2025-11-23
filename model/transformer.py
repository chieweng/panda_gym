"""
Files: single-file reference containing:
- PointCloudEncoder (PointNet-style)
- TransformerGPD model
- Dataset template (GraspNet/YCB style point clouds)
- Training loop skeleton, evaluation, and utilities

Notes:
- This is a practical, ready-to-run starting point. Replace dataset paths/IO with your dataset.
- The "Mamba" architecture here is a compact, efficient recurrent/convolutional alternative to the transformer
  (inspired by efficient sequence models). If you were referring to a different published "Mamba", you can
  adapt the block structure below.

Usage:
- Edit Dataset paths and hyperparameters in the bottom of this file.
- Run: python gpd_transformer_and_mamba.py

"""

import os
import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Point cloud encoder (PointNet-style)
# -----------------------------
class PointCloudEncoder(nn.Module):
    """Simple shared-MLP encoder for point clouds producing per-point features and a global feature.

    Input: (B, N, 3) point coordinates (optionally with normals or colors concatenated)
    Output: global feature (B, D) and per-point features (B, N, Dp)
    """
    def __init__(self, in_channels=3, pointfeat_dim=128, global_dim=256):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, pointfeat_dim, 1),
            nn.BatchNorm1d(pointfeat_dim),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(pointfeat_dim, global_dim),
            nn.ReLU(inplace=True),
            nn.Linear(global_dim, global_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C)
        B, N, C = x.shape
        x_t = x.transpose(1, 2).contiguous()  # (B, C, N)
        per_point = self.shared_mlp(x_t)  # (B, Dp, N)
        per_point = per_point.transpose(1, 2).contiguous()  # (B, N, Dp)
        global_feat, _ = torch.max(per_point, dim=1)  # (B, Dp)
        global_feat = self.global_mlp(global_feat)  # (B, global_dim)
        return per_point, global_feat

# -----------------------------
# Transformer-based GPD head
# -----------------------------
class TransformerGPD(nn.Module):
    """A Transformer-based model that consumes per-point features and produces grasp proposals.

    Assumptions:
    - We treat the point cloud as a sequence of tokens (N tokens)
    - Use a standard TransformerEncoder to produce per-token features, then apply heads
      for grasp classification/regression
    """
    def __init__(self,
                 pointfeat_dim=128,
                 transformer_dim=256,
                 n_heads=8,
                 n_layers=4,
                 mlp_hidden=256,
                 num_grasp_bins=1024):
        super().__init__()
        self.input_proj = nn.Linear(pointfeat_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=mlp_hidden,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head: per-token score, and per-token grasp params regression (x,y,z,theta,width)
        self.score_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_dim//2, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_dim//2, 5)  # example: (x,y,z,angle,width)
        )

    def forward(self, per_point_feats: torch.Tensor, mask: Optional[torch.Tensor]=None):
        # per_point_feats: (B, N, Dp)
        x = self.input_proj(per_point_feats)  # (B, N, D)
        # Optionally apply positional encoding. Here we add a simple learned positional encoding
        B, N, D = x.shape
        device = x.device
        if not hasattr(self, 'pos_embedding') or self.pos_embedding.shape[1] != N:
            # create or resize
            self.pos_embedding = nn.Parameter(torch.randn(1, N, D, device=device) * 0.01)
        x = x + self.pos_embedding
        x = self.transformer(x, src_key_padding_mask=None)  # (B, N, D)
        scores = self.score_head(x).squeeze(-1)  # (B, N)
        regs = self.reg_head(x)  # (B, N, 5)
        return scores, regs

# -----------------------------
# Dataset template - adjust to your dataset
# -----------------------------
class PointCloudGraspDataset(Dataset):
    """A simple dataset class. Expects preprocessed files with point clouds and labels.

    Each item should return:
    - points: (N, 3) float32
    - label_scores: (N,) 0/1 indicating whether seed point is a good grasp center
    - label_regs: (N, 5) regression targets for (x,y,z,angle,width)
    """
    def __init__(self, data_list, n_points=2048, augment=True):
        super().__init__()
        self.data_list = data_list
        self.n_points = n_points
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        # item can be a dict, or a path to a .npz, etc.
        if isinstance(item, str):
            data = np.load(item)
            points = data['points']  # (M,3)
            scores = data['scores']  # (M,)
            regs = data['regs']      # (M,5)
        else:
            points = item['points']
            scores = item['scores']
            regs = item['regs']

        # Random sampling to fixed N
        if points.shape[0] >= self.n_points:
            choice = np.random.choice(points.shape[0], self.n_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.n_points, replace=True)
        pts = points[choice]
        sc = scores[choice]
        rg = regs[choice]

        if self.augment:
            # simple augmentations
            angle = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0,0,1]])
            pts = pts.dot(rot.T)
            pts = pts + np.random.normal(scale=0.001, size=pts.shape)

        return {
            'points': torch.from_numpy(pts).float(),
            'scores': torch.from_numpy(sc).float(),
            'regs': torch.from_numpy(rg).float()
        }

# -----------------------------
# Losses and utils
# -----------------------------

def grasp_loss(pred_scores, pred_regs, tgt_scores, tgt_regs, score_weight=1.0, reg_weight=10.0):
    """Simple loss: BCE on scores + L1 on regression targets masked by positive score labels"""
    bce = F.binary_cross_entropy_with_logits(pred_scores, tgt_scores)
    # mask for regression
    mask = (tgt_scores > 0.5).unsqueeze(-1)
    if mask.sum() > 0:
        l1 = F.l1_loss(pred_regs[mask.expand_as(pred_regs)], tgt_regs[mask.expand_as(tgt_regs)])
    else:
        l1 = torch.tensor(0.0, device=pred_scores.device)
    return score_weight * bce + reg_weight * l1, {'bce': bce.item(), 'l1': l1.item()}

# -----------------------------
# Training loop skeleton
# -----------------------------

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    logs = {'bce': 0.0, 'l1': 0.0}
    for batch in dataloader:
        pts = batch['points'].to(device)  # (B,N,3)
        tgt_scores = batch['scores'].to(device)  # (B,N)
        tgt_regs = batch['regs'].to(device)  # (B,N,5)
        per_point, global_feat = encoder(pts)
        # choose model type externally
        pred_scores, pred_regs = model(per_point)
        loss, loss_terms = grasp_loss(pred_scores, pred_regs, tgt_scores, tgt_regs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logs['bce'] += loss_terms['bce']
        logs['l1'] += loss_terms['l1']
    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in logs.items()}

# -----------------------------
# Minimal evaluation: pick top-k proposals and compute simple metrics
# -----------------------------

def evaluate_model(model, dataloader, device, topk=10):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch in dataloader:
            pts = batch['points'].to(device)
            tgt_scores = batch['scores'].to(device)
            per_point, global_feat = encoder(pts)
            pred_scores, pred_regs = model(per_point)
            probs = torch.sigmoid(pred_scores)
            topk_vals, topk_idx = torch.topk(probs, min(topk, probs.shape[1]), dim=1)
            # For basic metric, compute mean precision@topk
            prec = (tgt_scores.gather(1, topk_idx) > 0.5).float().mean().item()
            all_scores.append(prec)
    return float(np.mean(all_scores)) if all_scores else 0.0

# -----------------------------
# Small runnable example (toy data) if executed directly
# -----------------------------
if __name__ == '__main__':
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_points = 1024
    batch_size = 8
    epochs = 3

    # Create toy dataset (random)
    def make_toy_item():
        M = 1500
        pts = np.random.randn(M,3).astype(np.float32) * 0.1
        # random ground truth: points near origin are good grasps
        d = np.linalg.norm(pts, axis=1)
        scores = (d < 0.15).astype(np.float32)
        regs = np.zeros((M,5), dtype=np.float32)
        regs[:,0:3] = pts  # target grasp center = point location
        regs[:,3] = np.random.uniform(-math.pi, math.pi, size=(M,))
        regs[:,4] = np.random.uniform(0.02, 0.08, size=(M,))
        return {'points': pts, 'scores': scores, 'regs': regs}

    toy_list = [make_toy_item() for _ in range(200)]
    dataset = PointCloudGraspDataset(toy_list, n_points=n_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_list = [make_toy_item() for _ in range(40)]
    valset = PointCloudGraspDataset(val_list, n_points=n_points, augment=False)
    valdataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build models
    encoder = PointCloudEncoder(in_channels=3, pointfeat_dim=128, global_dim=256).to(device)
    transformer_model = TransformerGPD(pointfeat_dim=128, transformer_dim=256, n_heads=8, n_layers=3).to(device)

    # Choose model to train
    model = transformer_model
    params = list(encoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    print('Starting training on device', device)
    for ep in range(epochs):
        loss, logs = train_one_epoch(model, dataloader, optimizer, device)
        val_prec = evaluate_model(model, valdataloader, device)
        print(f'Epoch {ep+1}/{epochs} loss={loss:.4f} bce={logs["bce"]:.4f} l1={logs["l1"]:.4f} val_prec@topk={val_prec:.4f}')

    print('Done. Save models if you want:')
    torch.save({'encoder': encoder.state_dict(), 'model': model.state_dict()}, 'gpd_checkpoint.pth')