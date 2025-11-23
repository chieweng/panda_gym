"""
PointMamba GPD - Point-Mamba implementation (PyTorch)

This file updates the earlier GPD Transformer / Mamba reference with a more faithful
Point-Mamba style backbone inspired by the PointMamba paper and the Mamba selective SSM.

Key features implemented here:
- Space-filling curve tokenization (Morton/Z-order) to serialize point clouds into sequences.
- A faithful (but readable) Selective SSM / MambaBlock implementation using per-token gates
  that compute a simple linear recurrence: h_t = a_t * h_{t-1} + b_t * x_t.
  This follows the selective SSM idea from the Mamba paper; it is functionally equivalent
  to a time-varying state-space model but implemented in straightforward PyTorch for clarity.
- PointMamba-style encoder: tokenization -> input projection -> several MambaBlocks (non-hierarchical)
- GPD heads: per-token score and regression (x,y,z,quat,width) — changeable to your preferred param.
- Toy training loop and notes on where to replace with optimized selective-scan implementations.

References:
- PointMamba: "PointMamba: A Simple State Space Model for Point Cloud Analysis" (NeurIPS 2024). See arXiv:2402.10739.
- Mamba: "Linear-Time Sequence Modeling with Selective State Spaces" (2023). See arXiv:2312.00752.

Notes on fidelity and performance:
- The true high-performance Mamba uses a hardware-aware selective-scan algorithm to compute
  the state evolution in parallel with O(n) time and good memory locality. Here I implement
  a clear, correct selective-SSM recurrence using Python loops for readability. That implementation
  is slower on long sequences but is correct and easy to debug. If you want, I can replace the loop
  with a batched parallel selective-scan kernel (or use an available optimized Mamba implementation).

Requirements:
- Python 3.8+
- PyTorch
- numpy

Install minimal:
    pip install torch numpy

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities: Morton (Z-order) 3D interleave for space-filling curve tokenization
# -----------------------------

def _part1by2(n):
    # 10-bit split helper for inputs assumed in [0,1023]
    n &= 0x3ff
    n = (n | (n << 16)) & 0x30000ff
    n = (n | (n << 8)) & 0x300f00f
    n = (n | (n << 4)) & 0x30c30c3
    n = (n | (n << 2)) & 0x9249249
    return n

def morton3D(x, y, z, max_val=1023):
    # x,y,z arrays of same shape with values normalized to [0, max_val]
    xi = np.clip((x * max_val).astype(np.int64), 0, max_val)
    yi = np.clip((y * max_val).astype(np.int64), 0, max_val)
    zi = np.clip((z * max_val).astype(np.int64), 0, max_val)
    code = (_part1by2(xi) << 2) | (_part1by2(yi) << 1) | _part1by2(zi)
    return code

def space_filling_token_order(points: np.ndarray, method='morton'):
    # points: (N,3) in object/local coordinates. Returns permutation indices for serialization.
    if method == 'morton':
        # normalize points to [0,1] bounding box
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        spans = maxs - mins
        spans[spans == 0] = 1e-6
        norm = (points - mins) / spans
        codes = morton3D(norm[:,0], norm[:,1], norm[:,2], max_val=1023)
        order = np.argsort(codes)
        return order
    else:
        raise ValueError('Unknown method')

# -----------------------------
# Selective SSM (Mamba) building block (readable / correct version)
# -----------------------------
class SelectiveSSMBlock(nn.Module):
    """A readable Selective SSM block implementing a simple per-token linear recurrence.

    Recurrence (per-feature channel):
        h_t = a_t * h_{t-1} + b_t * u_t
        y_t = W_o h_t
    where a_t and b_t are scalars (or vectors) computed from the input token u_t.

    Note: This is a simplified but faithful realization of the selective SSM idea.
    The true Mamba uses a batched parallel selective-scan algorithm for speed; here we
    implement the recurrence in a loop for clarity. For production, replace it with
    a parallel implementation (or use the state-spaces/mamba codegen).
    """
    def __init__(self, dim, state_dim=None, gate_hidden=64, use_vector_a=True):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim or dim
        # input projection
        self.in_proj = nn.Linear(dim, self.state_dim)
        # compute gates a_t and b_t from token
        self.gate_net = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, self.state_dim * 2)
        )
        # output projection
        self.out_proj = nn.Linear(self.state_dim, dim)
        # optional layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, dim)
        B, N, D = x.shape
        u = self.in_proj(x)       # (B, N, state_dim)
        gates = self.gate_net(x)  # (B, N, 2*state_dim)
        gates = gates.view(B, N, 2, self.state_dim)
        a = torch.sigmoid(gates[:,:,0,:])  # forget/propagate gate in (0,1)
        b = torch.sigmoid(gates[:,:,1,:])  # input scaling gate

        # initialize state
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        # naive recurrence (loop) - correct but not the fast selective-scan used in Mamba
        for t in range(N):
            ut = u[:,t,:]            # (B, state_dim)
            at = a[:,t,:]
            bt = b[:,t,:]
            # elementwise recurrence
            h = at * h + bt * ut
            y_t = self.out_proj(h)   # (B, dim)
            outputs.append(y_t.unsqueeze(1))
        y = torch.cat(outputs, dim=1)  # (B, N, dim)
        y = self.norm(y + x)
        return y

# -----------------------------
# PointMamba backbone
# -----------------------------
class PointMambaBackbone(nn.Module):
    def __init__(self, in_dim=3, pointfeat_dim=128, hidden_dim=256, n_blocks=6, token_method='morton'):
        super().__init__()
        self.token_method = token_method
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, pointfeat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pointfeat_dim, pointfeat_dim)
        )
        self.input_proj = nn.Linear(pointfeat_dim, hidden_dim)
        self.blocks = nn.ModuleList([SelectiveSSMBlock(hidden_dim, state_dim=hidden_dim) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, pts):
        # pts: (B, N, 3) raw coordinates
        B, N, C = pts.shape
        # Tokenize each sample with space-filling curve
        device = pts.device
        pts_np = pts.detach().cpu().numpy()
        orders = [space_filling_token_order(pts_np[b], method=self.token_method) for b in range(B)]
        # apply encoder
        x = self.encoder(pts)  # (B, N, pointfeat_dim)
        # reorder per sample
        x_ordered = x.clone()
        for b in range(B):
            idx = torch.from_numpy(orders[b]).to(device)
            x_ordered[b] = x[b, idx, :]
        # project to hidden
        h = self.input_proj(x_ordered)
        # pass through Mamba blocks
        for blk in self.blocks:
            h = blk(h)
        return h  # (B, N, hidden_dim)

# -----------------------------
# GPD model using PointMamba backbone
# -----------------------------
class PointMambaGPD(nn.Module):
    def __init__(self, in_dim=3, pointfeat_dim=128, hidden_dim=256, n_blocks=6):
        super().__init__()
        self.backbone = PointMambaBackbone(in_dim, pointfeat_dim, hidden_dim, n_blocks)
        # heads
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 1)
        )
        # regression: position (relative to point), quaternion (4), width
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, 8)  # dx,dy,dz, qw,qx,qy,qz, width
        )

    def forward(self, pts):
        # pts: (B,N,3)
        h = self.backbone(pts)
        scores = self.score_head(h).squeeze(-1)
        regs = self.reg_head(h)
        return scores, regs

# -----------------------------
# Dataset template (same as before, but with ordering note)
# -----------------------------
class PointCloudGraspDataset(Dataset):
    def __init__(self, data_list, n_points=2048, augment=True):
        super().__init__()
        self.data_list = data_list
        self.n_points = n_points
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        if isinstance(item, str):
            data = np.load(item)
            points = data['points']
            scores = data['scores']
            regs = data['regs']
        else:
            points = item['points']
            scores = item['scores']
            regs = item['regs']

        # sample N points
        if points.shape[0] >= self.n_points:
            choice = np.random.choice(points.shape[0], self.n_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.n_points, replace=True)
        pts = points[choice]
        sc = scores[choice]
        rg = regs[choice]

        if self.augment:
            angle = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0,0,1]])
            pts = pts.dot(rot.T)
            pts = pts + np.random.normal(scale=0.001, size=pts.shape)

        return {'points': torch.from_numpy(pts).float(), 'scores': torch.from_numpy(sc).float(), 'regs': torch.from_numpy(rg).float()}

# -----------------------------
# Loss and simple train loop (toy)
# -----------------------------

def grasp_loss(pred_scores, pred_regs, tgt_scores, tgt_regs, score_weight=1.0, reg_weight=10.0):
    bce = F.binary_cross_entropy_with_logits(pred_scores, tgt_scores)
    mask = (tgt_scores > 0.5).unsqueeze(-1)
    if mask.sum() > 0:
        l1 = F.l1_loss(pred_regs[mask.expand_as(pred_regs)], tgt_regs[mask.expand_as(tgt_regs)])
    else:
        l1 = torch.tensor(0.0, device=pred_scores.device)
    return score_weight * bce + reg_weight * l1, {'bce': bce.item(), 'l1': l1.item()}

# Toy run when executed directly
if __name__ == '__main__':
    import random
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 4
    N = 1024
    # toy data
    def make_toy_item():
        M = 1500
        pts = np.random.randn(M,3).astype(np.float32) * 0.1
        d = np.linalg.norm(pts, axis=1)
        scores = (d < 0.15).astype(np.float32)
        regs = np.zeros((M,8), dtype=np.float32)
        regs[:,0:3] = pts
        regs[:,3:7] = np.array([1,0,0,0], dtype=np.float32)  # identity quat
        regs[:,7] = np.random.uniform(0.02, 0.08, size=(M,))
        return {'points': pts, 'scores': scores, 'regs': regs}

    toy_list = [make_toy_item() for _ in range(32)]
    ds = PointCloudGraspDataset(toy_list, n_points=N)
    dl = DataLoader(ds, batch_size=B, shuffle=True)

    model = PointMambaGPD(in_dim=3, pointfeat_dim=64, hidden_dim=128, n_blocks=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(2):
        model.train()
        total = 0.0
        for batch in dl:
            pts = batch['points'].to(device)
            sc = batch['scores'].to(device)
            rg = batch['regs'].to(device)
            pred_sc, pred_rg = model(pts)
            loss, terms = grasp_loss(pred_sc, pred_rg, sc, rg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print('epoch', epoch, 'loss', total / len(dl))

# End of file

