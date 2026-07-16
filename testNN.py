import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.p = nn.AdaptiveAvgPool2d(1)
        self.f = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.p(x).view(b, c)
        y = self.f(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RoutingLayer(nn.Module):
    def __init__(self, in_caps, out_caps, in_dim, out_dim, iters=3):
        super().__init__()
        self.iters = iters
        self.W = nn.Parameter(torch.randn(out_caps, in_caps, out_dim, in_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_caps, in_caps, out_dim))
    def forward(self, x):
        b, in_caps, in_dim = x.shape
        out_caps, _, out_dim, _ = self.W.shape
        u = torch.einsum('b i d, o i k d -> b o i k', x, self.W)
        u = u + self.b.unsqueeze(0)
        u = u.reshape(b, out_caps, in_caps, out_dim)
        logits = torch.zeros(b, out_caps, in_caps, device=x.device)
        for _ in range(self.iters):
            c = F.softmax(logits, dim=1)
            s = torch.einsum('b o i k, b o i -> b o k', u, c)
            v = F.normalize(s, p=2, dim=-1)
            logits = logits + torch.einsum('b o i k, b o k -> b o i', u, v)
        return v

class MultiHeadFusion(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dp = nn.Dropout(0.1)
    def forward(self, q, k, v):
        b, t, _ = q.shape
        q = self.wq(q).view(b, t, self.n_head, self.d_k).transpose(1, 2)
        k = self.wk(k).view(b, t, self.n_head, self.d_k).transpose(1, 2)
        v = self.wv(v).view(b, t, self.n_head, self.d_k).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = F.softmax(att, dim=-1)
        att = self.dp(att)
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out)

class HyperComplexBlock(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.att = MultiHeadFusion(dim, head)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        self.gate = nn.Parameter(torch.ones(1))
    def forward(self, x):
        h = self.att(self.n1(x), self.n1(x), self.n1(x))
        x = x + self.gate * h
        x = x + self.ff(self.n2(x))
        return x

class DeepMegaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.c3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.se = SEBlock(256)
        self.caps_in = nn.Conv2d(256, 32 * 8, 1, bias=False)
        self.caps_primary = RoutingLayer(32, 16, 8, 12, iters=4)
        self.proj_caps = nn.Linear(12, 64)
        self.pos_enc = nn.Parameter(torch.randn(1, 256, 64) * 0.02)
        self.tf_blocks = nn.ModuleList([HyperComplexBlock(64, 4) for _ in range(6)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 10)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = F.silu(self.c1(x))
        x = F.silu(self.c2(x))
        x = F.silu(self.c3(x))
        x = self.se(x)
        b, c, h, w = x.shape
        x = self.caps_in(x).view(b, -1, 32, 8).mean(dim=2)
        x = self.caps_primary(x)
        x = self.proj_caps(x)
        x = x.view(b, -1, 64) + self.pos_enc[:, :x.size(1), :]
        for block in self.tf_blocks:
            x = block(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.head(x)

def mixup_criterion(y_pred, y_true, lam):
    y_true_a, y_true_b = y_true, y_true.flip(0)
    return lam * F.cross_entropy(y_pred, y_true_a) + (1 - lam) * F.cross_entropy(y_pred, y_true_b)

def train_chaos():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepMegaNet().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CyclicLR(opt, base_lr=1e-5, max_lr=1e-2, step_size_up=200, mode='triangular2')
    x = torch.randn(1024, 3, 32, 32)
    y = torch.randint(0, 10, (1024,))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
    for epoch in range(10):
        for i, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device), yb.to(device)
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(xb.size(0))
            xb_mix = lam * xb + (1 - lam) * xb[idx]
            opt.zero_grad()
            out = model(xb_mix)
            loss = mixup_criterion(out, yb, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            if i % 50 == 0:
                acc = (out.argmax(1) == yb).float().mean().item()
                print(f'E{epoch} I{i} L{loss.item():.3f} A{acc:.3f} LR{scheduler.get_last_lr()[0]:.6f}')

if __name__ == '__main__':
    train_chaos()