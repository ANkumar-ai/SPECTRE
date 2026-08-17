"""Minimal GPT with two interchangeable input pathways:
  arm='vanilla' : learned nn.Embedding * sqrt(d)  +  sinusoidal additive PE
  arm='spectre' : SPECTRE codec + projection, slot twist, NO additive PE
Everything downstream (attention, MLP, head) is identical.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectre import SpectreEmbedding


def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, d, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=dropout,
                                          batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(dropout))
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        T = x.size(1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T],
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_strings, d_model=128, n_layer=4, n_head=4,
                 block_size=64, arm='vanilla', K=8, dropout=0.0):
        super().__init__()
        self.arm, self.block_size = arm, block_size
        V = len(vocab_strings)
        if arm == 'vanilla':
            self.tok = nn.Embedding(V, d_model)
            self.register_buffer('pe', sinusoidal_pe(block_size, d_model))
            self.scale = math.sqrt(d_model)          # the classic scaling trick
        elif arm == 'spectre':
            self.tok = SpectreEmbedding(vocab_strings, d_model, K=K,
                                        max_slots=block_size,
                                        twist_at='embedding')
        else:
            raise ValueError(arm)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, V, bias=False)
        if arm == 'vanilla':
            self.head.weight = self.tok.weight       # tied (nanoGPT-style)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        if self.arm == 'vanilla':
            x = self.tok(idx) * self.scale + self.pe[:T]
        else:
            x = self.tok(idx)                         # twist happens inside
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss