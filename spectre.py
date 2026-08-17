"""SPECTRE: Spectral Phase-Encoded Complex Token Representations.

Deterministic byte->box / codepoint->angle codec + one learned projection,
with sequence position injected as a multiplicative phase twist (no additive PE).

Pipeline per token (as designed in the walkthrough spreadsheet):
  NFC -> codepoints (positions p) -> each byte drops a unit arrow e^{-i w_k p}
  into its byte box, at K clock speeds w_k -> arrows sum within (box, lens)
  -> whole grid divided by its global norm -> flatten (2*256*K reals)
  -> learned Linear -> d_model -> slot twist: pair j rotated by theta_j * m.
"""
import math
import unicodedata
import torch
import torch.nn as nn


def geometric_ladder(n, floor_frac, w_max_frac=0.9):
    """n speeds from w_max_frac*pi down to floor_frac*pi (radians)."""
    w_max = w_max_frac * math.pi
    w_min = floor_frac * math.pi
    if n == 1:
        return torch.tensor([w_max])
    r = (w_min / w_max) ** (1.0 / (n - 1))
    return w_max * (r ** torch.arange(n, dtype=torch.float64))


def codec_vector(token_str, omegas, n_boxes=256):
    """The fixed codec for one token string -> flat float tensor [2*n_boxes*K].

    Layout: box-major, then lens, then (Re, Im). Matches the spreadsheet.
    """
    s = unicodedata.normalize('NFC', token_str)
    K = omegas.numel()
    grid = torch.zeros(n_boxes, K, 2, dtype=torch.float64)
    for p, ch in enumerate(s):                    # p = CODEPOINT position
        ang = -omegas * p                          # [K]
        c, si = torch.cos(ang), torch.sin(ang)
        for b in ch.encode('utf-8'):               # bytes choose the boxes
            grid[b, :, 0] += c
            grid[b, :, 1] += si
    norm = grid.pow(2).sum().sqrt()
    if norm > 0:
        grid = grid / norm                         # one global divisor
    return grid.flatten().to(torch.float32)


class SpectreEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding(+PE). Codec fixed; W learned."""

    def __init__(self, token_strings, d_model, K=8, L_max=None,
                 max_slots=1024, w_max_frac=0.9, twist_at='embedding'):
        super().__init__()
        assert d_model % 2 == 0
        assert twist_at in ('embedding', 'qk', 'none')
        self.d_model, self.K, self.twist_at = d_model, K, twist_at

        lengths = [len(unicodedata.normalize('NFC', t)) for t in token_strings]
        if L_max is None:                          # 99.9th percentile rule
            srt = sorted(lengths)
            L_max = max(2, srt[min(len(srt) - 1, int(0.999 * len(srt)))])
        self.L_max = L_max

        omegas = geometric_ladder(K, 1.0 / L_max, w_max_frac)
        self.register_buffer('omegas', omegas.to(torch.float32))

        table = torch.stack([codec_vector(t, omegas) for t in token_strings])
        self.register_buffer('codec', table)       # [V, 2*256*K], fixed
        self.proj = nn.Linear(2 * 256 * K, d_model, bias=False)

        thetas = geometric_ladder(d_model // 2, 1.0 / max_slots, w_max_frac)
        self.register_buffer('thetas', thetas.to(torch.float32))

    def rotate(self, x, positions):
        """Rotate pair j of x by thetas[j] * m. x: [B,T,d], positions: [T]."""
        B, T, d = x.shape
        ang = positions.to(x.dtype)[:, None] * self.thetas[None, :]   # [T, d/2]
        c, s = torch.cos(ang), torch.sin(ang)
        xp = x.view(B, T, d // 2, 2)
        re, im = xp[..., 0], xp[..., 1]
        out = torch.stack((re * c - im * s, re * s + im * c), dim=-1)
        return out.view(B, T, d)

    def forward(self, idx, positions=None):
        x = self.proj(self.codec[idx])              # [B,T,d]
        if self.twist_at == 'embedding':
            if positions is None:
                positions = torch.arange(idx.size(1), device=idx.device)
            x = self.rotate(x, positions)
        return x