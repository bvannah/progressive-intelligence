from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class NgramConfig:
    arch: str = "legacy"
    vocab_rows: int = 1 << 18
    emb_dim: int = 64
    orders: tuple[int, ...] = (2, 3, 4)
    heads: int = 8
    proj_dim: int = 256
    score_hidden: int = 128
    dropout: float = 0.1
    num_options: int = 8
    hash_seed: int = 7
    order_path: bool = True
    opt_order: bool = True
    opt_gate: bool = True
    order_d_model: int = 32
    order_heads: int = 2
    order_layers: int = 1
    max_len: int = 2048
    num_heads: int = 4
    num_layers: int = 2
    dim_ff: int = 256
    use_sdpa: bool = False
    qk_norm: bool = False
    rope: bool = False
    rope_base: float = 500_000.0
    span_w: int = 64
    span_k: int = 8
    ckpt: bool = False
    chunk_attn: bool = False


def chunked_attn(q, k, v, mask, scale, chunk=64, dropout=None):
    """Memory-efficient masked attention via query chunking + recompute.

    Equivalent to softmax((q@k^T)*scale, dim=-1) @ v with `mask` selecting
    valid keys ((...,1,1,S)).  Queries are processed in `chunk`-sized blocks
    gated through torch.utils.checkpoint, so the full score tensor is never
    retained for backward -- turns O(L*S) activation memory into O(chunk*S).
    Dropout is applied to the unnormalized exp values, which is exactly
    equivalent to the standard softmax -> Dropout -> @v (scale-invariant mask),
    and empty-key rows produce zeros.  Missing-fill is broadcast over heads.
    """
    B, H, L, dh = q.shape
    S = k.shape[-2]
    outs = []
    for i in range(0, L, chunk):
        qc = q[:, :, i:i + chunk]

        def _run(qc, k, v, mask, has):
            s = (qc @ k.transpose(-2, -1)) * scale
            s = s.masked_fill(~mask, float("-inf"))
            m = s.max(-1, keepdim=True).values
            m_safe = torch.where(has, m, torch.zeros_like(m))
            e = torch.where(mask, torch.exp(s - m_safe), torch.zeros_like(s))
            denom = e.sum(-1, keepdim=True).clamp_min(1.0)
            if dropout is not None and dropout.p > 0.0:
                e = dropout(e)
            num = e @ v
            denom = torch.where(has, denom, torch.ones_like(denom))
            return num / denom

        has = mask.any(dim=-1, keepdim=True)  # (...,1,1,1)
        outs.append(
            torch.utils.checkpoint.checkpoint(
                _run, qc, k, v, mask, has,
                use_reentrant=False,
            )
        )
    return torch.cat(outs, dim=2)


def make_multipliers(config: NgramConfig) -> dict:
    rng = random.Random(config.hash_seed)
    multis = {}
    for order in config.orders:
        for head in range(config.heads):
            multis[(order, head)] = tuple(
                rng.randrange(1, 2**63) | 1 for _ in range(order)
            )
    return multis


def hash_stream(stream: bytes, order: int, head: int, multis: dict, rows: int) -> list[int]:
    ms = multis[(order, head)]
    n = len(stream)
    limit = n - order + 1
    return [
        (sum((stream[i + j] + 1) * ms[j] for j in range(order))) % rows
        for i in range(max(0, limit))
    ]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_inv_freq(dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rope_apply(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    h = d // 2
    fr = pos.float().unsqueeze(-1) * inv_freq
    cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class OrderPath(nn.Module):
    def __init__(self, config: NgramConfig, embed: nn.Embedding):
        super().__init__()
        self.config = config
        self.embed = embed
        self.pos = nn.Embedding(config.max_len, config.emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.emb_dim,
            nhead=config.order_heads,
            dim_feedforward=4 * config.emb_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=config.order_layers)
        self.pq = nn.Linear(config.proj_dim, config.emb_dim)
        self.pk = nn.Linear(config.emb_dim, config.emb_dim)
        self.pv = nn.Linear(config.emb_dim, config.emb_dim)

    def forward(self, idx: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B, O, H, T = idx.shape
        idx = idx.reshape(B * O * H, T)
        mask = mask.reshape(B * O * H, T)
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B * O * H, T)
        x = self.embed(idx) + self.pos(pos)
        x = self.enc(x, src_key_padding_mask=~mask)
        qk = self.pq(query).repeat_interleave(O * H, dim=0)
        k = self.pk(x)
        v = self.pv(x)
        s = (k * qk.unsqueeze(1)).sum(dim=-1)
        s = s.masked_fill(~mask, float("-inf"))
        row_has = mask.any(dim=-1)
        s = torch.where(row_has.unsqueeze(-1), s, torch.zeros_like(s))
        a = torch.softmax(s, dim=-1)
        a = a * row_has.unsqueeze(-1).float()
        out = (a.unsqueeze(-1) * v).sum(dim=1)
        return out.view(B, O, H, -1)


class LegacyDecisionNet(nn.Module):
    def __init__(self, config: NgramConfig):
        super().__init__()
        self.config = config
        n_flat = len(config.orders) * config.heads * config.emb_dim
        self.embed = nn.Embedding(config.vocab_rows, config.emb_dim)
        self.order = OrderPath(config, self.embed) if config.order_path else None
        self.proj = nn.Sequential(
            nn.Linear(n_flat, config.proj_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.proj_hybrid = nn.Sequential(
            nn.Linear(2 * n_flat, config.proj_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.opt_gate = (
            nn.Sequential(nn.Linear(2 * config.proj_dim, config.proj_dim), nn.Sigmoid())
            if config.opt_gate
            else None
        )
        self.score = nn.Sequential(
            nn.Linear(3 * config.proj_dim, config.score_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.score_hidden, 1),
        )

    def encode_bag(self, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(idx)
        emb = mask.unsqueeze(-1) * emb
        emb = emb.sum(dim=-2)
        flat = emb.reshape(*emb.shape[:-3], -1)
        return self.proj(flat)

    def encode_context(self, ctx_idx, ctx_mask, query=None):
        emb = self.embed(ctx_idx)
        emb = ctx_mask.unsqueeze(-1) * emb
        emb = emb.sum(dim=-2)
        flat_bag = emb.reshape(*emb.shape[:-3], -1)
        if self.order is None:
            return self.proj(flat_bag)
        order = self.order(ctx_idx, ctx_mask, query).reshape(*emb.shape[:-3], -1)
        return self.proj_hybrid(torch.cat([flat_bag, order], dim=-1))

    def encode_option(self, idx, mask, query):
        emb = self.embed(idx)
        emb = mask.unsqueeze(-1) * emb
        emb = emb.sum(dim=-2)
        flat_bag = emb.reshape(*emb.shape[:-3], -1)
        if self.order is None or not self.config.opt_order:
            return self.proj(flat_bag)
        B, N, O, H, M = idx.shape
        order = self.order(
            idx.reshape(B * N, O, H, M),
            mask.reshape(B * N, O, H, M),
            query.repeat_interleave(N, dim=0),
        )
        order = order.reshape(B * N, -1)
        return self.proj_hybrid(torch.cat([flat_bag.reshape(B * N, -1), order], dim=-1)).view(B, N, -1)

    def forward(self, ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask):
        q = self.encode_bag(q_idx, q_mask)
        m = self.encode_context(ctx_idx, ctx_mask, query=q)
        g = self.encode_option(opt_idx, opt_mask, q)
        if self.opt_gate is not None:
            qg = self.opt_gate(
                torch.cat(
                    [
                        q[:, None, :].expand(g.shape[0], g.shape[1], -1),
                        g,
                    ],
                    dim=-1,
                )
            )
            g = g * qg
        hq = torch.cat(
            [
                m[:, None, :].expand(g.shape[0], g.shape[1], -1),
                q[:, None, :].expand(g.shape[0], g.shape[1], -1),
                g,
            ],
            dim=-1,
        )
        logits = self.score(hq).squeeze(-1)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config: NgramConfig):
        super().__init__()
        d = config.emb_dim
        self.cfg = config
        self.heads = config.num_heads
        self.scale = d ** -0.5
        self.use_sdpa = config.use_sdpa
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(config.dropout)
        self.ff1 = nn.Linear(d, config.dim_ff)
        self.ff2 = nn.Linear(config.dim_ff, d)
        self.resid_drop = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        dh = d // config.num_heads
        if config.qk_norm:
            self.qkn_q = RMSNorm(dh)
            self.qkn_k = RMSNorm(dh)
        else:
            self.qkn_q = self.qkn_k = None
        if config.rope:
            self.register_buffer(
                "inv_freq",
                make_inv_freq(dh, config.rope_base),
                persistent=False,
            )
        else:
            self.inv_freq = None

    def _qk(self, x: torch.Tensor, pos: torch.Tensor | None):
        B, T, d = x.shape
        H, dh = self.heads, d // self.heads
        q = self.wq(x).view(B, T, H, dh).transpose(1, 2)
        k = self.wk(x).view(B, T, H, dh).transpose(1, 2)
        if self.qkn_q is not None:
            q = self.qkn_q(q)
            k = self.qkn_k(k)
        if self.inv_freq is not None and pos is not None:
            q = rope_apply(q, pos, self.inv_freq)
            k = rope_apply(k, pos, self.inv_freq)
        return q, k

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        B, T, d = x.shape
        dh = d // self.heads
        q, k = self._qk(x, pos)
        v = self.wv(x).view(B, T, self.heads, dh).transpose(1, 2)
        if self.use_sdpa:
            if self.cfg.chunk_attn:
                a = chunked_attn(
                    q, k, v,
                    mask=key_padding_mask[:, None, None, :],
                    scale=self.scale,
                    dropout=self.attn_drop if self.training else None,
                )
            else:
                mask = key_padding_mask[:, None, None, :].expand(B, 1, T, T)
                a = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=self.cfg.dropout if self.training else 0.0, scale=self.scale
                )
                row_has = key_padding_mask.any(dim=-1)
                a = a * row_has[:, None, None, None].float()
        else:
            s = (q @ k.transpose(-2, -1)) * self.scale
            km = (~key_padding_mask)[:, None, None, :]
            s = s.masked_fill(km, float("-inf"))
            row_has = key_padding_mask.any(dim=-1)
            s = torch.where(row_has[:, None, None, None], s, torch.zeros_like(s))
            a = torch.softmax(s, dim=-1)
            a = a * row_has[:, None, None, None].float()
            a = self.attn_drop(a)
            a = a @ v
        out = self.wo(a.transpose(1, 2).reshape(B, T, d))
        x = self.norm1(x + self.resid_drop(out))
        f = self.ff2(torch.relu(self.ff1(x)))
        x = self.norm2(x + self.resid_drop(f))
        return x


class TransformerStack(nn.Module):
    def __init__(self, config: NgramConfig):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        if config.ckpt:
            self._ckpt = True
        else:
            self._ckpt = False

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        for blk in self.blocks:
            if self._ckpt and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, key_padding_mask, pos, use_reentrant=False
                )
            else:
                x = blk(x, key_padding_mask, pos)
        return x


class V7Net(nn.Module):
    def __init__(self, config: NgramConfig):
        super().__init__()
        self.config = config
        d = config.emb_dim
        self.embed = nn.Embedding(config.vocab_rows, d)
        self.pos = nn.Embedding(config.max_len, d)
        self.enc = TransformerStack(config)
        self.pool_q = nn.Linear(d, config.proj_dim)
        self.qk_attn = nn.Linear(config.proj_dim, d)
        self.csum_k = nn.Linear(d, d)
        self.csum_v = nn.Linear(d, d)
        self.csum_head = nn.Linear(d, config.proj_dim)
        self.opt_ca = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
        self.opt_proj = nn.Linear(2 * d, config.proj_dim)
        self.opt_gate = (
            nn.Sequential(nn.Linear(2 * config.proj_dim, config.proj_dim), nn.Sigmoid())
            if config.opt_gate
            else None
        )
        self.score = nn.Sequential(
            nn.Linear(3 * config.proj_dim, config.score_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.score_hidden, 1),
        )

    def _embed(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[-1]
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        return self.embed(idx) + self.pos(pos)

    def _enc(self, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        idx = idx.clone()
        mask = mask.clone()
        empty = ~mask.any(dim=-1)
        if empty.any():
            if idx.shape[-1] == 0:
                idx = idx.new_zeros((idx.shape[0], 1))
                mask = mask.new_ones((mask.shape[0], 1), dtype=mask.dtype)
            else:
                idx[empty, 0] = 0
                mask[empty, 0] = True
        x = self._embed(idx)
        return self.enc(x, key_padding_mask=mask)

    def _mean_pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask):
        B, N = opt_idx.shape[0], opt_idx.shape[1]
        if ctx_mask.shape[-1] == 0:
            ctx_idx = ctx_idx.new_zeros((ctx_idx.shape[0], 1))
            ctx_mask = ctx_mask.new_ones((ctx_mask.shape[0], 1), dtype=ctx_mask.dtype)
        H_c = self._enc(ctx_idx, ctx_mask)
        H_q = self._enc(q_idx, q_mask)
        q = self.pool_q(self._mean_pool(H_q, q_mask))

        qk = self.qk_attn(q)
        k = self.csum_k(H_c)
        v = self.csum_v(H_c)
        s = (k * qk.unsqueeze(1)).sum(dim=-1)
        s = s.masked_fill(~ctx_mask, float("-inf"))
        row_has = ctx_mask.any(dim=-1)
        s = torch.where(row_has.unsqueeze(-1), s, torch.zeros_like(s))
        a = torch.softmax(s, dim=-1)
        cctx = (a.unsqueeze(-1) * v).sum(dim=1)
        cctx = self.csum_head(cctx)

        M = opt_idx.shape[2]
        o_rows = opt_idx.reshape(B * N, M)
        om_rows = opt_mask.reshape(B * N, M)
        H_o = self._enc(o_rows, om_rows)
        Tc = H_c.shape[1]
        H_c_rows = H_c.unsqueeze(1).expand(B, N, Tc, self.config.emb_dim).reshape(B * N, Tc, self.config.emb_dim)
        ca, _ = self.opt_ca(H_o, H_c_rows, H_c_rows)
        own = self._mean_pool(H_o, om_rows)
        cross = self._mean_pool(ca, om_rows)
        g = self.opt_proj(torch.cat([own, cross], dim=-1)).view(B, N, -1)

        if self.opt_gate is not None:
            qg = self.opt_gate(torch.cat([q[:, None, :].expand(B, N, -1), g], dim=-1))
            g = g * qg
        ctx_r = cctx[:, None, :].expand(B, N, -1)
        q_r = q[:, None, :].expand(B, N, -1)
        hq = torch.cat([ctx_r, q_r, g], dim=-1)
        return self.score(hq).squeeze(-1)


class SpanCrossAttn(nn.Module):
    """Options cross-attend over the top-k most-relevant context span windows.

    Retrieval is per-option: each option's pooled encoding (gated by the
    question) scores every context span window; only the top-`span_k` windows
    (width `span_w` tokens each) become attention keys.  This bounds the
    cross-attention cost at ~span_k*span_w keys regardless of context length,
    so long contexts (8k+) keep the option path at training-length cost.
    """

    def __init__(self, config: NgramConfig):
        super().__init__()
        d = config.emb_dim
        self.cfg = config
        self.heads = config.num_heads
        self.scale = d ** -0.5
        self.use_sdpa = config.use_sdpa
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(config.dropout)
        self.sel_k = nn.Linear(d, d)
        dh = d // config.num_heads
        if config.qk_norm:
            self.qkn_q = RMSNorm(dh)
            self.qkn_k = RMSNorm(dh)
            self.sel_norm = RMSNorm(d)
        else:
            self.qkn_q = self.qkn_k = self.sel_norm = None
        if config.rope:
            self.register_buffer("inv_freq", make_inv_freq(dh, config.rope_base), persistent=False)
        else:
            self.inv_freq = None

    def select(self, ctx_h: torch.Tensor, ctx_mask: torch.Tensor, sel_q: torch.Tensor) -> torch.Tensor:
        B, N, _ = sel_q.shape
        Tc = ctx_h.shape[1]
        cfg = self.cfg
        sk = self.sel_k(ctx_h)
        sq = sel_q
        if self.sel_norm is not None:
            sk = self.sel_norm(sk)
            sq = self.sel_norm(sq)
        score = (sk.unsqueeze(1) * sq.unsqueeze(2)).sum(-1)  # (B, N, Tc)
        score = score.masked_fill(~ctx_mask[:, None, :], float("-inf"))
        w = cfg.span_w
        n_w = (Tc + w - 1) // w
        pad = n_w * w - Tc
        if pad:
            score = torch.nn.functional.pad(score, (0, pad), value=float("-inf"))
            cm = torch.nn.functional.pad(ctx_mask, (0, pad))
        else:
            cm = ctx_mask
        ws = score.view(B, N, n_w, w)
        valid = cm[:, None, :, None].reshape(B, 1, n_w, w)
        cnt = valid.sum(-1).clamp_min(1)
        wscore = ws.masked_fill(~valid, 0.0).sum(-1) / cnt
        if n_w > cfg.span_k:
            top = wscore.topk(cfg.span_k, dim=-1).indices
        else:
            top = torch.arange(n_w, device=score.device).expand(B, N, n_w)
        keep = torch.zeros(B, N, n_w, dtype=torch.bool, device=score.device).scatter(-1, top, True)
        keep = keep[..., None].expand(B, N, n_w, w).reshape(B, N, n_w * w)[..., :Tc]
        keep = keep & ctx_mask[:, None, :]
        return keep

    def forward(self, opt_h, ctx_h, ctx_mask, sel_q, ctx_pos):
        B, N, M, d = opt_h.shape
        cfg = self.cfg
        H, dh = self.heads, d // self.heads
        keep = self.select(ctx_h, ctx_mask, sel_q)  # (B, N, Tc)

        K = self.wk(ctx_h).unsqueeze(1).expand(B, N, -1, -1).reshape(B * N, -1, d)
        V = self.wv(ctx_h).unsqueeze(1).expand(B, N, -1, -1).reshape(B * N, -1, d)

        S = int(keep.sum(-1).max().clamp_min(1).item())
        Kb = K.new_zeros(B * N, S, d)
        Vb = V.new_zeros(B * N, S, d)
        Pb = ctx_pos.new_zeros(B * N, 1, S)
        lens = torch.zeros(B * N, dtype=torch.long, device=keep.device)
        with torch.no_grad():
            for i in range(B * N):
                b, n = divmod(i, N)
                idx = keep[b, n].nonzero(as_tuple=False).squeeze(-1)
                si = idx.numel()
                if si == 0:
                    si = 1
                    idx = idx.new_zeros(1)
                Kb[i, :si] = K[i, idx]
                Vb[i, :si] = V[i, idx]
                Pb[i, 0, :si] = ctx_pos[idx]
                lens[i] = si

        q = self.wq(opt_h).view(B * N, M, H, dh).transpose(1, 2)
        if self.qkn_q is not None:
            q = self.qkn_q(q)
        if self.inv_freq is not None:
            q = rope_apply(q, torch.arange(M, device=q.device), self.inv_freq)

        Kb = Kb.view(B * N, S, H, dh).transpose(1, 2)
        Vb = Vb.view(B * N, S, H, dh).transpose(1, 2)
        if self.qkn_k is not None:
            Kb = self.qkn_k(Kb)
        if self.inv_freq is not None:
            Kb = rope_apply(Kb, Pb, self.inv_freq)

        key_mask = torch.zeros(B * N, 1, 1, S, dtype=torch.bool, device=q.device)
        key_mask[:, 0, 0, :] = torch.arange(S, device=q.device)[None, :] < lens[:, None]
        if self.use_sdpa:
            if self.cfg.chunk_attn:
                a = chunked_attn(
                    q, Kb, Vb,
                    mask=key_mask,
                    scale=self.scale,
                    dropout=self.attn_drop if self.training else None,
                )
            else:
                a = torch.nn.functional.scaled_dot_product_attention(
                    q, Kb, Vb, attn_mask=key_mask,
                    dropout_p=cfg.dropout if self.training else 0.0, scale=self.scale,
                )
        else:
            s = (q @ Kb.transpose(-2, -1)) * self.scale
            s = s.masked_fill(~key_mask, float("-inf"))
            row_has = key_mask.any(dim=-1, keepdim=True).to(s.dtype)  # (B*N,1,1,1)
            s = torch.where(row_has.bool(), s, torch.zeros_like(s))
            a = torch.softmax(s, dim=-1)
            a = a * row_has
            a = self.attn_drop(a)
            a = a @ Vb
        out = self.wo(a.transpose(1, 2).reshape(B * N, M, d))
        return out.view(B, N, M, d)


class V8Net(nn.Module):
    def __init__(self, config: NgramConfig):
        super().__init__()
        self.config = config
        d = config.emb_dim
        self.embed = nn.Embedding(config.vocab_rows, d)
        self.enc = TransformerStack(config)
        self.pool_q = nn.Linear(d, config.proj_dim)
        self.qk_attn = nn.Linear(config.proj_dim, d)
        self.csum_k = nn.Linear(d, d)
        self.csum_v = nn.Linear(d, d)
        self.csum_head = nn.Linear(d, config.proj_dim)
        if config.qk_norm:
            self.cpool_qn = RMSNorm(d)
            self.cpool_kn = RMSNorm(d)
        else:
            self.cpool_qn = self.cpool_kn = None
        self.opt_sel = nn.Linear(2 * d, d)
        self.q_proj = nn.Linear(config.proj_dim, d)
        self.span_ca = SpanCrossAttn(config)
        self.opt_proj = nn.Linear(2 * d, config.proj_dim)
        self.opt_gate = (
            nn.Sequential(nn.Linear(2 * config.proj_dim, config.proj_dim), nn.Sigmoid())
            if config.opt_gate
            else None
        )
        self.score = nn.Sequential(
            nn.Linear(3 * config.proj_dim, config.score_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.score_hidden, 1),
        )

    def _enc(self, idx: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        idx = idx.clone()
        mask = mask.clone()
        empty = ~mask.any(dim=-1)
        if empty.any():
            if idx.shape[-1] == 0:
                idx = idx.new_zeros((idx.shape[0], 1))
                mask = mask.new_ones((mask.shape[0], 1), dtype=mask.dtype)
            else:
                idx[empty, 0] = 0
                mask[empty, 0] = True
        x = self.embed(idx)
        T = idx.shape[-1]
        if pos is None:
            pos = torch.arange(T, device=idx.device)
        return self.enc(x, key_padding_mask=mask, pos=pos)

    def encode_context(self, ctx_idx: torch.Tensor, ctx_mask: torch.Tensor) -> dict:
        if ctx_mask.shape[-1] == 0:
            ctx_idx = ctx_idx.new_zeros((ctx_idx.shape[0], 1))
            ctx_mask = ctx_mask.new_ones((ctx_mask.shape[0], 1), dtype=ctx_mask.dtype)
        H_c = self._enc(ctx_idx, ctx_mask)
        k = self.csum_k(H_c)
        v = self.csum_v(H_c)
        if self.cpool_qn is not None:
            k = self.cpool_kn(k)
        return {"H_c": H_c, "k": k, "v": v, "ctx_mask": ctx_mask}

    def score_with_context(self, st: dict, ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask) -> torch.Tensor:
        H_c = st["H_c"]; k = st["k"]; v = st["v"]; ctx_mask = st["ctx_mask"]
        B, N = opt_idx.shape[0], opt_idx.shape[1]
        H_q = self._enc(q_idx, q_mask)
        q = self.pool_q(self._mean_pool(H_q, q_mask))

        qk = self.qk_attn(q)
        if self.cpool_qn is not None:
            qk = self.cpool_qn(qk)
        s = (k * qk.unsqueeze(1)).sum(dim=-1)
        s = s.masked_fill(~ctx_mask, float("-inf"))
        row_has = ctx_mask.any(dim=-1)
        s = torch.where(row_has.unsqueeze(-1), s, torch.zeros_like(s))
        a = torch.softmax(s, dim=-1)
        a = a * row_has.unsqueeze(-1).float()
        cctx = (a.unsqueeze(-1) * v).sum(dim=1)
        cctx = self.csum_head(cctx)

        M = opt_idx.shape[2]
        o_rows = opt_idx.reshape(B * N, M)
        om_rows = opt_mask.reshape(B * N, M)
        H_o = self._enc(o_rows, om_rows, pos=torch.arange(M, device=opt_idx.device))
        own = self._mean_pool(H_o, om_rows).view(B, N, -1)

        sel_q = self.opt_sel(torch.cat([self.q_proj(q)[:, None, :].expand(B, N, -1), own], dim=-1))
        Tc = ctx_mask.shape[1]
        ctx_pos = torch.arange(Tc, device=opt_idx.device)
        ca = self.span_ca(H_o.view(B, N, M, self.config.emb_dim), H_c, ctx_mask, sel_q, ctx_pos)
        cross = self._mean_pool(ca.view(B * N, M, self.config.emb_dim), om_rows).view(B, N, -1)
        g = self.opt_proj(torch.cat([own, cross], dim=-1))

        if self.opt_gate is not None:
            qg = self.opt_gate(torch.cat([q[:, None, :].expand(B, N, -1), g], dim=-1))
            g = g * qg
        ctx_r = cctx[:, None, :].expand(B, N, -1)
        q_r = q[:, None, :].expand(B, N, -1)
        hq = torch.cat([ctx_r, q_r, g], dim=-1)
        return self.score(hq).squeeze(-1)

    def forward(self, ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, prev_ctx=None):
        if prev_ctx is None:
            return self.score_with_context(
                self.encode_context(ctx_idx, ctx_mask), ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask
            )
        B = ctx_idx.shape[0]
        states = [None] * B
        miss = []
        for i in range(B):
            if prev_ctx[i] is None:
                miss.append(i)
            else:
                states[i] = prev_ctx[i]
        if miss:
            sub = self.encode_context(ctx_idx[miss], ctx_mask[miss])
            keys = list(sub)
            for j, i in enumerate(miss):
                states[i] = {key: val[j:j + 1] for key, val in sub.items()}
        outs = [
            self.score_with_context(
                states[i], ctx_idx[i:i + 1], ctx_mask[i:i + 1],
                q_idx[i:i + 1], q_mask[i:i + 1], opt_idx[i:i + 1], opt_mask[i:i + 1],
            )
            for i in range(B)
        ]
        return torch.cat(outs), states

    def _mean_pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)

    def forward(self, ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask):
        return self.score_with_context(
            self.encode_context(ctx_idx, ctx_mask), ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask
        )

    def forward_with_ctx_cache(self, ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, prev_ctx=None):
        B = ctx_idx.shape[0]
        states = [None] * B
        miss = []
        for i in range(B):
            if prev_ctx[i] is None:
                miss.append(i)
            else:
                states[i] = prev_ctx[i]
        if miss:
            sub = self.encode_context(ctx_idx[miss], ctx_mask[miss])
            for j, i in enumerate(miss):
                states[i] = {key: val[j:j + 1] for key, val in sub.items()}
        outs = [
            self.score_with_context(
                states[i], ctx_idx[i:i + 1], ctx_mask[i:i + 1],
                q_idx[i:i + 1], q_mask[i:i + 1], opt_idx[i:i + 1], opt_mask[i:i + 1],
            )
            for i in range(B)
        ]
        return torch.cat(outs), states


class DecisionNet(nn.Module):
    def __init__(self, config: NgramConfig):
        super().__init__()
        self.config = config
        if config.arch == "v7":
            self.net = V7Net(config)
        elif config.arch == "v8":
            self.net = V8Net(config)
        else:
            self.net = LegacyDecisionNet(config)

    def forward(self, ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, prev_ctx=None):
        if prev_ctx is None or not hasattr(self.net, "forward_with_ctx_cache"):
            return self.net(ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask)
        return self.net.forward_with_ctx_cache(ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, prev_ctx)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.config.arch == "legacy" and not any(k.startswith("net.") for k in state_dict):
            state_dict = {"net." + k: v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)