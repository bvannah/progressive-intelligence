from __future__ import annotations

import json
import re
import unicodedata

import torch
from torch.utils.data import Dataset

from model import NgramConfig, make_multipliers, hash_stream
from tokenizer import Vocab, build_vocab

_WS = re.compile(r"\s+")


def normalize_text(s) -> bytes:
    if isinstance(s, (bytes, bytearray)):
        s = bytes(s).decode("utf-8", "replace")
    t = unicodedata.normalize("NFKC", s).lower()
    t = _WS.sub(" ", t).strip()
    return t.encode("utf-8")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_IDX_DTYPE = torch.int32


def encode_stream(stream_bytes: bytes, n_orders: int, heads: int, multis: dict, rows: int, max_len: int):
    sbytes = stream_bytes[:max_len]
    length = len(sbytes)
    idx = torch.zeros((n_orders, heads, length), dtype=_IDX_DTYPE)
    mask = torch.zeros((n_orders, heads, length), dtype=torch.bool)
    for oi in range(n_orders):
        order = 2 + oi
        n_pos = max(0, length - order + 1)
        with torch.no_grad():
            for hi in range(heads):
                hs = hash_stream(sbytes, order, hi, multis, rows)
                idx[oi, hi, :n_pos] = torch.tensor(hs, dtype=_IDX_DTYPE)
                mask[oi, hi, :n_pos] = True
    return idx, mask


def build_example(row: dict, cfg: NgramConfig, multis: dict, max_ctx: int):
    ctx_idx, ctx_mask = encode_stream(normalize_text(row.get("context", "")), len(cfg.orders), cfg.heads, multis, cfg.vocab_rows, max_ctx)
    q_idx, q_mask = encode_stream(normalize_text(row.get("question", "")), len(cfg.orders), cfg.heads, multis, cfg.vocab_rows, 512)

    options = row.get("options", [])
    n_opt = len(options)
    O, H = len(cfg.orders), cfg.heads
    opt_idxs = []
    opt_masks = []
    for ob in options:
        oi_idx, oi_mask = encode_stream(normalize_text(ob), O, H, multis, cfg.vocab_rows, 256)
        opt_idxs.append(oi_idx)
        opt_masks.append(oi_mask)
    M = max(x.shape[2] for x in opt_idxs)
    opt_idx = torch.zeros((n_opt, O, H, M), dtype=_IDX_DTYPE)
    opt_mask = torch.zeros((n_opt, O, H, M), dtype=torch.bool)
    for i, (vi, vm) in enumerate(zip(opt_idxs, opt_masks)):
        n = vi.shape[2]
        opt_idx[i, :, :, :n] = vi[:, :, :n]
        opt_mask[i, :, :, :n] = vm[:, :, :n]

    probs = row.get("teacher_probs")
    correct = row.get("correct_index")
    if probs is None:
        if correct is None:
            probs = [1.0 / n_opt] * n_opt
        else:
            probs = [0.05 / (n_opt - 1)] * n_opt
            probs[correct] = 0.95
    probs_t = torch.tensor(probs, dtype=torch.float32)
    correct_t = torch.tensor(correct if correct is not None else -1, dtype=torch.long)
    family = row.get("meta", {}).get("family", "unknown")
    return ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, probs_t, correct_t, family


def build_example_v7(row: dict, cfg: NgramConfig, vocab: Vocab, max_ctx: int, opt_cap: int | None = None):
    cids = vocab.encode(row.get("context", ""))
    qids = vocab.encode(row.get("question", ""))
    opt_cap = min(cfg.max_len, opt_cap or cfg.max_len)
    opt_lists = [vocab.encode(o)[:opt_cap] for o in row.get("options", [])]
    n_opt = len(opt_lists)
    M = max((len(x) for x in opt_lists), default=1)

    ctx_idx = torch.tensor(cids[:max_ctx], dtype=_IDX_DTYPE)
    ctx_mask = torch.ones(ctx_idx.shape[0], dtype=torch.bool)
    q_idx = torch.tensor(qids[: cfg.max_len], dtype=_IDX_DTYPE)
    q_mask = torch.ones(q_idx.shape[0], dtype=torch.bool)
    opt_idx = torch.zeros((n_opt, M), dtype=_IDX_DTYPE)
    opt_mask = torch.zeros((n_opt, M), dtype=torch.bool)
    for i, li in enumerate(opt_lists):
        li = li[: cfg.max_len]
        k = len(li)
        if k:
            opt_idx[i, :k] = torch.tensor(li, dtype=_IDX_DTYPE)
            opt_mask[i, :k] = True

    probs = row.get("teacher_probs")
    correct = row.get("correct_index")
    if probs is None:
        if correct is None:
            probs = [1.0 / n_opt] * n_opt
        else:
            probs = [0.05 / (n_opt - 1)] * n_opt
            probs[correct] = 0.95
    probs_t = torch.tensor(probs, dtype=torch.float32)
    correct_t = torch.tensor(correct if correct is not None else -1, dtype=torch.long)
    family = row.get("meta", {}).get("family", "unknown")
    return ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, probs_t, correct_t, family


def collate(batch):
    if batch[0][0].ndim == 1:
        return _collate_v7(batch)
    return _collate_legacy(batch)


def _collate_legacy(batch):
    ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, probs, correct, families = zip(*batch)
    B = len(ctx_idx)
    O, H = ctx_idx[0].shape[:2]
    T = max(x.shape[2] for x in ctx_idx)
    bb = torch.zeros((B, O, H, T), dtype=_IDX_DTYPE)
    bm = torch.zeros((B, O, H, T), dtype=torch.bool)
    for i, x in enumerate(ctx_idx):
        n = x.shape[2]
        bb[i, :, :, :n] = x
        bm[i, :, :, :n] = ctx_mask[i]
    Tq = max(x.shape[2] for x in q_idx)
    qb = torch.zeros((B, O, H, Tq), dtype=_IDX_DTYPE)
    qm = torch.zeros((B, O, H, Tq), dtype=torch.bool)
    for i, x in enumerate(q_idx):
        n = x.shape[2]
        qb[i, :, :, :n] = x
        qm[i, :, :, :n] = q_mask[i]
    N = max(x.shape[0] for x in opt_idx)
    M = max(x.shape[3] for x in opt_idx)
    ob = torch.zeros((B, N, O, H, M), dtype=_IDX_DTYPE)
    om = torch.zeros((B, N, O, H, M), dtype=torch.bool)
    for i, x in enumerate(opt_idx):
        n = x.shape[0]
        nn = x.shape[3]
        ob[i, :n, :, :, :nn] = x
        om[i, :n, :, :, :nn] = opt_mask[i]
    pk = torch.zeros((B, N))
    ck = torch.full((B, N), -1, dtype=torch.long)
    for i, x in enumerate(probs):
        n = x.shape[0]
        pk[i, :n] = x
        ck[i, :n] = correct[i]
    probs = pk
    correct = ck
    return bb, bm, qb, qm, ob, om, probs, correct, list(families)


def _collate_v7(batch):
    ctx_idx, ctx_mask, q_idx, q_mask, opt_idx, opt_mask, probs, correct, families = zip(*batch)
    B = len(ctx_idx)
    Tc = max(x.shape[0] for x in ctx_idx)
    bb = torch.zeros((B, Tc), dtype=_IDX_DTYPE)
    bm = torch.zeros((B, Tc), dtype=torch.bool)
    for i, x in enumerate(ctx_idx):
        n = x.shape[0]
        bb[i, :n] = x
        bm[i, :n] = ctx_mask[i]
    Tq = max(x.shape[0] for x in q_idx)
    qb = torch.zeros((B, Tq), dtype=_IDX_DTYPE)
    qm = torch.zeros((B, Tq), dtype=torch.bool)
    for i, x in enumerate(q_idx):
        n = x.shape[0]
        qb[i, :n] = x
        qm[i, :n] = q_mask[i]
    N = max(x.shape[0] for x in opt_idx)
    M = max(x.shape[1] for x in opt_idx)
    ob = torch.zeros((B, N, M), dtype=_IDX_DTYPE)
    om = torch.zeros((B, N, M), dtype=torch.bool)
    for i, x in enumerate(opt_idx):
        n, nn = x.shape
        ob[i, :n, :nn] = x
        om[i, :n, :nn] = opt_mask[i]
    pk = torch.zeros((B, N))
    ck = torch.full((B, N), -1, dtype=torch.long)
    for i, x in enumerate(probs):
        n = x.shape[0]
        pk[i, :n] = x
        ck[i, :n] = correct[i]
    return bb, bm, qb, qm, ob, om, pk, ck, list(families)


class DecisionDataset(Dataset):
    def __init__(self, rows: list[dict], cfg: NgramConfig, max_ctx: int = 2048,
                 vocab: Vocab | None = None, opt_cap: int | None = None):
        self.cfg = cfg
        self.vocab = vocab
        self.max_ctx = max_ctx
        if cfg.arch in ("v7", "v8"):
            if vocab is None:
                raise ValueError("v7 DecisionDataset requires vocab")
            self.multis = None
            self.items = [build_example_v7(r, cfg, vocab, max_ctx, opt_cap=opt_cap) for r in rows]
        else:
            self.multis = make_multipliers(cfg)
            self.items = [build_example(r, cfg, self.multis, max_ctx) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]