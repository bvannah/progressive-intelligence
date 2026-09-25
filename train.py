from __future__ import annotations

import argparse
import os
import pathlib
import random
import subprocess
import sys
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DecisionDataset, collate, load_jsonl, build_vocab
from tokenizer import Vocab
from model import DecisionNet, NgramConfig
from constants import TRAIN_DATA, MODEL_DATA_V8, PUBLIC_CACHE


def newtonschulz(m: torch.Tensor, steps: int = 5) -> torch.Tensor:
    dim = m.ndim
    if dim == 2:
        m = m[None, ...]
    pad_r = max(m.shape[-2], m.shape[-1]) - m.shape[-2]
    pad_c = max(m.shape[-2], m.shape[-1]) - m.shape[-1]
    if pad_r or pad_c:
        m = torch.nn.functional.pad(m, (0, pad_c, 0, pad_r))
    a, b, c = 1.0, -0.5562, 0.3334
    x = m.float()
    for _ in range(steps):
        x = a * x.bmm(b * x.bmm(x.transpose(-2, -1)) + c * x.bmm(x.transpose(-2, -1)).bmm(x.bmm(x.transpose(-2, -1))))
    if pad_r or pad_c:
        x = x[..., : x.shape[-2] - pad_r, : x.shape[-1] - pad_c]
    if dim == 2:
        x = x[0]
    return x.to(m.dtype)


class MuonAdamMix:
    """Muon (orthogonalized momentum) on 2-D weights + AdamW on the rest.

    https://arxiv.org/abs/2502.16982 — Muon's documented strength is
    associative-memory (retrieval-style) learning, our model's core job.
    """

    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, muon_beta=0.95,
                 adam_betas=(0.9, 0.95), weight_decay=0.01, ns_steps=5):
        muon_p, adam_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and "embed" not in name and "norm" not in name.lower():
                muon_p.append(p)
            else:
                adam_p.append(p)
        self.adam = torch.optim.AdamW(adam_p, lr=adam_lr, betas=adam_betas, weight_decay=weight_decay)
        self.muon_p = muon_p
        self.muon_lr = muon_lr
        self.beta = muon_beta
        self.ns_steps = ns_steps
        self.state = {p: torch.zeros_like(p) for p in muon_p}

    def step(self):
        with torch.no_grad():
            for p in self.muon_p:
                if p.grad is None:
                    continue
                z = self.state[p]
                z.mul_(self.beta).add_(p.grad, alpha=1 - self.beta)
                p.add_(newtonschulz(z, self.ns_steps), alpha=-self.muon_lr)
        self.adam.step()

    def zero_grad(self, set_to_none: bool = True):
        self.adam.zero_grad(set_to_none=set_to_none)
        for p in self.muon_p:
            if p.grad is not None:
                p.grad = None

    def group_counts(self) -> dict:
        return {"muon": sum(p.numel() for p in self.muon_p),
                "adam": sum(p.numel() for p in self.adam.param_groups[0]["params"])}


def make_optimizer(model, lr, attn_mult, embed_mult, head_mult=1.0):
    groups = {"attn": [], "surface": [], "head": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("enc.", "opt_ca", "qk_attn", "csum_", "order.", "span_ca")):
            groups["attn"].append(p)
        elif any(k in name for k in ("embed.", "pos.")):
            groups["surface"].append(p)
        else:
            groups["head"].append(p)
    return torch.optim.AdamW(
        [
            {"params": groups["attn"], "lr": lr * attn_mult},
            {"params": groups["head"], "lr": lr * head_mult},
            {"params": groups["surface"], "lr": lr * embed_mult},
        ],
        lr=lr,
        weight_decay=1e-5,
    ), {k: sum(p.numel() for p in v) for k, v in groups.items()}


def masked_kl(logits, probs, om):
    """KL over valid options only (ignores padded option slots).

    `om` is the per-option token mask (B,N,M): pad options are all-False. Masking the
    logits to -inf makes the softmax ignore pads, but F.kl_div yields 0 * (-inf) = NaN
    on those slots, so the elementwise output is re-zeroed there. Mean is over rows,
    exactly like `batchmean` on all-valid data (parity => identical when N==8 everywhere).
    """
    valid = om.any(dim=-1)  # (B, N) — True for real, non-empty options
    logits_m = logits.masked_fill(~valid, float("-inf"))
    kl = F.kl_div(F.log_softmax(logits_m, dim=1), probs, reduction="none")  # (B, N)
    kl = kl.masked_fill(~valid, 0.0)
    return kl.sum(dim=1).mean()


# --- periodic checkpoint / resume ------------------------------------------------


def opt_state_dict(opt) -> dict:
    if isinstance(opt, MuonAdamMix):
        return {"muon": [t.detach().cpu().clone() for t in opt.state.values()],
                "adam": opt.adam.state_dict()}
    return {"adam": opt.state_dict()}


def opt_load_state(opt, sd: dict | None):
    sd = sd or {}
    if isinstance(opt, MuonAdamMix):
        adam = sd.get("adam")
        if adam:
            opt.adam.load_state_dict(adam)
        vals = sd.get("muon") or []
        if len(vals) == len(opt.muon_p):
            opt.state = {p: vals[i].to(p.device) for i, p in enumerate(opt.muon_p)}
    else:
        adam = sd.get("adam")
        if adam:
            opt.load_state_dict(adam)


def save_resume(path, cfg, vocab, model, opt, epoch, best_loss, best_state, stale,
                device, args):
    """Full state allowing kill-and-resume. Also loadable by eval/public_eval/
    public_eval.py (config/model/vocab keys are theirs)."""
    torch.save({
        "config": asdict(cfg),
        "model": model.state_dict(),
        "vocab": vocab.to_state(),
        "best": best_state,
        "best_loss": best_loss,
        "stale": stale,
        "epoch": epoch,  # last completed epoch (0-based)
        "optimizer": opt_state_dict(opt),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
        "rng_py": random.getstate(),
        "args": vars(args),
    }, path)


def save_best_artifact(path, cfg, vocab, model, seed, args):
    """Evaluable best-model snapshot (config/model/vocab + train metadata)."""
    torch.save({"config": asdict(cfg), "model": model.state_dict(), "seed": seed,
                "vocab": vocab.to_state(),
                "train": {"lr": args.lr, "lr_attn_mult": args.lr_attn_mult,
                          "lr_embed_mult": args.lr_embed_mult, "lr_head_mult": args.lr_head_mult,
                          "epochs": args.epochs, "patience": args.patience, "clip": args.clip,
                          "valid": args.valid}}, path)


def run_public_eval(tmp_path, which, cache_dir, device, out_base) -> float:
    """Score current model on the public eval sets (eval-half only) via public_eval.py.
    Returns wall-clock seconds spent."""
    t0 = time.perf_counter()
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve().parent / "public_eval.py"),
           "--model", tmp_path, "--which", which,
           "--cache-dir", cache_dir, "--device", device]
    res = subprocess.run(cmd, capture_output=True, text=True)
    stdout = (res.stdout or "").strip()
    print(stdout if stdout else "(public eval produced no stdout)")
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        print(f"WARN: public eval returned {res.returncode}: {stderr[-500:]}")
    else:
        print(f"[public eval] done in {time.perf_counter() - t0:.1f}s -> {which}", flush=True)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(TRAIN_DATA))
    ap.add_argument("--out", default=str(MODEL_DATA_V8))
    ap.add_argument("--valid", default=None, help="true held-out rows file for early stopping")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--max-opt", type=int, default=128, help="cap option token length")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-fraction", type=float, default=0.15)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--lr-attn-mult", type=float, default=3.0)
    ap.add_argument("--lr-embed-mult", type=float, default=0.5)
    ap.add_argument("--lr-head-mult", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--trace-every", type=int, default=0)
    ap.add_argument("--arch", choices=["v7", "v8"], default="v8")
    ap.add_argument("--no-sdpa", action="store_true", help="manual attention instead of scaled_dot_product_attention")
    ap.add_argument("--no-qkn", action="store_true", help="disable per-head QK norm")
    ap.add_argument("--no-rope", action="store_true", help="disable rotary position embeddings")
    ap.add_argument("--rope-base", type=float, default=500000.0, help="RoPE frequency base (high = length extrapolation)")
    ap.add_argument("--span-w", type=int, default=64, help="context span window width in tokens")
    ap.add_argument("--span-k", type=int, default=8, help="max number of retrieved span windows per option")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing on encoder + cross-attention")
    ap.add_argument("--chunk-attn", action="store_true", help="chunked memory-efficient attention (query chunking + recompute)")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--muon-adam-lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=5,
                    help="save a kill/resume checkpoint (best-so-far best model) every N epochs; 0 disables")
    ap.add_argument("--resume", default="",
                    help="path to a .resume.pt checkpoint from a previous run; continues from its epoch")
    ap.add_argument("--public-eval-every", type=int, default=10,
                    help="run public eval (eval-half only) every N epochs; 0 disables")
    ap.add_argument("--public-eval-which", default="arc-challenge,mmlu-astronomy,race-middle")
    ap.add_argument("--public-eval-cache", default=str(PUBLIC_CACHE))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = NgramConfig()
    resume_epoch = 0
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(args.resume, map_location="cpu")
        cfg = NgramConfig(**resume_ck["config"])
        cfg.max_len = min(cfg.max_len, args.max_ctx)
        saved_epoch = int(resume_ck.get("epoch", -1))
        resume_epoch = saved_epoch + 1
        cfg.arch = resume_ck["config"].get("arch", "v8")
        torch.set_rng_state(resume_ck["rng_torch"])
        random.setstate(resume_ck["rng_py"])
        if args.device == "cuda" and resume_ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(resume_ck["rng_cuda"])
        print(f"resumed {args.resume} from epoch {saved_epoch} "
              f"(best_loss {resume_ck.get('best_loss', float('nan')):.4f})")
    else:
        cfg.arch = args.arch
        cfg.emb_dim = 64
        cfg.num_heads = 4
        cfg.num_layers = 2
        cfg.dim_ff = 256
        cfg.use_sdpa = not args.no_sdpa
        cfg.qk_norm = not args.no_qkn
        cfg.rope = not args.no_rope
        cfg.rope_base = args.rope_base
        cfg.span_w = args.span_w
        cfg.span_k = args.span_k
        cfg.ckpt = args.ckpt
        cfg.chunk_attn = args.chunk_attn
        cfg.max_len = min(cfg.max_len, args.max_ctx)

    rows = load_jsonl(args.data)
    random.shuffle(rows)
    if args.valid is not None:
        train_rows = rows
        eval_rows = load_jsonl(args.valid)
    else:
        n_eval = max(1, int(len(rows) * args.eval_fraction))
        eval_rows = rows[:n_eval]
        train_rows = rows[n_eval:]

    if resume_ck is not None:
        vocab = Vocab.from_state(resume_ck["vocab"])
        cfg.vocab_rows = vocab.n_rows()
    else:
        texts = []
        for r in train_rows:
            texts.append(r.get("context", ""))
            texts.append(r.get("question", ""))
            for o in r.get("options", []):
                texts.append(o)
        vocab = build_vocab(texts, top_words=args.vocab_size)
        cfg.vocab_rows = vocab.n_rows()

    train_ds = DecisionDataset(train_rows, cfg, max_ctx=args.max_ctx, vocab=vocab, opt_cap=args.max_opt)
    eval_ds = DecisionDataset(eval_rows, cfg, max_ctx=args.max_ctx, vocab=vocab, opt_cap=args.max_opt)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = DecisionNet(cfg).to(args.device)
    if args.optimizer == "muon":
        opt = MuonAdamMix(model, muon_lr=args.muon_lr, adam_lr=args.muon_adam_lr)
        opt_label = f"muon(muon_lr={args.muon_lr})"
        gcounts = opt.group_counts()
    else:
        opt, gcounts = make_optimizer(model, args.lr, args.lr_attn_mult, args.lr_embed_mult, args.lr_head_mult)
        opt_label = "adamw"

    if resume_ck is not None:
        model.load_state_dict(resume_ck["model"])
        opt_load_state(opt, resume_ck.get("optimizer"))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={args.device} arch={cfg.arch} params={n_params / 1e6:.2f}M "
          f"table={cfg.vocab_rows}x{cfg.emb_dim} rows={len(train_rows)} vocab={len(vocab)} "
          f"opt={opt_label} sdpa={cfg.use_sdpa} qk_norm={cfg.qk_norm} rope={cfg.rope} "
          f"rope_base={cfg.rope_base:.0f} span={cfg.span_k}x{cfg.span_w} ckpt={cfg.ckpt} "
          f"chunk_attn={cfg.chunk_attn} "
          f"max_ctx={args.max_ctx} max_opt={args.max_opt} batch={args.batch_size} "
          f"ckpt_every={args.ckpt_every} pub_eval_every={args.public_eval_every}")
    print(f"lr groups: attn={args.lr * args.lr_attn_mult:.2e} head={args.lr * args.lr_head_mult:.2e} "
          f"embed={args.lr * args.lr_embed_mult:.2e} | {gcounts}")
    print(f"families: " + ", ".join(sorted({r['meta']['family'] for r in rows})))
    print(f"valid rows: {len(eval_rows)} from {'<in-train split>' if args.valid is None else args.valid}")

    best_loss = float(resume_ck["best_loss"]) if resume_ck is not None else float("inf")
    best_state = resume_ck.get("best") if resume_ck is not None else None
    stale = int(resume_ck["stale"]) if resume_ck is not None else 0
    resume_path = f"{args.out}.resume.pt"

    run_start = time.perf_counter()
    for epoch in range(resume_epoch, args.epochs):
        model.train()
        total_loss_t = torch.zeros((), dtype=torch.float32, device=args.device)
        total_correct_t = torch.zeros((), dtype=torch.int64, device=args.device)
        total = 0
        t_collate = t_fwd = t_bwd = t_opt = t_eval = t_save = t_pub = 0.0
        epoch_t0 = time.perf_counter()
        t_last = epoch_t0
        for step, (bb, bm, qb, qm, ob, om, probs, correct, families) in enumerate(train_dl):
            t_collate += time.perf_counter() - t_last
            bb, bm, qb, qm, ob, om, probs, correct = (
                bb.to(args.device), bm.to(args.device), qb.to(args.device), qm.to(args.device),
                ob.to(args.device), om.to(args.device), probs.to(args.device), correct.to(args.device),
            )
            _f = time.perf_counter()
            logits = model(bb, bm, qb, qm, ob, om)
            t_fwd += time.perf_counter() - _f
            loss_raw = masked_kl(logits, probs, om)
            _b = time.perf_counter()
            (loss_raw / args.accum).backward()
            t_bwd += time.perf_counter() - _b
            total_loss_t = total_loss_t + loss_raw.detach() * len(probs)
            total += len(probs)
            if (step + 1) % args.accum == 0:
                _o = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                opt.zero_grad()
                t_opt += time.perf_counter() - _o
            if args.trace_every and step % args.trace_every == 0:
                tag = {n.split(".")[-1]: t for n, t in gcounts.items()}
                _g = {"attn": [], "surface": [], "head": []}
                for name, p in model.named_parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    if any(k in name for k in ("enc.", "opt_ca", "qk_attn", "csum_", "order.")):
                        _g["attn"].append(p.grad)
                    elif any(k in name for k in ("embed.", "pos.")):
                        _g["surface"].append(p.grad)
                    else:
                        _g["head"].append(p.grad)
                gn = {k: (sum((g * g).sum().item() for g in v) ** 0.5) for k, v in _g.items()}
                mx = max(p.abs().max().item() for p in model.parameters())
                print(f"  step {step:5d} | grad attn={gn['attn']:.3f} head={gn['head']:.3f} "
                      f"embed={gn['surface']:.3f} | max|W|={mx:.3f}")
            pred = logits.masked_fill(~om.any(dim=-1), float("-inf")).argmax(dim=1)
            ci = correct[:, 0]
            ok = ci >= 0
            if ok.any():
                total_correct_t = total_correct_t + (pred[ok] == ci[ok]).sum()
            if step % 50 == 0:
                acc = float(total_correct_t) / total if total else 0.0
                print(f"epoch {epoch} step {step} loss {float(total_loss_t) / max(1, total or 1):.4f} "
                      f"acc {acc:.3f} ({time.perf_counter() - epoch_t0:.0f}s elapsed)", flush=True)
            t_last = time.perf_counter()

        model.eval()
        _e0 = time.perf_counter()
        eval_correct, eval_total = 0, 0
        eval_loss = 0.0
        with torch.no_grad():
            for bb, bm, qb, qm, ob, om, probs, correct, families in eval_dl:
                bb, bm, qb, qm, ob, om, probs, correct = (
                    bb.to(args.device), bm.to(args.device), qb.to(args.device), qm.to(args.device),
                    ob.to(args.device), om.to(args.device), probs.to(args.device), correct.to(args.device),
                )
                logits = model(bb, bm, qb, qm, ob, om)
                eval_loss += float(masked_kl(logits, probs, om)) * len(probs)
                pred = logits.masked_fill(~om.any(dim=-1), float("-inf")).argmax(dim=1)
                ci = correct[:, 0]
                ok = ci >= 0
                if ok.any():
                    eval_correct += int((pred[ok] == ci[ok]).sum())
                    eval_total += int(ok.sum())
        t_eval = time.perf_counter() - _e0
        eval_loss_v = eval_loss / max(1, len(eval_ds))
        eval_acc_v = eval_correct / max(1, eval_total)
        improved = eval_loss_v < best_loss - 1e-6
        if improved:
            best_loss = eval_loss_v
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
            mark = "  (best)"
        else:
            stale += 1
            mark = f"  (stale {stale}/{args.patience})"
        print(f"epoch {epoch} eval_loss {eval_loss_v:.4f} eval_acc {eval_acc_v:.3f}{mark}")

        if args.ckpt_every and (epoch + 1) % args.ckpt_every == 0:
            _s0 = time.perf_counter()
            if best_state is not None:
                model.load_state_dict(best_state)  # save the BEST so far
            save_resume(resume_path, cfg, vocab, model, opt, epoch, best_loss, best_state,
                        stale, args.device, args)
            save_best_artifact(f"{args.out}.best_e{epoch + 1}.pt", cfg, vocab, model,
                               args.seed, args)
            t_save = time.perf_counter() - _s0
            print(f"[ckpt] epoch {epoch + 1}: best-so-far saved -> {resume_path}, "
                  f"{args.out}.best_e{epoch + 1}.pt ({t_save:.1f}s)", flush=True)
            model.train()

        if args.public_eval_every and (epoch + 1) % args.public_eval_every == 0:
            _s0 = time.perf_counter()
            tmp = f"{args.out}.tmp_pub.pt"
            torch.save({"config": asdict(cfg), "model": model.state_dict(),
                        "vocab": vocab.to_state()}, tmp)
            build_ckpt_s = time.perf_counter() - _s0
            t_pub += run_public_eval(tmp, args.public_eval_which, args.public_eval_cache,
                                     args.device, args.out)
            t_pub += build_ckpt_s
            if os.path.exists(tmp):
                os.remove(tmp)

        print(f"  time: collate={t_collate:6.1f}s fwd={t_fwd:6.1f}s bwd={t_bwd:6.1f}s "
              f"opt={t_opt:6.1f}s eval={t_eval:6.1f}s save={t_save:6.1f}s pub={t_pub:6.1f}s "
              f"| epoch total {time.perf_counter() - epoch_t0:6.1f}s", flush=True)

        if stale >= args.patience:
            print(f"early stop after {args.patience} epoch(s) without eval improvement")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best checkpoint (eval_loss {best_loss:.4f})")
    torch.save({"config": asdict(cfg), "model": model.state_dict(), "seed": args.seed, "vocab": vocab.to_state(),
            "train": {"lr": args.lr, "lr_attn_mult": args.lr_attn_mult, "lr_embed_mult": args.lr_embed_mult,
                      "lr_head_mult": args.lr_head_mult,
                      "epochs": args.epochs, "patience": args.patience, "clip": args.clip,
                      "valid": args.valid}}, args.out)
    print(f"saved {args.out} ({n_params / 1e6:.2f}M params) | total run {time.perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    main()