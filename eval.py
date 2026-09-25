from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DecisionDataset, collate, load_jsonl
from tokenizer import Vocab
from model import DecisionNet, NgramConfig
from constants import EVAL_DATA


def ece(probs, correct):
    ok = correct >= 0
    if not ok.any():
        return 0.0, 0
    p = probs[ok]
    c = correct[ok]
    conf, pred = p.max(dim=1)
    acc = (pred == c).float()
    bins = torch.linspace(0, 1, 11).to(p.device)
    total = 0.0
    counts = 0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        sel = (conf >= lo) & (conf < hi)
        if i == 9:
            sel = (conf >= lo) & (conf <= hi)
        n = int(sel.sum())
        if n == 0:
            continue
        total += n * abs(acc[sel].mean() - conf[sel].mean())
        counts += n
    return float(total / max(1, counts)), int(ok.sum())


def brier(probs, ref):
    d = probs - ref
    return float((d * d).sum(dim=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(EVAL_DATA))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-ctx", type=int, default=768)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    conf = ckpt["config"]
    cfg = NgramConfig(**{**conf, "opt_order": conf.get("opt_order", False), "opt_gate": conf.get("opt_gate", False)})
    vocab = None
    if cfg.arch in ("v7", "v8"):
        vocab = Vocab.from_state(ckpt["vocab"])
        cfg.vocab_rows = vocab.n_rows()
    model = DecisionNet(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = load_jsonl(args.data)
    ds = DecisionDataset(rows, cfg, max_ctx=args.max_ctx, vocab=vocab)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    all_probs = []
    all_correct = []
    all_families = []
    all_n = []
    total_logloss = 0.0
    n_ref = 0
    ref_sum = 0.0
    perm_max_err = 0.0

    with torch.no_grad():
        for bb, bm, qb, qm, ob, om, probs, correct, families in dl:
            bb, bm, qb, qm, ob, om = (bb.to(args.device), bm.to(args.device), qb.to(args.device),
                                      qm.to(args.device), ob.to(args.device), om.to(args.device))
            logits = model(bb, bm, qb, qm, ob, om)
            valid = om.any(dim=-1)
            logits = logits.masked_fill(~valid, float("-inf"))
            p = torch.softmax(logits, dim=1)
            all_probs.append(p.cpu())
            all_correct.append(correct)
            all_families += families
            all_n.append(valid.sum(dim=1).cpu())

            ok = correct >= 0
            ci = correct[:, 0]
            pc = p.gather(1, ci.clamp(min=0).to(args.device).unsqueeze(1)).squeeze(1)
            total_logloss += float((-pc.clamp_min(1e-9).log()[ok[:, 0]].sum()))
            n_ref += int(ok[:, 0].sum())

            ref_sum += float(((p.cpu() - probs) ** 2).sum(dim=1).sum())

            num = ob.shape[1]
            perm = torch.randperm(num)
            ob_p = ob[:, perm]
            om_p = om[:, perm]
            logits2 = model(bb, bm, qb, qm, ob_p, om_p)
            p2 = torch.softmax(
                logits2.masked_fill(~om_p.any(dim=-1), float("-inf")), dim=1)
            inv = torch.argsort(perm)
            perm_max_err = max(perm_max_err, float((p2[:, inv] - p).abs().max()))

    probs_all = torch.cat(all_probs)
    correct_all = torch.cat(all_correct)

    ci_all = correct_all[:, 0]
    ok = ci_all >= 0
    acc = float((probs_all[ok].argmax(dim=1) == ci_all[ok]).float().mean())
    avg_ll = total_logloss / max(1, n_ref)
    avg_brier = ref_sum / max(1, len(rows))
    ece_val, ece_n = ece(probs_all, correct_all[:, 0])

    print(f"samples={len(rows)} accuracy={acc:.4f} logloss={avg_ll:.4f} "
          f"brier={avg_brier:.4f} ece={ece_val:.4f} (n={ece_n})")
    print(f"permutation-invariance max|dp| = {perm_max_err:.2e}")

    fam_acc = {}
    for fam in sorted(set(all_families)):
        idx = [i for i, f in enumerate(all_families) if f == fam]
        pi = probs_all[idx]
        ci = correct_all[idx, 0]
        k = ci >= 0
        fa = float((pi[k].argmax(dim=1) == ci[k]).float().mean()) if k.any() else float("nan")
        fam_acc[fam] = fa
    for fam, fa in fam_acc.items():
        print(f"  {fam:12s} acc {fa:.4f}")

    all_n_t = torch.cat(all_n)
    by_names = sorted(set(all_n_t.tolist()))
    if len(by_names) > 1:
        print("  by option count (accuracy on determinate rows):")
        for nv in by_names:
            sel = (all_n_t == nv) & ok
            if not sel.any():
                continue
            na = float((probs_all[sel].argmax(dim=1) == ci_all[sel]).float().mean())
            print(f"    n={nv:2d} acc {na:.4f} (n={int(sel.sum())})")

    device = args.device
    if device == "cuda":
        bb, bm, qb, qm, ob, om, probs, correct, fam = next(iter(DataLoader([ds[0]], batch_size=1, collate_fn=collate)))
        bb, bm, qb, qm, ob, om = bb.to(device), bm.to(device), qb.to(device), qm.to(device), ob.to(device), om.to(device)
        model(bb, bm, qb, qm, ob, om)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_rep = 50
        for _ in range(n_rep):
            model(bb, bm, qb, qm, ob, om)
        torch.cuda.synchronize()
        cuda_ms = (time.perf_counter() - t0) / n_rep * 1000
        print(f"latency cuda  {cuda_ms:.3f} ms/call (sample 'single')")

        cpu_model = DecisionNet(cfg)
        cpu_model.load_state_dict(ckpt["model"])
        cpu_model.eval()
        bb, bm, qb, qm, ob, om = (bb.to("cpu"), bm.to("cpu"), qb.to("cpu"), qm.to("cpu"), ob.to("cpu"), om.to("cpu"))
        with torch.no_grad():
            cpu_model(bb, bm, qb, qm, ob, om)
            t0 = time.perf_counter()
            for _ in range(n_rep):
                cpu_model(bb, bm, qb, qm, ob, om)
        cpu_ms = (time.perf_counter() - t0) / n_rep * 1000
        print(f"latency cpu   {cpu_ms:.3f} ms/call")


if __name__ == "__main__":
    main()