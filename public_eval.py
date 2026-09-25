from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from data import DecisionDataset, collate
from tokenizer import Vocab
from model import DecisionNet, NgramConfig
from constants import MODEL_DATA, PUBLIC_CACHE
from pub_eval_data import load_or_fetch, SOURCES


def _ece(conf_c: list[float], conf_w: list[float]) -> float:
    if not conf_c and not conf_w:
        return float("nan")
    bins = 10
    allc = conf_c + conf_w
    total = len(allc)
    lo, hi = min(allc), max(allc)
    ece = 0.0
    for b in range(bins):
        l = lo + (hi - lo) * b / bins
        r = lo + (hi - lo) * (b + 1) / bins
        idx = [i for i, v in enumerate(allc) if l <= v < r]
        if not idx:
            continue
        n_c = sum(1 for i in idx if i < len(conf_c))
        acc = n_c / len(idx)
        conf = sum(allc[i] for i in idx) / len(idx)
        ece += len(idx) / total * abs(acc - conf)
    return ece


def _article_key(ex: dict) -> str:
    return ex["meta"].get("article") or ex["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_DATA))
    ap.add_argument("--which", default="arc-challenge,mmlu-astronomy,race-middle")
    ap.add_argument("--cache-dir", default=str(PUBLIC_CACHE))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-ctx", type=int, default=2048)
    ap.add_argument("--max-rows", type=int, default=20_000,
                    help="max public rows per set (full test sets by default)")
    ap.add_argument("--frac-train", type=float, default=0.5,
                    help="fraction of each public eval set allowed into training; "
                         "those rows are excluded from these scores (must match fetch_public)")
    ap.add_argument("--include-train-half", action="store_true",
                    help="DEBUG: also score the train-half (leaks the eval; never use for real numbers)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    conf = ckpt.get("config") or {}
    cfg = NgramConfig(**{**conf, "opt_order": conf.get("opt_order", False), "opt_gate": conf.get("opt_gate", False)})
    vocab = None
    if cfg.arch in ("v7", "v8"):
        vocab = Vocab.from_state(ckpt["vocab"])
        cfg.vocab_rows = vocab.n_rows()
    m = DecisionNet(cfg)
    m.load_state_dict(ckpt["model"])
    m.to(args.device)
    m.eval()

    pathlib.Path(args.cache_dir).mkdir(exist_ok=True)
    print(f"model {args.model} -- public MC (no-overlap policy: only eval-half of each set is scored)")
    for which in [w.strip() for w in args.which.split(",")]:
        if which not in SOURCES:
            print(f"skip unknown source {which}")
            continue
        entries = load_or_fetch(which, args.cache_dir, args.max_rows)
        n_eval_half = sum(1 for e in entries if e["split"] == "eval-half")
        n_train_half = len(entries) - n_eval_half
        examples_raw = [e["example"] for e in entries
                        if e["split"] == ("eval-half" if not args.include_train_half else "train-half")]
        flag = "include-train-half(DEBUG, leaked)" if args.include_train_half else "scored"
        examples = [dict(ex) for ex in examples_raw]
        for i, ex in enumerate(examples):
            ex.setdefault("id", f"{which}-{i}")
        n = len(examples)
        n_opt = len(examples[0]["options"]) if examples else None
        ds = DecisionDataset(examples, cfg, max_ctx=args.max_ctx, vocab=vocab)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

        acc = logloss = brier = 0.0
        n = ll_n = 0
        conf_c, conf_w = [], []
        by_n: dict = {}
        keys = [_article_key(ex) for ex in examples]
        ctx_cache: dict = {}
        n_ctx_enc = 0
        t0 = time.perf_counter()
        with torch.no_grad():
            for k, (bb, bm, qb, qm, ob, om, probs, correct, fam) in enumerate(dl):
                bb = bb.to(args.device); bm = bm.to(args.device)
                qb = qb.to(args.device); qm = qm.to(args.device)
                ob = ob.to(args.device); om = om.to(args.device)
                bstart = k * args.batch
                prev = [ctx_cache.get(keys[bstart + j]) for j in range(bb.shape[0])]
                if hasattr(m.net, "forward_with_ctx_cache"):
                    logits, states = m(bb, bm, qb, qm, ob, om, prev_ctx=prev)
                    for j, st in enumerate(states):
                        if prev[j] is None:
                            ctx_cache[keys[bstart + j]] = st
                            n_ctx_enc += 1
                    while len(ctx_cache) > 128:
                        ctx_cache.pop(next(iter(ctx_cache)))
                else:
                    logits = m(bb, bm, qb, qm, ob, om)
                valid = om.any(dim=-1)
                p = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=1)
                for j in range(p.shape[0]):
                    cc = int(correct[j][0]) if correct.dim() == 2 else int(correct[j])
                    pred = int(p[j].argmax())
                    if cc >= 0:
                        ll_n += 1
                        conf = float(p[j][cc])
                        logloss += -math.log(max(conf, 1e-9))
                        brier += float(((p[j] - torch.nn.functional.one_hot(torch.tensor(cc), num_classes=p.shape[1]).float().to(p.device)) ** 2).sum())
                        if pred == cc:
                            acc += 1
                            conf_c.append(float(p[j][pred]))
                        else:
                            conf_w.append(float(p[j][pred]))
                        slot = by_n.setdefault(int(valid[j].sum().item()), [0, 0])
                        slot[1] += 1
                        if pred == cc:
                            slot[0] += 1
                    n += 1
        dt = (time.perf_counter() - t0) / max(1, n) * 1000
        ece = _ece(conf_c, conf_w)
        cache_note = f" ctx_enc={n_ctx_enc}/{n}" if hasattr(m.net, "forward_with_ctx_cache") else ""
        excl = f" ({n_train_half} train-half rows excluded from score)" if not args.include_train_half else ""
        print(f"--- {which:16s} scored={ll_n:4d}/{n} pool={len(entries)}(train-half={n_train_half}, "
              f"eval-half={n_eval_half}) nopt={n_opt} acc={acc / max(1, ll_n):.3f} "
              f"logloss={logloss / max(1, ll_n):.3f} brier={brier / max(1, ll_n):.3f} "
              f"ece={ece:.3f} latency={dt:.3f} ms/row{cache_note}{excl} ---", flush=True)
        print("    honest note: options are pre-frozen in dataset order; no A/B swap, "
              "no teacher prob leakage; train-half rows are excluded from this score "
              "(no-overlap policy)")
        if by_n:
            parts = "  ".join(f"n={k} {c}/{t}={c / max(1, t):.3f}" for k, (c, t) in sorted(by_n.items()))
            print(f"    by option count: {parts}")


if __name__ == "__main__":
    main()