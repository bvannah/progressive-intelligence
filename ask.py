from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from data import collate, build_example_v7
from tokenizer import Vocab
from model import DecisionNet, NgramConfig
from constants import MODEL_DATA_V8


def build_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    conf = ckpt["config"]
    cfg = NgramConfig(**{**conf, "opt_order": conf.get("opt_order", False), "opt_gate": conf.get("opt_gate", False)})
    vocab = None
    if cfg.arch in ("v7", "v8"):
        vocab = Vocab.from_state(ckpt["vocab"])
    model = DecisionNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return cfg, vocab, model


def encode(cfg, vocab, context, question, options, max_ctx):
    n = len(options)
    if not 2 <= n <= 8:
        raise ValueError(f"need 2-8 options, got {n}")
    row = {
        "context": context,
        "question": question,
        "options": options,
        "teacher_probs": None,
        "correct_index": None,
        "meta": {"family": "interactive"},
    }
    ex = build_example_v7(row, cfg, vocab, max_ctx, opt_cap=cfg.max_len)
    return ex


def predict(cfg, vocab, model, context, question, options, max_ctx, device, cache):
    ex = encode(cfg, vocab, context, question, options, max_ctx)
    bb, bm, qb, qm, ob, om, _, _, _ = collate([ex])
    bb, bm, qb, qm, ob, om = (t.to(device, non_blocking=True) for t in (bb, bm, qb, qm, ob, om))
    key = (context, max_ctx)
    st = cache.pop(key, None)
    with torch.no_grad():
        logits, states = model(bb, bm, qb, qm, ob, om, prev_ctx=[st])
    cache[key] = states[0]
    probs = F.softmax(logits, dim=1)[0]
    return probs, bb.shape[1]


def show(context, question, options, probs, n_ctx=0):
    print("\n" + "-" * 60)
    if context:
        print(f"context ({n_ctx} tokens): {context[:120]}{'...' if len(context) > 120 else ''}")
    print(f"question: {question[:120]}{'...' if len(question) > 120 else ''}")
    k = len(options)
    order = sorted(range(k), key=lambda i: -probs[i].item())
    for rank, i in enumerate(order):
        bar = "#" * int(round(probs[i].item() * 30))
        marker = " <--" if rank == 0 else ""
        print(f"  {chr(65 + i)}. {probs[i].item():.4f} {bar:<30}  {options[i][:60]}{marker}")
    print("-" * 60)


def interactive(cfg, vocab, model, max_ctx, device):
    cache = {}
    print("Ask the model. Empty input at any prompt ends the session.")
    print("Options: comma-separated list of 2-8 choices.\n")
    while True:
        try:
            context = input("context (optional): ").strip()
            question = input("question: ").strip()
            if not question:
                print("bye")
                break
            raw = input("options (comma-separated): ").strip()
            if not raw:
                print("bye")
                break
            options = [o.strip() for o in raw.split(",") if o.strip()]
            if not 2 <= len(options) <= 8:
                print(f"  need 2-8 options, got {len(options)}")
                continue
            probs, n_ctx = predict(cfg, vocab, model, context, question, options, max_ctx, device, cache)
            show(context, question, options, probs, n_ctx)
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break


def main():
    ap = argparse.ArgumentParser(description="Query the trained model with a question + 2-8 options.")
    ap.add_argument("--model", default=str(MODEL_DATA_V8))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--context", default=None, help="optional context passage")
    ap.add_argument("--question", default=None, help="question text")
    ap.add_argument("-o", "--option", action="append", dest="option_list", default=None,
                    help="answer option (repeatable, or use --options with commas)")
    ap.add_argument("--options", default=None, help="comma-separated options")
    ap.add_argument("--max-ctx", type=int, default=8000, help="max context tokens (default 8000; RoPE is unbounded)")
    ap.add_argument("--chunk", action="store_true", help="use chunked attention (lower memory, slower; for very long contexts)")
    args = ap.parse_args()

    cfg, vocab, model = build_model(args.model, args.device)
    if args.chunk:
        cfg.chunk_attn = True
    max_ctx = args.max_ctx

    cache = {}
    if args.question:
        options = args.options or []
        if args.option_list:
            options = options + args.option_list if options else args.option_list
        if isinstance(options, str):
            options = [o.strip() for o in options.split(",") if o.strip()]
        probs, n_ctx = predict(cfg, vocab, model, args.context or "", args.question, options, max_ctx, args.device, cache)
        show(args.context or "", args.question, options, probs, n_ctx)
    else:
        interactive(cfg, vocab, model, max_ctx, args.device)


if __name__ == "__main__":
    main()