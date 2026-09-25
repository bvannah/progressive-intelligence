from __future__ import annotations

import argparse
import json
import pathlib

from constants import PUBLIC_CACHE
from pub_eval_data import load_or_fetch, export_corpus_row, SOURCES


def main():
    ap = argparse.ArgumentParser(
        description="Fetch public eval sets, deterministically split 50/50, export the train-half "
                    "as tier=fact corpus rows (scored eval uses the other half only).")
    ap.add_argument("--cache-dir", default=str(PUBLIC_CACHE))
    ap.add_argument("--train-out", default="", help="jsonl of train-half rows (default: <data>/public_eval_train.jsonl)")
    ap.add_argument("--which", default=",".join(SOURCES))
    ap.add_argument("--frac-train", type=float, default=0.5)
    ap.add_argument("--max-rows", type=int, default=20_000)
    ap.add_argument("--overwrite", action="store_true", help="refetch even if cache exists")
    args = ap.parse_args()

    cache_dir = pathlib.Path(args.cache_dir)
    train_out = pathlib.Path(args.train_out) if args.train_out else \
        pathlib.Path(PUBLIC_CACHE).parent / "public_eval_train.jsonl"
    train_out.parent.mkdir(parents=True, exist_ok=True)

    total_train = total_eval = 0
    with open(train_out, "w", encoding="utf-8") as f:
        for name in [w.strip() for w in args.which.split(",") if w.strip()]:
            if name not in SOURCES:
                print(f"skip unknown dataset {name}")
                continue
            entries = load_or_fetch(name, cache_dir, args.max_rows, force_refresh=args.overwrite)
            train = [e for e in entries if e["split"] == "train-half"]
            eval_ = [e for e in entries if e["split"] == "eval-half"]
            exported = 0
            for idx, e in enumerate(train):
                row = export_corpus_row(name, e["example"], idx)
                if row is None:
                    continue
                f.write(json.dumps(row) + "\n")
                exported += 1
            total_train += exported
            total_eval += len(eval_)
            print(f"{name:16s} total={len(entries):5d} train-half(->train)={exported:5d} "
                  f"eval-half(scored)={len(eval_):5d}")
    print(f"train-half rows exported ({total_train}) -> {train_out}")
    print(f"eval-half rows reserved for scoring ({total_eval}); "
          f"any row used for training is excluded from public-eval scores")


if __name__ == "__main__":
    main()