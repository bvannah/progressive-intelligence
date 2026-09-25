"""Part D step 3 fix: enforce the 90/10 per-family train/eval split.

- removes exact duplicate rows (same row_key) within each file and across
  files (train copy kept)
- moves the minimum deterministic set of rows from train -> eval so every
  family reaches eval_fraction (default 0.09)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from merge_data import load_rows, row_key


def dedup(rows):
    seen = set()
    out = []
    for r in rows:
        k = row_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out, len(rows) - len(out)


def by_family(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["meta"]["family"]].append(r)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--min-eval-fraction", type=float, default=0.09)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    train, tdup = dedup(load_rows(args.train))
    eval_, edup = dedup(load_rows(args.eval))
    train_keys = {row_key(x) for x in train}
    cross = len([1 for r in eval_ if row_key(r) in train_keys])
    eval_ = [r for r in eval_ if row_key(r) not in train_keys]

    ef = args.min_eval_fraction
    tfam = by_family(train)
    efam = by_family(eval_)

    removed = 0
    for fam, rows in list(tfam.items()):
        have = len(efam.get(fam, []))
        total = len(rows) + have
        need_eval = int(round(total * ef)) + 1
        deficit = need_eval - have
        if deficit <= 0:
            continue
        rng = random.Random(f"{args.seed}:rebal:{fam}")
        order = list(range(len(rows)))
        rng.shuffle(order)
        take = set(order[:deficit])
        moved = [rows[i] for i in sorted(take)]
        rows[:] = [r for i, r in enumerate(rows) if i not in take]
        efam[fam].extend(moved)
        removed += len(moved)
        print(
            f"  {fam}: move {len(moved)} -> eval (train {len(rows)} / eval {len(efam[fam])})",
            flush=True,
        )

    new_train = [r for rows in tfam.values() for r in rows]
    new_eval = [r for rows in efam.values() for r in rows]
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in new_train:
            f.write(json.dumps(r) + "\n")
    with open(args.out_eval, "w", encoding="utf-8") as f:
        for r in new_eval:
            f.write(json.dumps(r) + "\n")
    print(f"intra-file duplicates dropped: train={tdup} eval={edup}")
    print(f"cross-file duplicate(s) removed from eval: {cross}")
    print(f"rows moved train->eval: {removed}")
    print(f"final written train={len(new_train)} eval={len(new_eval)}")


if __name__ == "__main__":
    main()