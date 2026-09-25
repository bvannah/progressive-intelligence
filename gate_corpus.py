"""Corpus gate (Part D step 3) — run BEFORE training on the merged train/eval files.

Checks (train and eval):
  1. every row validates via merge_data.validate_row (2..8 options, probs, tier, provenance)
  2. teacher_probs: len == n, sums to 1, correct_index < n, gold index has >= 0.9 prob
  3. no empty / NA-style option marked correct
  4. meta.tier present everywhere; every tier=fact row carries provenance; fact_flagged
     (shape) rows are counted (rewrite/neutralize gate)
  5. n-histogram per family: variable-N families actually contain n<8 rows (train; --require-var-n)
  6. family table; overall n-hist; tier mix; cross-file dedup; per-family eval fraction (90/10)

Exit code 0 = gate passed, 1 = checks failed.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from merge_data import load_rows, row_key, _fact_flagged, is_na_padding, topic_of
from prompts import FAMILY_OPTION_COUNTS


def gate(rows: list[dict], label: str, require_var_n: bool) -> bool:
    ok = True
    n_all = Counter(len(r["options"]) for r in rows)
    tiers = Counter(r["meta"].get("tier", "<none>") for r in rows)
    no_tier = sum(1 for r in rows if "tier" not in r["meta"])
    fact_no_prov = sum(
        1 for r in rows
        if r["meta"].get("tier") == "fact"
        and not any(r["meta"].get(k) for k in ("source_page", "dataset", "support_span", "source"))
    )
    flagged_shape = sum(
        1 for r in rows
        if r["meta"].get("tier") != "fact" and (r["meta"].get("fact_flagged") or _fact_flagged(r))
    )
    bad_probs = bad_na_correct = gold_low = 0
    for r in rows:
        n = len(r["options"])
        p = r["teacher_probs"]
        if len(p) != n or abs(sum(p) - 1.0) > 1e-2:
            bad_probs += 1
            continue
        if r["correct_index"] is not None:
            if not (0 <= r["correct_index"] < n):
                bad_probs += 1
            else:
                txt = r["options"][r["correct_index"]]
                if is_na_padding(txt):
                    bad_na_correct += 1
                if p[r["correct_index"]] < 0.5:
                    bad_probs += 1
                elif p[r["correct_index"]] < 0.9:
                    gold_low += 1

    fam = Counter(r["meta"]["family"] for r in rows)
    var_n_ok = True
    print(f"\n== {label} gate ==")
    print(f"  rows={len(rows)}  n-hist={dict(sorted(n_all.items()))}")
    print(f"  tiers={dict(tiers)}  missing-tier={no_tier}  fact-no-provenance={fact_no_prov}  "
          f"shape-fact-flagged={flagged_shape}")
    print(f"  bad-prob/correct-index={bad_probs}  gold-soft(0.5-0.9)={gold_low}  "
          f"NA-marked-correct={bad_na_correct}")
    print("  per-family:")
    for f, c in fam.most_common():
        ns = Counter(len(r["options"]) for r in rows if r["meta"]["family"] == f)
        n_wide = sum(v for k, v in ns.items() if k < 8)
        target = FAMILY_OPTION_COUNTS.get(f)
        fam_ok = "ok" if not require_var_n or target is None or n_wide > 0 else "ALL-8!"
        if target is not None and require_var_n and n_wide == 0:
            var_n_ok = False
        print(f"    {f:14s} {c:6d}  n<8:{n_wide:5d}  n4:{ns[4]:5d}  n8:{ns[8]:5d}  "
              f"n-hist {dict(sorted(ns.items()))}  [{fam_ok}]")

    for name, cond in [
        ("no bad probs / correct-index", bad_probs == 0),
        ("no NA/empty option marked correct", bad_na_correct == 0),
        ("tier present on all rows", no_tier == 0),
        ("every tier=fact row has provenance", fact_no_prov == 0),
        ("variable-N families actually vary (n<8 rows exist)", var_n_ok),
    ]:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--require-var-n", action="store_true",
                    help="require variable-N rows (n<8) in the option-counts families")
    args = ap.parse_args()

    train = load_rows(args.train)
    eval_ = load_rows(args.eval)

    t_ok = gate(train, "train", args.require_var_n)
    e_ok = gate(eval_, "eval", False)

    seen = set()
    dups = 0
    for r in train + eval_:
        k = row_key(r)
        if k in seen:
            dups += 1
        seen.add(k)
    print(f"\n  cross-file exact-duplicates={dups}")

    topics_train: dict[str, set] = defaultdict(set)
    topics_eval: dict[str, set] = defaultdict(set)
    for r in train:
        t = topic_of(r)
        if t:
            topics_train[r["meta"]["family"]].add(t)
    for r in eval_:
        t = topic_of(r)
        if t:
            topics_eval[r["meta"]["family"]].add(t)
    overlaps = {
        fam: (topics_train[fam] & topics_eval[fam])
        for fam in topics_train
        if fam in topics_eval and (topics_train[fam] & topics_eval[fam])
    }
    if overlaps:
        print("  topic-overlap between train and eval (informational):")
        for fam in sorted(overlaps):
            print(f"    {fam}: {len(overlaps[fam])} shared topic(s)")
    else:
        print("  topic-disjoint split: no topic shared between train and eval")

    fam_train = Counter(r["meta"]["family"] for r in train)
    fam_eval = Counter(r["meta"]["family"] for r in eval_)
    low_eval = []
    for f in sorted(set(fam_train) | set(fam_eval)):
        total = fam_train[f] + fam_eval[f]
        frac = fam_eval[f] / total if total else 0
        if 0 < total and frac < 0.09:
            low_eval.append(f"{f}({frac:.3f})")
    print(f"  per-family eval fraction >= 0.09: "
          f"{'all ok' if not low_eval else 'LOW: ' + ', '.join(low_eval)}")
    e_ok = e_ok and not low_eval and dups == 0

    print(f"\nOVERALL: {'PASS' if (t_ok and e_ok) else 'FAIL'}")
    raise SystemExit(0 if (t_ok and e_ok) else 1)


if __name__ == "__main__":
    main()