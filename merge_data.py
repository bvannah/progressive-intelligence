from __future__ import annotations

import argparse
import json
import random
import re
import uuid
from collections import Counter, defaultdict
from statistics import median

from constants import PUBLIC_CACHE

KNOWN_FAMILIES = {
    "retrieval", "paraphrase", "order", "ambiguity", "permuted",
    "numeric", "spatial", "compare", "oddoneout", "category", "property", "negation", "conditional",
    "define", "analogy", "cause", "summary", "tone", "intent", "knowledge", "consistency",
    "wiki_fact",
    "arithmetic", "counting", "sets", "beforeafter", "ordinal", "sequence_next",
    "timeclock", "dates", "negprop", "synonym", "antonym", "transitivity", "distribution",
    "wikicat", "wikibox", "wikiret",
    "openbookqa", "sciq", "commonsense", "hellaswag", "medqa",
    "open", "longread",
    "chain_arithmetic", "chain_logic", "chain_sequence", "chain_retrieval",
    "reformulate", "near_miss", "noise",
    "quality_qa", "qasc", "medmcqa", "mmlu", "gsm_mc", "math_mc",
    "arc-challenge", "mmlu-astronomy", "race-middle",
}

# Families whose content asserts real-world facts and must carry disk/dataset provenance.
FACT_FAMILIES = {
    "wiki_fact",
    "openbookqa", "sciq", "commonsense", "hellaswag", "medqa",
    "quality_qa", "qasc", "medmcqa", "mmlu", "gsm_mc", "math_mc",
    "arc-challenge", "mmlu-astronomy", "race-middle",
}

_FACT_YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20[0-2]\d)\b")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip()).lower()


def row_key(row: dict) -> tuple:
    return (
        _norm(row.get("context", "")),
        _norm(row.get("question", "")),
        tuple(_norm(o) for o in row.get("options", [])),
    )


def _fact_flagged(row: dict) -> bool:
    """Heuristic for tier=shape rows that read like encyclopedia fact-assertions
    (year + capitalized proper-noun phrase). Such rows are flagged (meta.fact_flagged)
    so the rewrite pass can neutralize them or the merge can drop them."""
    text = f"{row.get('context', '')} {row.get('question', '')}"
    if not _FACT_YEAR.search(text):
        return False
    names = re.findall(r"[A-Z][A-Za-z']+(?:[\s-]+[A-Z][A-Za-z']+){1,}", text)
    return bool(names)


def validate_row(row: dict, source: str = "<llm>") -> dict:
    errors = []
    if not isinstance(row, dict):
        raise ValueError(f"{source}: row is not a JSON object")
    if not isinstance(row.get("context"), str):
        errors.append("context missing/empty")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        errors.append("question missing/empty")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 8 or any(
        not isinstance(o, str) or not o.strip() for o in options
    ):
        errors.append("options must be a list of 2..8 non-empty strings")
    ci = row.get("correct_index")
    if ci is not None and (
        not isinstance(ci, int) or isinstance(ci, bool) or not 0 <= ci < len(options)
    ):
        errors.append("correct_index must be null or an int in 0..len(options)-1")
    if errors:
        raise ValueError(f"{source}: " + "; ".join(errors))

    family = row.get("meta", {}).get("family", "unknown")
    if family not in KNOWN_FAMILIES:
        family = "unknown"

    if row.get("teacher_probs") is None:
        if ci is None:
            probs = [1.0 / len(options)] * len(options)
        else:
            probs = [0.05 / (len(options) - 1)] * len(options)
            probs[ci] = 0.95
    else:
        probs = list(row["teacher_probs"])
        if len(probs) != len(options):
            raise ValueError(f"{source}: teacher_probs must have {len(options)} entries")
        s = sum(probs)
        if s <= 0:
            raise ValueError(f"{source}: teacher_probs sum must be > 0")
        probs = [p / s for p in probs]

    ls = dict(row.get("label_source") or {})
    ls.setdefault("n_samples", 1)
    ls.setdefault("vote", "direct")
    ls.setdefault("model", "unknown-model")
    ls.setdefault("temperature", 0.7)

    meta = dict(row.get("meta") or {})
    meta.setdefault("family", family)
    meta.setdefault("generated_by", f"llm:{ls.get('model')}")

    meta.setdefault("tier", "fact" if family in FACT_FAMILIES else "shape")
    if meta["tier"] == "fact":
        if not any(meta.get(k) for k in ("source_page", "dataset", "support_span", "source")):
            raise ValueError(
                f"{source}: tier=fact row ({family!r}) missing provenance "
                "(meta.source_page | meta.dataset | meta.support_span)"
            )
        meta.setdefault("fact_flagged", False)
    elif _fact_flagged(row):
        meta["fact_flagged"] = True

    return {
        "id": row.get("id") or str(uuid.uuid4()),
        "task": row.get("task") or family,
        "difficulty": row.get("difficulty") or "medium",
        "context": row["context"].strip(),
        "question": row["question"].strip(),
        "options": [o.strip() for o in options],
        "correct_index": ci,
        "teacher_probs": probs,
        "label_source": ls,
        "meta": meta,
    }


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(validate_row(row, f"{path}:{ln}"))
    return rows


_NA_SPELLINGS = {
    "n/a", "na", "n a",
    "not available", "not-available",
    "none", "nil", "unknown", "no answer",
}


def is_na_padding(txt: str) -> bool:
    """True if an option is empty or generator NA-padding.

    "Na" (the chemical symbol for sodium) is a legitimate answer in legacy
    knowledge rows, so it is deliberately never treated as padding.
    """
    t = (txt or "").strip()
    if not t:
        return True
    if t == "Na":
        return False
    return t.lower() in _NA_SPELLINGS


def print_stats(rows: list[dict], label: str) -> None:
    """Post-generation diversity sanity check (A6 n-histogram + B5 family mix /
    context-length / bigram-drift). The merge gate relies on these numbers."""
    if not rows:
        print(f"-- {label}: 0 rows --")
        return
    fam = Counter(r["meta"]["family"] for r in rows)
    n_hist = Counter(len(r["options"]) for r in rows)
    tiers = Counter(r["meta"].get("tier", "?") for r in rows)
    flagged = sum(1 for r in rows if r["meta"].get("fact_flagged"))
    ctx_len: dict[str, list[float]] = {}
    for r in rows:
        ctx_len.setdefault(r["meta"]["family"], []).append(len(r.get("context", "").split()))
    print(f"-- {label}: n={len(rows)} --")
    print("  tiers:", " ".join(f"{t}={c}" for t, c in sorted(tiers.items())),
          f" fact_flagged={flagged}")
    print("  n-hist:", " ".join(f"n{k}={n_hist[k]}" for k in sorted(n_hist)))
    print("  family mix:")
    for f, c in fam.most_common():
        m = median(ctx_len[f]) if ctx_len[f] else 0.0
        print(f"    {f:12s} {c:6d}  ctx-len median {m:5.1f}")
    toks = [w for r in rows for w in re.findall(r"[A-Za-z0-9']+", r.get("context", "").lower())]
    bigr = Counter(zip(toks, toks[1:])).most_common(10)
    if bigr:
        print("  top context bigrams:", "  ".join(f"{a} {b}:{c}" for (a, b), c in bigr))


TOPIC_KEYS = ("topic", "source_topic", "dataset", "source_page")


def topic_of(r: dict) -> str | None:
    m = r.get("meta") or {}
    for k in TOPIC_KEYS:
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return _norm(v)
    return None


def split_rows_family(args, fam: str, rows: list, rng: random.Random):
    """Returns (train_rows, eval_add) for one family, honoring --topic-disjoint."""
    if args.train_only:
        return rows, []
    # split_with hint: derived rows (e.g. noise variants) must follow their
    # source row's split so train/eval never share the same question+gold.
    forced_eval = [r for r in rows if (r.get("meta") or {}).get("split_with") == "eval"]
    forced_train = [r for r in rows if (r.get("meta") or {}).get("split_with") == "train"]
    free = [r for r in rows if (r.get("meta") or {}).get("split_with") not in ("train", "eval")]
    if not free:
        return forced_train, forced_eval
    if not args.topic_disjoint:
        n_eval = max(1, int(round(len(free) * args.eval_fraction)))
        shuffled = list(free)
        rng.shuffle(shuffled)
        return forced_train + shuffled[n_eval:], forced_eval + shuffled[:n_eval]
    # Topic-disjoint: assign WHOLE topics to eval, never splitting a topic.
    groups: dict = defaultdict(list)
    for r in free:
        groups[topic_of(r)].append(r)
    keyed = [g for k, g in groups.items() if k is not None]
    unkeyed = groups.get(None, [])
    target = max(1, int(round(len(free) * args.eval_fraction)))
    assigned = 0
    eval_add: list = []
    if keyed:
        rng.shuffle(keyed)
        for g in keyed:
            eval_add.extend(g)
            assigned += len(g)
            if assigned >= target:
                break
    unkeyed_shuffled = list(unkeyed)
    rng.shuffle(unkeyed_shuffled)
    deficit = max(0, target - assigned)
    eval_add.extend(unkeyed_shuffled[:deficit])
    used = {id(r) for r in eval_add}
    return forced_train + [r for r in free if id(r) not in used], forced_eval + eval_add


def main():
    ap = argparse.ArgumentParser(description="Merge LLM-generated rows into train/eval splits.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--new", action="append", required=True, help="LLM-output jsonl(s) to merge in")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-eval", required=True)
    ap.add_argument("--eval-fraction", type=float, default=0.15)
    ap.add_argument("--train-only", action="store_true",
                    help="route ALL new rows to train; leave eval unchanged")
    ap.add_argument("--topic-disjoint", action="store_true",
                    help="route eval rows per family by whole topics (meta.topic / "
                         "meta.source_topic / meta.dataset / meta.source_page), so train and "
                         "eval never share a topic within a family; rows without a topic "
                         "fall back to the random split")
    ap.add_argument("--drop-fact-flagged", action="store_true",
                    help="drop tier=shape rows flagged as encyclopedia-style fact assertions")
    ap.add_argument("--drop-gold-low", type=float, default=0.0,
                    help="drop rows where correct_index is set but gold teacher prob < this "
                         "(legacy rows that violate the 0.9-gold convention)")
    ap.add_argument("--drop-na-correct", action="store_true",
                    help="drop rows whose correct option is empty or NA-padding")
    ap.add_argument("--exclude-existing", action="store_true",
                    help="drop new rows whose question+options already exist in current train/eval "
                         "(cross-dataset overlap guard)")
    ap.add_argument("--exclude-public-eval", default="",
                    help="comma-list of public eval sets whose eval-half rows must never be in "
                         "training (no-overlap policy); matched by question+options identity")
    ap.add_argument("--public-eval-cache", default="",
                    help="cache dir with {set}.jsonl containing split tags (default: PUBLIC_CACHE)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    exclude_keys: set = set()
    if args.exclude_public_eval:
        import pub_eval_data as ped
        cache_dir = args.public_eval_cache or str(PUBLIC_CACHE)
        for name in [n.strip() for n in args.exclude_public_eval.split(",") if n.strip()]:
            for ent in ped.load_or_fetch(name, cache_dir):
                if ent["split"] == "eval-half":
                    ex = ent["example"]
                    exclude_keys.add((
                        _norm(ex.get("question", "")),
                        tuple(_norm(o) for o in ex.get("options", [])),
                    ))
        print(f"no-overlap guard: {len(exclude_keys)} eval-half rows reserved (never in training)")

    def _hits_public_eval(r: dict) -> bool:
        return (_norm(r.get("question", "")),
                tuple(_norm(o) for o in r.get("options", []))) in exclude_keys

    def _gold_too_low(r: dict) -> bool:
        if args.drop_gold_low <= 0 or r["correct_index"] is None:
            return False
        return r["teacher_probs"][r["correct_index"]] < args.drop_gold_low

    def _gold_na_pad(r: dict) -> bool:
        if not args.drop_na_correct or r["correct_index"] is None:
            return False
        return is_na_padding(r["options"][r["correct_index"]])

    train = load_rows(args.train)
    eval_ = load_rows(args.eval)
    dropped = 0
    excluded = 0
    if args.drop_gold_low > 0 or args.drop_na_correct:
        before = len(train) + len(eval_)
        train = [r for r in train if not _gold_too_low(r) and not _gold_na_pad(r)]
        eval_ = [r for r in eval_ if not _gold_too_low(r) and not _gold_na_pad(r)]
        dropped = before - (len(train) + len(eval_))
        if dropped:
            print(f"cleanup: dropped {dropped} legacy row(s) (drop-gold-low "
                  f"{args.drop_gold_low}, drop-na-correct={args.drop_na_correct})")
    seen = {row_key(r) for r in train} | {row_key(r) for r in eval_}

    def _qkey(r: dict) -> tuple:
        return (_norm(r.get("question", "")), tuple(_norm(o) for o in r.get("options", [])))

    seen2 = {_qkey(r) for r in train} | {_qkey(r) for r in eval_}

    if exclude_keys:
        leaks = [r for r in train if _hits_public_eval(r)] + \
                [r for r in eval_ if _hits_public_eval(r)]
        if leaks:
            print(f"FATAL [no-overlap]: {len(leaks)} existing corpus row(s) match reserved "
                  f"public eval-half rows; refusing to merge.")
            raise SystemExit(1)

    new_rows = []
    rejected = 0
    for path in args.new:
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = validate_row(json.loads(line), f"{path}:{ln}")
                except ValueError as e:
                    rejected += 1
                    if rejected < 6 or rejected % 100 == 0:
                        print(f"  reject {f'{path}:{ln}'}: {e}")
                    continue
                if _hits_public_eval(row):
                    excluded += 1
                    continue
                if args.exclude_existing and _qkey(row) in seen2:
                    excluded += 1
                    continue
                if args.drop_fact_flagged and row["meta"].get("fact_flagged"):
                    dropped += 1
                    continue
                if _gold_too_low(row):
                    dropped += 1
                    continue
                if _gold_na_pad(row):
                    dropped += 1
                    continue
                if row_key(row) in seen:
                    dropped += 1
                    continue
                seen.add(row_key(row))
                new_rows.append(row)

    train_ids = {r["id"] for r in train} | {r["id"] for r in eval_}
    for r in new_rows:
        if r["id"] in train_ids:
            r["id"] = f"{r['id']}-{uuid.uuid4().hex[:6]}"
        train_ids.add(r["id"])

    from collections import defaultdict

    by_family = defaultdict(list)
    for r in new_rows:
        by_family[r["meta"]["family"]].append(r)

    for fam, rows in by_family.items():
        rng = random.Random(f"{args.seed}:{fam}")
        train_rows, eval_add = split_rows_family(args, fam, rows, rng)
        train.extend(train_rows)
        eval_.extend(eval_add)
        print(f"{fam:12s} +{len(train_rows):4d} train  +{len(eval_add):3d} eval")

    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(args.out_eval, "w", encoding="utf-8") as f:
        for r in eval_:
            f.write(json.dumps(r) + "\n")
    print(f"train={len(train)} eval={len(eval_)} dropped={dropped} rejected={rejected} "
          f"public-eval-excluded={excluded}")
    print_stats(train, "train-stats")
    print_stats(eval_, "eval-stats")


if __name__ == "__main__":
    main()