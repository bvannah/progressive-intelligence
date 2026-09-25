"""Post-processing passes that multiply value and variety of existing rows:

  --mode near_miss    rewrite ONE distractor into a near-miss of the correct option
                      (minimal-edit wrong answer); closeness auto-checked.
  --mode reformulate  rewrite context+question into a different register; option set,
                      order, correct_index and teacher_probs must stay verbatim.
  --mode noise        procedural: inject 1-3 irrelevant filler sentences into the
                      context; gold unchanged (cheap, high-volume).

All modes keep the source row's meta (topic, source, provenance, tier) and flush per
row, so partial runs are restartable with --append.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import time

import generate as G
from merge_data import load_rows, validate_row, row_key
from prompts import REFORMULATE, SCHEMA

_WS = re.compile(r"\s+")

NEAR_MISS_XFORM = """\
You turn one wrong option into a NEAR-MISS for reading-comprehension training.
Input is a JSON array of rows. For EACH input row produce EXACTLY ONE row that:
- keeps "context", "question", "options" ORDER, "correct_index" and "teacher_probs" EXACTLY
  as given, EXCEPT: replace the SINGLE option occupying the index of the input row's
  `meta.near_miss_target` with a transformed version;
- the new option differs from the CORRECT option by one small edit ONLY: add or remove a
  single qualifier ("not", "only", "almost", "barely"), swap an entity/number by +-1 or a
  synonym, reverse two words, or negate one clause. It must SHARE most words with the
  correct option and remain grammatically correct and plausible;
- the new option must NOT be equivalent to the correct option and must NOT be closer than
  the correct option to the text's answer;
- "correct_index" stays unchanged; the correct option text stays UNCHANGED.
Output a JSON ARRAY following the SCHEMA. No prose, no fences, no trailing commas.
"""

FILLERS = [
    "The kettle whistled twice before anyone noticed.",
    "A notice on the wall explained the new parking rules.",
    "Grey curtains hung over the tall windows.",
    "The caretaker kept the keys in a tin by the door.",
    "Rain had been forecast for the whole week.",
    "A poster advertised the neighbourhood's annual fair.",
    "The clock on the shelf was ten minutes fast.",
    "Somewhere upstairs a door closed quietly.",
    "The delivery van took the long route around the square.",
    "Fresh bread was laid out on the counter each morning.",
    "A small dog watched from the porch.",
    "The telephone rang once and then stopped.",
    "Papers for the meeting were still being printed.",
    "The hallway light flickered for a moment.",
    "An umbrella leaned against the coat stand.",
    "The monthly newsletter arrived on Friday.",
    "Someone had left the gate open overnight.",
    "Three chairs were stacked near the window.",
    "The heating came on just after dawn.",
    "A map of the region hung behind the desk.",
]


def _norm(s: str) -> str:
    return _WS.sub(" ", s.strip()).lower()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _near_miss_ok(correct_opt: str, new_dist: str) -> bool:
    c, n = _norm(correct_opt), _norm(new_dist)
    r = _ratio(c, n)
    if r >= 0.999:
        return False
    lr = max(len(c), 1) / max(len(n), 1)
    if not 0.25 <= lr <= 4.0:
        return False
    return 0.30 <= r <= 0.95


def _prompt_user(instruction: str, rows: list[dict], model_id: str) -> str:
    return (
        instruction
        + f"\n\nInput rows ({len(rows)}):\n"
        + json.dumps(rows, ensure_ascii=False)
        + "\n\nOutput one row per input row, in the same order, as a JSON array. "
        f'Set "label_source"."model" = "{model_id}".\n'
        "Output only the JSON array now."
    )


def _transform_call(args, rows: list[dict], instruction: str, model: str, token, base, headers) -> list[dict]:
    sys = (
        "You transform existing training rows. Follow the schema exactly. Your entire "
        "reply must be a single JSON array. No prose, no fences, no trailing commas.\n\n"
        + SCHEMA
    )
    user = _prompt_user(instruction, rows, model)
    for attempt in range(args.retries):
        try:
            reply = G.chat(base, token, model, [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ], args.temperature, args.max_tokens, headers)
            out = G.extract_json_array(reply)
            return [validate_row(r, f"transform:{model}") for r in out]
        except (Exception) as e:
            if getattr(e, "code", None) in (429, 500, 502, 503, 504, 529) or isinstance(e, TimeoutError):
                time.sleep(2 ** attempt + 1)
                continue
            print(f"    [transform] call failed (attempt {attempt+1}): {e}", flush=True)
            if attempt == args.retries - 1:
                return []
    return []


def _accept_near_miss(row: dict, src: dict) -> bool:
    target = row["meta"].get("near_miss_target")
    ci = row["correct_index"]
    if not isinstance(target, int) or ci is None or target == ci:
        return False
    if len(row["options"]) != len(src["options"]):
        return False
    if _norm(row["options"][ci]) != _norm(src["options"][ci]):
        return False
    for i in range(len(row["options"])):
        if i == target:
            continue
        if _norm(row["options"][i]) != _norm(src["options"][i]):
            return False
    return _near_miss_ok(row["options"][ci], row["options"][target])


def _add_meta(src: dict, new: dict, family: str) -> dict:
    meta = dict(src.get("meta") or {})
    meta["family"] = family
    meta["source_family"] = src.get("meta", {}).get("family", "unknown")
    new = dict(new)
    new["meta"] = {**(new.get("meta") or {}), **meta}
    return new


def run_near_miss(args, rows: list, models, token, base, headers, f) -> None:
    rng = random.Random(args.seed)
    done = accepted = 0
    batch_n = max(1, args.count)
    for start in range(0, len(rows), batch_n):
        batch = list(rows[start:start + batch_n])
        for r in batch:
            n = len(r["options"])
            ci = r["correct_index"]
            targets = [i for i in range(n) if i != ci] or []
            r["meta"]["near_miss_target"] = rng.choice(targets)
        model = models[rng.randrange(len(models))]
        got = _transform_call(args, batch, NEAR_MISS_XFORM, model, token, base, headers)
        if len(got) != len(batch):
            done += len(batch)
            print(f"  [near_miss] batch {start} malformed; skipping", flush=True)
            continue
        for r, src in zip(got, batch):
            r["meta"]["near_miss_target"] = src["meta"].get("near_miss_target")
            if not _accept_near_miss(r, src):
                continue
            out = _add_meta(src, r, "near_miss")
            f.write(json.dumps(out) + "\n")
            f.flush()
            accepted += 1
        done += len(batch)
        print(f"  [near_miss] {accepted} accepted / {done} processed", flush=True)


def run_reformulate(args, rows: list, models, token, base, headers, f) -> None:
    rng = random.Random(args.seed)
    done = accepted = 0
    batch_n = max(1, args.count)
    for start in range(0, len(rows), batch_n):
        batch = list(rows[start:start + batch_n])
        model = models[rng.randrange(len(models))]
        got = _transform_call(args, batch, REFORMULATE, model, token, base, headers)
        if len(got) != len(batch):
            done += len(batch)
            print(f"  [reformulate] batch {start} malformed; skipping", flush=True)
            continue
        for r, src in zip(got, batch):
            if _norm(src["question"]) == _norm(r["question"]):
                continue
            if [_norm(o) for o in src["options"]] != [_norm(o) for o in r["options"]]:
                continue
            if src["correct_index"] != r["correct_index"]:
                continue
            out = _add_meta(src, r, "reformulate")
            f.write(json.dumps(out) + "\n")
            f.flush()
            accepted += 1
        done += len(batch)
        print(f"  [reformulate] {accepted} accepted / {done} processed", flush=True)


def _match_source(batch: list[dict], row: dict) -> dict:
    for s in batch:
        if row_key(s)[1:] == row_key(row)[1:]:
            return s
    return batch[0]


def run_noise(args, rows: list, f, rng=None) -> None:
    rng = rng or random.Random(args.seed)
    printed = 0
    for r in rows:
        n_inj = rng.choice([1, 1, 2, 2, 3])
        fillers = [f_ for f_ in rng.sample(FILLERS, 3) if not any(_norm(o) in _norm(f_) for o in r["options"])]
        use = fillers[:min(n_inj, len(fillers))]
        if not use:
            continue
        meta = dict(r.get("meta") or {})
        meta.update(family="noise", source_family=meta.get("family", "unknown"),
                    noise_injected=len(use))
        out = dict(r)
        out["context"] = (r.get("context", "") + " " + " ".join(use)).strip()
        out["meta"] = meta
        f.write(json.dumps(out) + "\n")
        f.flush()
        printed += 1
        if printed % 5000 == 0:
            print(f"  [noise] {printed}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("near_miss", "reformulate", "noise"))
    ap.add_argument("--in", dest="inp", required=True, help="source jsonl pool")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="cline", choices=sorted(G.PROVIDERS))
    ap.add_argument("--models", default=G.DEFAULT_MODEL)
    ap.add_argument("--count", type=int, default=4, help="rows per LLM completion call")
    ap.add_argument("--limit", type=int, default=0, help="max source rows to process (0=all)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--token", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    pool = load_rows(args.inp)
    if args.limit:
        rng = random.Random(args.seed)
        pool = rng.sample(pool, min(args.limit, len(pool)))
    print(f"{args.mode}: pool {len(pool)} rows")

    mode_f = "a" if args.append else "w"
    with open(args.out, mode_f, encoding="utf-8") as f:
        if args.mode == "noise":
            run_noise(args, pool, f)
            return
        token = args.token or G.load_token(args.provider)
        base = args.base or G.PROVIDERS[args.provider]["base"]
        headers = dict(G.PROVIDERS[args.provider]["headers"])
        if args.provider == "cline":
            headers = {**headers, **G.CLINE_HEADERS}
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if args.mode == "near_miss":
            run_near_miss(args, pool, models, token, base, headers, f)
        elif args.mode == "reformulate":
            run_reformulate(args, pool, models, token, base, headers, f)
    print(f"{args.mode} done -> {args.out}")


if __name__ == "__main__":
    main()