import argparse
import json
import random
import uuid
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("/mnt/1BBFC6E12FC77EDC/data/progressive_intelligence_data")
OUT_DEFAULT = DATA_ROOT / "external.jsonl"


def _norm(s):
    return " ".join((s or "").replace("\n", " ").split())


def _row(fam, context, question, options, correct, dataset, tag):
    n = len(options)
    probs = [0.05 / (n - 1)] * n
    probs[correct] = 0.95
    return {
        "id": str(uuid.uuid4()),
        "task": fam,
        "difficulty": "medium",
        "context": context,
        "question": question,
        "options": options,
        "correct_index": correct,
        "teacher_probs": probs,
        "label_source": {"model": f"public-dataset:{dataset}", "temperature": 0.0,
                         "n_samples": 1, "vote": "gold"},
        "meta": {"family": fam, "generated_by": tag, "dataset": dataset,
                 "source_page": dataset, "tier": "fact"},
    }


def split_choices(choices):
    if isinstance(choices, dict):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
    else:
        items = list(choices)
        labels = [c.get("label") for c in items]
        texts = [c.get("text") for c in items]
    return texts, labels


def convert(rng, fam, split="train", limit=-1):
    print(f"== {fam} ({split}) ==", flush=True)
    from datasets import load_dataset

    ds_id, config = {
        "openbookqa": ("allenai/openbookqa", "main"),
        "sciq": ("allenai/sciq", None),
        "commonsense": ("tau/commonsense_qa", None),
        "hellaswag": ("Rowan/hellaswag", None),
        "medqa": ("GBaker/MedQA-USMLE-4-options-hf", None),
    }[fam]
    ds = (load_dataset(ds_id, config, split=split) if config
          else load_dataset(ds_id, split=split))
    tag = f"external:{fam}"
    questions = []
    for ex in ds:
        if 0 < limit <= len(questions):
            break
        if fam == "openbookqa":
            q = _norm(ex["question_stem"])
            texts, labels = split_choices(ex["choices"])
            key = ex.get("answerKey", "")
            if key not in labels:
                continue
            ci, ctx, qtext = labels.index(key), "", q
        elif fam == "sciq":
            q = _norm(ex["question"])
            texts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
            ci, ctx, qtext = 0, _norm(ex.get("support", "")), q
        elif fam == "commonsense":
            q = _norm(ex["question"])
            texts, labels = split_choices(ex["choices"])
            key = ex.get("answerKey", "")
            if key not in labels:
                continue
            ci, ctx, qtext = labels.index(key), "", q
        elif fam == "hellaswag":
            ctx = _norm(ex.get("ctx", "")) or _norm(ex.get("context", ""))
            if not ctx:
                continue
            texts = list(ex["endings"])
            key = ex.get("label")
            ci = None
            if key is not None:
                try:
                    ci = int(key)
                except (TypeError, ValueError):
                    ci = None
            if ci is None or not (0 <= ci < len(texts)):
                gold = ex.get("gold_label")
                if isinstance(gold, int):
                    ci = gold
                else:
                    continue
            qtext = "Which ending is the most sensible continuation of the passage?"
        elif fam == "medqa":
            q = _norm((ex.get("sent1") or "") + " " + (ex.get("sent2") or ""))
            texts = [str(ex.get(f"ending{i}") or "") for i in range(4)]
            ans = ex.get("answer", ex.get("label"))
            ci = None
            if isinstance(ans, int):
                ci = ans
            elif isinstance(ans, str):
                a = ans.strip().upper()
                if a and a[0] in "ABCD":
                    ci = "ABCD".index(a[0])
            if ci is None or not (0 <= ci < len(texts)):
                continue
            ctx, qtext = "", q
        else:
            continue

        options = []
        seen = set()
        for t in texts:
            nrm = _norm(t)
            k = nrm.lower()
            if not k or k in seen:
                continue
            seen.add(k)
            options.append(nrm)
        if len(options) < 2 or not (0 <= ci < len(options)):
            continue
        questions.append(_row(fam, ctx, qtext, options, ci, ds_id, tag))
    return questions


def main():
    ap = argparse.ArgumentParser(description="Fetch gold-label real MC datasets (train splits) as tier=fact rows.")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--families", default="openbookqa,sciq,commonsense,hellaswag,medqa")
    ap.add_argument("--limit", type=int, default=-1, help="max rows per family (default: all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows = []
    for fam in args.families.split(","):
        if fam.strip():
            all_rows.extend(convert(rng, fam, "train", args.limit))
    seen = set()
    kept = []
    for r in all_rows:
        k = (_norm(r["context"]), _norm(r["question"]), tuple(_norm(o) for o in r["options"]))
        if k not in seen:
            seen.add(k)
            kept.append(r)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    for fam, cnt in Counter(r["meta"]["family"] for r in kept).most_common():
        print(f"{fam}: {cnt}")
    print(f"wrote {len(kept)} rows -> {args.out}")


if __name__ == "__main__":
    main()