from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import uuid
from collections import Counter

DATA = "/mnt/1BBFC6E12FC77EDC/data/progressive_intelligence_data/public_new"
DEFAULT_OUT = pathlib.Path(DATA)


def _row(fam, context, question, options, correct, dataset, source_page, topic=None, tag=None):
    n = len(options)
    probs = [0.05 / (n - 1)] * n
    probs[correct] = 0.95
    meta = {"family": fam, "generated_by": tag or f"external:{fam}", "dataset": dataset,
            "source_page": source_page, "tier": "fact"}
    if topic:
        meta["topic"] = topic
    return {"id": str(uuid.uuid4()), "task": fam, "difficulty": "medium",
            "context": context, "question": question, "options": options,
            "correct_index": correct, "teacher_probs": probs,
            "label_source": {"model": f"public-dataset:{dataset}", "temperature": 0.0,
                             "n_samples": 1, "vote": "gold"},
            "meta": meta}


def _texts(choices):
    if isinstance(choices, dict):
        labels = list(choices.get("label", []))
        texts = list(choices.get("text", []))
    else:
        items = list(choices)
        labels = [c.get("label") for c in items]
        texts = [c.get("text") for c in items]
    return texts, labels


def convert_quality(rng, limit=-1, split="train"):
    from datasets import load_dataset
    ds = load_dataset("emozilla/quality", "default", split=split)
    rows, taken = [], 0
    for i, ex in enumerate(ds):
        if 0 < limit <= taken:
            break
        article = ex.get("article") or ""
        question = ex.get("question") or ""
        opts = ex.get("options") or []
        ans = ex.get("answer")
        if not article or len(opts) < 2 or not isinstance(ans, int) or not (0 <= ans < len(opts)):
            continue
        rows.append(_row("quality_qa", article, question, opts, ans,
                         "quality", f"quality:{i}", topic=f"quality:{i // 20}"))
        taken += 1
    return rows


def convert_qasc(rng, limit=-1, split="train"):
    from datasets import load_dataset
    ds = load_dataset("allenai/qasc", split=split)
    rows, taken = [], 0
    for i, ex in enumerate(ds):
        if 0 < limit <= taken:
            break
        texts, labels = _texts(ex.get("choices"))
        if len(texts) < 2:
            continue
        ans = ex.get("answerKey")
        correct = labels.index(ans) if ans in labels else -1
        if correct < 0 or correct >= len(texts):
            continue
        facts = [f.get("text") for f in ex.get("supporting_facts") or []]
        ctx = " ".join(x for x in facts if x) or ex.get("combined_fact") or ""
        rows.append(_row("qasc", ctx, ex.get("question", ""), texts, correct,
                         "qasc", f"qasc:{i}", topic=f"qasc:{i}"))
        taken += 1
    return rows


def convert_medmcqa(rng, limit=-1, split="train"):
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    rows, taken = [], 0
    for ex in ds:
        if 0 < limit <= taken:
            break
        opts = [ex.get(k) or "" for k in ("opa", "opb", "opc", "opd")]
        cop = ex.get("cop")
        if len(opts) < 2 or not cop or not (1 <= cop <= 4):
            continue
        correct = cop - 1
        topic = ex.get("subject_name") or ex.get("topic_name") or "medmcqa"
        rows.append(_row("medmcqa", ex.get("exp") or "", ex.get("question", ""), opts, correct,
                         "medmcqa", f"medmcqa:{ex.get('id', taken)}", topic=topic))
        taken += 1
    return rows


MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"]


def convert_mmlu(rng, limit=-1):
    from datasets import load_dataset
    rows, taken = [], 0
    total_cap = limit if limit > 0 else 40000
    subjects = [s for s in MMLU_SUBJECTS if s != "astronomy"]
    per_subj = max(1, total_cap // len(subjects))
    for subj in subjects:
        if 0 < limit <= taken:
            break
        try:
            ds = load_dataset("cais/mmlu", subj, split="train")
        except Exception as e:
            print(f"  [mmlu:{subj}] skip ({e})", flush=True)
            continue
        sub_taken = 0
        for ex in ds:
            if taken >= total_cap or sub_taken >= per_subj:
                break
            texts = ex.get("choices") or []
            ans = ex.get("answer")
            if len(texts) < 2 or not isinstance(ans, int) or not (0 <= ans < len(texts)):
                continue
            rows.append(_row("mmlu", ex.get("question", ""), ex.get("question", ""), texts,
                             ans, "mmlu", f"mmlu:{subj}", topic=subj))
            taken += 1
            sub_taken += 1
    return rows


def convert_mc_eval(rng, name, limit=-1, split="train"):
    fam = {"gsm8k-mc": "gsm_mc", "math-mc": "math_mc"}[name]
    path = pathlib.Path("/tmp/opencode/dev/data") / name / f"{split}.jsonl"
    rows, taken = [], 0
    for line in open(path, encoding="utf-8"):
        if 0 < limit <= taken:
            break
        ex = json.loads(line)
        labels = ["A", "B", "C", "D"]
        texts = [ex.get(k) or "" for k in labels]
        ans = ex.get("Answer") or ex.get("answer")
        if len([t for t in texts if t]) < 2:
            continue
        correct = labels.index(ans) if ans in labels else -1
        if correct < 0 or correct >= len(texts):
            continue
        rows.append(_row(fam, ex.get("Question", ""), ex.get("Question", ""), texts, correct,
                         name, f"{name}:{taken}", topic=f"{name}:{taken}"))
        taken += 1
    return rows


CONVERTERS = {
    "quality_qa": lambda rng, lim: convert_quality(rng, lim),
    "qasc": lambda rng, lim: convert_qasc(rng, lim),
    "medmcqa": lambda rng, lim: convert_medmcqa(rng, lim),
    "mmlu": lambda rng, lim: convert_mmlu(rng, lim),
    "gsm_mc": lambda rng, lim: convert_mc_eval(rng, "gsm8k-mc", lim),
    "math_mc": lambda rng, lim: convert_mc_eval(rng, "math-mc", lim),
}

CAPS = {"quality_qa": 6700, "qasc": 8134, "medmcqa": 30000, "mmlu": 40000,
        "gsm_mc": 7000, "math_mc": 7000}


def main():
    ap = argparse.ArgumentParser(description="Fetch additional public MC training datasets (tier=fact).")
    ap.add_argument("--only", default=",".join(CAPS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    which = [f.strip() for f in args.only.split(",") if f.strip()]
    for fam in which:
        fn = CONVERTERS[fam]
        cap = CAPS[fam]
        print(f"== {fam} (cap {cap}) ==", flush=True)
        try:
            rows = fn(rng, cap)
        except Exception as e:
            print(f"  [{fam}] FAILED: {e}", flush=True)
            continue
        rng.shuffle(rows)
        with open(out_dir / f"{fam}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows)} -> {out_dir / (fam + '.jsonl')}", flush=True)


if __name__ == "__main__":
    main()