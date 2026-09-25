from __future__ import annotations

import json
from collections import Counter, defaultdict

DATA = "/mnt/1BBFC6E12FC77EDC/data/progressive_intelligence_data"
OUT = "/home/brian/code/progressive-intelligence/DATASET_BREAKDOWN_M4.md"

TOPIC_KEYS = ("topic", "source_topic", "dataset")


def topic_of(r: dict):
    m = r.get("meta") or {}
    for k in TOPIC_KEYS:
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())
    return "(unset)"


def _norm(s):
    return " ".join((s or "").split())


def short(s, n):
    s = _norm(str(s))
    return s if len(s) <= n else s[:n] + "…"


def fmt_row(r):
    opts = r.get("options", [])
    ci = r.get("correct_index")
    gold = r.get("teacher_probs", [None] * len(opts))[ci] if ci is not None else None
    gold_s = f"{gold:.2f}" if gold is not None else "?"
    m = r.get("meta") or {}
    lines = [
        f"  - **question:** {short(r.get('question'), 140)}",
        f"    options: {short(' | '.join(f'{i}. {o}' for i, o in enumerate(opts)), 170)}",
        f"    correct: idx={ci} → {short(opts[ci], 70) if ci is not None else '?'} (gold={gold_s})",
        f"    context: {short(r.get('context') or '', 150)}",
        f"    tier={m.get('tier')} source={m.get('dataset') or m.get('generated_by')}",
    ]
    return "\n".join(lines)


def one_row_compact(r):
    return None


def scan(split):
    path = f"{DATA}/{split}.jsonl"
    counts = Counter()
    topics = defaultdict(Counter)
    examples_fam = {}
    examples_top = {}
    total = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        fam = r["meta"]["family"]
        counts[fam] += 1
        total += 1
        topics[fam][topic_of(r)] += 1
        if fam not in examples_fam:
            examples_fam[fam] = r
        t = topic_of(r)
        if (fam, t) not in examples_top:
            examples_top[(fam, t)] = r
    return total, counts, topics, examples_fam, examples_top


def main():
    tr = scan("train")
    ev = scan("eval")
    _, train_counts, train_top, train_ef, train_et = tr
    _, eval_counts, eval_top, eval_ef, eval_et = ev

    EXTERNAL = {"hellaswag", "commonsense", "sciq", "openbookqa", "arc-challenge",
                "race-middle", "mmlu-astronomy", "medqa", "qasc", "medmcqa",
                "gsm_mc", "math_mc", "quality_qa"}
    WIKI = {"wiki_fact", "wikibox", "wikicat", "wikiret"}
    LLM_GEN = {"chain_arithmetic", "chain_logic", "chain_retrieval", "chain_sequence",
               "near_miss", "reformulate", "noise"}

    def bucket(f):
        if f in EXTERNAL:
            return "external public MC"
        if f in WIKI:
            return "wiki-derived"
        if f in LLM_GEN:
            return "llm-generated (round-3/noise)"
        return "synthetic templated"

    all_fams = sorted({*train_counts, *eval_counts}, key=lambda f: (-train_counts[f], f))

    train_total = tr[0]
    eval_total = ev[0]

    out = []
    out.append("# Progressive-Intelligence — Dataset Breakdown")
    out.append("")
    out.append(f"_Generated {__import__('datetime').date.today()} — corpus after round-4 merge "
               f"(public datasets added). Gate: OVERALL PASS._")
    out.append("")
    out.append(f"- **train.jsonl:** {train_total:,} rows")
    out.append(f"- **eval.jsonl (own eval):** {eval_total:,} rows")
    out.append("")
    out.append("## 1. Source (family) proportions")
    out.append("")
    out.append("| family | bucket | train | % of train | eval | % of eval |")
    out.append("|---|---|---:|---:|---:|---:|")
    for f in all_fams:
        tc = train_counts[f]
        ec = eval_counts[f]
        out.append(f"| {f} | {bucket(f)} | {tc:,} | {100 * tc / train_total:.2f}% | "
                   f"{ec:,} | {100 * ec / eval_total:.2f}% |")
    out.append("")
    out.append("### Totals by bucket")
    out.append("")
    out.append("| bucket | train | % of train | eval | % of eval |")
    out.append("|---|---:|---:|---:|---:|")
    for b in ["synthetic templated", "wiki-derived", "external public MC", "llm-generated (round-3/noise)"]:
        tc = sum(train_counts[f] for f in all_fams if bucket(f) == b)
        ec = sum(eval_counts[f] for f in all_fams if bucket(f) == b)
        out.append(f"| {b} | {tc:,} | {100 * tc / train_total:.2f}% | {ec:,} | {100 * ec / eval_total:.2f}% |")
    out.append("")

    out.append("## 2. Per-source detail (topics + examples)")
    out.append("")
    for f in all_fams:
        tc = train_counts[f]
        ec = eval_counts[f]
        out.append(f"### {f}")
        out.append("")
        out.append(f"train {tc:,} · eval {ec:,} · bucket: {bucket(f)}")
        out.append("")
        fam_eg = train_ef.get(f) or eval_ef.get(f)
        if fam_eg:
            out.append("**Example row (source-level):**")
            out.append("")
            out.append(fmt_row(fam_eg))
            out.append("")
        merged = Counter(train_top.get(f, {})) + Counter(eval_top.get(f, {}))
        out.append(f"**Topics ({len(merged)} distinct):**")
        out.append("")
        out.append("| topic | train | eval |")
        out.append("|---|---:|---:|")
        for t, cnt in merged.most_common():
            out.append(f"| `{t}` | {train_top.get(f, {}).get(t, 0):,} | "
                       f"{eval_top.get(f, {}).get(t, 0):,} |")
        out.append("")

    out.append("## 3. Public eval sets (held out, never trained on eval-half)")
    out.append("")
    out.append("These drive `public_eval.py` scoring; only their **train-half** rows may be in the "
               "corpus, eval-half rows are excluded by the no-overlap guard.")
    out.append("")
    out.append("| eval set | source dataset | config | split | total cached | train-half | eval-half |")
    out.append("|---|---|---|---|---:|---:|---:|")
    for name, ds, cfg, sp in [("arc-challenge", "allenai/ai2_arc", "ARC-Challenge", "test"),
                              ("mmlu-astronomy", "cais/mmlu", "astronomy", "test"),
                              ("race-middle", "ehovy/race", "middle", "test")]:
        try:
            rows = [json.loads(l) for l in open(f"{DATA}/.public-cache/{name}.jsonl", encoding="utf-8")]
        except OSError:
            continue
        half = Counter(r.get("split") for r in rows)
        out.append(f"| {name} | {ds} | {cfg} | {sp} | {len(rows):,} | "
                   f"{half.get('train-half', 0):,} | {half.get('eval-half', 0):,} |")
    out.append("")
    for name in ["arc-challenge", "mmlu-astronomy", "race-middle"]:
        rows = [json.loads(l) for l in open(f"{DATA}/.public-cache/{name}.jsonl", encoding="utf-8")]
        evh = next((r for r in rows if r.get("split") == "eval-half"), rows[0])
        raw = evh["row"]["row"]
        ch = raw.get("choices")
        if isinstance(ch, dict):
            opts = list(ch.get("text", []))
            labels = list(ch.get("label", []))
        elif isinstance(ch, list):
            opts = list(ch)
            labels = list(range(len(ch)))
        else:
            opts = list(raw.get("options", []))
            labels = list(range(len(opts)))
        ans = raw.get("answerKey") or raw.get("answer")
        if isinstance(ans, str) and len(ans) == 1 and ans.upper() in "ABCDEFG":
            ci = ord(ans.upper()) - ord("A")
        elif isinstance(ans, str):
            ci = labels.index(ans) if ans in labels else (int(ans) if ans.strip().isdigit() else -1)
        elif isinstance(ans, int):
            ci = ans
        else:
            ci = -1
        if not (0 <= ci < len(opts)):
            ci = None
        eg = {"question": raw.get("question", ""), "options": opts, "correct_index": ci,
              "context": raw.get("context", ""), "meta": {"tier": "fact", "dataset": name}}
        out.append(f"### {name} — example eval-half row")
        out.append("")
        out.append(fmt_row(eg))
        out.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()