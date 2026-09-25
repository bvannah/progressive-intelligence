from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import socket
import time
import urllib.error
import urllib.request

from generate import chat, load_token, extract_json_array, PROVIDERS, CLINE_HEADERS, DEFAULT_MODEL
from prompts import SCHEMA
from merge_data import validate_row

DEFAULT_CORPUS = "/mnt/9a99846e-8002-475e-9eba-b992cd54e718/kb_full/corpus.jsonl"
_OUT_DEFAULT = pathlib.Path(
    "/mnt/1BBFC6E12FC77EDC/data/progressive_intelligence_data/wiki_fact_llm.jsonl")

WIKI_EXTRACT = """\
You extract multiple-choice quiz questions from a Wikipedia passage.
For EACH question produce ONE JSON row where:
- "context" is the passage text (verbatim),
- "question" asks about a fact STRICTLY ENTAILED by the passage,
- "options" are 2-8 short, plausible answer strings (vary the count across the batch,
  keep at least 4),
- "correct_index" is the index of the one entailed answer,
- "teacher_probs" places ~0.95 probability on the correct option and the rest thinly
  on the others,
- meta.support_span is the VERBATIM contiguous quote (1-3 sentences) from the passage
  that establishes the answer,
- meta.family is "wiki_fact".
DO NOT invent facts absent from the passage. DO NOT use outside knowledge. Output a
JSON ARRAY following the schema. No prose, no fences, no trailing commas.
"""

_SENT = re.compile(r"(?<=[.!?])\s+")
_BAD = re.compile(r"\{\{|\}\}|\[\[|\]\]|==|href=|http")


def _norm(s):
    return " ".join(s.replace("\n", " ").split()).strip()


def _chunks(text, min_len=180, max_len=700, max_sent=4):
    sents = [_norm(s) for s in _SENT.split(text) if len(_norm(s)) >= 40]
    cur, cur_len = [], 0
    for s in sents:
        sl = len(s)
        if cur and (cur_len + sl > max_len or len(cur) >= max_sent):
            if min_len <= cur_len <= max_len and not _BAD.search(" ".join(cur)):
                yield " ".join(cur)
            cur, cur_len = [s], sl
        else:
            cur.append(s)
            cur_len += sl
    if cur and min_len <= cur_len <= max_len and not _BAD.search(" ".join(cur)):
        yield " ".join(cur)


def yield_articles(corpus_path, max_scan=-1):
    seen = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if 0 < max_scan <= seen:
                return
            seen += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("redirect_target"):
                continue
            title = (r.get("title") or "").strip()
            text = r.get("text") or ""
            if not (3 <= len(title) <= 80) or not (80 <= len(text) <= 40000):
                continue
            yield title, text


def main():
    ap = argparse.ArgumentParser(description="LLM content-grounded fact extraction from the local wiki corpus (B8).")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=str(_OUT_DEFAULT))
    ap.add_argument("--limit", type=int, default=100, help="chunks to process")
    ap.add_argument("--max-scan", type=int, default=-1)
    ap.add_argument("--chunks-per-article", type=int, default=1)
    ap.add_argument("--provider", default="cline", choices=sorted(PROVIDERS))
    ap.add_argument("--models", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--base", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    token = args.token or load_token(args.provider)
    base = args.base or PROVIDERS[args.provider]["base"]
    headers = dict(PROVIDERS[args.provider]["headers"])
    if args.provider == "cline":
        headers = {**headers, **CLINE_HEADERS}
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rng = random.Random(args.seed)
    system = SCHEMA + "\n\n" + WIKI_EXTRACT
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    produced = rejected = calls = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for title, text in yield_articles(args.corpus, args.max_scan):
            chunks = list(_chunks(text))
            if not chunks:
                continue
            rng.shuffle(chunks)
            for chunk in chunks[: args.chunks_per_article]:
                if produced >= args.limit:
                    print(f"reached --limit {args.limit}")
                    print(f"wrote {produced} rows, rejected {rejected}, calls {calls} -> {args.out}")
                    return
                model = models[rng.randrange(len(models))]
                user = f"Here is the passage:\n\n<<<{chunk}>>>\n\n\nNow emit 3-5 JSON rows per the instructions."
                got = []
                for attempt in range(args.retries):
                    try:
                        reply = chat(base, token, model, [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ], args.temperature, args.max_tokens, headers)
                        got = extract_json_array(reply)
                        calls += 1
                        break
                    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                            socket.timeout, OSError) as e:
                        status = getattr(e, "code", "conn")
                        if status in (429, 500, 502, 503, 504, 529) or status == "conn":
                            time.sleep(2 ** attempt + 1)
                            continue
                        break
                    except (ValueError, json.JSONDecodeError):
                        continue
                if not got:
                    rejected += 1
                    continue
                for r0 in got:
                    meta = dict(r0.get("meta") or {})
                    meta.update({"family": "wiki_fact", "source_page": title,
                                 "generated_by": f"llm:{model}", "tier": "fact"})
                    r0["meta"] = meta
                    try:
                        row = validate_row(r0, f"wiki-llm:{title[:40]}")
                    except ValueError:
                        rejected += 1
                        continue
                    if not row["meta"].get("support_span"):
                        rejected += 1
                        continue
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    produced += 1
                print(f"  {produced}/{args.limit} rows (rejected {rejected})", flush=True)
    print(f"wrote {produced} rows, rejected {rejected}, calls {calls} -> {args.out}")


if __name__ == "__main__":
    main()