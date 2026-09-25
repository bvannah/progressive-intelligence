from __future__ import annotations

import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

# Deterministic 50% train / 50% eval-half split for EVERY public eval set.
# A row is tagged eval-half iff it is allowed to be scored; train-half rows may
# enter the training corpus but are then EXCLUDED from public-eval scoring.
# Same content -> same tag, independent of fetch order or tooling.
SALT = "progressive-intelligence-public-eval-v1"

DS_BASE = "https://datasets-server.huggingface.co"
DEFAULT_MAX_ROWS = 20_000

SOURCES = {
    "arc-challenge": {
        "dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test",
        "remap": "arc", "family": "arc-challenge",
    },
    "mmlu-astronomy": {
        "dataset": "cais/mmlu", "config": "astronomy", "split": "test",
        "remap": "mmlu", "family": "mmlu-astronomy",
    },
    "race-middle": {
        "dataset": "ehovy/race", "config": "middle", "split": "test",
        "remap": "race", "family": "race-middle",
    },
}


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def fetch_rows(src: dict, max_rows: int = DEFAULT_MAX_ROWS) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < max_rows:
        length = min(100, max_rows - len(rows))
        q = urllib.parse.urlencode({
            "dataset": src["dataset"], "config": src["config"], "split": src["split"],
            "offset": offset, "length": length,
        })
        with urllib.request.urlopen(f"{DS_BASE}/rows?{q}", timeout=60) as r:
            data = json.load(r)
        chunk = data.get("rows", [])
        if not chunk:
            break
        rows.extend(chunk)
        offset += len(chunk)
    return rows[:max_rows]


def remap_arc(rec: dict, idx: int) -> dict:
    row = rec["row"]
    table = row.get("choices") or {}
    if isinstance(table, dict):
        text = [str(t).strip() for t in (table.get("text") or [])]
        labels = [str(l).strip().upper() for l in (table.get("label") or [])]
    else:
        text = [str(c.get("text", "")).strip() for c in table]
        labels = [str(c.get("label", "")).strip().upper() for c in table]
    ci = None
    ans = str(row.get("answerKey", "")).strip().upper()
    for i, lab in enumerate(labels):
        if lab == ans:
            ci = i
            break
    q = str(row.get("question", "")).strip()
    return {"context": q, "question": q, "options": text,
            "correct_index": ci, "meta": {"family": "arc-challenge"}}


def remap_mmlu(rec: dict, idx: int) -> dict:
    row = rec["row"]
    options = [str(c).strip() for c in (row.get("choices") or [])]
    ans = row.get("answer")
    q = str(row.get("question", "")).strip()
    ci = int(ans) if isinstance(ans, (int, float)) and 0 <= int(ans) < len(options) else None
    return {"context": q, "question": q, "options": options,
            "correct_index": ci, "meta": {"family": "mmlu-astronomy"}}


def remap_race(rec: dict, idx: int) -> dict:
    row = rec["row"]
    options = [str(o).strip() for o in (row.get("options") or [])] or \
              [str(o).strip() for o in (row.get("choices") or [])]
    ans = str(row.get("answer", "")).strip().upper()
    ci = None
    if ans:
        for i, o in enumerate(options):
            if str(o).strip().upper() == ans:
                ci = i
                break
        if ci is None and ans[0] in "ABCD":
            ci = "ABCD".find(ans[0])
    article = str(row.get("article", "")).strip()
    return {"context": article,
            "question": str(row.get("question", "")).strip(),
            "options": options, "correct_index": ci,
            "meta": {"family": "race-middle", "article": article}}


REMAPPERS = {"arc": remap_arc, "mmlu": remap_mmlu, "race": remap_race}


def canonical_key(ex: dict) -> bytes:
    return (
        f"{ex['meta'].get('family')}::{_norm(ex.get('context', ''))}::"
        f"{_norm(ex.get('question', ''))}::"
        f"{chr(30).join(_norm(o) for o in ex.get('options', []))}"
    ).encode()


def split_tag(ex: dict, frac_train: float = 0.5) -> str:
    """Deterministic hash of row content -> 'train-half' | 'eval-half'."""
    h = int(hashlib.sha256(SALT.encode() + canonical_key(ex)).hexdigest(), 16)
    return "train-half" if h % 1000 < int(round(frac_train * 1000)) else "eval-half"


def load_or_fetch(name: str, cache_dir: str | pathlib.Path, max_rows: int = DEFAULT_MAX_ROWS,
                  force_refresh: bool = False) -> list[dict]:
    """Return [{"example": ..., "split": "train-half"|"eval-half", "raw": rec}].
    `rec` is the datasets-server element {"row": <hf row>}; cache lines are
    {"row": <server element>, "split": tag}. Tagless caches are refetched."""
    src = SOURCES[name]
    cache_f = pathlib.Path(cache_dir) / f"{name}.jsonl"
    recs: list[dict] = []
    if not force_refresh and cache_f.exists():
        recs = [json.loads(l) for l in cache_f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if recs and "split" not in recs[0]:
            recs = []
    fresh = not recs
    if fresh:
        recs = list(fetch_rows(src, max_rows))
    out = []
    for i, rec in enumerate(recs):
        ex = REMAPPERS[src["remap"]](rec, i)
        tag = rec.get("split") or split_tag(ex)
        out.append({"example": ex, "split": tag, "raw": rec})
    if fresh:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        with cache_f.open("w", encoding="utf-8") as f:
            for e in out:
                f.write(json.dumps({"row": e["raw"], "split": e["split"]}) + "\n")
    return out


def export_corpus_row(name: str, ex: dict, idx: int) -> dict | None:
    """train-half public-eval example -> tier=fact corpus row (gold probs + provenance)."""
    ci = ex.get("correct_index")
    options = ex.get("options") or []
    n = len(options)
    if ci is None or not (0 <= ci < n) or n < 2:
        return None
    probs = [0.05 / (n - 1)] * n
    probs[ci] = 0.95
    src = SOURCES[name]
    family = ex["meta"].get("family", name)
    return {
        "id": f"pub:{name}:{idx}",
        "task": family,
        "difficulty": "medium",
        "context": ex.get("context", ""),
        "question": ex.get("question", ""),
        "options": options,
        "correct_index": ci,
        "teacher_probs": probs,
        "label_source": {"model": "public-dataset", "temperature": 0.0,
                         "n_samples": 1, "vote": "gold"},
        "meta": {
            "family": family,
            "pub_set": name,
            "pub_split": "train-half",
            "dataset": f"{src['dataset']}:{src['config']}:{src['split']}",
            "tier": "fact",
            "generated_by": "public-dataset",
        },
    }