import argparse
import json
import random
import re
import sys
import uuid
from pathlib import Path

DEFAULT_CORPUS = "/mnt/9a99846e-8002-475e-9eba-b992cd54e718/kb_full/corpus.jsonl"
DATA_ROOT = Path("/mnt/1BBFC6E12FC77EDC/data/progressive_intelligence_data")

_SKIP_NS = {"file", "image", "category", "wikipedia", "template", "help", "portal", "module", "media",
            "special", "user", "draft", "talk", "wp", "wiktionary", "wikisource"}

_CAT_SKIP_PREFIX = (
    "articles", "cs1", "all ", "pages ", "wikipedia ", "category articles", "use dmy", "use british",
    "good articles", "featured ", "redirects ", "stub ", "orphaned ", "cleanup ", "unreferenced",
    "b-class", "start-class", "stub-class", "c-class", "ga-class", "fa-class", "living people",
    "short description", "coordinates", "hidden categories", "commons category", "disambiguation",
    "redirect-class", "non-talk", "tracked", "ambiguous", "kml", "articles needing",
)

FIELD_Q = {
    "capital": "What is the capital of {title}?",
    "official_language": "What is the official language of {title}?",
    "languages": "What is the chief language of {title}?",
    "currency": "What is the currency of {title}?",
    "established": "When was {title} established?",
    "founded": "When was {title} founded?",
    "area_km2": "What is the total area of {title}?",
    "population": "What is the population of {title}?",
    "elevation_m": "What is the elevation of {title}?",
}
FIELD_ALIAS = {"area_total_km2": "area_km2", "area_total": "area_km2", "established_date": "established",
               "founded_date": "founded", "native_name_lang": None, "founded_year": "founded"}

CAT_Q = "Which category is this article classified under?"
RET_Q = "Which of these topics is mentioned in the passage?"
KB_Q = "Which article is this passage from?"


def norm(s):
    return " ".join((s or "").replace("\n", " ").split())


def balanced_spans(text):
    spans, depth, start, n = [], 0, None, len(text)
    i = 0
    while i < n - 1:
        if text[i] == "{" and text[i + 1] == "{":
            if depth == 0:
                start = i
            depth += 1
            i += 2
            continue
        if text[i] == "}" and text[i + 1] == "}":
            if depth == 1:
                spans.append((start, i + 2))
            depth = max(0, depth - 1)
            i += 2
            continue
        i += 1
    return spans


def strip_spans(text, spans):
    if not spans:
        return text
    parts, last = [], 0
    for a, b in spans:
        parts.append(text[last:a])
        parts.append(" ")
        last = b
    parts.append(text[last:])
    return "".join(parts)


def clean_wiki(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"\[\d+(?:[–-]\d+)*\]", " ", text)
    for _ in range(8):
        spans = balanced_spans(text)
        if not spans:
            break
        text = strip_spans(text, spans)
    text = text.replace("{{", " ").replace("}}", " ")
    text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", lambda m: m.group(2), text)
    text = re.sub(r"\[\[[^\[\]]*\]\]", " ", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"[ \t\r]+", " ", text)
    text = " ".join(text.splitlines())
    return text.strip()


def link_labels(text):
    out = []
    for m in re.finditer(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]", text):
        target, label = m.group(1) or "", m.group(2) or m.group(1) or ""
        if not target or not label:
            continue
        ns = target.strip().split(":", 1)[0].lower()
        if ns in _SKIP_NS and ":" in target.strip():
            continue
        lbl = label.replace("_", " ").strip()
        if 2 <= len(lbl) <= 48 and "#" not in lbl and lbl != target.strip():
            out.append(lbl)
    return out


def split_top(body, sep="|"):
    chunks, cur, depth, n = [], [], 0, len(body)
    i = 0
    while i < n:
        if body[i] == "{" and i + 1 < n and body[i + 1] == "{":
            depth += 1
            cur.append("{{")
            i += 2
            continue
        if body[i] == "}" and i + 1 < n and body[i + 1] == "}":
            depth -= 1
            cur.append("}}")
            i += 2
            continue
        if body[i] == "[" and i + 1 < n and body[i + 1] == "[":
            j = body.find("]]", i)
            if j == -1:
                cur.append(body[i])
                i += 1
                continue
            cur.append("LINK")
            i = j + 2
            continue
        if body[i] == sep and depth <= 0:
            chunks.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(body[i])
        i += 1
    if cur:
        chunks.append("".join(cur))
    return chunks


def clean_value(v):
    v = re.sub(r"<ref[^>]*>.*?</ref>", " ", v, flags=re.S)
    v = re.sub(r"<!--.*?-->", " ", v, flags=re.S)
    v = re.sub(r"\{\{[a-zA-Z_ ]*\|([^|}]+)", r"\1", v)
    for _ in range(4):
        spans = balanced_spans(v)
        if not spans:
            break
        v = strip_spans(v, spans)
    v = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", lambda m: m.group(2), v)
    v = re.sub(r"\[\[[^\[\]]*\]\]", " ", v)
    v = v.replace("'''", "").replace("''", "").replace("{{", " ").replace("}}", " ")
    v = re.sub(r"[ \t\r]+", " ", v)
    return " ".join(v.splitlines()).strip()


def infobox_pairs(text):
    pairs = {}
    for a, b in balanced_spans(text):
        inner = text[a + 2:b - 2]
        if not inner.lstrip().startswith("Infobox"):
            continue
        chunks = split_top(inner[len("Infobox"):], "|")
        for ch in chunks[1:]:
            k, _, v = ch.partition("=")
            k = k.strip()
            v = clean_value(v)
            if not k or not v or "=" in v or len(v) > 48:
                continue
            if re.search(r"[{}\[\]<>&\|]", v):
                continue
            k = FIELD_ALIAS.get(k, k)
            if k is None or k not in FIELD_Q:
                continue
            pairs.setdefault(k, []).append(v)
    out = {}
    for k, vs in pairs.items():
        seen = set()
        for v in vs:
            if v not in seen:
                seen.add(v)
                out.setdefault(k, []).append(v)
    return out


def split_sentences(text):
    return [s.replace("\n", " ").strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_seed(r):
    title = r.get("title", "")
    text = r.get("text", "")
    if not title or not text:
        return None
    if "{{disambiguation" in text or "{{dab" in text:
        return None
    idx = text.find("\n==")
    lead = text[:idx] if idx >= 0 else text
    lead = "\n".join(l for l in lead.splitlines() if not l.strip().startswith("#"))

    raw_sents = split_sentences(lead)
    raw_3 = " ".join(raw_sents[:3])

    cleaned = clean_wiki(raw_3)
    sents = split_sentences(cleaned)
    lead1 = sents[0][:260] if sents else ""
    lead3 = cleaned[:520]

    m = re.search(r"\{\{Short description\|([^}]+)\}\}", text)
    short = clean_wiki(m.group(1)) if m and m.group(1).strip() else ""
    defn = ""
    if sents:
        if 40 <= len(sents[0]) <= 260:
            defn = sents[0]
        elif len(" ".join(sents[:2])) <= 280:
            defn = " ".join(sents[:2])
    if len(defn) < 40 and short and len(short) >= 15:
        defn = short

    links = [l for l in link_labels(raw_3) if l.lower() in lead3.lower()][:5]

    cats = []
    for cm in re.finditer(r"\[\[Category:([^\]]+)\]\]", text):
        c = cm.group(1).strip()
        cl = c.lower()
        pre = c.split("|", 1)[0].strip()
        if 3 <= len(c) <= 60 and not cl.startswith(_CAT_SKIP_PREFIX) and not toml_meta(cl):
            cats.append(pre)
    cats = cats[:16]

    box = infobox_pairs(text)

    if not (defn or links or cats or box) and not short:
        return None
    return {"title": title, "lead1": lead1, "lead3": lead3, "def": defn,
            "links": links, "cats": cats, "box": box}


def toml_meta(cl):
    return cl.endswith("stubs") or " infobox" in cl


def reservoir(stream, cap, rng, sample_p=1.0):
    keep = []
    n = 0
    t0 = __import__("time").time()
    for line in stream:
        n += 1
        if sample_p < 1.0 and rng.random() > sample_p:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if n % 500_000 == 0 or __import__("time").time() - t0 > 60:
            t0 = __import__("time").time()
            print(f"scanned {n} docs, kept {len(keep)}", flush=True)
        if len(keep) < cap:
            keep.append(r)
        else:
            j = rng.randrange(0, n)
            if j < cap:
                keep[j] = r
    return keep, n


def build_seeds(docs, rng, max_seeds):
    rng.shuffle(docs)
    seeds = []
    for r in docs:
        s = extract_seed(r)
        if s is None:
            continue
        seeds.append(s)
        if len(seeds) >= max_seeds:
            break
    return seeds


def opts8(rng, correct, pool, n_opt=8, exclude=None):
    exclude = exclude or set()
    opts = [correct]
    seen = {norm(correct).lower()}
    snorm = norm(correct).lower()
    cand = rng.sample(pool, min(len(pool), 60))
    for c in cand:
        if len(opts) >= n_opt:
            break
        cn = norm(c).lower()
        if cn and cn != snorm and cn not in seen and cn not in exclude:
            seen.add(cn)
            opts.append(c)
    for _ in range(40):
        if len(opts) >= n_opt:
            break
        c = rng.choice(pool)
        cn = norm(c).lower()
        if cn and cn != snorm and cn not in seen and cn not in exclude:
            seen.add(cn)
            opts.append(c)
    rng.shuffle(opts)
    pos = opts.index(correct)
    return opts, pos


def make_row(fam, tag, context, question, options, correct, title, diff="medium"):
    probs = [0.05 / 7] * 8
    probs[correct] = 0.95
    return {
        "id": str(uuid.uuid4()),
        "task": fam,
        "difficulty": diff,
        "context": context,
        "question": question,
        "options": options,
        "correct_index": correct,
        "teacher_probs": probs,
        "label_source": {"model": "wikipedia-corpus", "temperature": 0.0, "n_samples": 1, "vote": "template"},
        "meta": {"family": fam, "generated_by": tag, "source_page": title[:80]},
    }


def row_key(r):
    return (norm(r["context"]), norm(r["question"]), tuple(norm(o) for o in r["options"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--max-scan", type=int, default=-1)
    ap.add_argument("--seed-cap", type=int, default=320000)
    ap.add_argument("--parse-sample", type=float, default=1.0,
                    help="parse only this fraction of lines (candidate sampling)")
    ap.add_argument("--out", default=str(DATA_ROOT / "mined.jsonl"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cap-def", type=int, default=60000)
    ap.add_argument("--cap-kb", type=int, default=60000)
    ap.add_argument("--cap-cat", type=int, default=60000)
    ap.add_argument("--cap-box", type=int, default=80000)
    ap.add_argument("--cap-ret", type=int, default=80000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("phase A: streaming corpus...", flush=True)
    with open(args.corpus, encoding="utf-8") as f:
        if args.max_scan > 0:
            def stream():
                for i, line in enumerate(f):
                    if i >= args.max_scan:
                        break
                    yield line
            docs, _ = reservoir(stream(), args.seed_cap, rng, sample_p=args.parse_sample)
        else:
            docs, _ = reservoir(f, args.seed_cap, rng, sample_p=args.parse_sample)
    print(f"phase A done: {len(docs)} docs retained from corpus", flush=True)

    seeds = build_seeds(docs, rng, args.seed_cap)
    print(f"seeds built: {len(seeds)}", flush=True)

    def_pool, title_pool, cat_pool = [], [], []
    box_pools = {}
    for s in seeds:
        if s["def"] and len(def_pool) < 50000:
            def_pool.append(s["def"])
        if s["title"] and len(title_pool) < 60000:
            title_pool.append(s["title"])
        for c in s["cats"]:
            if len(cat_pool) < 50000:
                cat_pool.append(c)
        for fld, vals in s["box"].items():
            bp = box_pools.setdefault(fld, [])
            for v in vals:
                if len(bp) < 40000:
                    bp.append(v)

    def_pool = list(dict.fromkeys(def_pool))
    title_pool = list(dict.fromkeys(title_pool))
    cat_pool = list(dict.fromkeys(cat_pool))
    box_pools = {f: list(dict.fromkeys(vs)) for f, vs in box_pools.items()}
    print(f"pools: def={len(def_pool)} title={len(title_pool)} cat={len(cat_pool)} "
          f"box={ {k: len(v) for k, v in box_pools.items()} }", flush=True)

    counts = {"define": 0, "knowledge": 0, "wikicat": 0, "wikibox": 0, "wikiret": 0}
    caps = {"define": args.cap_def, "knowledge": args.cap_kb, "wikicat": args.cap_cat,
            "wikibox": args.cap_box, "wikiret": args.cap_ret}

    rng.shuffle(seeds)
    rows, seen = [], set()
    for s in seeds:
        t = s["title"]
        done = all(counts[k] >= caps[k] for k in counts)
        if done:
            break
        if counts["define"] < caps["define"] and s["def"]:
            opts, pos = opts8(rng, s["def"], def_pool)
            r = make_row("define", "template:wikilead-def", "",
                         f"Which of these best defines {t}?", opts, pos, t)
            k = row_key(r)
            if k not in seen:
                seen.add(k)
                rows.append(r)
                counts["define"] += 1
        if counts["knowledge"] < caps["knowledge"] and s["def"]:
            opts, pos = opts8(rng, t, title_pool)
            r = make_row("knowledge", "template:wikilead-title", s["def"],
                         KB_Q, opts, pos, t)
            k = row_key(r)
            if k not in seen:
                seen.add(k)
                rows.append(r)
                counts["knowledge"] += 1
        if counts["wikicat"] < caps["wikicat"] and s["cats"]:
            cat = rng.choice(s["cats"])
            others = set(norm(c).lower() for c in s["cats"])
            opts, pos = opts8(rng, cat, cat_pool, exclude=others)
            ctx = f"{t}. {s['lead1']}".strip() if s["lead1"] else t
            r = make_row("wikicat", "template:wikicat-membership", ctx,
                         CAT_Q, opts, pos, t)
            k = row_key(r)
            if k not in seen:
                seen.add(k)
                rows.append(r)
                counts["wikicat"] += 1
        if counts["wikibox"] < caps["wikibox"] and s["box"]:
            flds = [f for f in s["box"] if s["box"][f] and f in box_pools and len(box_pools[f]) >= 8]
            rng.shuffle(flds)
            for fld in flds[:2]:
                if counts["wikibox"] >= caps["wikibox"]:
                    break
                val = rng.choice(s["box"][fld])
                q = FIELD_Q[fld].format(title=t)
                opts, pos = opts8(rng, val, box_pools[fld])
                r = make_row("wikibox", f"template:wikibox-{fld}", s["lead1"],
                             q, opts, pos, t, diff="hard")
                k = row_key(r)
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
                    counts["wikibox"] += 1
        if counts["wikiret"] < caps["wikiret"] and s["links"]:
            link = rng.choice(s["links"])
            others = set(norm(l).lower() for l in s["links"])
            opts, pos = opts8(rng, link, title_pool, exclude=others)
            r = make_row("wikiret", "template:wikiret-link", s["lead3"],
                         RET_Q, opts, pos, t)
            k = row_key(r)
            if k not in seen:
                seen.add(k)
                rows.append(r)
                counts["wikiret"] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("counts:", counts)
    print(f"wrote {len(rows)} mined rows -> {args.out}")


if __name__ == "__main__":
    main()