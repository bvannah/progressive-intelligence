from __future__ import annotations

import argparse
import json
import random
import re
import uuid
from pathlib import Path

from constants import DATA_ROOT, WIKI_FACT_DATA

WIKI_FACT_DATA = DATA_ROOT / "wiki_fact.jsonl"
DEFAULT_CORPUS = "/mnt/9a99846e-8002-475e-9eba-b992cd54e718/kb_full/corpus.jsonl"

FAMILY = "wiki_fact"
YEAR = re.compile(r"\b((?:1[5-9]|20)\d{2})\b")
PREP = re.compile(r"\b(?:in|on|by|during|from|since|around|between|until)\b"
                  r"\s+(?:the\s+)?\b((?:1[5-9]|20)\d{2})\b")
LOC_PREP = re.compile(r"\b(?:is|are|was|were)\s+(?:located|based|headquartered|situated)\s+in\s+(.*)", re.I)
BAD = re.compile(r"\{\{|\}\}|\[\[|\]\]|http")
CITATION = re.compile(r"\s*\[(?:notes|refs|\d+)\]\s*")
BAD_START = {"this", "these", "their", "its", "they", "as", "although",
             "while", "then", "but", "until", "for", "when", "where"}

_QUESTIONS = [
    "In which year did this happen?",
    "During which year did this occur?",
    "In what year did this take place?",
]

_PLACE_SPLIT = re.compile(r"[,;.]|\b(?:and|the|of|by|at|as|in|on|for|during|called|until|was|were|is|are|that|which|with)\b")
_NOISY = re.compile(r"[;-]|\d|\[")


_PLACE_Q = "Where is the entity mentioned in the passage located?"

_NUMBER_RE = re.compile(
    r"(?:population of\s+|population\s+of\s+)?(\d[\d,.]*(?: million| billion| thousand)?)"
    r"(?=\s*(?:people|inhabitants|residents|sq|\bkm|kilomet|kilom|\bm\b|meters|metres|miles|hectares|acres|euros|dollars|GB|TB|MHz|watts)?)"
)
_NUMBER_QUESTION = "What number is explicitly stated in this sentence?"
_NUMBER_QVARIANTS = [
    "What number is explicitly stated in this sentence?",
    "Which number is given here?",
    "What figure does the passage state?",
]

_WHO_RE = re.compile(
    r"\b(?:founded|established|created|developed|built|introduced|composed|led|owned|chaired)"
    r"\s+by\s+([A-Z][\w.' -]*?)\s*(?=[,;.(]|\bin\s|\bon\s|\band\s|\bwith\s|\bwho\b|\bthe\b|\bbefore\b|$)",
    re.I,
)
_WHO_QUESTIONS = {
    "founded": "Who is credited with founding this entity?",
    "established": "Who established this entity?",
    "created": "Who created the subject of the sentence?",
    "developed": "Who developed the subject of the sentence?",
    "built": "Who built the subject of the sentence?",
    "introduced": "Who introduced the subject of the sentence?",
    "composed": "Who composed the subject of the sentence?",
    "led": "Who is described as leading the subject?",
    "owned": "Who is described as owning the subject?",
    "chaired": "Who chairs the subject?",
}

_CAT_RE = re.compile(
    r"\bis\s+(?:a|an)\s+(?:kind|type|form|sort|species|genus|class|member)\s+of\s+([a-z][a-z -]{1,34})",
    re.I,
)
_CAT_QUESTIONS = [
    "What category is assigned to the subject?",
    "What kind of thing is described?",
    "To which general category does this belong?",
]


def pick_place(sent: str) -> tuple[str, str] | None:
    m = LOC_PREP.search(sent)
    if m is None or _NOISY.search(sent[: min(80, len(sent))]):
        return None
    rest = m.group(1).strip()
    toks = [t for t in _PLACE_SPLIT.split(rest) if t.strip()]
    if not toks:
        return None
    place = toks[0].strip(" '.")
    if len(place) < 2 or len(place) > 45 or place[0].islower():
        return None
    if BAD.search(place) or not re.match(r"^[A-Z][\w.' -]*$", place):
        return None
    return place, _PLACE_Q

_PREP_FOR_OPT = re.compile(r"\b(?:in|on|by|during|from|since|around|between|until)\b")


def clean_sentence(s: str) -> str:
    s = CITATION.sub("", s)
    s = s.replace("'''", "").replace("''", "")
    s = s.replace("(s)", "").replace("/s", "")
    s = re.sub(r"\bThe\s*$", "", s)
    s = re.sub(r"\.The\s*$", ".", s)
    s = s.strip("· ")
    return s.strip()


def facts_from_text(text: str, kinds: set) -> list[tuple]:
    text = re.sub(r"^#+ .*$", "", text, flags=re.M)
    text = re.sub(r"\n+", " ", text)
    out = []
    for s in re.split(r"\. ", text):
        s = clean_sentence(s.strip())
        if not (50 <= len(s) <= 320):
            continue
        if not s or s[0] != s[0].upper():
            continue
        if BAD.search(s):
            continue
        first = s.split()[0].strip("(").lower().rstrip(",")
        if first in BAD_START:
            continue
        if "year" in kinds:
            ys = YEAR.findall(s)
            if len(ys) == 1 and PREP.search(s):
                out.append(("year", s, int(ys[0]), None, None))
                continue
        if "place" in kinds:
            pl = pick_place(s)
            if pl is not None:
                out.append(("place", s, None, pl[0], pl[1]))
                continue
        if "number" in kinds:
            nums = _NUMBER_RE.findall(s)
            if "(" in s or ")" in s:
                continue
            if len(nums) == 1:
                num = nums[0].strip()
                if re.fullmatch(r"\d[\d,.]*(?: million| billion| thousand)?", num):
                    out.append(("number", s, None, num, None))
                    continue
        if "who" in kinds:
            m = _WHO_RE.search(s)
            if m is not None:
                name = m.group(1).strip().rstrip(".,;")
                if re.fullmatch(r"[A-Z][\w.' -]{1,39}", name) and name.count(" ") <= 3:
                    verb = m.group(0).split()[0].lower()
                    out.append(("who", s, None, name, verb))
                    continue
        if "cat" in kinds:
            m = _CAT_RE.search(s)
            if m is not None:
                cat = m.group(1).strip().lower()
                if 2 <= len(cat) <= 35 and not re.search(r"\d", cat):
                    out.append(("cat", s, None, cat, None))
    return out


def yield_articles(corpus_path: Path, max_scan: int = -1):
    seen = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if max_scan > 0 and seen >= max_scan:
                return
            seen += 1
            r = json.loads(line)
            if r["redirect_target"]:
                continue
            text = r.get("text") or ""
            if len(text) < 80 or len(text) > 40000:
                continue
            yield r["title"], text


def build_options_years(rng, year):
    options = {str(year)}
    for _ in range(80):
        options.add(str(rng.randint(1500, 2030)))
        if len(options) >= 8:
            break
    options = list(options)[:8]
    rng.shuffle(options)
    if str(year) not in options:
        options[0] = str(year)
        rng.shuffle(options)
    return options, options.index(str(year))


def build_options_places(rng, place, pool):
    options = {place}
    for _ in range(200):
        p = rng.choice(pool)
        if p != place and p not in place and place not in p:
            options.add(p)
        if len(options) >= 8:
            break
    while len(options) < 8:
        options.add(f"Port {rng.choice('ABCDEFGHIJ')}{rng.randint(1, 99)}")
        if len(options) >= 8:
            break
    options = list(options)[:8]
    rng.shuffle(options)
    if place not in options:
        options[0] = place
        rng.shuffle(options)
    return options, options.index(place)


def build_options_numbers(rng, num, pool):
    options = {num}
    for _ in range(300):
        p = rng.choice(pool)
        if p != num:
            options.add(p)
        if len(options) >= 8:
            break
    while len(options) < 8:
        options.add(str(rng.randint(1, 10000)))
    options = list(options)[:8]
    rng.shuffle(options)
    if num not in options:
        options[0] = num
        rng.shuffle(options)
    return options, options.index(num)


def build_options_names(rng, name, pool):
    options = set()
    for _ in range(400):
        p = rng.choice(pool)
        if p != name:
            options.add(p)
        if len(options) >= 7:
            break
    while len(options) < 7:
        options.add(f"{rng.choice('ABCDEFGHIJKLM')}{rng.choice('abcdefghijklm')}{rng.randint(10, 99)}")
    options = list(options)[:7]
    rng.shuffle(options)
    pos = rng.randrange(len(options) + 1)
    options.insert(pos, name)
    return options, pos


def build_options_cats(rng, cat, pool):
    options = set()
    for _ in range(300):
        p = rng.choice(pool)
        if p != cat:
            options.add(p)
        if len(options) >= 7:
            break
    while len(options) < 7:
        options.add(f"{rng.choice('abcdefghijklm')}{rng.choice('abcdefghijklm')} {rng.randint(1, 99)}")
    options = list(options)[:7]
    rng.shuffle(options)
    pos = rng.randrange(len(options) + 1)
    options.insert(pos, cat)
    return options, pos


def make_row(rng, title, kind, sentence, payload):
    tags = {"year": "template:wiki-fact-year", "place": "template:wiki-fact-located",
            "number": "template:wiki-fact-number", "who": "template:wiki-fact-who",
            "cat": "template:wiki-fact-category"}
    tag = tags[kind]
    if kind == "year":
        year = payload
        if not (1500 <= year <= 2030):
            return None
        question = rng.choice(_QUESTIONS)
        options, correct = build_options_years(rng, year)
    elif kind == "number":
        num = payload
        question = rng.choice(_NUMBER_QVARIANTS)
        options, correct = build_options_numbers(rng, num, NUMBER_POOL)
    elif kind == "who":
        name, verb = payload
        question = _WHO_QUESTIONS.get(verb, "Who is named in the passage?")
        options, correct = build_options_names(rng, name, NAME_POOL)
    elif kind == "cat":
        cat = payload
        question = rng.choice(_CAT_QUESTIONS)
        options, correct = build_options_cats(rng, cat, CAT_POOL)
    else:
        place, question = payload
        options, correct = build_options_places(rng, place, PLACE_POOL)
    probs = [0.05 / 7] * 8
    probs[correct] = 0.95
    return {
        "id": str(uuid.uuid4()),
        "task": FAMILY,
        "difficulty": "easy",
        "context": sentence,
        "question": question,
        "options": options,
        "correct_index": correct,
        "teacher_probs": probs,
        "label_source": {"model": "wikipedia-corpus", "temperature": 0.0,
                         "n_samples": 1, "vote": "template"},
        "meta": {"family": FAMILY, "generated_by": tag, "source_page": title},
    }


PLACE_POOL: list[str] = []
NUMBER_POOL: list[str] = []
NAME_POOL: list[str] = []
CAT_POOL: list[str] = []


def pool_append(pool, val, cap=16384):
    if len(pool) < cap:
        pool.append(val)
    elif random.randrange(2) == 0:
        pool[random.randrange(len(pool))] = val


def main():
    ap = argparse.ArgumentParser(description="Emit wiki fact-extraction rows from the local Wikipedia corpus.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--limit", type=int, default=3000, help="max rows to emit")
    ap.add_argument("--max-scan", type=int, default=-1, help="cap articles scanned (default: all)")
    ap.add_argument("--kind", default="year,place,number,who,cat", help="comma-separated fact kinds")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(WIKI_FACT_DATA))
    args = ap.parse_args()

    kinds = {k.strip() for k in args.kind.split(",") if k.strip()}
    rng = random.Random(args.seed)
    out = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for title, text in yield_articles(Path(args.corpus), max_scan=args.max_scan):
            for kid, sent, year, payload_a, q in facts_from_text(text, kinds):
                if kid == "place":
                    pool_append(PLACE_POOL, payload_a)
                    payload = (payload_a, q)
                elif kid == "year":
                    pool_append(NUMBER_POOL, str(year))
                    payload = year
                elif kid == "number":
                    pool_append(NUMBER_POOL, payload_a)
                    payload = payload_a
                elif kid == "who":
                    name, verb = payload_a, q
                    pool_append(NAME_POOL, name)
                    payload = (name, verb)
                elif kid == "cat":
                    pool_append(CAT_POOL, payload_a)
                    payload = payload_a
                row = make_row(rng, title, kid, sent, payload)
                if row is None:
                    continue
                f.write(json.dumps(row) + "\n")
                out += 1
                if out >= args.limit:
                    f.flush()
                    print(f"wrote {out} wiki_fact rows -> {args.out}")
                    return
    print(f"wrote {out} wiki_fact rows -> {args.out}")


if __name__ == "__main__":
    main()