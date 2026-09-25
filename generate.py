from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import secrets
import socket
import time
import urllib.error
import urllib.request
from collections import Counter

from merge_data import validate_row
from prompts import FAMILY_INSTRUCTIONS, FAMILY_OPTION_COUNTS, build_prompt, build_variator_prompt
from constants import LLM_DATA
from composition import SEEDED_FAMILIES, make_seed, seed_line, verify_seeded_row

KILO_GATEWAY = "https://api.kilo.ai/api/gateway"
CLINE_BASE = "https://api.cline.bot/api/v1"

KILO_HEADERS = {
    "X-KILOCODE-EDITORNAME": "Pi",
    "User-Agent": "pi-free-providers",
}

CLINE_HEADERS = {
    "HTTP-Referer": "https://cline.bot",
    "X-Title": "Cline",
    "X-PLATFORM": "Visual Studio Code",
    "X-PLATFORM-VERSION": "1.109.3",
    "X-CLIENT-TYPE": "VSCode Extension",
    "X-CLIENT-VERSION": "4.1.10",
    "X-CORE-VERSION": "4.1.10",
    "X-Is-Multiroot": "false",
    "User-Agent": "Cline/4.1.10",
}

PROVIDERS = {
    "kilo": {"base": KILO_GATEWAY, "headers": KILO_HEADERS},
    "cline": {"base": CLINE_BASE, "headers": {}},
    "local": {"base": "http://127.0.0.1:8080/v1", "headers": {}},
}

DEFAULT_MODEL = "google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,nex-agi/nex-n2.5-mini:free"
DEFAULT_WIKI_CORPUS = "/mnt/9a99846e-8002-475e-9eba-b992cd54e718/kb_full/corpus.jsonl"

# Broad seed topics for the topic sampler (B2); mixed with wiki-corpus headlines.
SEED_TOPICS = [
    "volcanology", "jazz", "baking", "sailing", "beekeeping", "volleyball", "chess",
    "public parks", "budgeting", "trains", "poetry", "robotics", "geology", "knitting",
    "home aquariums", "gardening", "astronomy", "cinema", "mountaineering", "ceramics",
    "origami", "fencing", "pottery", "first aid", "freshwater fish", "coffee roasting",
    "clockmaking", "lighthouses", "recycling", "yoga", "archery", "sushi", "bridges",
    "tulips", "meteorology", "archaeology", "blacksmithing", "cycling", "soap making",
    "rivers", "orchards", "power outages", "street markets", "postal services",
    "orphanages", "libraries", "nursing homes", "fire stations", "municipal drains",
]


def load_wiki_topics(corpus_path: str | None, cap: int = 5000) -> list[str]:
    if not corpus_path:
        return []
    topics: list[str] = []
    try:
        with open(corpus_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= cap:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("redirect_target"):
                    continue
                title = (r.get("title") or "").strip()
                if not 3 <= len(title) <= 60:
                    continue
                if not re.match(r"^[A-Za-z][A-Za-z0-9 ,'\-()\u00e9\u00e8]*$", title):
                    continue
                topics.append(title)
    except OSError as e:
        print(f"  wiki-topic load skipped: {e}")
    return topics


def _ulid() -> str:
    chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    ts = ""
    t = int(time.time() * 1000)
    for _ in range(10):
        ts = chars[t % 32] + ts
        t //= 32
    return ts + "".join(secrets.choice(chars) for _ in range(16))


CLINE_HEADERS["X-Task-ID"] = _ulid()


def load_token(provider: str) -> str:
    env = {
        "kilo": "KILO_API_KEY",
        "cline": "CLINE_API_KEY",
        "local": "",
    }[provider]
    if not env:
        return ""
    val = os.environ.get(env, "").strip()
    if val:
        return val
    pi_dir = pathlib.Path.home() / ".pi"
    auth_file = pi_dir / "agent" / "auth.json"
    if auth_file.exists():
        try:
            auth = json.loads(auth_file.read_text())
            entry = auth.get(provider, {})
            tok = entry.get("access") if provider == "kilo" else entry.get("key")
            if tok:
                return tok
        except Exception:
            pass
    free_file = pi_dir / "free.json"
    if free_file.exists():
        try:
            key = json.loads(free_file.read_text()).get(f"{provider}_api_key")
            if key:
                return key
        except Exception:
            pass
    if provider == "cline":
        try:
            cli = pathlib.Path.home() / ".cline" / "data" / "settings" / "providers.json"
            if cli.exists():
                raw = json.loads(cli.read_text())
                for name in ("cline-pass", "cline"):
                    key = raw.get("providers", {}).get(name, {}).get("settings", {}).get("apiKey")
                    if key:
                        return key
        except Exception:
            pass
    raise RuntimeError(f"no {provider} token found (set {env} or login via Pi)")


def _request(base: str, path: str, method: str, token: str, body=None, headers=None, timeout: int = 60):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            **headers,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def list_models(provider: str, base: str, token: str) -> list[str]:
    if provider == "cline":
        return _list_cline_models(base)
    data = _request(base, "/models", "GET", token, headers=PROVIDERS[provider]["headers"])
    models = data.get("data", data) if isinstance(data, dict) else data
    ids = [m["id"] for m in models]
    free = [m["id"] for m in models if m.get("id", "").endswith(":free")]
    print(f"catalog: {len(ids)} models, {len(free)} free")
    for mid in free:
        print("  free:", mid)
    return ids


def _list_cline_models(base: str) -> list[str]:
    ids: list[str] = []
    hdrs = {**PROVIDERS["cline"]["headers"], **CLINE_HEADERS, "Accept": "application/json"}
    try:
        j = _request(base, "/ai/cline/recommended-models", "GET", "public", headers=hdrs, timeout=30)
        free = j.get("free") or []
        ids.extend(m["id"] for m in free if m.get("id"))
        print(f"cline recommended free: {len(free)}")
    except Exception as e:
        print(f"  recommended-models fetch failed: {e}")
    store = pathlib.Path.home() / ".pi" / "agent" / "models-store.json"
    if store.exists():
        try:
            cline = json.loads(store.read_text()).get("cline", {}).get("models", [])
            for m in cline:
                mid = m.get("id", "")
                cost = m.get("cost", {})
                if mid.endswith(":free") or (cost.get("input", 1) == 0 and cost.get("output", 1) == 0):
                    if mid not in ids:
                        ids.append(mid)
            print(f"cline cached free: {len(ids)}")
        except Exception:
            pass
    for mid in ids:
        print("  free:", mid)
    return ids


def chat(base: str, token: str, model: str, messages: list, temperature: float, max_tokens: int, headers=None) -> str:
    data = _request(
        base,
        "/chat/completions",
        "POST",
        token,
        body={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers=headers,
    )
    payload = data
    if isinstance(data, dict) and "choices" not in data and isinstance(data.get("data"), dict):
        payload = data["data"]
    msg = payload["choices"][0]["message"]
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") for p in content if p.get("type") == "text"
        )
    if not content.strip() and msg.get("reasoning"):
        content = msg["reasoning"]
    return content


def extract_json_array(text: str) -> list:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array in model reply")
    return json.loads(text[start : end + 1])


def generate_family(
    family: str,
    count: int,
    model: str,
    token: str,
    base: str,
    headers: dict,
    temperature: float,
    max_tokens: int,
    retries: int,
    option_count: int | None = None,
    topic: str | None = None,
    extra: str = "",
) -> list[dict]:
    system, user = build_prompt(family, count, model, option_count, topic)
    if extra:
        user = user.rstrip() + "\n\n" + extra
    for attempt in range(retries):
        try:
            reply = chat(base, token, model, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], temperature, max_tokens, headers)
            rows = extract_json_array(reply)
            return [validate_row(r, f"llm:{model}") for r in rows]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
            status = getattr(e, "code", "conn")
            if status in (429, 500, 502, 503, 504, 529) or status == "conn":
                backoff = 2 ** attempt + 1
                print(f"    [{family}] HTTP {status} (attempt {attempt+1}), retry in {backoff}s")
                time.sleep(backoff)
                continue
            raise
        except (ValueError, json.JSONDecodeError) as e:
            print(f"    [{family}] parse/validate failure (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                print(f"    [{family}] giving up this batch")
                return []
    return []


def generate_variator(
    rows: list[dict],
    model: str,
    token: str,
    base: str,
    headers: dict,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> list[dict]:
    system, user = build_variator_prompt(rows, model)
    for attempt in range(retries):
        try:
            reply = chat(base, token, model, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], temperature, max_tokens, headers)
            out = extract_json_array(reply)
            return [validate_row(r, f"variator:{model}") for r in out]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout, OSError) as e:
            status = getattr(e, "code", "conn")
            if status in (429, 500, 502, 503, 504, 529) or status == "conn":
                backoff = 2 ** attempt + 1
                print(f"    [variator] HTTP {status} (attempt {attempt+1}), retry in {backoff}s")
                time.sleep(backoff)
                continue
            raise
        except (ValueError, json.JSONDecodeError) as e:
            print(f"    [variator] parse/validate failure (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return []
    return []


def run_variator(args, models, token, base, headers) -> None:
    from merge_data import load_rows

    seed = load_rows(args.variator_from)
    print(f"variator: {len(seed)} seed rows")
    rng = random.Random(args.seed)
    batch_n = max(1, args.count)
    produced = 0
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for start in range(0, len(seed), batch_n):
            batch = seed[start:start + batch_n]
            model = models[rng.randrange(len(models))]
            result = None
            for _ in range(1):
                rows = generate_variator(batch, model, token, base, headers,
                                         args.temperature, args.max_tokens, args.retries)
                if len(rows) == len(batch) and all(
                    len(rows[i]["options"]) == len(batch[i]["options"]) for i in range(len(batch))
                ):
                    result = rows
                    break
                print(f"  [variator] batch {start} failed count/option-count check; retrying another model")
                model = models[rng.randrange(len(models))]
            if result is None:
                print(f"  [variator] batch {start} abandoned")
                continue
            for r in result:
                f.write(json.dumps(r) + "\n")
                f.flush()
            produced += len(result)
            print(f"  [variator] +{len(result)} (total {produced})", flush=True)
    print(f"variator wrote {produced} rows -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Generate decision-model training rows via an LLM gateway.")
    ap.add_argument("--provider", default="cline", choices=sorted(PROVIDERS),
                    help="gateway provider (reuses Pi's config for auth/headers/baseUrl)")
    ap.add_argument("--models", default=DEFAULT_MODEL, help="comma-separated ordered model list")
    ap.add_argument("--families", default=","
                    .join(f for f in FAMILY_INSTRUCTIONS if f != "reformulate"))
    ap.add_argument("--count", type=int, default=5, help="rows requested per single completion call")
    ap.add_argument("--per-family", type=int, default=50)
    ap.add_argument("--out", default=str(LLM_DATA))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--base", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--append", action="store_true", help="append to --out instead of overwriting")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--option-counts", default="",
                    help="comma list of option counts to cycle per call: ints 2..8 or 'mix' "
                         "(default: per-family mix from FAMILY_OPTION_COUNTS)")
    ap.add_argument("--variator-from", default="",
                    help="jsonl of existing rows to diversify via the variator path (B1.3)")
    ap.add_argument("--no-topics", action="store_true",
                    help="disable random topic injection (B2)")
    ap.add_argument("--wiki-corpus", default=DEFAULT_WIKI_CORPUS)
    ap.add_argument("--wiki-topics-cap", type=int, default=5000,
                    help="max corpus lines scanned for headline topics")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.provider not in PROVIDERS:
        ap.error(f"unknown provider {args.provider!r}; choose from {sorted(PROVIDERS)}")

    token = args.token or load_token(args.provider)
    base = args.base or PROVIDERS[args.provider]["base"]
    headers = PROVIDERS[args.provider]["headers"]
    if args.provider == "cline":
        headers = {**headers, **CLINE_HEADERS}

    if args.list_models:
        list_models(args.provider, base, token)
        return
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    existing = Counter()
    if args.append:
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing[json.loads(line).get("meta", {}).get("family", "?")] += 1
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        print(f"resume mode: {sum(existing.values())} rows already in {args.out}")

    if args.variator_from:
        run_variator(args, models, token, base, headers)
        return

    topics: list[str] = []
    if not args.no_topics:
        topics = SEED_TOPICS[:] + load_wiki_topics(args.wiki_corpus, args.wiki_topics_cap)
    rng = random.Random(args.seed)
    print(f"topic bank: {len(topics)} topics "
          f"({'mixed' if not args.no_topics else 'disabled'})")

    if args.option_counts.strip():
        option_seq: list = []
        for token_ in args.option_counts.split(","):
            token_ = token_.strip().lower()
            if token_ == "mix":
                option_seq.append(None)
            else:
                n = int(token_)
                if not 2 <= n <= 8:
                    ap.error(f"--option-counts element must be 2..8 or 'mix', got {token_!r}")
                option_seq.append(n)
    else:
        option_seq = []

    produced = 0
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for family in families:
            remaining = max(0, args.per_family - existing.get(family, 0))
            if remaining <= 0:
                print(f"-- {family}: already complete ({existing.get(family, 0)} rows), skipping --", flush=True)
                continue
            fam_done = 0
            model_idx = 0
            call_idx = 0
            seq = option_seq or FAMILY_OPTION_COUNTS.get(family, [None])
            print(f"-- {family}: target {remaining} more rows (have {existing.get(family, 0)}), option counts {seq} --", flush=True)
            while fam_done < remaining:
                model = models[model_idx % len(models)]
                n_req = min(args.count, remaining - fam_done)
                n_opt = seq[call_idx % len(seq)]
                call_idx += 1
                topic = rng.choice(topics) if topics else None
                extra = ""
                seed = None
                if family in SEEDED_FAMILIES:
                    seed = make_seed(rng, family)
                    extra = seed_line(family, seed)
                rows = generate_family(family, n_req, model, token, base, headers,
                                       args.temperature, args.max_tokens, args.retries,
                                       option_count=n_opt, topic=topic, extra=extra)
                if not rows:
                    model_idx += 1
                    if model_idx >= len(models) * 2:
                        print(f"  [{family}] all models exhausted; stopping family")
                        break
                    continue
                kept = []
                for r in rows:
                    if seed is not None and family in SEEDED_FAMILIES:
                        if not verify_seeded_row(family, r, seed):
                            continue
                        r["meta"]["seed"] = seed
                    if topic:
                        r["meta"]["topic"] = topic
                    kept.append(r)
                if seed is not None and len(kept) < len(rows):
                    print(f"  [{family}] hard-gold verify kept {len(kept)}/{len(rows)}", flush=True)
                if not kept:
                    model_idx += 1
                    continue
                for r in kept:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                fam_done += len(kept)
                produced += len(kept)
                print(f"  [{family}] {fam_done}/{remaining}", flush=True)
        print(f"\nwrote {produced} validated rows -> {args.out}")


if __name__ == "__main__":
    main()