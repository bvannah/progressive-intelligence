from __future__ import annotations

import argparse
import json
import pathlib
import re
import socket
import time
import urllib.error
import urllib.request

from generate import chat, load_token, extract_json_array, PROVIDERS, CLINE_HEADERS, DEFAULT_MODEL
from prompts import build_rewriter_prompt
from merge_data import load_rows, validate_row


def _parse_choice(text: str, n: int) -> int | None:
    if not text:
        return None
    t = text.strip()
    m = re.match(r"(\d+)", t)
    if m:
        idx = int(m.group(1))
        return idx if idx < n else None
    m = re.match(r"[^\w]*([A-Ha-h])", t)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        return idx if idx < n else None
    return None


def _gate_fail(row, model, token, base, headers, temperature, retries):
    options = row["options"]
    lines = "\n".join(f"{i}) {o}" for i, o in enumerate(options))
    system = ("Answer a multiple-choice question with ONLY an integer index. "
              "No explanation.")
    user = (f"context: {row['context']}\n"
            f"question: {row['question']}\n"
            f"options:\n{lines}\n\n"
            "answer index:")
    for attempt in range(retries):
        try:
            reply = chat(base, token, model, [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ], temperature, 8, headers)
            return _parse_choice(reply, len(options))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                socket.timeout, OSError) as e:
            status = getattr(e, "code", "conn")
            if status in (429, 500, 502, 503, 504, 529) or status == "conn":
                time.sleep(2 ** attempt + 1)
                continue
            return None
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description="LLM rewrite pass: format variation + re-answer gate (B7).")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="cline", choices=sorted(PROVIDERS))
    ap.add_argument("--models", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--base", default="")
    ap.add_argument("--token", default="")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    ap.add_argument("--no-gate", action="store_true", help="skip the re-answer gate")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    token = args.token or load_token(args.provider)
    base = args.base or PROVIDERS[args.provider]["base"]
    headers = dict(PROVIDERS[args.provider]["headers"])
    if args.provider == "cline":
        headers = {**headers, **CLINE_HEADERS}
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = load_rows(args.inp)
    if args.max_rows:
        rows = rows[: args.max_rows]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    rewrote = kept_original = gated_out = failed = 0
    with open(args.out, mode, encoding="utf-8") as f:
        from random import Random
        rng = Random(args.seed)
        for i, orig in enumerate(rows):
            model = models[rng.randrange(len(models))]
            system, user = build_rewriter_prompt([orig], model)
            got = None
            for attempt in range(args.retries):
                try:
                    reply = chat(base, token, model, [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ], args.temperature, args.max_tokens, headers)
                    out_rows = [validate_row(r, f"rewrite:{model}") for r in extract_json_array(reply)]
                    got = out_rows
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                        socket.timeout, OSError) as e:
                    status = getattr(e, "code", "conn")
                    if status in (429, 500, 502, 503, 504, 529) or status == "conn":
                        time.sleep(2 ** attempt + 1)
                        continue
                    failed += 1
                    break
                except (ValueError, json.JSONDecodeError):
                    continue
            if got is None or len(got) != 1:
                failed += 1
                f.write(json.dumps(orig) + "\n")
                f.flush()
                continue
            new = got[0]
            if new["meta"].get("family") != orig["meta"].get("family"):
                new["meta"]["family"] = orig["meta"].get("family", "unknown")
            if args.no_gate:
                new["meta"]["rewritten"] = True
                f.write(json.dumps(new) + "\n")
                f.flush()
                rewrote += 1
                continue
            idx = _gate_fail(new, model, token, base, headers, args.temperature, args.retries)
            if idx == new["correct_index"]:
                new["meta"]["rewritten"] = True
                f.write(json.dumps(new) + "\n")
                f.flush()
                rewrote += 1
            else:
                kept_original += 1
                f.write(json.dumps(orig) + "\n")
                f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} rewrote={rewrote} kept-orig={kept_original} "
                      f"fail={failed}", flush=True)
    print(f"done: rewrote={rewrote} kept-original(gate)={kept_original} fail={failed} -> {args.out}")


if __name__ == "__main__":
    main()