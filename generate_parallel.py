"""generate_parallel.py — multiprocess + multi-host LLM generation runner.

Spawns N independent generate.py workers (disjoint family sets, one output file
each) as detached sessions, monitors them, and auto-restarts any worker that dies
(pygrep-style) using generate.py's --append resume logic. Fully restart-safe:
rerunning this command with the same args will adopt still-alive workers (by PID
file) instead of duplicating work.

Usage example:
  python3 generate_parallel.py \
    --state-dir /tmp/opencode/gg \
    --worker "w1|cline|cline-free/gemini-3.8-flash,cline-free/deepseek-v4.1-flash|order,ambiguity,permuted,cause,summary,tone,intent|150|/mnt/.../llm_rd2.jsonl" \
    --worker "w2|cline|cline-free/deepseek-v4.1-flash,cline-free/gemini-3.8-flash|define,knowledge,analogy,consistency,open,longread|150|/mnt/.../llm_rd2b.jsonl"

Worker format:  name|provider|models|csv-families|per_family|out.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from collections import Counter

HERE = pathlib.Path(__file__).parent
STOP = False


def sig_handler(sig, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def worker_targets(out: pathlib.Path, families: list[str]) -> Counter[str, int]:
    """Valid rows per family currently in the worker's output file."""
    cnt: Counter = Counter()
    if not out.exists():
        return cnt
    with open(out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cnt[json.loads(line).get("meta", {}).get("family", "?")] += 1
            except json.JSONDecodeError:
                pass
    return cnt


def worker_done(out: pathlib.Path, families: list[str], per_family: int) -> bool:
    cnt = worker_targets(out, families)
    return all(cnt.get(f, 0) >= per_family for f in families)


def spawn(worker: dict, state_dir: pathlib.Path, common: dict) -> subprocess.Popen | None:
    name = worker["name"]
    provider = worker["provider"]
    models = worker["models"]
    families = worker["families"]
    per_family = worker["per_family"]
    out = worker["out"]
    pid_f = state_dir / f"{name}.pid"
    if pid_f.exists():
        try:
            pid = int(pid_f.read_text().strip())
            os.kill(pid, 0)
            return None  # already running; manager will pick it up
        except (OSError, ValueError):
            pass
    log_f = state_dir / f"{name}.log"
    cmd = [
        sys.executable, str(HERE / "generate.py"),
        "--provider", provider,
        "--models", models,
        "--families", families,
        "--per-family", str(per_family),
        "--out", str(out),
        "--append",
        "--count", str(worker.get("count", common["count"])),
        "--max-tokens", str(worker.get("max_tokens", common["max_tokens"])),
        "--retries", str(common["retries"]),
        "--temperature", str(worker.get("temperature", common["temperature"])),
        "--seed", str(worker.get("seed", common["seed"])),
    ]
    with open(log_f, "ab") as f:
        f.write(f"\n=== restart {name} at {time.strftime('%F %T')} ===\n".encode())
        f.flush()
    try:
        p = subprocess.Popen(
            cmd, cwd=str(HERE), stdout=open(log_f, "ab"), stderr=open(log_f, "ab"),
            start_new_session=True,
        )
        pid_f.write_text(str(p.pid))
        return p
    except OSError as e:
        print(f"[{name}] spawn failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="append", required=True,
                    help="name|provider|models|families|per_family|out  (repeatable)")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--check-sec", type=float, default=20.0)
    ap.add_argument("--max-restarts", type=int, default=50)
    args = ap.parse_args()

    state_dir = pathlib.Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    workers = []
    for w in args.worker:
        parts = w.split("|")
        if len(parts) < 6 or len(parts) > 9:
            print(f"bad --worker spec: {w} (6..9 fields: "
                  f"name|provider|models|families|per_family|out|count|max_tokens|temperature)")
            sys.exit(2)
        name, provider, models, families, per_family, out = parts[:6]
        over = {}
        if len(parts) >= 7 and parts[6].strip():
            over["count"] = int(parts[6])
        if len(parts) >= 8 and parts[7].strip():
            over["max_tokens"] = int(parts[7])
        if len(parts) >= 9 and parts[8].strip():
            over["temperature"] = float(parts[8])
        workers.append({
            "name": name, "provider": provider, "models": models,
            "families": families, "per_family": int(per_family),
            "out": pathlib.Path(out), "restarts": 0, **over,
        })

    common = {"count": args.count, "max_tokens": args.max_tokens,
              "retries": args.retries, "temperature": args.temperature, "seed": args.seed}

    procs: dict[str, subprocess.Popen | None] = {}
    for w in workers:
        procs[w["name"]] = spawn(w, state_dir, common)

    print("launched:", ", ".join(f"{w['name']}({'running' if procs[w['name']] is not None else 'adopted-alive'})"
                                 for w in workers), flush=True)

    while not STOP:
        for w in workers:
            name = w["name"]
            done = worker_done(w["out"], w["families"].split(","), w["per_family"])
            pid_f = state_dir / f"{name}.pid"
            alive = False
            if pid_f.exists():
                try:
                    pid = int(pid_f.read_text().strip())
                    os.kill(pid, 0)
                    alive = True
                except (OSError, ValueError):
                    alive = False
            if done:
                if alive:
                    os.kill(pid, signal.SIGTERM)
                continue
            if not alive and not done:
                if w["restarts"] >= args.max_restarts:
                    print(f"[{name}] hitting max restarts; giving up", flush=True)
                    continue
                w["restarts"] += 1
                print(f"[{name}] worker died (restart #{w['restarts']}); resuming", flush=True)
                procs[name] = spawn(w, state_dir, common)

        summary = {}
        for w in workers:
            cnt = worker_targets(w["out"], w["families"].split(","))
            need = w["per_family"]
            fam_rows = {f: min(cnt.get(f, 0), need) for f in w["families"].split(",")}
            total = sum(fam_rows.values())
            want = need * len(w["families"].split(","))
            summary[w["name"]] = {"rows": fam_rows, "pct": int(100 * total / max(1, want))}
        tot = sum(min(cnt.get(f, 0), w["per_family"]) for w in workers for f in w["families"].split(","))
        print(f"[{time.strftime('%F %T')}] " + "  ".join(
            f"{n}({v['pct']}%->{v['rows']})" for n, v in summary.items()) +
            f"  total_capped={tot}", flush=True)
        with open(state_dir / "status.json", "w") as f:
            json.dump({"time": time.time(), "workers": summary}, f)
        time.sleep(args.check_sec)

    print("terminating workers...", flush=True)
    for w in workers:
        pid_f = state_dir / f"{w['name']}.pid"
        if pid_f.exists():
            try:
                os.kill(int(pid_f.read_text().strip()), signal.SIGTERM)
            except (OSError, ValueError):
                pass
    print("done")


if __name__ == "__main__":
    main()