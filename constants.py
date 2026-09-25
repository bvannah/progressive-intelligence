from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("/mnt/1BBFC6E12FC77EDC/data/progressive_intelligence_data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

SYNTHETIC_DATA = DATA_ROOT / "data.jsonl"
LLM_DATA = DATA_ROOT / "llm_rows.jsonl"
WIKI_FACT_DATA = DATA_ROOT / "wiki_fact.jsonl"
BASE_DATA = DATA_ROOT / "base.jsonl"
TRAIN_DATA = DATA_ROOT / "train.jsonl"
EVAL_DATA = DATA_ROOT / "eval.jsonl"

MODEL_DATA = DATA_ROOT / "model_v7.pt"
MODEL_DATA_V8 = DATA_ROOT / "model_v8.pt"
PUBLIC_CACHE = DATA_ROOT / ".public-cache"