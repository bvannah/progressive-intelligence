# Progressive Intelligence

A System-1-style multiple-choice "decision" model: given a passage of context and
a small list of candidate answers (2–8), it returns a probability distribution
over the candidates in a single forward pass. No token generation.

The bet ([`ideation.md`](ideation.md)) is that a large share of real decisions
are "Fast" — answerable from shallow pattern-level evidence without the
expensive multi-step reasoning that autoregressive LLMs are built for. This
project optimizes for that Fast side on a local-GPU budget: the whole model is
1.78M parameters.

## Current model

`model_v8.pt` is the frozen production model:

- Architecture `v8`: 2-layer RoPE transformer (base 500k), QK-norm, SDPA, and a
  per-option top-k span cross-attention (`span_w=64`, `span_k=8`) that keeps key
  memory bounded regardless of context length.
- `emb_dim=64`, 4 heads, `dim_ff=256`, `proj_dim=256`, custom word/number
  tokenizer (~16k word entries), option text capped at 256 tokens, up to 8
  options.
- Trained with uniform AdamW (lr 1e-3), masked-KL over option probabilities,
  early stop on eval loss (patience 3): 33 epochs, best at epoch 30,
  **0.3989 eval-loss / 0.805 accuracy** on its own held-out eval set.
- Context is RoPE-based, so the encoder accepts arbitrarily long context;
  training used ≤512 tokens but inference extrapolates cleanly to 8k
  (see [`ask.py`](#usage) and [`v8-results-bc-design.md`](v8-results-bc-design.md)).

Per directive in `AGENTS.md`: the model is intended to be **generic**, not
tuned toward any specific evaluation set.

## Training data

Sources are balanced per family (no source above ~7% of the corpus, no family
above 3x the median) and checked so nothing overlaps the held-out public eval
sets:

- **Programmatic generation** — `synthetic.py`, `composition.py` (deterministic
  template and compositional-chain families with hard-gold verification).
- **LLM generation** — `generate.py`, `generate_parallel.py` (batch LLM round-3
  data), `wiki_fact_llm.py`.
- **Mined Wikipedia facts** — `wiki_fact.py`, `mine_corpus.py`.
- **External public MC datasets** — `fetch_public_new.py` (QuALITY, QASC,
  MedMCQA, GSM-MC, MATH-MC; `cais/mmlu` omitted, no train split).
- **Transformations / noise** — `transform.py`, `rewrite.py`.

`merge_data.py` merges all sources, enforces per-family 90/10 train/eval with a
topic-disjoint split that routes new rows to sensible families, and excludes
public-eval overlap. `rebalance_eval.py` dedups and rebalances the eval split.
`gate_corpus.py` is the pre-training gate (`gate_corpus.py --train T --eval E
--require-var-n`). `make_breakdown.py` regenerates the corpus census
(`DATASET_BREAKDOWN_M4.md`).

The latest merged corpus (train 638,382 / eval 69,069) passes the gate; a v9
training run on it uses `train.jsonl` / `eval.jsonl`.

## Usage — querying the model (`ask.py`)

`ask.py` loads a checkpoint and lets you ask a question (with optional context)
plus 2–8 answer choices and get probabilities back.

Defaults: model `model_v8.pt`, device `cuda` if available (else `cpu`),
context up to 8000 tokens.

### Single question, comma-separated options

```
python3 ask.py --question "what is the capital of France?" \
  --options "Berlin, Paris, Madrid, London"
```

### With a context passage and repeated `-o` flags

```
python3 ask.py --context "A whale is a large marine mammal that breathes air
  and gives live birth. Dolphins are small toothed whales." \
  --question "Is a dolphin a mammal?" -o yes -o no -o maybe
```

Output (ranked, with an ASCII confidence bar):

```
------------------------------------------------------------
context (21 tokens): A whale is a large marine mammal that breathes air...
question: Is a dolphin a mammal?
  A. 0.8585 ##########################      yes <--
  C. 0.0774 ##                              maybe
  B. 0.0642 ##                              no
------------------------------------------------------------
```

### Interactive REPL

Run with no `--question` to get a prompt loop; each turn takes an optional
context, a question, and comma-separated options. Empty input ends the session.

```
python3 ask.py
Ask the model. Empty input at any prompt ends the session.
Options: comma-separated list of 2-8 choices.

context (optional): The quick brown fox jumps over the lazy dog.
question: which animal jumps over the dog?
options (comma-separated): fox, cat, dog, bird
```

### Keys to know

- **Context caching.** When you reuse the same context for several questions,
  the encoded context (`forward_with_ctx_cache` in `model.py`) is reused, so
  follow-up questions are ~30x faster (e.g. 4200-token context: 0.58s cold →
  0.02s per follow-up).
- **Long context.** `--max-ctx` defaults to 8000 (RoPE extrapolates; an 8000
  token query takes ~0.6s / ~300MB on the 6GB GPU). Pass `--chunk` for chunked
  attention if you want to trade speed for memory on extreme lengths.
- **Options:** 2–8 choices are enforced; option text is truncated at 256 tokens.
- **Other flags:** `--model path.pt` (e.g. `model_v7.pt`), `--device cpu|cuda`,
  `--max-ctx N`.

## Training

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
  --data data/train.jsonl --valid data/eval.jsonl --out data/model_v9.pt \
  --arch v8 --seed 1 --batch-size 128 --lr 1e-3 \
  --lr-attn-mult 1.0 --lr-embed-mult 1.0 --lr-head-mult 1.0 \
  --max-ctx 512 --epochs 40 --patience 3 --device cuda
```

Checkpoints are saved at every 5 epochs plus a best-state artifact; the final
`--out` file holds the best model, its config, and the tokenizer vocab.

## Evaluation

- `eval.py --model model_v8.pt --data eval.jsonl` — accuracy, ECE, Brier on a
  dataset.
- `public_eval.py` — held-out public sets (ARC-challenge, MMLU astronomy,
  RACE-middle) via the cache under `data/.public-cache/`; these are excluded
  from training data.

## Status

- `model_v8.pt` — frozen, production.
- v9 (expanded corpus above) — training run in progress / paused; a second
  interface-relevant change near v9 was a fix for empty-context rows in
  `model.py` (`encode_context` normalizes zero-width context to a single
  padding token).