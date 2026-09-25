# Progressive Intelligence — Ideation

Status: early brainstorming / v0
Goal of this doc: capture the idea, the reference points we learned from, the
(very real) open questions, and the set of concrete design decisions to make
next. Nothing here is decided.

---

## 1. Thesis

Most "intelligence" we actually deploy is not open-ended generation — it's
choosing between a handful of options given some context. Yet we currently pay
LLM prices (auto-regressive token generation) for that.

We want a model class that:

- **Inputs**: a long blob of context (text/bytes) + a small list of candidate
  answers (N = 8 to start).
- **Outputs**: a softmax probability vector over the candidates. All the
  "thinking" happens in hidden layers of a single (or handful of) forward
  pass(es). No token generation.
- **Training data**: generated programmatically by LLMs.
- **Constrain**: fit inside the local machine's compute/memory budget, be
  dramatically faster and cheaper than any LLM on the same decision task.

The bet (stolen from Kahneman's "System 1", and from what Jev/TypeSafe is
trying to do) is that a large share of real decisions are *Fast* — answerable
from local, shallow, pattern-level evidence — and do not require the *Slow*
multi-step reasoning that auto-regressive models are built for. We give up the
Slow side entirely and optimize the Fast side to the extreme.

---

## 2. What we learned from the Jev model (web research)

Jev (TypeSafe AI, launched 2026-09-15) is the first "System One Model":

- Unstructured **state** + typed **questions** in → typed answers with
  **calibrated probabilities** out, in **70–500 ms**, at $0.042 / 1M input
  tokens with free output. Claims of ~40–200× speedups over frontier LLMs on
  "System One-shaped" tasks (independent checks put the honest number closer
  to ~5× vs a cheap LLM — the big multiples are vs *reasoning* models).
- **No text generation at all** → outputs are pre-defined type-safe values;
  "hallucination" (of schema) is structurally impossible.
- Three question primitives: **Choice** (pick one of a fixed list, get per-option
  probabilities), **Score** (rate on a scale), **Boolean** (P(true)).
- Supports many questions per request, answered in **parallel** (single query,
  one forward pass) — this is the efficiency core. Choice cardinality up to 255
  via a 2-stage "score all then choose" trick.
- Trained with **RLCD** = Reinforcement Learning for Calibrated Decisions:
  rewards epistemically honest probabilities, not just argmax accuracy.
- Answers always carry confidence; they are meant to be **gated** on
  (high confidence → act, low → escalate), like a "smart if-statement".

What's genuinely transferable to us:
1. **Give up text generation → restrict the output space → win speed/calibration.**
2. **Parallel scoring** of all candidates in one pass (score-per-option, then
   softmax) is the natural architecture.
3. **Calibration is a first-class training objective**, not a byproduct.
4. **Serving reality**: decisions are composed into workflows/gates, so
   calibrated probabilities + speed > raw argmax accuracy.

What's *not* transferable: we can't assume a frontier-scale training run or a
distributed parallel sampler. We have to make the whole thing fit a single
laptop.

---

## 3. The system shape

```
context (long bytes/string)
        + [answer_1, answer_2, ..., answer_8]
                    |
                    v
        +---------------------------+
        |  encoder (fixed-size vec) |   <- cheapest thing that works
        +---------------------------+
                    |
     score(context, answer_i) -> logit_i   (i = 1..N, batched/parallel)
                    |
                    v
             softmax -> P(answer_i)
```

Key properties:
- **One (or few) forward pass(es)**. No auto-regression, no KV cache, no
  sampling loop.
- **Fixed input size**: the context must be compressed to a fixed-length vector
  no matter how long it is (this is *the* hard design decision — see §6).
- **Fixed output size**: N logits, softmaxed. Posteriors are cheap to produce
  (the softmax over N is ~free).
- **Option-order invariance** is a desirable property: permuting the answer list
  should permute the probabilities identically. That pushes us toward scoring
  each (context, option) pair independently rather than doing joint attention
  over all options. (This is also what Jev effectively does.)

---

## 4. Architecture sketches (design options, in increasing cost)

### A. Feature-hash + MLP ("fastText-style") — cheapest
- Operate directly on bytes / character n-grams (2..5-grams), no tokenizer.
- Hash n-grams into a large embedding table (e.g., 2^17..2^19 buckets), sum/average
  → fixed context vector `h` (e.g., 256–512 dims).
- Score function: `s_i = MLP([h ⊕ g_i])` where `g_i` is the same hashed
  encoding of option i. Options are batched into one forward pass.
- **Cost**: a few hash lookups + one small MLP → sub-millisecond to low
  milliseconds on CPU. Possible to do in pure numpy/C with no deep-learning
  framework in the loop.
- **Weakness**: no sequence/context modeling, weak on anything requiring
  word-order-sensitive inference.

### B. Small transformer encoder + pooling — moderate
- 2–4 layers, hidden 256–512, on top of the hashed or a small BPE vocab.
- Mean/CLS pooling → context vector `h`; score each option as in A.
- **Cost**: milliseconds, still far cheaper than any autoregressive LLM.
- **Strength**: locality + long-range-ish correspondence, better at clauses
  like "X happened *after* Y".

### D. Memory + attention hybrid (DeepSeek V4.1 / Engram-style) — the target
The frontier is moving exactly the way you're pointing. DeepSeek V4.1 Flash
(Sept 2026) is 763B params but only ~8B *active* per token; 196B of the 763B
are n-gram weights in what DeepSeek calls a "conditional memory module". Qwen
3.8-Flash-Next (180B, with 51B n-grams) and Google's Gemma PLE (per-layer
embeddings) pursue the same split: a cheap, deterministic *memory* path + a
*computation* path (MoE / attention) for actual thinking.

DeepSeek's Engram (Jan 2026 paper, Apache-2.0 reference code) is the mechanism
to steal wholesale:
- Hashes token 2-/3-grams to embedding rows via O(1), collision-resistant
  hashing: multiple heads, prime moduli per head, different seed per layer.
- Multi-scale + multi-head embedding → tolerant of hash collisions, richer
  retrieval.
- Injects the retrieved vectors into selected transformer layers (they augment
  layers 1 and 15 in their config).
- Core claim (their words): **decouple memory from computation.** The active
  network stays small while the lookup table can be enormous — offloadable to
  host RAM — because fetching a row costs "a few dozen table lookups", not a
  full sweep over the weights. (The Register's numbers: the 763B model needs
  only ~567GB of GPU memory.)
- Selective layer injection: Engram attaches only at *chosen* layers (layers 1
  and 15 in their demo) — early layers enrich with lookups, the middle stays
  pure computation. "Which layers get memory" is a fine-grained knob on the
  memory/compute split.

Mapping onto our design — your instinct is the recipe:
- **The n-gram path does everything it's capable of.** Hash the context (and
  each option) into row IDs, pull the stored association vectors, aggregate →
  a cheap "recall" signal `m`. Order-blind, deterministic, ~free per row.
- **Attention/order adds the residue.** A small order-sensitive encoder consumes
  the sequence *and* `m`, so token order and composition are only added where
  the n-gram lookup comes up short. Complementary by construction: n-gram
  memory is position-independent and offloadable; attention is
  position-dependent and compute-heavy.
- **Optional asymmetric readout**: `logit_i = α·match(m, g_i) + scorer([o, g_i])`
  — a near-parameter-free similarity term (retriever-shaped: recalls stored
  patterns) plus an attention readout (handles the residue). This gives the
  memory path an *explicitly retrieval-shaped* job rather than only latent
  geometry.

### C. The prototype plan
- v0: build the n-gram **memory path** (feature-hash encoding + a *learned* row
  table, subsection D's cheap half) + decision MLP + softmax. Sub-ms, no
  framework needed.
- Then measure whether the attention/order path (B/D) lifts accuracy enough to
  justify its cost *on our eval set*. Don't theorize — measure.
- Treat A/B/D as one continuous axis (memory × order) and slide along it
  against the eval set + latency budget.

Important: representation of the **question/answer semantics**. The context
should embed "this is a classification task" or the actual question, so the
model isn't trained per-task. We can also prepend a task-instruction token or a
hardcoded task description as part of the context — same cost.

---

## 5. Capacity budget: how far does a few GB + a few seconds get us?

Constraint (agreed): **a few GB of RAM and a few seconds of wall-clock on a
modest consumer machine** (CPU-driven, possibly a mid-range GPU). No datacenter.

### Depth is (almost) free — generation was the tax

An LLM answers a multiple-choice question with one forward pass to *read* the
context, then **one forward pass per output token** to *write* the answer —
hundreds to thousands more, and chain-of-thought multiplies that further. We do
**one** pass (a handful at most). So an LLM's *generation* budget becomes our
*depth* budget: a decision net that spends as much compute as an LLM spends on
a single token can be as deep as we like.

### The flop math

A dense forward pass ≈ `2 × params × tokens`:

| budget item | cost |
|---|---|
| 1B params × 500 chars | ≈ 1 TFLOP |
| 1B params × 8k chars | ≈ 8 TFLOP |
| CPU multi-thread (≈ 100–300 GFLOPs, AVX2) | seconds for the 8k case |
| mid GPU (≈ 10–50 TFLOPs) | tens of ms |

Takeaway: **params are cheap; context length is the flop killer.** A deep model
over a long sequence blows the budget (attention is O(n²)). Depth only becomes
"free" when its cost is decoupled from context length.

### Reader / thinker split (the key structural choice)

- **Reader** — cost scales with context *length* but is cheap per unit:
  feature-hash / n-gram pooling, or a small sliding-window / linear-attention
  encoder. Emits the fixed vector `h`.
- **Thinker** — a deep, fixed-shape stack over `h` (MLP / transformer blocks on
  the compressed vector). Cost is **independent of context length**: this is
  where depth and the bulk of the params live, and its budget only grows with
  params, never with input length.

So "a few seconds" ≈ the reader swallows even very long contexts cheaply while
the entire time budget goes to the thinker. E.g., a 500M-param thinker on a
512-dim `h` ≈ 0.5 TFLOP ≈ 1–3 s on CPU — identical whether the context was 500
or 50,000 chars.

### Iterating is allowed (depth-by-time, cheap memory)

Multiple passes within budget are fair game, and are memory-cheap if weights are
**shared** (Universal-Transformer / pondering style): run the same thinker block
over `h` k times, stopping on a confidence gate. Each iteration is a tiny
fixed-shape pass, not a token emission, so this buys recognizable multi-step
behavior at ~1/50–1/100 of an autoregressive budget — and turns "a few seconds"
into an adaptive quality knob.

### How to spend ~2GB (draft allocation)

Memory is the cheap half of the budget; compute is the scarce half. Spend
accordingly:

- n-gram memory table (learned, Engram-style input rows + optionally a wide
  recall matrix) — keep it **as large as the RAM budget allows**: fetching a
  row is a handful of lookups, so table size barely affects latency (the
  DeepSeek V4.1 trick — 196B of its 763B params are n-grams *because* they're
  cheap to query). This is our offloadable "encyclopedia".
- ~200–400MB — reader / attention path (order-sensitive; compute-heavy, keep
  small).
- ~1–1.5GB — thinker: the deep stack over the fixed vector.

### What this buys (the honest intelligence claim)

At ~1B+ params and a few seconds, the claim is: **small-LLM-level comprehension
(≈ Llama-3.2-1B / GPT-2-class) pointed 100% at deciding, well-calibrated, at
10²–10³× cheaper per call than the same LLM making the same decision.** With
pondering, the System One line pushes up toward short chain-of-thought problems;
genuinely independent multi-hop reasoning still needs per-step escalation (gate
back to an LLM) or giving up speed.

### Choosing the split: sweep, don't guess (Engram's U-shape)

Engram formalizes this exact question as an **allocation ratio**
α = memory_capacity / total_capacity, and reports a **U-shaped scaling law**:
too little memory and early layers waste their "effective depth" reconstructing
repetitive patterns from scratch; too much memory and the reasoning stack is
starved. For next-token LM workloads their measured sweet spot is
α* ≈ 0.15–0.25 of capacity in memory — higher for knowledge-heavy domains
(0.20–0.30), lower for reasoning-heavy ones (0.10–0.20).

Implications for us:

- **Compute our own α* — don't adopt theirs.** Our probe mix is known by
  construction (retrieval vs paraphrase vs order vs ambiguity rows). Run
  **iso-parameter and iso-FLOP sweeps** over (table rows × dim) vs (attention
  depth/width) holding the total budget fixed, measured **per probe group**.
  The per-group optimum naturally points toward bigger memory for
  retrieval-heavy mixes and more attention for order-heavy ones — the
  allocation *is* the answer to "how much space for each side".
- **Budget the two axes differently.** Memory is RAM-bound (row fetch is O(1)
  and offloadable; grow the table until held-out n-gram coverage plateaus).
  Computation is latency-bound (grow attention only while the few-seconds
  budget allows). Two independent constraints, not one number (see §5
  allocation).
- **Verify with path ablations.** After training, zero-out each path and
  measure per-probe-group accuracy and logit shift. Clean separation =
  memory carries retrieval rows, attention carries order rows. This serves
  both the allocation measurement and the specialization check (see §7).

Capacity-probe questions (→ §12): where does accuracy/flops saturate? Under a
2GB cap, do we hit "more layers" diminishing returns before "more memory" does?

---

## 6. The core hard problem: encoding arbitrary-length text into a fixed vector

This is where "just use an MLP" stops being trivial. Options:

1. **Lexical bag / n-gram hashing** (A/D above). Fast, order-agnostic, bounded
   size — and with a *learned* table (D) it becomes a genuine memory path that
   surfaces stored associations, not raw counts (the Engram trick).
2. **Fixed-window + pooling**: truncate / chunk the context into a fixed number
   of windows (e.g., 8 windows of 512 tokens), encode each → pool. Handles
   longer context than a single window, still fixed-size.
3. **A freely-scaling encoder** (small transformer with sliding window) → gives
   a fixed-size pooled vector regardless of input length.
4. **No fixed-size requirement for scoring**: if the score function is
   `f(option_i, context)` and it's all read-only over the context, we could make
   the *weights* depend on context length — but that breaks the token-count-
   independent flop budget. Safer to fix the encoding dim.

Open question: how long is "long context," realistically, and do we cap it (e.g.,
8k chars like Jev's playground) with a documented degradation strategy past the
cap (truncate middle, or chunk-and-pool)?

---

## 7. Training data generation with LLMs (the "teacher")

Training data must be synthesized. Basic recipe:

- Define **task templates / domains** → we want breadth: factual QA, trivia,
  intent/routing, binary property checks, story-consistency questions, define/
  synonym, arithmetic-grade recall, classification, "which is the odd one out".
- For each: LLM writes a context (+ question) and N plausible answers with
  exactly one correct one. This is cheap and guarantees labels (the LLM
  generated the story, so it knows the answer — correctness is *constructive*,
  not judged).
- **Teacher signal options**:
  - (a) **Hard label only**: train with plain cross-entropy on the correct index.
  - (b) **Soft teacher probabilities**: ask the LLM for a full distribution over
    the options (or sample it several temperatures / few seeds and average).
    More informative (calibration signal + near-miss info), slightly more
    expensive per sample. Strongly favored for a *calibration*-first model.
  - (c) **RLCD-style objective**: define a reward combining accuracy *and*
    calibration (e.g., penalize confidence-vs-correctness mismatch) and do RL or
    best-of-N over the softmax output. Heavier; probably a later step.
- **Contamination control**: hold out whole task categories from training and
  evaluate on held-out ones to check generalization beyond `(template → answer
  template)` memorization. Also evaluate on *fresh LLM-generated* data not seen
  in training.

Volume: to cover "a humongous variety" of problems with a modest single-pass
model, we may need millions of examples. That's a real cost in generation time —
but it's a **one-time training cost**, amortized over inference that is ~1000×
cheaper than an LLM. This is the core economic argument to keep stating in the
doc.

### Steering facts into memory, logic into attention (bias, don't force)

Where knowledge lands isn't something you can hard-code on a neural net, but
you can *bias* it and then *verify* the division happened:

- **Data tags**: the teacher annotates each row with `reasoning_type`
  (lexical / retrieval / paraphrase / order / ambiguity). Converts "which skill
  does this row need?" from guesswork into a label we can weight and report on.
- **Curriculum**: train retrieval rows first (fast, forces the table to learn
  verbatim storage and association); escalate to paraphrase and order rows —
  attention gradients concentrate there because memory already solves the easy
  half.
- **Auxiliary heads** that specialize each path: a retrieval head on `m`
  ("is the answer stated verbatim in the context?") keeps the table storing
  explicit facts; an order head on `o` ("which event came first?") keeps
  attention encoding sequence. Both cheap, and both double as diagnostics.
- **Structural bias**: memory injected at early layers (Engram puts it at
  layers 1 and 15) + a retriever-shaped readout (`α·match(m, g_i)`) pushes
  recall geometry into the table; later layers compose and reason.
- **Don't force too hard**: explicit stop-gradients or hard gating between
  paths are strong inductive priors that can cap end-to-end accuracy. DeepSeek
  does *not* force the division — it allocates capacity and lets the split
  emerge, then measures. Prefer the allocation knob (§5) and soft biases over
  hard enforcement.

---

## 8. Evaluation & calibration (how we decide "good enough")

- Accuracy (argmax), top-2 coverage.
- **Log-loss / NLL** on the held-out set — the honest measure of "probabilities
  are useful".
- Expected Calibration Error (ECE), Brier score, reliability diagrams.
- **Option-shuffle invariance** test: permute the answer order in eval; posteriors
  must permute identically. (Catches a whole class of degenerate architectures.)
- **Speed/behavioral probes** on the local machine: wall-clock vs context length,
  memory, #params, flops. Publish targets: v0 hash baseline in ms; full deep
  model ≤ a few seconds on 8k chars, CPU.
- Baselines to compare against: a small local LLM (e.g., llama.cpp) forced into a
  multiple-choice harness via constrained JSON/prompting — measure *its* accuracy
  AND latency to quantify the trade.

---

## 9. Design decisions to make (numbered, decision-oriented)

1. **Memory/compute blend**: how much order-insensitive n-gram memory (A/D) vs
   order-sensitive attention (B/D) — the DeepSeek-style mix. Formalize as
   α = memory / total capacity; find α* per probe group via iso-parameter +
   iso-FLOP sweeps (§5). → Start with the n-gram memory path + shared eval
   harness; add the attention path where measurements say it pays.
2. **Fixed N=8 vs variable cardinality**: Jev supports up to 255 via a 2-stage
   scheme. Start at 8? Design the interface so N-variable doesn't break the
   option-order invariant.
3. **N-gram input space**: byte/char n-grams (no tokenizer, OOV-safe, simplest)
   vs compressed-token n-grams (Engram normalizes + dedups the vocabulary ~70%
   before hashing — closer to DeepSeek's proven recipe) vs mini-BPE.
4. **Teacher signal**: hard labels vs soft probabilities vs RLCD. Recommend
   starting with (b) soft teacher, since calibration is the goal.
5. **Teacher model**: cheapest capable local LLM vs frontier API. Mix: mass-produce
   easy data with a small model, curate hard/edge data with a stronger one.
6. **Loss**: plain CE vs KL-to-soft-targets; add temperature-scaling /
   isotonic-regression post-hoc either way.
7. **Context handling**: max length, chunking policy, pooling scheme, and the
   cost of long context on the fixed-size abstraction.
8. **Capacity/latency budget**: ~2GB RAM / few seconds wall-clock on a modest
   CPU-or-mid-GPU machine. Split the spend: reader (context-length cost) vs
   thinker (depth cost); decide params target, fp16 vs int8, and whether to
   allow adaptive weight-shared pondering within the time budget (see §5).
9. **Scope of task breadth** (see Risks, §11): how broad is "humongous variety"
   in *v0*? A single-pass net will not do multi-hop reasoning; define the boundary
   explicitly.
10. **Deployment format**: framework-free inference (numpy/onnx, single-file),
    or a light framework wrapper.

---

## 10. Open questions (need answers as much as decisions)

- **How far can a single-pass / pondering model get on System-One-style
  multiple choice?** This is empirical and it's the first thing to build a probe
  for.
- How big can the n-gram memory table get before more rows stop helping?
  (Engram reports a sparsity/"U-shaped" allocation law: how to divide budget
  between memory and computation.) Measure it within the few-GB budget.
- What is *our* α* (memory/total capacity), per probe group — and does it track
  Engram's ~0.15–0.25 for language modeling? (Compute via iso-parameter +
  iso-FLOP sweeps on our own data, not adoption.)
- Do path ablations show the desired fact→memory / logic→attention split, or
  does one path silently do all the work while the other barely matters?
- Do n-gram tables trained on synthetic decision data *transfer* to unseen
  question styles, or just memorize template co-occurrences? (The order path
  must carry the composition; test on held-out task kinds.)
- What's the *accuracy ceiling* vs context-length and vs answer-list size for a
  fixed compute budget?
- Does option-scoring generalize to **unseen answer strings** (like a retriever),
  or does it overfit per-position/"template clues"? A factorization that encodes
  the option textually should transfer far better than a positional classifier.
- Where do the errors concentrate? (lexical-shortcut fallibility vs reasoning
  beyond capacity) — decides whether to chase more data vs more model.
- How much does *global word-order* matter in the target tasks? (Determines
  whether transformer-C is worth it.)
- Can a well-calibrated shallow model beat an uncalibrated LLM harness on
  *decision quality* (e.g., expected log-loss) even where raw accuracy is close?
  (That's the Jev-style pitch, and the honest benchmark to aim at.)
- Data angle: how many synthetic samples does it take to reach "good" — and is
  the generation cost with an LLM actually worth it vs. labeling real corpora?
- Is there existing related work (hashed-bag-of-words classifiers, distillation
  into shallow nets, "teacher-student" QA compression) we should be reusing
  instead of rediscovering? (Literature search TODO.)
- Evaluation hygiene: how do we keep the eval set clean of generation templates
  and give the model zero chance of "peeking" at template structure?

---

## 11. Risks / known failure modes (state them early)

- **Compositional/multi-hop reasoning** is the line we *cannot* cross with a
  single-pass model. If the target set really needs it, we lose. Scope
  must say "System One problems only," honestly.
- **Order / position artifacts** (the model keys on answer index instead of
  answer content) → mitigated by option-order invariance eval + per-option
  text encoding.
- **Teacher bias**: all synthetic data comes from LLM priors + prompt shapes;
  the model inherits their blind spots. Mitigate with task diversity + held-out
  fresh generation.
- **Data-viability trap**: "millions of examples" might not converge to "good"
  at the chosen capacity in practice → keep a *smaller, carefully scoped* v0
  target.
- **Cost illusion**: generating the training set with an LLM is only worth it if
  the trained shallow model is used *a lot* (or is an order of magnitude cheaper
  per call). State the amortization math in the README once it's measured.
- **"Zero hallucination" overclaim**: we can claim bounded outputs (no free-form
  text) but NOT truthfulness. A false low-uncertainty answer is still wrong; only
  the *format* is safe.
- **Memory-table brittleness**: an n-gram lookup is only as smart as the
  training distribution — cold (rare) n-grams return garbage rows, and the
  network must learn to distrust low-coverage recall. Held-out question styles
  stress this hardest.

---

## 12. Suggested next steps (v0 milestones)

1. Lock the environment (hardware/software), then run a **capacity probe**: find
   the max params × depth that fits ~2GB / a few seconds on it, and lock the
   reader/thinker split + pondering budget in numbers.
2. Baseline harness: pick/slice a public multiple-choice eval set (e.g., a subset
   of ARC, MMLU, RACE) **and** generate a fresh LLM eval set. Measure LLM
   (constrained) accuracy + latency on it as the reference.
3. Implement **v0 = architecture A** end-to-end (encode → score → softmax),
   no framework if possible. Establish flops/latency numbers.
4. Generate a first synthetic training set with a local LLM (hard labels first),
   train v0, measure: accuracy, log-loss, ECE, shuffle-invariance.
5. Compare v0 vs the constrained-LLM baseline on the *same* eval; report the
   accuracy/cost Pareto point, productively.
6. Only then: try B (small transformer), soft-teacher distillation, and
   variable-N as separate experiments driven by what the measurements showed.
## §8b public-eval baseline (honest; milestone 2.1)
Frozen v4 (`model_v4.pt`, 9.16M params), public HuggingFace slice fetched at
inference-time via datasets-server (no cached teacher probs, no option shuffle,
no teacher-prob leakage; variable N=4 options handled via padded collate).
- arc-challenge:    n=400  acc=0.242 logloss=1.420 brier=0.768 ece=0.092
- mmlu-astronomy:   n=400  acc=0.276 logloss=1.487 brier=0.786 ece=0.073
- race-middle:      n=400  acc=0.310 logloss=1.396 brier=0.753 ece=0.065
Latency ~0.03-0.29 ms/row (CUDA), ~val-token entropy dominates.
Interpretation: task model generalizes *negatively* on MC public slice vs its
own synthetic slice (0.91): expected — the v4 decision path is a bag-of-ngram
teacher-margin re-ranker over frozen option text, not a meaning-comprehender.
