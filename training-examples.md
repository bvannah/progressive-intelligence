# Training examples: how problems and answers get encoded and handled

Status: design illustrations (v0).
All probability / hash numbers below are hand-drawn illustrations, not measured
outputs. They exist to make the pipeline concrete, so we can argue about it and
build the harness around it.

---

## 0. The contract (recap)

Inputs (one request):

- `context` — a long blob of text (the thing to read; can be 100s–100k+ chars).
- `options` — exactly **N** candidate answers (N = 8 for v0).

Output (one response):

- `probabilities[8]` — a softmax over the options, plus a calibration-aware
  confidence score. No text is generated, ever.

Internal handling:

1. `question` is concatenated ahead of the passage → `stream = question + "\n" + passage`.
2. `options` are encoded as **separate short byte strings**, not part of the
   stream, so each option can be scored against the same context encoding.
3. All thinking happens in one (or a few) forward pass(es) — never
   auto-regressively.

---

## 1. Training example format (JSON schema)

Each training row is what the harness hands to the trainer (and what an
inference wrapper strips down for serving).

```json
{
  "id": "ex-000001",
  "task": "trivia-fact",
  "difficulty": "easy",

  "context": "The capital of France is Paris. France sits in Western Europe...",
  "question": "What is the capital of France?",

  "options": [
    "London", "Paris", "Berlin", "Madrid",
    "Rome", "Vienna", "Lisbon", "Brussels"
  ],

  "correct_index": 1,

  "teacher_probs": [0.01, 0.93, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005],

  "label_source": {
    "model": "local-llm-8b",
    "temperature": 0.6,
    "n_samples": 4,
    "vote": "mean-of-softmax"
  },

  "meta": {
    "generated_by": "template:capital-of-country",
    "prompt_hash": "a3f9...",
    "holds_out_group": "probe-1"
  }
}
```

Field notes:

- `teacher_probs` is the **training target** (soft). `correct_index` is only for
  evaluation / hard-label experiments. Design decision §9 (item 4): start with
  soft teacher probabilities so calibration is rewarded from day one.
- `holds_out_group` / `task` let us quarantine whole task families (template
  contamination control).
- `options` is always length 8 in v0; shorter real option sets get
  LLM-generated distractors up to 8.

The equivalent raw **inference request** (no teacher fields, no labels):

```json
{
  "context": "The capital of France is Paris. France sits in Western Europe...",
  "question": "What is the capital of France?",
  "options": ["London", "Paris", "Berlin", "Madrid",
              "Rome", "Vienna", "Lisbon", "Brussels"]
}
```

---

## 2. The encoding pipeline (how text → vectors)

The pipeline is shared by context and options. For each byte string:

1. **Normalize** — NFKC, lowercase, collapse whitespace (keeps bytes/strings
   robust, shrinks the effective n-gram space; mirrors Engram's
   CompressedTokenizer idea).
2. **Extract character n-grams** (v0: 2-, 3-, 4-, 5-grams) → the n-gram *bag*.
3. **Hash** each n-gram to a table row id via a collision-resistant hash
   (multi-head, prime moduli, per-layer seed — Engram recipe, §4-D). O(1),
   deterministic, order-insensitive.
4. **Look up** the learned row embeddings (memory path) and pool them
   (mean / weighted mean) → **memory vector** `m`.
5. *(v1)* **Order path** — run the sequence through a small position-aware
   encoder → **order vector** `o`. v0 skips this.
6. Combine → context representation `h` (v0: `h = m`; v1: `h = [m; o]`).

Then the **scorer**, once per option:

```
logit_i = MLP([ h  ⊕  encode(option_i) ])
probs   = softmax(logit_1 .. logit_8)
```

Because every option is scored **independently against the same `h`**, the
output is option-order invariant by construction: permuting `options` only
permutes the logits.

Worked micro-illustration (Example 1 below) — the first ~120 bytes of the
stream, char-3-grams, `2^20` buckets, 256-dim rows:

```
stream: "what is the capital of france?\nThe capital of France is Paris. ..."

3-grams → hash bucket (multi-head, illustrative):
  "wh " -> 0x1A3F shot...
  "hat" -> 0x93D1
  "cap" -> 0x27FA2
  "api" -> 0x00B7
  "of " -> 0x44C9
  "fra" -> 0x7E210
  "ran" -> 0x11E66
  "anc" -> 0x29C44  ← these rows converge (in learned space) on the
  "nce" -> 0x88B10     "capital ↔ city" association cluster
  ...
mean over ~500 row embeddings  →  m ∈ R^256
```

The learned table's job, stated honestly: rows end up encoding co-occurrence
statistics seen in training data ("the capital of X is Y" → the row for `cap`
lies near rows for city names). The memory vector `m` is a sum over the
*rows that are actually present* — so unseen-but-common text already activates
useful geometry, and cold (rare) n-grams default to near-random rows the
scorer learns to discount (§11 risk: memory-table brittleness).

---

## 3. Worked training examples

### 3.1 Retrieval-dominant — "the n-gram memory path shines here"

The answer is stated almost verbatim in the context. The memory path alone
should — in principle — get this right.

```json
{
  "id": "ex-000101",
  "task": "trivia-fact",
  "difficulty": "easy",
  "context": "The capital of France is Paris. France sits in Western Europe, "
             "bordered by Belgium, Germany, Luxembourg, Switzerland, Italy, "
             "Spain, and a small strip of the Netherlands. Its official "
             "language is French.",
  "question": "What is the capital of France?",
  "options": ["London", "Paris", "Berlin", "Madrid",
              "Rome", "Vienna", "Lisbon", "Brussels"],
  "correct_index": 1,
  "teacher_probs": [0.005, 0.97, 0.005, 0.005, 0.005, 0.005, 0.0025, 0.0025],
  "label_source": { "model": "local-llm-8b", "temperature": 0.3, "n_samples": 3 },
  "meta": { "generated_by": "template:capital-of-country", "holds_out_group": "probe-1" }
}
```

Pipeline trace:

- `stream = "what is the capital of france?\nThe capital of France is Paris. ..."`
- n-grams: `cap`, `api`, `of `, `fra`, `ran`, `anc`, `nce`, `par`, `ari`, ... →
  rows that (in the learned table) are strongly weighted toward "Paris".
- `h = m` — the pooled memory vector already points near the "Paris" cluster.
- Scorer: `logit(Paris) ≈ 4.3`, all others ≤ −2 → softmax `[≈0, 0.97, ≈0, ...]`.
- Confidence (max prob) ≈ 0.97 → a gate could act automatically.

What this tests:

- Does the memory path *alone* solve strict lexical-association cases?
  If not, the table isn't learning co-occurrence — fix the table, not the model.

---

### 3.2 Semantic / paraphrase — "memory plus meaning"

No option string appears in the context; the right answer matches *content*,
not word overlap. A raw bag-overlap model would fail here; the learned rows must
carry semantic proximity.

```json
{
  "id": "ex-000207",
  "task": "review-intent",
  "difficulty": "medium",
  "context": "Battery drains in under two hours on a full charge, and the "
             "screen cracked on the first drop even with the included case. "
             "Support did not answer my email. I want to return it.",
  "question": "What is the customer's primary objective?",
  "options": [
    "Request a refund for the device",
    "Get a replacement battery",
    "Complain about screen durability",
    "Ask for tech support hours",
    "Request a firmware update",
    "Cancel their account",
    "Leave a negative store review",
    "Ask for shipping costs back"
  ],
  "correct_index": 0,
  "teacher_probs": [0.82, 0.06, 0.05, 0.02, 0.01, 0.01, 0.01, 0.02],
  "label_source": { "model": "local-llm-8b", "temperature": 0.6, "n_samples": 4 },
  "meta": { "generated_by": "template:review-intent", "holds_out_group": "probe-2" }
}
```

Pipeline trace:

- Options share few n-grams with the context (`refund` ↔ `return`, `battery` ↔
  `battery`, ... only partial overlap).
- The useful signal is *association*, not overlap: rows for `return`, `want`,
  `refund`-adjacent words live near each other from training data (routing
  templates), so `m` and the option encodings land in nearby regions.
- The scorer MLP learns that nearness ≈ "this option matches the intent".
- Output softmax: `[0.82, 0.06, 0.05, ...]` — high, though not one-hot.

What this tests:

- **Transfer to paraphrases** — the hard open question (§10/§11). We must
  measure whether the memory path transfers to wording the teacher never
  generated, or whether it only memorizes template co-occurrences.
- Distractor hygiene: "Get a replacement battery" is a *plausible wrong* answer;
  the teacher should watermark such rows to watch calibration under near-misses.

---

### 3.3 Order-sensitive / compositional — "the attention path's domain"

Here the *order* of events decides the answer. A bag of n-grams cannot see
long-range order; `{delivered, then, refund}` vs `{refund, then, delivered}`
produce nearly identical memory vectors. This is the compelling case for the
v1 order path (§4-D), and it should appear in every eval probe.

```json
{
  "id": "ex-000312",
  "task": "sequence-disambiguation",
  "difficulty": "hard",
  "context": "The package was delivered on Tuesday at 4pm. The customer "
             "noticed the damage on Wednesday morning and requested a refund. "
             "The courier picked the item up on Friday.",
  "question": "What happened immediately after the customer noticed the damage?",
  "options": [
    "Refund was requested",
    "Package was delivered",
    "Courier picked up the item",
    "Support was called",
    "A replacement was shipped",
    "The order was cancelled",
    "The refund was approved",
    "The box was inspected"
  ],
  "correct_index": 0,
  "teacher_probs": [0.88, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01],
  "label_source": { "model": "local-llm-8b", "temperature": 0.5, "n_samples": 4 },
  "meta": { "generated_by": "template:sequence-timeline", "holds_out_group": "probe-3" }
}
```

Pipeline trace (v0, memory-only) — expect failure:

- `m` mixes `delivered`, `refund`, `picked up` roughly uniformly → logits
  flat; the model guesses. **This is why v0 alone is not "good answers to
  complex problems" — it's the baseline that justifies the orders path.**

Pipeline trace (v1, memory + order path):

- The order path reads the sequence positionally: `noticed the damage`
  (position ~17) is immediately followed by `requested a refund`.
- `o` encodes "the event after X is Y"; scorer fuses `[m; o]` and picks
  "Refund was requested".
- Order path cost scales with sequence length — keep it a small/linear-attn
  encoder so long contexts stay within the §5 budget.

What this tests:

- Whether the **order path earns its FLOPs** on our eval set (the core
  measurement in design decision #1).
- Objective teachable signpost: permute `delivered` and `refund` clauses,
  re-run — the correct option must flip. If the model is order-blind, this
  test exposes it immediately.

---

### 3.4 Ambiguous — "why we train to soft probabilities"

Deliberately underdetermined. There is no unique right answer; a good model
says so with a spread distribution. This row keeps calibration honest and is a
natural place for the confidence gate to escalate to a human.

```json
{
  "id": "ex-000410",
  "task": "ambiguity-probe",
  "difficulty": "hard",
  "context": "The front door was left unlocked again. The keys were found on "
             "the kitchen counter by the coffee maker.",
  "question": "Who most likely left the keys on the counter?",
  "options": [
    "The owner, in a hurry", "The cleaning service",
    "The previous tenant",     "A visitor",
    "A repair technician",     "The delivery driver",
    "A family member",         "The pet sitter"
  ],
  "correct_index": null,
  "teacher_probs": [0.34, 0.14, 0.09, 0.19, 0.08, 0.05, 0.08, 0.03],
  "label_source": { "model": "frontier-llm", "temperature": 0.9, "n_samples": 12,
                    "vote": "mean-of-softmax" },
  "meta": { "generated_by": "template:underdetermined", "holds_out_group": "probe-4" }
}
```

Pipeline trace:

- `m` is diffuse (no strong association cluster fires). The scorer naturally
  outputs a flat-ish distribution.
- Note `correct_index: null` — the harness must support *no* hard label. Loss is
  still well-defined (KL to teacher_probs).
- A model trained only on one-hot labels would collapse this to a confident
  wrong pick; soft targets + a calibration loss (RLCD-style, §7) keep the
  spread.

What this tests:

- **Calibration**: does the model stay honest when it can't know? (ECE / Brier
  on `probe-4`-type rows specifically.)
- **Gate behavior**: at max-prob ≈ 0.34 the deployed config should route to a
  human rather than act.

---

### 3.5 Same problem, permuted options — "option-order invariance"

Exactly Example 1, but the answer list is shuffled. The output probabilities
must shuffle identically. This is enforced by architecture (§2 per-option
scoring) but must be *verified* during training and eval.

```json
{
  "id": "ex-000102",
  "task": "trivia-fact",
  "difficulty": "easy",
  "context": "The capital of France is Paris. France sits in Western Europe, "
             "bordered by Belgium, Germany, Luxembourg, Switzerland, Italy, "
             "Spain, and a small strip of the Netherlands. Its official "
             "language is French.",
  "question": "What is the capital of France?",
  "options": ["Brussels", "Rome", "Madrid", "Paris",
              "Vienna", "Berlin", "Lisbon", "London"],
  "correct_index": 3,
  "teacher_probs": [0.0025, 0.0025, 0.005, 0.97, 0.005, 0.005, 0.005, 0.005],
  "label_source": { "model": "local-llm-8b", "temperature": 0.3, "n_samples": 3 },
  "meta": { "generated_by": "template:capital-of-country", "holds_out_group": "probe-1",
            "permuted_from": "ex-000101" }
}
```

Pipeline trace:

- `encode(Paris)` is identical to before → `logit(Paris)` unchanged (≈ 4.3);
  the context vector `h` is unchanged.
- Every other option's logit is equally invariant, so
  `probs = permute(probs_of_ex_000101)` exactly.

What this tests:

- **Zero plug-in leakage**: the model must never key on answer *position*.
  A classifier that learns "option at index 1 wins when the question mentions
  'capital'" would fail this row — and that failure must be visible in the
  eval harness every run.

---

## 4. What this set is designed to test (eval mapping)

| Example | What it stresses | Eval hook |
|---|---|---|
| 3.1 | Memory-only lexical retrieval | fast-path latency ceiling; table quality |
| 3.2 | Semantic/paraphrase transfer | held-out wording families (probe-2) |
| 3.3 | Order/composition (v1) | permute-clause invariance test |
| 3.4 | Calibration / ambiguity | ECE + gate thresholds (probe-4) |
| 3.5 | Option-order invariance | permutation check, every run |

The five rows are the seed set for the harness: the same five shapes should be
re-generated at scale by the teacher LLM (thousands per family), with the same
evals computed on held-out versions.

---

## 5. Open points these examples surface (small, concrete)

- Python/JSON for rows, or binary (numpy memmap) once we're at millions of rows?
  (Schema above is JSON; the trainer may prefer a packed format.)
- Should `question` be a first-class field (as drawn) or always folded into
  `context`? The model only ever consumes the concatenated stream — but keeping
  it separate is better bookkeeping.
- `teacher_probs` vs hard label for `loss`: §9 (item 6) recommends KL to soft
  targets; example 3.4 requires it.
- Do we ever want `N < 8` in serving while training at exactly 8? (Interface
  freedom; see design decision #2 in ideation.md.)