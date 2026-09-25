# Notes: Useful public training datasets + no-overlap ingestion plan

Decision: license terms are not a blocker (academic / fair-use research). The
goal is a BALANCED amount of each dataset so no single task-family dominates,
and none of it overlaps the eval holdout sets (arc-challenge, mmlu-astronomy,
race-middle) or data already in the corpus.

## Directly usable MCQ datasets (fit our row schema as-is)

| dataset | HF id | size | row shape | notes |
|---|---|---|---|---|
| QuALITY | `allenai/quality` | 6.7k | context=long article (2k-8k tok), 4-way MC | hard long-document comprehension; fits our `longread`/comprehension goals |
| QASC | `allenai/qasc` | ~8.1k train (9.9k total) | 8-way MC, grade-school science | sentence-composition / multi-hop; keeps 17M-sentence corpus optional |
| MedMCQA | `openlifescienceinformatics/medmcqa` | ~182k train | 4-way medical MC | cap hard; distinct from existing `medqa` (GBaker/MedQA-USMLE) |
| MMLU (non-astronomy) | `cais/mmlu` | 99,842 train / 56 subjects | 4-way MC | take ALL subjects EXCEPT astronomy (eval); keep subject in meta.topic |
| GSM-MC | `geralt-targaryen/MC-Evaluation` | 7,468 train | 4-way MC of GSM8K | math word problems in MC form |
| MATH-MC | `geralt-targaryen/MC-Evaluation` | 7,278 train | 4-way MC of MATH | competition math in MC form |
| Open-Platypus | `garage-bAInd/Open-Platypus` | ~24.9k | mixed reasoning/QA (mostly MC-style) | deduped by source in the paper |

## Chat / free-format datasets (deferred: need an adapter or a chat target)
These don't fit context+options rows directly. Only pursue if we later adopt a
free-response or dialogue training format; otherwise skip.
- `allenai/tulu-3-sft-mixture` (~939k, 18 sources, the open SFT backbone used
  by OLMo-2/Marin)
- `CohereForAI/aya_dataset` (~204k multilingual, Apache-2.0)
- UltraChat-200k, oasst1/oasst2, databricks-dolly-15k
- CCQA (~60M English web QA pairs; pre-training scale)

## Not useful / excluded
- SPIQA (multimodal figure QA; text-only model), MainframeBench (COBOL),
  others on HF MC ledger that are eval-only or non-English.
- ARC, RACE, MMLU-astronomy (already eval holdout), plus any eval split.
- MMLU-CF / RepLiQA (built as benchmarks; eval-only by design).

## No-overlap ingestion plan (order matters)

1. **Fetch** with a `fetch_external.py`-style `_row` converter: tier=`fact`,
   `meta.dataset`=<canonical name>, `meta.source_page`=<article/subject id>,
   `generated_by`="external:<fam>", and `generate` teacher_probs (0.95 gold /
   thin rest). New families: `quality_qa`, `qasc`, `medmcqa`, `mmlu`,
   `gsm_mc`, `math_mc`. Use a NEW file per source:
   `${DATA}/public_new/<fam>.jsonl`.
   - Only fetch `train` splits; never fetch eval/validation-only content.
   - MMLU: skip the astronomy subject at fetch time; keep subject as meta.topic.
2. **Exclude eval identity** at merge time (already implemented):
   `--exclude-public-eval arc-challenge,mmlu-astronomy,race-middle` removes any
   row whose question+options match the eval caches.
3. **Dedup vs existing corpus**: extend `merge_data.py` with
   `--exclude-existing train,eval` (question-hash + options + correct_index)
   so nothing already in corpus (incl. legacy external rows) is re-added
   under a new family. Also rely on existing in-set cross-dedup (gate dup=0).
4. **Route eval without leakage**:
   `--topic-disjoint --eval-fraction 0.15` with meta.topic from source
   (MMLU subject, QuALITY article id, QASC question id, GSM/MATH problem
   source). Keeps train/eval topic-disjoint per family.
5. **Same cleanup flags** as prior rounds: `--drop-gold-low 0.5`,
   `--drop-na-correct`, then `gate_corpus.py --require-var-n`.
6. **Balance caps per source** (avoid task overfitting; corpus train is ~592k):
   - QuALITY ~6.7k, QASC ~8.1k, GSM-MC ~7k, MATH-MC ~7k (their full trains)
   - MedMCQA cap at 30k, MMLU cap at 40k (stratified across subjects),
     Open-Platypus ~24.9k
   - total new ≈ 120k direct rows; NO single source > 40k, no source > ~7% of
     final corpus.
   - If chat format is adopted later, cap each chat source at ~20-30k and re-check
     balance.
7. **Post-merge balance check**: report per-family counts; flag any family above
   the median-by >3x; re-run gate; keep the drop/dup/eval-fraction invariants.
   Optionally add a diversity ratio row to gate_corpus.py.

Optional: the new pools also feed the near_miss / reformulate / noise
transform passes for extra robustness volume.

Status: deferred until the user says go. This note is the pickup point.