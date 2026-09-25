# Idea (not built yet): Wikipedia-passage Q&A with comprehension + knowledge halves

Read random portions of random Wikipedia articles, and have an LLM generate a
question + answers based on the passage. Two halves:

1. **Comprehension** — answer is strictly entailed by the passage.
   → ALREADY EXISTS as `wiki_fact_llm.py` (B8), family `wiki_fact`, fact tier.
   Random chunks (180–700 chars, <=4 sentences) of random wiki articles; LLM
   emits 3–5 MCQs per chunk; hard-verified via `meta.support_span` = a verbatim
   contiguous quote from the passage that establishes the answer. So far 200
   rows generated, 32 rejected.

2. **Knowledge** — answer is NOT in the passage / omitted from context (a
   real-world-knowledge question; the passage is just a prompt surface).
   → NOT BUILT. Could be added as a `wiki_know` mode in `wiki_fact_llm.py`.
   Verification ideas:
   - assert the correct answer does NOT appear verbatim (or near-verbatim)
     anywhere in the given passage (string/substring check);
   - no `support_span` required inside the given passage; optionally allow
     `support_span` from a *different* article as provenance;
   - keep fact tier + provenance (source page / support span elsewhere);
   - reject rows where any of the derived/gold answer tokens are spans of the
     passage.

Closest existing analogues (not this idea):
- `knowledge` family: templated statement-completion from a fixed fact bank,
  no passage (`synthetic.py` `gen_knowledge_tmpl`).
- `wikibox` / `wikicat`: procedural recall from infobox/category facts.

Status: deferred. See this file to pick it up later.