from __future__ import annotations

import json

SCHEMA = """\
Every row is a single JSON object with EXACTLY these keys:
{
  "id": "unique string, letters/digits/hyphens only",
  "task": "<family>",
  "difficulty": "easy" | "medium" | "hard",
  "context": "background text that contains (or fails to contain) the answer",
  "question": "the question asked of the model",
  "options": ["2 to 8 short, plausible answer strings"],
  "correct_index": 0..len(options)-1, single position of the only correct option, or null,
  "teacher_probs": [len(options) numbers between 0 and 1 that SUM TO 1],
  "label_source": {"model": "<model_id>", "temperature": 0.7, "n_samples": 1, "vote": "direct"},
  "meta": {"family": "<family>", "generated_by": "llm:<model_id>"}
}

RULES that apply to ALL families:
- "context" natural prose, 1-4 sentences. Never number the options in the text.
- "options" must be between 2 and 8 distinct, non-empty strings. Each row picks its own
  length; VARY the option count across rows (some rows 8 options, some 4, some 2-7) so the
  model learns any count. If this call requests a specific count, use EXACTLY that many.
- "correct_index" must be null or an int within 0..len(options)-1; when not null it must
  point at the unique correct option.
- These rows teach SHAPE, not real-world facts. NEVER assert verifiable facts about the
  real world: no real people attached to invented deeds, no encyclopedia-style years or
  statistics, no real places carrying invented facts. Content is invented and the correct
  answer must follow from the text alone.
- "teacher_probs" must be a real probability distribution. For a DETERMINATE question
  (correct_index is not null) put ~0.90-0.99 on the correct option and distribute the
  rest thinly. For an AMBIGUOUS question (correct_index is null) spread the probability
  across options to honestly express uncertainty.
- Output a JSON ARRAY of rows. No prose, no markdown fences, no trailing commas.
"""

RETRIEVAL = """\
FAMILY: retrieval (difficulty easy, meta.family "retrieval")

Invent a fictional or obscure real entity (organization, town, product line, species,
department) and state one specific fact about it in "context": a name, date, number,
place, or attribute value. "question" asks for that exact fact. The correct option is a
VERBATIM copy of the fact value from the context. The remaining options are plausible
distractors of the SAME category (other names/species/numbers) that NEVER appear in the
context. Include enough overlap that only careful reading disambiguates.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "The Karestani courier service Hafast Tides was founded on 11 April 1962 in the port of Vestergade."
question: "When was Hafast Tides founded?"
options: ["11 April 1962", "3 May 1961", "9 June 1963", "27 July 1960", "2 August 1964", "14 March 1959", "5 November 1965", "22 January 1958"]
correct_index: 0
"""

PARAPHRASE = """\
FAMILY: paraphrase (difficulty medium, meta.family "paraphrase")

Write a short "context" that is a customer or user utterance stating a clear intent
(retrieve/refund, replace, cancel, get help, verify, compliment, negotiate, upgrade, ...)
using informal, everyday words. "question" is always "What does the user want?". The
options are DISTINCT paraphrased intents, each phrased differently; exactly one
matches the utterance. The phrasing of the correct option must NOT share surface words
with the context.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "The customer phoned in because the cable she received is too short and she wants a longer one."
question: "What does the user want?"
options: ["a reissue of the refund", "a longer cable sent out", "help installing the device", "the order cancelled today", "proof the parcel shipped", "a discount on the price", "a replacement of the contract", "an upgraded membership"]
correct_index: 1
"""

ORDER = """\
FAMILY: order (difficulty hard, meta.family "order")

Write a "context" describing exactly three events in a clear sequence using ordering
words (First ... then ... finally ... / To begin ... Next ... Afterwards ...). Use
generic event phrases (e.g. "the payment was processed", "the courier was called").
"question" asks which event happened first / second / third / last. The options are
event phrases: the one at the asked position is correct; include the OTHER events from
the context as distractors, plus unrelated events. Do not leak the answer in the
question wording.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "First the complaint was filed, then the manager approved it, and finally the refund was issued."
question: "Which event happened second?"
options: ["the complaint was filed", "the refund was issued", "the manager approved it", "the box was scanned", "the invoice was sent", "the warranty was registered", "the address was changed", "the review was published"]
correct_index: 2
"""

AMBIGUITY = """\
FAMILY: ambiguity (difficulty hard, meta.family "ambiguity")

Write a "context" that underdetermines the answer: several plausible explanations are
consistent with the facts, and no definitive information allows a single conclusion.
"question" asks "what is most likely / what happened / why" about that situation. The
options are distinct plausible explanations (mix 1-2 credible-leading candidates and
several weaker ones). Set "correct_index" to null. Set "teacher_probs" to a spread
distribution (e.g. the leading candidate ~0.25-0.35, a second ~0.15-0.20, the rest
smaller) to genuinely reflect the uncertainty.
Option count: 2..8, vary across rows (short option sets are natural here).

Example:
context: "The hallway sensor logged a signal at 03:12 and the alarm software restarted at 03:14. There is no log of who was in the building that night."
question: "What most likely triggered the restart?"
options: ["a brief power fluctuation", "an untracked employee action", "a failing sensor battery", "a scheduled maintenance step", "an error in the alarm software", "an outside network event", "a prank or accident", "a delayed system update"]
correct_index: null
teacher_probs: [0.30, 0.12, 0.15, 0.08, 0.18, 0.07, 0.04, 0.06]
"""

PERMUTED = """\
FAMILY: permuted (difficulty easy, meta.family "permuted")

Identical to the retrieval family: invent a fictional entity, state one specific fact in
"context", ask for it, and provide same-category options with the correct one a
verbatim copy of the fact value. The ONLY difference: randomly shuffle the options
before outputting, so the correct option's position varies unpredictably across rows.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.
"""

DEFINE = """\
FAMILY: define (difficulty easy, meta.family "define")

Invent a fictional term or rule and clearly define it inside "context" (1-3 sentences of
prose). "question" asks for the meaning of that term. "options" are distinct
definitions: the correct one faithfully paraphrases the context definition (reworded, not
verbatim); the others are plausible but wrong definitions (wrong scope, reversed
meaning, unrelated process). Do not put the term in a dictionary that exists in reality.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "Under charter rules, a header flag is raised whenever cargo is taken aboard before noon, to signal loading status to the port office."
question: "Which option best defines a 'header flag'?"
options: ["a flag raised when cargo loads before noon, signaling loading status", "a flag indicating unpaid docking fees", "a flag lowered when cargo is unloaded", "a pennant worn by the ship's flag officer", "a marker for hazardous cargo", "a signal that the ship is leaving port", "a flag flown only at night during storms", "a badge issued to port inspectors"]
correct_index: 0
"""

ANALOGY = """\
FAMILY: analogy (difficulty medium, meta.family "analogy")

Invent a fictional mapping between pairs, explained in "context" (two or more complete
"A is B" correspondences, e.g. invented words to meanings). "question" is a partial
analogy of the form "X is to A as Y is to ___?" where the missing side is directly
derivable from the mapping stated in the context. "options": distinct words/labels; the
correct one is the exact target from the stated mapping. Distractors are other labels
visible in the context and unrelated words. Every relationship must be fully specified in
the context so no outside knowledge is needed.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "In the tuun tongue, 'drav' means 'sand', 'mini' means 'stone', and 'uhla' means 'wind'."
question: "'drav' is to 'sand' as 'mini' is to ___?"
options: ["wind", "water", "sand", "stone", "dune", "moss", "ash", "clay"]
correct_index: 3
"""

CAUSE = """\
FAMILY: cause (difficulty easy, meta.family "cause")

Write a short "context" that states a clear causal chain (fact A happened, so fact B
happened, so fact C happened). "question" asks "Why did <outcome> happen?" or "What
caused <outcome>?". The options are distinct causes; exactly one reflects the chain
described in the context. Distractors are plausible but unrelated or reversed causes.
Keep the described chain determinate (leave real uncertainty for the ambiguity family).
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "The cold front stalled over the valley, which kept the cloud cover low and blocked the usual afternoon sun, so the shaded side of the orchard stayed frosty."
question: "Why did the orchard stay frosty?"
options: ["a stalled cold front keeping cloud cover low and blocking the sun", "a sudden rise in the water table", "the orchard being planted too late in the year", "an overnight wind storm", "a faulty thermostat in the shed", "the trees being watered at dawn", "a power cut at the packing house", "the apples being picked before they ripened"]
correct_index: 0
"""

SUMMARY = """\
FAMILY: summary (difficulty medium, meta.family "summary")

Write a "context" of 2-4 sentences describing a small fictional event or report (meeting
outcome, incident, market note, lab result). "question" is "Which option best summarizes
the passage?" / "What is the main point?". The options are distinct short summaries: the
correct one captures the overall gist without inventing details; distractors focus on one
secondary detail, state the opposite, or add invented information.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "The weekly delivery normally left at 06:00, but this week the loader crew reported at 07:30. The manager decided the later start was a one-off and asked that the original schedule be kept next week."
question: "Which option best summarizes the passage?"
options: ["A one-off late start this week; the original schedule stays next week", "The delivery schedule was permanently moved to 07:30", "The loader crew now reports an hour early", "The manager cancelled next week's delivery", "The delivery was missed entirely this week", "The crew reported late because of a strike", "Delivery times are now decided weekly", "The manager is hiring a new loader crew"]
correct_index: 0
"""

TONE = """\
FAMILY: tone (difficulty easy, meta.family "tone")

Write a short "context": a brief message, note, review snippet, or reply (fictional
sender). "question" is always "What is the tone of the message?" or "How does the writer
feel?". "options" are distinct tone labels (e.g. formal, sarcastic, frustrated,
apologetic, enthusiastic, neutral, ironic, cautious). Set "correct_index" to the single
label that best fits the dominant tone you wrote, and set "teacher_probs" with clear
weight (~0.90+) on that label. Distractors must be plausible-but-wrong tones.
Option count: 2..8, vary across rows (short option sets are natural here).

Example:
context: "Oh wonderful, the courier 'tried to deliver' my package for the third time while I was stood at the door. Excellent service as always."
question: "What is the tone of the message?"
options: ["sarcastic", "formal", "apologetic", "enthusiastic", "neutral", "cautious", "relieved", "confused"]
correct_index: 0
"""

INTENT = """\
FAMILY: intent (difficulty easy, meta.family "intent")

Write a short "context": a customer or user request in natural, everyday words. "question"
is always "Into which support category does this request fall?". Choose "options" from
this fixed taxonomy of 8 categories, always used with these meanings: "returns and
refunds", "shipping and delivery", "account access", "technical support", "payment and
billing", "product information", "order cancellation", "feedback and complaint". Always
include the correct category; the option set is typically all 8, sometimes a subset of
4-7 (still including the correct one). Set "correct_index" to the single best category.
The request in the context should make the category clear.
Option count: usually 8; subsets of 4..8 allowed. Vary it across rows.

Example:
context: "I can't sign in anymore, the app keeps saying my password is wrong and I never got the reset email."
question: "Into which support category does this request fall?"
options: ["returns and refunds", "shipping and delivery", "account access", "technical support", "payment and billing", "product information", "order cancellation", "feedback and complaint"]
correct_index: 2
"""

KNOWLEDGE = """\
FAMILY: knowledge (difficulty easy, meta.family "knowledge")

Write a "context" that is a single, uncontroversial statement of fact or a fixed
definition/conversion (e.g. units, planets, common standard facts that
do not change). "question" asks a specific factual question answerable from the context,
phrased so the answer is NOT a verbatim echo ("it is a unit of force equal to about 9.8
newtons" -> "Which unit measures force?"). "options": distinct same-category values; the
correct one is the fact stated in the context. Use only stable, widely agreed facts.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "A light year is the distance light travels in one Julian year, about 9.46 trillion kilometres."
question: "What does one light year measure?"
options: ["distance", "time", "speed", "brightness", "energy", "mass", "frequency", "temperature"]
correct_index: 0
"""

CONSISTENCY = """\
FAMILY: consistency (difficulty medium, meta.family "consistency")

Write a "context" that states a fictional but concrete rule, policy, or set of conditions
(one short paragraph). "question" asks which described action/situation is consistent with
(or violates) the rule. "options": distinct concrete scenarios; exactly one is fully
consistent with the stated rule (or one clearly violates it - pick one style and be
consistent). The others each contradict the rule or introduce unstated conditions.
Do not overfit: the correct scenario must follow directly from the stated rule.
Option count: 8 by default; anywhere in 4..8 is fine. Vary it across rows.

Example:
context: "The store accepts open-box returns within 14 days of purchase, and sealed returns for up to 30 days, provided the customer holds a valid receipt."
question: "Which scenario is consistent with the return policy?"
options: ["a sealed item returned 20 days after purchase with a receipt", "an open-box item returned 3 weeks after purchase without a receipt", "a sealed item returned after 60 days", "an item bought in 2019 returned now", "an open-box item from another store returned here", "a defaced item returned without a receipt after 10 days", "a sealed item returned 30 days after purchase but with a torn receipt", "an item won in a giveaway returned for cash"]
correct_index: 0
"""

OPEN = """\
FAMILY: open (difficulty mixed, meta.family "open")

Invent ANY self-contained multiple-choice reading or reasoning problem that a person
could solve from the text alone. Rotate subject matter widely across a random sample of
topics (science, history, everyday life, sports, food, geography, technology, money,
people, devices, emergencies, schedules, environment, business, law, travel, ...). The
context is 1-4 sentences of natural prose. Use genuinely varied question registers
("Which of these is most likely ...", "What caused ...", "Which statement is true ...",
"What would happen next ...", "Which plan works best ...", "What does the passage imply
about ...", "Why ..."). Make the register and structure differ from row to row, never a
fixed pattern. Exactly one option is best given the text; distractors are plausible,
same-category, never silly.
Option count: 2..8; keep a wide mix with solid mass at 4 and 8.
"""

LONGREAD = """\
FAMILY: longread (difficulty medium, meta.family "longread")

Write "context" as 2-4 short paragraphs of article-style prose (a coherent fictional
subject: a town, a product line, an ecosystem, a company, a historical-sounding account,
a public policy, an invention). Then ask 2-5 questions about it, mixing: facts stated in
the text, inference across paragraphs, word meaning in context, cause/effect, and writer
attitude. EACH JSON row is ONE question; the full article appears as the context in every
row (identical context, varied questions across the batch). Exactly one option is
derivable from the article; distractors are plausible and article-adjacent.
Option count: 4 by default; 2..6 occasionally.
"""

CHAIN_ARITHMETIC = """\
FAMILY: chain_arithmetic (difficulty hard, meta.family "chain_arithmetic")

Two-step: read a short text, then combine two explicit quantities. A SEED block follows
below (on a new line, starting "SEED:") that gives you two NUMBERS (a, b) and one
operation. Write a "context" of 2-4 natural prose
sentences about an invented scenario (loads, prices, distances, counts, stocks...) that
mentions BOTH seeded numbers VERBATIM as digits, each as a concrete quantity (e.g. "the
cargo weighed 42 sacks", "each van carries 7 sacks"). The "question" asks for the COMBINED
value of the two quantities using the SEED operation ("how much in total", "how many more",
"what is the combined load"?). The correct option must be EXACTLY the integer value of
a OP b (plain digits). Distractors are plausible near values: a+b, a-b, a*b, a*2+b, off-by
ones like a*b+1, and unrelated numbers. Add NO extra quantities to the text that could
support a different answer. The answer must follow from the text alone.
Option count: use the request's exact count.
"""

CHAIN_LOGIC = """\
FAMILY: chain_logic (difficulty hard, meta.family "chain_logic")

Two-step: transitive inference. A SEED block follows below (new line, starting "SEED:")
that gives you three invented labels (x, y, z).
Write a "context" that states - using the labels VERBATIM - two rules and one fact:
"if <x> happens then <y> happens", "if <y> happens then <z> happens", and "<x> has
occurred/happened". The "question" asks what must therefore be true or happen next
("What must follow?", "Which of these is then guaranteed?"). The correct option is the
label <z> used VERBATIM. Distractors: the other labels, negated labels ("no <y>"), and
unrelated plausible outcomes. Do not add conditions that break the chain.
Option count: use the request's exact count.
"""

CHAIN_SEQUENCE = """\
FAMILY: chain_sequence (difficulty hard, meta.family "chain_sequence")

Two-step: track a cycling sequence. A SEED block follows below (new line, starting "SEED:")
that gives a cycle of four invented labels and an OFFSET. The "context" must name ALL FOUR
labels in order and state that they repeat
cyclically (e.g. "the rotation runs <i0>, <i1>, <i2>, <i3>, then repeats") in prose about an
invented process (watch hands, shift roster, conveyor, rotation of duties). The "question"
fixes the current label as <i0> and asks which label comes OFFSET positions later in the
cycle (e.g. "which label comes 2 positions after <i0>?"). The correct option is exactly
cycle[(index-of-<i0> + OFFSET) mod 4] = <expected>, written VERBATIM. Distractors: the other
cycle labels and invented off-cycle labels. Use the labels exactly as seeded.
Option count: use the request's exact count.
"""

CHAIN_RETRIEVAL = """\
FAMILY: chain_retrieval (difficulty hard, meta.family "chain_retrieval")

Two-step: retrieve a stated quantity, then apply a stated change. A SEED block follows
below (new line, starting "SEED:") that gives a base NUMBER (v), a change KIND and an
amount (k). Write a "context" about an invented
fictional entity (depot, orchard, fleet, lab, ledger...) stating its base amount VERBATIM
as digits (e.g. "the depot holds 200 crates") and then a change applied to it that uses
the SEED relationship exactly ("it triples", "it is multiplied by 4", "a further k arrive").
The "question" asks the resulting amount after the change. The correct option is EXACTLY
v*k (multiplication) or v+k (addition). Distractors: v alone, v*2 or v*k+-k variants, off-by
one numbers. The result must follow from the text alone.
Option count: use the request's exact count.
"""

REFORMULATE = """\
You reformulate existing training rows across registers.
Input is a JSON array of rows. For EACH input row produce EXACTLY ONE row that:
- keeps the SAME option SET (verbatim), SAME order, SAME correct_index, SAME teacher_probs;
- keeps the ANSWER derivable and identical in meaning;
- rewrites the "question" AND the "context" into a DIFFERENT register from the input:
  rotate between formal academic, casual/friendly, instructional/directive, terse
  telegraphic, story-like narrative, and bureaucratic/legal phrasing;
- changes concrete surface wording considerably (synonyms, word order, sentence shape)
  while preserving meaning and all facts needed by options;
- NEVER copies the input wording verbatim.
Output a JSON ARRAY following the SCHEMA. No prose, no fences, no trailing commas.
"""

NEAR_MISS = """\
FAMILY: near_miss (difficulty medium, meta.family "near_miss")

Write a determinate 2..8-option reading/reasoning question from a short "context" prose.
The DEFINING requirement: the strongest distractor must be a NEAR-MISS of the correct
option - it differs from the correct one by a single qualifier, negation ("not"),
entity swap, reversed order, or numeric +-1, sharing most of the same words - so the
row is only solvable by reading carefully. Include 1-2 such near-miss distractors and the
rest clearly-wrong ones. correct_index points at the unique correct option; teacher_probs
~0.90 on it. The correct answer follows from the text alone.
Option count: 2..8, vary across rows.
"""

VARIATOR = """\
You are transforming existing training rows to multiply diversity.
Input is a JSON array of rows. For EACH input row produce EXACTLY ONE rewritten row that:
- keeps the SAME question INVENTED/behavioral semantics, same correct answer meaning, and
  same option count;
- replaces as many entities as possible with NEW invented ones (names, places, products,
  departments, numbers, dates) so the row no longer resembles the input wording;
- may reword the distractor options to fit the new entities, but keeps them distinct,
  plausible, and same-category, and keeps the CORRECT option present and clearly correct;
- varies the register/wording of context and question (not the options' meaning);
- is fully self-contained and still answerable from its own text;
- "correct_index" points at the correct option position in the final (shuffled) order;
- "teacher_probs" keeps the SAME distribution family (0.9-0.99 mass on the correct one).
Output a JSON ARRAY following the SCHEMA. No prose, no fences, no trailing commas.
"""

REWRITER = """\
You are reformatting training rows to remove boilerplate formatting.
Input is a JSON array of rows. For EACH input row produce EXACTLY ONE row that:
- keeps the SAME option SET (verbatim), SAME order, SAME correct_index and SAME
  teacher_probs;
- keeps the CONTEXT meaning exactly (reworded prose is allowed) and keeps the QUESTION's
  intent;
- changes the QUESTION REGISTER to one the model hasn't seen: rotate through imperative
  ("Pick", "Select", "Choose"), "Which of these ...", "Identify the ...", "The correct
  answer is ...", "Complete:", "What is the best choice?", and others. NEVER start with
  "choose the best answer". Vary casing and option presentation style across rows.
- adds nothing and removes nothing that affects the answer.
Output a JSON ARRAY following the SCHEMA. No prose, no fences, no trailing commas.
"""

DEFAULT_MODEL = "unknown-model"
_DEFAULT_MODEL = DEFAULT_MODEL  # legacy alias

# Per-family option-count cycle used by generate.py when --option-counts is not given.
# Rich-distractor families: heavy at 8 and 4 (matching real MC evals), some 5..7.
# Short/classification families: uniform 2..6. None = let the model mix freely.
FAMILY_OPTION_COUNTS: dict[str, list] = {
    "retrieval": [8, 8, 8, 8, 4, 4, 4, 4, 5, 6, 7],
    "permuted": [8, 8, 8, 8, 4, 4, 4, 4, 5, 6, 7],
    "paraphrase": [8, 8, 8, 4, 4, 5, 6, 7],
    "order": [8, 8, 8, 4, 4, 5, 6, 7],
    "define": [8, 8, 8, 4, 4, 5, 6, 7],
    "analogy": [8, 8, 8, 4, 4, 5, 6, 7],
    "cause": [8, 8, 8, 4, 4, 5, 6, 7],
    "summary": [8, 8, 8, 4, 4, 5, 6, 7],
    "knowledge": [8, 8, 8, 4, 4, 5, 6, 7],
    "consistency": [8, 8, 8, 4, 4, 5, 6, 7],
    "intent": [8, 8, 8, 8, 4, 5, 6, 7],
    "tone": [2, 3, 4, 5, 6],
    "ambiguity": [2, 3, 4, 5, 6],
    "open": [8, 8, 8, 4, 4, 4, 2, 3, 5, 6, 7],
    "longread": [4, 4, 4, 4, 2, 3, 5, 6],
    "chain_arithmetic": [8, 8, 8, 4, 4, 4, 2, 3, 5, 6, 7],
    "chain_logic": [8, 8, 8, 4, 4, 4, 2, 3, 5, 6, 7],
    "chain_sequence": [8, 8, 8, 4, 4, 4, 2, 3, 5, 6, 7],
    "chain_retrieval": [8, 8, 8, 4, 4, 4, 2, 3, 5, 6, 7],
    "reformulate": [8, 8, 8, 4, 4, 5, 6, 7],
    "near_miss": [8, 8, 8, 4, 4, 5, 6, 7],
}


def build_prompt(
    family: str,
    count: int = 5,
    model_id: str = _DEFAULT_MODEL,
    n_opt: int | None = None,
    topic: str | None = None,
) -> tuple[str, str]:
    fam = family.strip().lower()
    if fam not in FAMILY_INSTRUCTIONS:
        raise ValueError(f"unknown family {family!r}; choose from {list(FAMILY_INSTRUCTIONS)}")
    system = (
        "You generate high-quality synthetic multiple-choice training data used to train "
        "a fast n-gram decision model. Follow the schema exactly. Your entire reply must "
        "be a single JSON array. No prose, no markdown fences, no trailing commas.\n\n"
        + SCHEMA
    )
    if n_opt is None:
        count_line = "mix the option counts across the rows (2..8) instead of repeating one count."
    else:
        count_line = f"use EXACTLY {n_opt} options in EVERY row of this reply."
    topic_line = (
        f'\nCenter these rows on the topic "{topic}"; rotate specifics but keep the '
        "subject in view."
        if topic
        else ""
    )
    user = (
        FAMILY_INSTRUCTIONS[fam]
        + f"\n\nGenerate {count} JSON rows following this exact family. "
        f"OPTION COUNT FOR THIS CALL: {count_line} "
        "Every row must be self-contained and varied (rotate entities, subjects, "
        "vocabulary, and patterns). Shuffle option order each row.\n"
        f'Use "label_source"."model" = "{model_id}" and "meta"."generated_by" = "llm:{model_id}".\n'
        f"{topic_line}\n"
        "Output only the JSON array now."
    )
    return system, user


def build_variator_prompt(rows: list[dict], model_id: str = _DEFAULT_MODEL) -> tuple[str, str]:
    system = (
        "You transform existing training rows to multiply diversity. Follow the schema "
        "exactly. Your entire reply must be a single JSON array. No prose, no fences, "
        "no trailing commas.\n\n" + SCHEMA
    )
    user = (
        VARIATOR
        + f"\n\nInput rows ({len(rows)}):\n"
        + json.dumps(rows, ensure_ascii=False)
        + "\n\nOutput one rewritten row per input row, in the same order, as a JSON array. "
        'Set "label_source"."model" = "' + model_id + '".\n'
        "Output only the JSON array now."
    )
    return system, user


def build_rewriter_prompt(rows: list[dict], model_id: str = _DEFAULT_MODEL) -> tuple[str, str]:
    system = (
        "You reformat training rows to remove boilerplate. Follow the schema exactly. "
        "Your entire reply must be a single JSON array. No prose, no fences, no trailing "
        "commas.\n\n" + SCHEMA
    )
    user = (
        REWRITER
        + f"\n\nInput rows ({len(rows)}):\n"
        + json.dumps(rows, ensure_ascii=False)
        + "\n\nOutput one rewritten row per input row, in the same order, as a JSON array. "
        'Set "label_source"."model" = "' + model_id + '".\n'
        "Output only the JSON array now."
    )
    return system, user


FAMILY_INSTRUCTIONS = {
    "retrieval": RETRIEVAL,
    "paraphrase": PARAPHRASE,
    "order": ORDER,
    "ambiguity": AMBIGUITY,
    "permuted": PERMUTED,
    "define": DEFINE,
    "analogy": ANALOGY,
    "cause": CAUSE,
    "summary": SUMMARY,
    "tone": TONE,
    "intent": INTENT,
    "knowledge": KNOWLEDGE,
    "consistency": CONSISTENCY,
    "open": OPEN,
    "longread": LONGREAD,
    "chain_arithmetic": CHAIN_ARITHMETIC,
    "chain_logic": CHAIN_LOGIC,
    "chain_sequence": CHAIN_SEQUENCE,
    "chain_retrieval": CHAIN_RETRIEVAL,
    "reformulate": REFORMULATE,
    "near_miss": NEAR_MISS,
}