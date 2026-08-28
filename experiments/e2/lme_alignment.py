"""Align Table 4 scoring with LongMemEval's own grading contract.

Table 3 (LoCoMo) is graded by ``quality.ANSWER_PROMPT`` / ``JUDGE_PROMPT``,
which assume a short factual answer compared for factual agreement. That
contract is wrong for two of LongMemEval's six columns and silently
under-reports a third:

* **SS-Pref gold answers are rubrics, not facts.** They read "The user would
  prefer suggestions of Sony-compatible accessories ...". A grader told to
  check factual agreement rejects a response that satisfies the rubric.
  Measured on TriDB/GEM: SS-Pref scored 36.67% under the LoCoMo contract
  against 96.67% for Mandol and 93.33% for EverMemOS in the paper -- a gap no
  capability difference explains.
* **The answer prompt caps responses at "under 5-6 words".** A five-word reply
  cannot demonstrate awareness of a user's preferences, so the cap makes the
  rubric columns unwinnable by construction.
* **No reference date is supplied.** LongMemEval questions carry
  ``question_date`` and their temporal phrasing ("yesterday", "last week") is
  relative to it. Without it, the Temporal column measures the harness.

The judge below is Mandol's own ``MEM0_JUDGE_PROMPT`` (benchmark_longmemeval/
task_eval/evaluation.py), transcribed verbatim except that ``{answer}`` and
``{response}`` are renamed to the keyword names ``quality.judge_answers``
already passes. Using the paper's grader is what makes "same backbone,
different systems" a comparison of systems rather than of graders.

The answer prompt is NOT copied from Mandol: theirs names their own three-tower
context sections (``<conversation_history>``, ``<episodic_facts>``,
``<entity_knowledge>``), which no other system produces. Only the three
system-agnostic properties are carried over -- reference date, no length cap,
and an explicit licence to decline.
"""

from __future__ import annotations

import re
from typing import Any

#: Mandol's judge, verbatim. It answers ``{"label": "yes"|"no"}``.
LME_JUDGE_PROMPT = """I will give you a question, a correct answer (or rubric), and a model response. Decide whether the model response is correct.

CORE PRINCIPLE — Semantic equivalence: Judge by MEANING, not exact words. Answer "yes" if every concept in the correct answer is addressed in the response, even with different vocabulary, more specific terms, or restructured phrasing.

IMPORTANT BIAS CHECK: You have a tendency to say "no" too quickly. Before concluding "no", you MUST verify the answer is truly wrong, not just differently worded. When in doubt, lean toward "yes".

Rules:

**Equivalence & Supersets**
- Equivalent or superset responses are correct. Extra details are fine unless proven to be factually wrong. Extra qualifiers are fine unless proven to be wrong. E.g., "a blue dress and a matching necklace" is correct when the answer is "a blue dress."
- If a response captures the most specific part (exact item/place/name) but omits a broader container, it's correct.
- Same factual meaning with different phrasing = correct (e.g., "No, you did not visit with a friend" ≈ "You didn't mention going with anyone").
- Adding scope qualifiers like "regular-season" or "excluding X" is fine as long as the core value is correct. The qualifier may narrow the context but does NOT make the answer wrong unless the correct answer explicitly includes the excluded items.

**Lists & Compound Terms**
- For list answers, match each item by semantic meaning. A concept is covered if restated via synonyms, sub-concepts, or related terms. Adding methodological detail or rewording verbs to near-synonyms is acceptable.
- A broad term like "A and B significance" is covered if the response addresses the topic area through related specific terms, even without naming each component literally.
- If some items as listed as "or"s, "maybe"s and potential answers, it's okay if the answer does not include those.
- If two items in a list achieve the same purpose, listing just one of them is fine.

IMPORTANT: The "anti-preference" items are very specific!
Eg. Someone "not interested in general AI topics" could be very interested in specific AI topics in general AI *conferences*; those are not the same thing and should be accepted! topics != conferences

**Numbers & Precision**
- Hedging ("at least 3", "approximately") is fine if the core number matches. A range that includes the correct answer is correct.
Generally, if the user themself would be satisfied by the response, it is acceptable. Ie. If the answer is conditional on information they would have (eg. their birthday, some hidden dependent information), and would be correct with that information, that is acceptable.
- More precise answers are correct: "22 days" matches "3 weeks"; "over $270" matches "$270."; "9 1/2 months" matches "9 months";

- Rough answers are correct: "about nine months" ≈ "9 months; "8 months and 20 days" matches "9 months";

- Off-by-one errors on days/weeks/months are acceptable.
- Approximate unit conversions are equivalent: "14 weeks" ≈ "3 months", "6 months" ≈ "half a year."
- Round time ranges generously: 7 months and 16 days ≈ 8 months.
- Notes instead of chords are acceptable when justified
- A correct number with added context (e.g., "about 5 months ago (around December 2022)") is correct — the parenthetical date is supplementary, not a contradiction.

**Dates & Temporal**
- Date format variations are equivalent: "February 1st" = "Feb 1, 2023" = "on February 1."
- Same-day event ordering swaps are acceptable.
- Outdated info alongside the correct updated answer is acceptable if the current value is identified.
- "recent" is upto 6 years ago, which means 2017+
- References like "last weekend", "last Wednesday", etc. are imprecise - people sometimes mean the weekend/Wednesday before the latest one if they're near it. "Last 3 months" can include boundary days of the 4th month back. "Last month" includes the current month so far. Be flexible with such timestamps

**Counting Edge Cases**
- If correct answer is "0" or "nothing found," model saying "not enough information" is also correct.
- Similarly, If correct answer is "not enough information", model saying "0" or "nothing found," is also correct.

**Preference/Personalization Rubrics** (apply in order):
1. Correct if the response demonstrates awareness of user's personal context (preferences, habits, interests). Need not satisfy every rubric point.
2. Primary criterion: do main suggestions align with what the user WANTS?
3. Anti-preferences: evaluate the OVERALL thrust, not keyword scanning. If the response largely suggests correct options, minor incidental references to "not-preferred" things are fine.
4. Mentioning a phone app as a MEANS to a preferred activity (e.g., meditation app for sleep) is not "suggesting phone use." Judge by the activity, not delivery mechanism.
5. "May not prefer" = mild preference, not hard prohibition. Secondary/context-dependent inclusion is fine.
6. Explicit acknowledgment of anti-preferences (e.g., "keep screens off") strengthens correctness.
7. Context-dependent suggestions are acceptable (reading is fine on a bus even if rubric flags visual attention activities). Adjacent genres alongside preferred ones are additive, not contradictory.
8. If the rubric mentions specific user resources/tools (e.g., "Suica card", "TripIt app"), the response is correct if it demonstrates awareness of the user's MAIN personal context even if it does not name every specific tool. The rubric is a guide, not a checklist.

**Abstention Matching**
- If correct answer = unanswerable/abstention, ANY phrasing that conveys "I don't have this information" is correct, regardless of what partial context is mentioned or omitted.
- Saying "not enough information" while mentioning partial related context = correct abstention.
- Saying "no record of X" or "only have plans for X, not actual dates" = correct abstention.
- The key test: does the response REFUSE to answer the question? If yes, it matches an abstention ground truth, period.
- This is a one-way rule: if the correct answer is a concrete fact, number, date, item, link, or preference rubric, a model response that refuses to answer ("not enough information", "no record", "cannot determine", "not specified") is NOT correct.

FINAL CHECK: Before answering "no," you MUST reason through these steps:
1. What is the core factual claim or intent of the correct answer?
2. Does the model response address that same claim, even in different words?
3. Is the response a superset (correct answer + extra details)?
4. For numbers: does the core number match, ignoring hedging/qualifiers?
5. For abstentions: does the response effectively decline to answer?
Only answer "no" if, after this analysis, a core concept is entirely unaddressed or contradicted.

Question: {question}

Correct Answer: {gold_answer}

Model Response: {generated_answer}

Return JSON only, with exactly two fields:
{{"reasoning": "brief explanation", "label": "yes"}} or {{"reasoning": "brief explanation", "label": "no"}}"""

#: System-agnostic answer prompt. One flat context block, because that is all
#: every adapter can be relied on to produce.
LME_ANSWER_PROMPT = """You are a memory-augmented assistant answering a question from retrieved memories.

# RETRIEVED MEMORIES
{context}

# QUESTION
{question}

# INSTRUCTIONS
- Ground the answer in the retrieved memories. Prefer the most recent entry
  when entries conflict.
- Be specific and direct. Do not pad, but do not truncate either: when the
  question asks for a suggestion or recommendation, give one that reflects
  what these memories say the user actually prefers.
- If the memories do not contain the answer, say so plainly.

Answer:"""


#: The judge must emit a `reasoning` field before `label`. quality.py caps the
#: judge at 48 output tokens, which was enough for LoCoMo's `{"label": ...}`
#: but truncates this one mid-sentence: 64.6% of replies came back unparseable
#: and every one of them fell through to WRONG. Measured on TriDB/GEM that
#: alone moved Overall from 74.20 to 34.60.
JUDGE_MAX_TOKENS = 512

_LABEL_RE = re.compile(r'"label"\s*:\s*"(yes|no|correct|wrong|true|false)"', re.I)
_TRAILING_RE = re.compile(r"\b(yes|no|correct|wrong)\b\s*[\"'}\]]*\s*$", re.I)


def judge_label(text: str) -> str:
    """Map the judge's reply onto quality.py's CORRECT/WRONG vocabulary.

    ``quality._judge_label`` accepts only CORRECT/WRONG and returns WRONG for
    anything else, so a reply this grader cannot parse is indistinguishable
    from a reply that says "wrong". Mandol's judge answers ``yes``/``no`` and
    puts a free-text ``reasoning`` field first, so both a vocabulary mismatch
    and a truncated JSON body land in that same silent bucket.

    Unparseable text therefore raises instead of defaulting. A grading run
    that cannot read its own grader must fail loudly, not report a number.
    """
    import json as _json

    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2:
            body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            candidate = "\n".join(body).strip()

    label = ""
    try:
        label = str(_json.loads(candidate).get("label", "")).strip().lower()
    except Exception:  # noqa: BLE001 - fall back to a regex over partial JSON
        match = _LABEL_RE.search(candidate) or _TRAILING_RE.search(candidate)
        if match:
            label = match.group(1).lower()

    if label in {"yes", "correct", "true"}:
        return "CORRECT"
    if label in {"no", "wrong", "false"}:
        return "WRONG"
    raise ValueError(f"unparseable judge reply: {candidate[:200]!r}")


def apply(quality_module: Any, question_dates: dict[str, str]) -> None:
    """Rebind the grading contract on the shared quality module.

    Plain attribute assignment of values that do not reference the attribute
    being replaced -- unlike the earlier ``_widen_timeout``, whose lambda
    resolved back to itself and recursed until the process was OOM-killed.
    """
    quality_module.ANSWER_PROMPT = LME_ANSWER_PROMPT
    quality_module.JUDGE_PROMPT = LME_JUDGE_PROMPT
    quality_module._judge_label = judge_label

    # Widen the judge's output budget. judge_answers hardcodes max_tokens=48.
    original_chat = quality_module._chat

    async def _chat_widened(client, endpoint, model, prompt, *, max_tokens):
        if max_tokens < JUDGE_MAX_TOKENS and prompt.startswith(
            LME_JUDGE_PROMPT[:40]
        ):
            max_tokens = JUDGE_MAX_TOKENS
        return await original_chat(
            client, endpoint, model, prompt, max_tokens=max_tokens
        )

    quality_module._chat = _chat_widened

    # The reference date rides inside the rendered context rather than as a
    # third template field: generate_answers calls
    # ``ANSWER_PROMPT.format(context=..., question=...)`` with exactly those
    # two keywords, so a {question_date} placeholder would raise KeyError for
    # every question.
    original = quality_module._contexts

    def _contexts_with_date(record: dict[str, Any]) -> tuple[list[str], str]:
        values, rendered = original(record)
        stamp = question_dates.get(str(record.get("question_id") or ""))
        if not stamp:
            return values, rendered
        header = (
            "# CURRENT REFERENCE TIME\n"
            f"The current time for this question is: **{stamp}**\n"
            'Treat it as "TODAY" / "NOW" and resolve every relative reference '
            '("yesterday", "last week", "3 days ago") against it, not against '
            "the real-world date.\n\n# MEMORIES\n"
        )
        return values, header + rendered

    quality_module._contexts = _contexts_with_date
