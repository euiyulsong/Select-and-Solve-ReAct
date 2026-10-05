#!/usr/bin/env python3
"""
plan_react_router_retrieval_eval_plaintext.py

Goal
----
Evaluate retrieval strategy selection on:
  - SQuAD validation:   100 single-hop-ish questions
  - HotpotQA validation/distractor: 100 multi-hop questions

Strategies
----------
1) always_react
   Original question -> Search -> ( "-" OR follow-up query ) -> Search ...
   Max 3 SEARCH rounds.

2) always_plan_react
   Plain-text plan/decomposition -> Search first planned query
   -> ( "-" OR follow-up query ) -> Search ...
   Max 3 SEARCH rounds.

3) router
   GPT-6 Luna prints ONLY:
       0  => ReAct
       1  => Plan -> ReAct
   Then executes that route.

4) static_hop   (optional diagnostic)
   SQuAD   => 0
   Hotpot  => 1
   This uses benchmark identity, so treat it as an oracle-like diagnostic,
   not a deployable router.

IMPORTANT: NO JSON FROM THE LLM
-------------------------------
Router:
    output exactly: 0
    or:             1

Plan:
    output only search queries, one per line.
    No bullets, no numbering, no explanation.

Verify / follow-up:
    if evidence is sufficient:
        -
    otherwise:
        one follow-up search query only

No LLM answer-generation call is used.
The experiment evaluates RETRIEVAL only.

Speed
-----
- BM25 is local.
- LLM calls are cached by exact prompt + kind.
  If always_react and router->ReAct reach the same state, the follow-up
  result is reused instead of paying/calling twice.
- reasoning effort defaults to "none".
- max SEARCH rounds defaults to exactly 3 as the hard cap.

Install
-------
pip install -U openai datasets rank-bm25 pandas numpy tqdm

Run
---
export OPENAI_API_KEY="..."

python plan_react_router_retrieval_eval_plaintext.py \
  --n-per-dataset 100 \
  --corpus-size 1000 \
  --top-k 3 \
  --max-search-rounds 3 \
  --methods always_react,always_plan_react,router \
  --output results.csv

To include dataset-static diagnostic:
  --methods always_react,always_plan_react,router,static_hop

For a quick smoke test:
python plan_react_router_retrieval_eval_plaintext.py \
  --n-per-dataset 5 \
  --corpus-size 300 \
  --top-k 3 \
  --max-search-rounds 3
"""

import argparse
import hashlib
import os
import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ============================================================================
# Global config
# ============================================================================

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
DEFAULT_SEED = 42

client = OpenAI()

# Exact-prompt cache.
# This makes cross-method comparison cheaper and fairer:
# same component + same prompt => exactly the same model output.
LLM_CACHE: Dict[str, Tuple[str, dict]] = {}


# ============================================================================
# Small helpers
# ============================================================================

def clean_text(x) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def fresh_meta() -> dict:
    return {
        "llm_calls": 0,
        "llm_cache_hits": 0,
        "latency_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def add_meta(a: dict, b: dict) -> dict:
    out = dict(a)
    for k in (
        "llm_calls",
        "llm_cache_hits",
        "latency_s",
        "input_tokens",
        "output_tokens",
    ):
        out[k] = out.get(k, 0) + b.get(k, 0)
    return out


def short(text: str, n: int = 150) -> str:
    text = clean_text(text)
    return text if len(text) <= n else text[: n - 3] + "..."


# ============================================================================
# OpenAI call -- PLAIN TEXT ONLY
# ============================================================================

def llm_text(
    *,
    kind: str,
    prompt: str,
    model: str,
    effort: str,
    max_output_tokens: int,
    use_cache: bool = True,
) -> Tuple[str, dict]:
    """
    Returns plain response.output_text.
    No structured outputs. No JSON schema.
    """
    cache_key = f"{model}\n{effort}\n{kind}\n{prompt}"

    if use_cache and cache_key in LLM_CACHE:
        text, original_meta = LLM_CACHE[cache_key]
        meta = fresh_meta()
        meta["llm_cache_hits"] = 1

        print(f"      [LLM:{kind}:CACHE] {text!r}", flush=True)
        return text, meta

    t0 = time.perf_counter()

    response = client.responses.create(
        model=model,
        reasoning={"effort": effort},
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    latency = time.perf_counter() - t0
    text = response.output_text.strip()

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    meta = {
        "llm_calls": 1,
        "llm_cache_hits": 0,
        "latency_s": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    if use_cache:
        LLM_CACHE[cache_key] = (text, meta)

    print(
        f"      [LLM:{kind}] {text!r} "
        f"(lat={latency:.2f}s, in={input_tokens}, out={output_tokens})",
        flush=True,
    )

    return text, meta


# ============================================================================
# LLM output parsing -- deliberately trivial
# ============================================================================

def parse_route(text: str) -> int:
    """
    Expected exact output: 0 or 1.
    Be slightly defensive if model adds whitespace/punctuation.
    """
    s = text.strip()

    if s == "0":
        return 0
    if s == "1":
        return 1

    m = re.search(r"(?<!\d)([01])(?!\d)", s)
    if m:
        print(f"      [WARN] Router output was not exact; parsed {m.group(1)!r}")
        return int(m.group(1))

    print(f"      [WARN] Could not parse router output {s!r}; fallback=0")
    return 0


def parse_plan_lines(text: str, question: str, max_lines: int = 3) -> List[str]:
    """
    Planner should return ONLY one search query per line.
    We strip accidental bullets/numbers, but there is no JSON parsing.
    """
    lines = []

    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue

        # Defensive cleanup for accidental bullets or numbering.
        s = re.sub(r"^\s*[-*•]\s*", "", s)
        s = re.sub(r"^\s*\d+[\.\)]\s*", "", s)
        s = s.strip()

        if s and s != "-":
            lines.append(s)

    if not lines:
        return [question]

    return lines[:max_lines]


def parse_followup(text: str) -> str | None:
    """
    Expected:
      "-"                    => stop
      "some follow-up query" => continue
    """
    s = text.strip()

    if s == "-":
        return None

    # Defensive: if model accidentally emits multiple lines, use first nonempty line.
    for line in s.splitlines():
        q = line.strip()
        if not q:
            continue
        if q == "-":
            return None
        q = re.sub(r"^\s*[-*•]\s*", "", q)
        q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
        q = q.strip()
        if q:
            return q

    return None


# ============================================================================
# Prompts
# ============================================================================

def router_prompt(question: str) -> str:
    return f"""Choose the retrieval execution strategy for the question.

Output exactly ONE character and nothing else:
0 = ReAct
1 = Plan then ReAct

Use 0 when:
- searching the original question first is a strong starting point, and
- observing the first retrieval result before deciding the next query is enough.

Use 1 when:
- the question has multiple distinct evidence needs, comparison/composition,
  or dependent facts, and
- decomposing BEFORE the first search is likely to improve retrieval coverage
  under a strict maximum of 3 searches.

Important:
- Do not guess the benchmark/dataset.
- A question that looks multi-hop does NOT automatically require 1.
- ReAct can discover an intermediate entity after search 1.
- Choose 1 only when upfront decomposition is genuinely useful.

Question:
{question}

Output only 0 or 1."""


def plan_prompt(question: str) -> str:
    return f"""Prepare a minimal retrieval plan for this question.

Output ONLY 1 to 3 search queries, one query per line.
No bullets.
No numbering.
No labels.
No explanation.
No JSON.

Rules:
- Line 1 must be directly searchable immediately.
- Later lines represent useful evidence needs for solving the question.
- Do not invent an unknown intermediate entity before retrieving it.
- Keep each line concise.
- Maximum 3 lines.

Question:
{question}"""


def followup_prompt(
    *,
    question: str,
    plan_queries: List[str],
    previous_queries: List[str],
    evidence_text: str,
) -> str:
    plan_text = "\n".join(plan_queries) if plan_queries else "(none)"
    prev_text = "\n".join(previous_queries) if previous_queries else "(none)"

    return f"""Decide whether another retrieval search is needed.

If the retrieved evidence is already sufficient to support the answer:
output exactly:
-

If one important fact/evidence is still missing:
output ONLY one focused follow-up search query.

No explanation.
No JSON.
No label such as "Query:".
Do not answer the original question.
Do not repeat a previous query.
Use entities discovered in the retrieved evidence when helpful.

Original question:
{question}

Initial plan queries, if any:
{plan_text}

Previous search queries:
{prev_text}

Retrieved evidence:
{evidence_text}

Output only "-" or one follow-up search query."""


# ============================================================================
# Dataset construction
# ============================================================================

def build_squad(
    n: int,
    corpus_size: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Evaluation questions come from SQuAD validation.
    Every selected question's gold context is inserted into the search corpus FIRST.
    Remaining corpus docs are validation distractors.
    """
    ds = load_dataset("rajpurkar/squad", split="validation")
    rng = random.Random(seed)

    all_indices = list(range(len(ds)))
    rng.shuffle(all_indices)

    chosen = []
    seen_gold_contexts = set()

    # One selected QA per unique context to make the gold-doc mapping simple.
    for idx in all_indices:
        context = clean_text(ds[idx]["context"])
        context_id = stable_hash(context)

        if context_id in seen_gold_contexts:
            continue

        seen_gold_contexts.add(context_id)
        chosen.append(idx)

        if len(chosen) >= n:
            break

    if len(chosen) < n:
        raise RuntimeError(f"Only found {len(chosen)} unique SQuAD contexts.")

    corpus_by_id: Dict[str, dict] = {}
    items = []

    # Gold contexts first: guarantees every selected validation QA has gold in corpus.
    for idx in chosen:
        row = ds[idx]

        context = clean_text(row["context"])
        doc_id = f"squad::{stable_hash(context)}"

        corpus_by_id[doc_id] = {
            "id": doc_id,
            "title": clean_text(row.get("title", "")),
            "text": context,
        }

        items.append({
            "qid": f"squad::{row['id']}",
            "dataset": "squad",
            "question": clean_text(row["question"]),
            "gold_ids": {doc_id},
            "static_route": 0,
        })

    # Fill with validation distractor contexts.
    distractor_indices = list(range(len(ds)))
    rng.shuffle(distractor_indices)

    for idx in distractor_indices:
        if len(corpus_by_id) >= corpus_size:
            break

        row = ds[idx]
        context = clean_text(row["context"])
        doc_id = f"squad::{stable_hash(context)}"

        if doc_id not in corpus_by_id:
            corpus_by_id[doc_id] = {
                "id": doc_id,
                "title": clean_text(row.get("title", "")),
                "text": context,
            }

    corpus = list(corpus_by_id.values())

    # Hard guarantee.
    corpus_ids = {d["id"] for d in corpus}
    for item in items:
        if not item["gold_ids"].issubset(corpus_ids):
            raise RuntimeError(f"SQuAD gold missing for {item['qid']}")

    return items, corpus


def hotpot_docs(row) -> List[dict]:
    """
    HotpotQA distractor config stores titles + sentence lists.
    We index one Wikipedia title/context as one document.
    """
    titles = row["context"]["title"]
    sentence_groups = row["context"]["sentences"]

    docs = []

    for title, sentences in zip(titles, sentence_groups):
        title = clean_text(title)
        text = clean_text(" ".join(sentences))

        docs.append({
            "id": f"hotpot::{title.lower()}",
            "title": title,
            "text": text,
        })

    return docs


def build_hotpot(
    n: int,
    corpus_size: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Evaluation questions come from HotpotQA distractor validation.
    Every selected example's native context is inserted first, so all supporting
    documents for the selected questions are guaranteed to exist in the corpus.

    NOTE:
    100 Hotpot examples can already contribute ~1000 unique native context docs.
    Therefore actual corpus size may exceed --corpus-size.
    We never delete gold/native docs just to hit the requested integer.
    """
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    rng = random.Random(seed + 1)
    chosen = rng.sample(range(len(ds)), n)

    corpus_by_id: Dict[str, dict] = {}
    items = []

    # Add ALL native docs from selected examples first.
    for idx in chosen:
        row = ds[idx]

        for d in hotpot_docs(row):
            corpus_by_id.setdefault(d["id"], d)

        gold_titles = {
            clean_text(t).lower()
            for t in row["supporting_facts"]["title"]
        }
        gold_ids = {f"hotpot::{title}" for title in gold_titles}

        items.append({
            "qid": f"hotpot::{row['id']}",
            "dataset": "hotpotqa",
            "question": clean_text(row["question"]),
            "gold_ids": gold_ids,
            "static_route": 1,
        })

    # Top up from other validation examples if requested corpus is larger.
    all_indices = list(range(len(ds)))
    rng.shuffle(all_indices)

    for idx in all_indices:
        if len(corpus_by_id) >= corpus_size:
            break

        for d in hotpot_docs(ds[idx]):
            corpus_by_id.setdefault(d["id"], d)

            if len(corpus_by_id) >= corpus_size:
                break

    corpus = list(corpus_by_id.values())

    # Hard guarantee: all supporting docs are in the search corpus.
    corpus_ids = {d["id"] for d in corpus}

    for item in items:
        missing = item["gold_ids"] - corpus_ids
        if missing:
            raise RuntimeError(
                f"Hotpot gold docs missing for {item['qid']}: {sorted(missing)}"
            )

    return items, corpus


# ============================================================================
# BM25
# ============================================================================

class BM25Search:
    def __init__(self, docs: List[dict]):
        self.docs = docs

        tokenized_corpus = [
            tokenize(f"{d.get('title', '')} {d['text']}")
            for d in docs
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int) -> List[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]

        results = []

        for local_rank, idx in enumerate(order, start=1):
            doc = self.docs[int(idx)]

            results.append({
                **doc,
                "score": float(scores[int(idx)]),
                "local_rank": local_rank,
            })

        return results


# ============================================================================
# Pretty printing of every intermediate result
# ============================================================================

def print_search_results(
    *,
    results: List[dict],
    gold_ids: set,
    retrieved_ids_so_far: List[str],
):
    for d in results:
        hit = "GOLD" if d["id"] in gold_ids else "----"
        already = " DUP" if d["id"] in retrieved_ids_so_far else ""

        print(
            f"          #{d['local_rank']} [{hit}{already}] "
            f"score={d['score']:.4f} | {d.get('title', '')}",
            flush=True,
        )
        print(
            f"              {short(d['text'], 180)}",
            flush=True,
        )


def print_running_metrics(gold_ids: set, retrieved_ids: List[str]):
    unique_ret = list(dict.fromkeys(retrieved_ids))
    hit = gold_ids & set(unique_ret)

    recall = len(hit) / len(gold_ids) if gold_ids else 0.0
    full = gold_ids.issubset(set(unique_ret))

    print(
        f"          cumulative gold={len(hit)}/{len(gold_ids)} "
        f"support_recall={recall:.3f} full_support={int(full)}",
        flush=True,
    )


# ============================================================================
# Evidence for follow-up generation
# ============================================================================

def format_evidence(
    evidence_by_id: Dict[str, dict],
    *,
    max_docs: int = 9,
    chars_per_doc: int = 900,
) -> str:
    blocks = []

    for i, d in enumerate(list(evidence_by_id.values())[:max_docs], start=1):
        title = d.get("title", "") or d["id"]
        blocks.append(
            f"[D{i}] {title}\n{d['text'][:chars_per_doc]}"
        )

    return "\n\n".join(blocks) if blocks else "(none)"


# ============================================================================
# Router / Planner / Follow-up components
# ============================================================================

def call_router(
    question: str,
    *,
    model: str,
    effort: str,
    use_cache: bool,
) -> Tuple[int, dict]:
    raw, meta = llm_text(
        kind="ROUTER",
        prompt=router_prompt(question),
        model=model,
        effort=effort,
        max_output_tokens=16,
        use_cache=use_cache,
    )

    route = parse_route(raw)

    print(
        f"      [ROUTER] route={route} "
        f"({'REACT' if route == 0 else 'PLAN_REACT'})",
        flush=True,
    )

    return route, meta


def call_plan(
    question: str,
    *,
    model: str,
    effort: str,
    use_cache: bool,
) -> Tuple[List[str], dict]:
    raw, meta = llm_text(
        kind="PLAN",
        prompt=plan_prompt(question),
        model=model,
        effort=effort,
        max_output_tokens=120,
        use_cache=use_cache,
    )

    lines = parse_plan_lines(raw, question, max_lines=3)

    print("      [PLAN]", flush=True)
    for i, q in enumerate(lines, start=1):
        print(f"          {i}. {q}", flush=True)

    return lines, meta


def call_followup(
    *,
    question: str,
    plan_queries: List[str],
    previous_queries: List[str],
    evidence_by_id: Dict[str, dict],
    model: str,
    effort: str,
    use_cache: bool,
) -> Tuple[str | None, dict]:
    raw, meta = llm_text(
        kind="FOLLOWUP",
        prompt=followup_prompt(
            question=question,
            plan_queries=plan_queries,
            previous_queries=previous_queries,
            evidence_text=format_evidence(evidence_by_id),
        ),
        model=model,
        effort=effort,
        max_output_tokens=64,
        use_cache=use_cache,
    )

    q = parse_followup(raw)

    if q is None:
        print("      [VERIFY] evidence sufficient -> -", flush=True)
    else:
        print(f"      [FOLLOW-UP QUERY] {q}", flush=True)

    return q, meta


# ============================================================================
# Core retrieval loops
# ============================================================================

def execute_react(
    *,
    item: dict,
    searcher: BM25Search,
    top_k: int,
    max_search_rounds: int,
    model: str,
    effort: str,
    use_cache: bool,
) -> dict:
    """
    ReAct:
      Search #1 = original question, no initial rewrite.
      Then after each search:
          "-" OR one follow-up query.
      Hard cap = max_search_rounds SEARCH calls.
    """
    meta = fresh_meta()

    previous_queries: List[str] = []
    retrieved_ids: List[str] = []
    evidence_by_id: Dict[str, dict] = {}

    current_query = item["question"]
    stopped_early = False

    for search_round in range(1, max_search_rounds + 1):
        print(
            f"      [SEARCH {search_round}/{max_search_rounds}] {current_query}",
            flush=True,
        )

        results = searcher.search(current_query, top_k=top_k)

        print_search_results(
            results=results,
            gold_ids=item["gold_ids"],
            retrieved_ids_so_far=retrieved_ids,
        )

        previous_queries.append(current_query)

        for d in results:
            retrieved_ids.append(d["id"])
            evidence_by_id.setdefault(d["id"], d)

        print_running_metrics(item["gold_ids"], retrieved_ids)

        # Search #3 is the hard stop: no verify call after last search.
        if search_round >= max_search_rounds:
            break

        followup, m = call_followup(
            question=item["question"],
            plan_queries=[],
            previous_queries=previous_queries,
            evidence_by_id=evidence_by_id,
            model=model,
            effort=effort,
            use_cache=use_cache,
        )
        meta = add_meta(meta, m)

        if followup is None:
            stopped_early = True
            break

        current_query = followup

    return {
        "queries": previous_queries,
        "retrieved_ids": retrieved_ids,
        "search_calls": len(previous_queries),
        "stopped_early": stopped_early,
        "meta": meta,
        "plan_queries": [],
    }


def execute_plan_react(
    *,
    item: dict,
    searcher: BM25Search,
    top_k: int,
    max_search_rounds: int,
    model: str,
    effort: str,
    use_cache: bool,
) -> dict:
    """
    Plan -> ReAct:
      1) Planner emits 1-3 plain search-query lines.
      2) Search #1 uses plan line #1.
      3) The whole plan is passed as hints to the follow-up controller.
      4) After each search:
           "-" OR one evidence-grounded follow-up query.
      5) Hard cap = same max_search_rounds as ReAct.

    We intentionally do NOT blindly execute plan lines #2/#3.
    They are hints; after seeing actual evidence, the follow-up controller may
    ground/replace them. This keeps the "ReAct" part genuinely reactive.
    """
    meta = fresh_meta()

    plan_queries, m = call_plan(
        item["question"],
        model=model,
        effort=effort,
        use_cache=use_cache,
    )
    meta = add_meta(meta, m)

    previous_queries: List[str] = []
    retrieved_ids: List[str] = []
    evidence_by_id: Dict[str, dict] = {}

    current_query = plan_queries[0] if plan_queries else item["question"]
    stopped_early = False

    for search_round in range(1, max_search_rounds + 1):
        print(
            f"      [SEARCH {search_round}/{max_search_rounds}] {current_query}",
            flush=True,
        )

        results = searcher.search(current_query, top_k=top_k)

        print_search_results(
            results=results,
            gold_ids=item["gold_ids"],
            retrieved_ids_so_far=retrieved_ids,
        )

        previous_queries.append(current_query)

        for d in results:
            retrieved_ids.append(d["id"])
            evidence_by_id.setdefault(d["id"], d)

        print_running_metrics(item["gold_ids"], retrieved_ids)

        if search_round >= max_search_rounds:
            break

        followup, m = call_followup(
            question=item["question"],
            plan_queries=plan_queries,
            previous_queries=previous_queries,
            evidence_by_id=evidence_by_id,
            model=model,
            effort=effort,
            use_cache=use_cache,
        )
        meta = add_meta(meta, m)

        if followup is None:
            stopped_early = True
            break

        current_query = followup

    return {
        "queries": previous_queries,
        "retrieved_ids": retrieved_ids,
        "search_calls": len(previous_queries),
        "stopped_early": stopped_early,
        "meta": meta,
        "plan_queries": plan_queries,
    }


# ============================================================================
# Retrieval metrics
# ============================================================================

def compute_retrieval_metrics(
    gold_ids: set,
    retrieved_ids: List[str],
) -> dict:
    # Preserve global retrieval order while removing duplicates.
    unique_ids = list(dict.fromkeys(retrieved_ids))
    unique_set = set(unique_ids)

    gold_hit = gold_ids & unique_set

    support_recall = (
        len(gold_hit) / len(gold_ids)
        if gold_ids
        else np.nan
    )

    full_support = (
        float(gold_ids.issubset(unique_set))
        if gold_ids
        else np.nan
    )

    hit_any = float(bool(gold_hit))

    first_gold_rank = None

    for rank, doc_id in enumerate(unique_ids, start=1):
        if doc_id in gold_ids:
            first_gold_rank = rank
            break

    mrr_first = (
        1.0 / first_gold_rank
        if first_gold_rank is not None
        else 0.0
    )

    return {
        "support_recall": support_recall,
        "full_support": full_support,
        "hit_any": hit_any,
        "mrr_first": mrr_first,
        "unique_docs_retrieved": len(unique_ids),
    }


# ============================================================================
# One method evaluation
# ============================================================================

def evaluate_method(
    *,
    item: dict,
    method: str,
    searcher: BM25Search,
    top_k: int,
    max_search_rounds: int,
    model: str,
    effort: str,
    use_cache: bool,
) -> dict:
    total_meta = fresh_meta()

    if method == "always_react":
        route = 0

    elif method == "always_plan_react":
        route = 1

    elif method == "static_hop":
        # Benchmark-informed diagnostic:
        # SQuAD -> ReAct, Hotpot -> Plan+ReAct.
        route = int(item["static_route"])

    elif method == "router":
        route, router_meta = call_router(
            item["question"],
            model=model,
            effort=effort,
            use_cache=use_cache,
        )
        total_meta = add_meta(total_meta, router_meta)

    else:
        raise ValueError(f"Unknown method: {method}")

    route_name = "REACT" if route == 0 else "PLAN_REACT"

    print(f"      [EXECUTE] {route_name}", flush=True)

    if route == 0:
        run = execute_react(
            item=item,
            searcher=searcher,
            top_k=top_k,
            max_search_rounds=max_search_rounds,
            model=model,
            effort=effort,
            use_cache=use_cache,
        )
    else:
        run = execute_plan_react(
            item=item,
            searcher=searcher,
            top_k=top_k,
            max_search_rounds=max_search_rounds,
            model=model,
            effort=effort,
            use_cache=use_cache,
        )

    total_meta = add_meta(total_meta, run["meta"])

    metrics = compute_retrieval_metrics(
        item["gold_ids"],
        run["retrieved_ids"],
    )

    print(
        "      [RESULT] "
        f"recall={metrics['support_recall']:.3f} "
        f"full={metrics['full_support']:.0f} "
        f"mrr={metrics['mrr_first']:.3f} "
        f"searches={run['search_calls']} "
        f"llm_calls={total_meta['llm_calls']} "
        f"cache_hits={total_meta['llm_cache_hits']} "
        f"lat={total_meta['latency_s']:.2f}s",
        flush=True,
    )

    return {
        "qid": item["qid"],
        "dataset": item["dataset"],
        "question": item["question"],
        "method": method,
        "route": route,
        "route_name": route_name,
        "static_route": item["static_route"],
        "route_matches_static_hop": float(route == item["static_route"]),
        **metrics,
        "search_calls": run["search_calls"],
        "stopped_early": float(run["stopped_early"]),
        "queries": " || ".join(run["queries"]),
        "plan_queries": " || ".join(run["plan_queries"]),
        "llm_calls": total_meta["llm_calls"],
        "llm_cache_hits": total_meta["llm_cache_hits"],
        "latency_s": total_meta["latency_s"],
        "input_tokens": total_meta["input_tokens"],
        "output_tokens": total_meta["output_tokens"],
    }


# ============================================================================
# Summary tables
# ============================================================================

def summarize(df: pd.DataFrame):
    metric_cols = [
        "support_recall",
        "full_support",
        "hit_any",
        "mrr_first",
        "search_calls",
        "unique_docs_retrieved",
        "stopped_early",
        "llm_calls",
        "llm_cache_hits",
        "latency_s",
        "input_tokens",
        "output_tokens",
    ]

    overall = (
        df.groupby("method")[metric_cols]
        .mean()
        .reset_index()
    )

    by_dataset = (
        df.groupby(["dataset", "method"])[metric_cols]
        .mean()
        .reset_index()
    )

    router_df = df[df["method"] == "router"].copy()

    if len(router_df):
        route_mix = (
            router_df.groupby(["dataset", "route_name"])
            .size()
            .rename("n")
            .reset_index()
        )

        route_mix["ratio"] = route_mix.groupby("dataset")["n"].transform(
            lambda x: x / x.sum()
        )

        router_agreement = router_df["route_matches_static_hop"].mean()
    else:
        route_mix = pd.DataFrame()
        router_agreement = np.nan

    return overall, by_dataset, route_mix, router_agreement


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-per-dataset", type=int, default=100)
    parser.add_argument("--corpus-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-search-rounds", type=int, default=3)

    parser.add_argument(
        "--methods",
        type=str,
        default="always_react,always_plan_react,router",
        help=(
            "Comma separated from: "
            "always_react,always_plan_react,router,static_hop"
        ),
    )

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)

    parser.add_argument(
        "--effort",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=str,
        default="plan_react_router_retrieval_results.csv",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable exact-prompt LLM cache.",
    )

    args = parser.parse_args()

    allowed_methods = {
        "always_react",
        "always_plan_react",
        "router",
        "static_hop",
    }

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    unknown = set(methods) - allowed_methods

    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    use_cache = not args.no_cache

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 100)
    print("CONFIG")
    print("=" * 100)
    print(f"model              : {args.model}")
    print(f"reasoning effort   : {args.effort}")
    print(f"n per dataset      : {args.n_per_dataset}")
    print(f"requested corpus   : {args.corpus_size}")
    print(f"top-k per search   : {args.top_k}")
    print(f"max search rounds  : {args.max_search_rounds}")
    print(f"methods            : {methods}")
    print(f"LLM prompt cache   : {use_cache}")
    print()

    print("[LOAD] SQuAD validation...", flush=True)
    squad_items, squad_corpus = build_squad(
        n=args.n_per_dataset,
        corpus_size=args.corpus_size,
        seed=args.seed,
    )

    print("[LOAD] HotpotQA distractor validation...", flush=True)
    hotpot_items, hotpot_corpus = build_hotpot(
        n=args.n_per_dataset,
        corpus_size=args.corpus_size,
        seed=args.seed,
    )

    print()
    print("=" * 100)
    print("CORPUS")
    print("=" * 100)
    print(f"SQuAD corpus docs : {len(squad_corpus)}")
    print(f"Hotpot corpus docs: {len(hotpot_corpus)}")
    print(
        "Gold guarantee    : every sampled evaluation question's gold/support docs "
        "are present in its corpus"
    )
    print()

    searchers = {
        "squad": BM25Search(squad_corpus),
        "hotpotqa": BM25Search(hotpot_corpus),
    }

    items = squad_items + hotpot_items
    random.Random(args.seed).shuffle(items)

    rows = []

    for q_index, item in enumerate(
        tqdm(items, desc="questions", total=len(items)),
        start=1,
    ):
        print()
        print("=" * 100)
        print(
            f"[QUESTION {q_index}/{len(items)}] "
            f"dataset={item['dataset']} qid={item['qid']}"
        )
        print(f"Q: {item['question']}")
        print(f"gold docs: {sorted(item['gold_ids'])}")
        print("=" * 100)

        for method in methods:
            print()
            print(f"  >>> METHOD: {method}")

            try:
                row = evaluate_method(
                    item=item,
                    method=method,
                    searcher=searchers[item["dataset"]],
                    top_k=args.top_k,
                    max_search_rounds=args.max_search_rounds,
                    model=args.model,
                    effort=args.effort,
                    use_cache=use_cache,
                )
                rows.append(row)

            except Exception as e:
                print(
                    f"      [ERROR] qid={item['qid']} method={method}: {repr(e)}",
                    flush=True,
                )

    if not rows:
        raise RuntimeError("No successful rows.")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    overall, by_dataset, route_mix, router_agreement = summarize(df)

    print()
    print("=" * 100)
    print("OVERALL")
    print("=" * 100)
    print(overall.to_string(index=False))

    print()
    print("=" * 100)
    print("BY DATASET")
    print("=" * 100)
    print(by_dataset.to_string(index=False))

    if len(route_mix):
        print()
        print("=" * 100)
        print("ROUTER ROUTE MIX")
        print("=" * 100)
        print(route_mix.to_string(index=False))

        print(
            "\nRouter agreement with dataset-static rule "
            "(SQuAD=ReAct, Hotpot=Plan+ReAct): "
            f"{router_agreement:.4f}"
        )
        print(
            "NOTE: this agreement is diagnostic only. "
            "End-to-end retrieval metrics are the main evaluation."
        )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"CSV saved to: {args.output}")
    print(f"Unique cached LLM prompts: {len(LLM_CACHE)}")


if __name__ == "__main__":
    main()
