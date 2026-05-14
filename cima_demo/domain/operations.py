"""Pure domain operations — no I/O, fully deterministic (KIMA_Domain_CIMA_v0.10 §11)."""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

from cima_demo.domain.entities import CItem, ContextPack, ContextView
from cima_demo.domain.value_objects import (
    BudgetReport,
    CItemFilter,
    ContextBudget,
    CoverageReport,
    CVSWeights,
    ForgetParams,
    ItemType,
    PhasePolicy,
    PromotionPolicy,
    QueryType,
    RetrievalPlan,
    ScoredCItem, BridgePolicy, QueryFacets, _STOPWORDS, DirectEvidenceStrategy,
)

# ── Importance scoring ────────────────────────────────────────────────────────

def compute_static_importance(
    item_type: str,
    confidence: float,
    validation_label: str | None,
) -> float:
    """Base importance from type + confidence (CIMA §3.2).

    DECISION / CONSTRAINT start higher (0.85); PROCEDURE at 0.75;
    FACT/HYPOTHESIS/OBSERVATION at 0.5.
    Multiplied by confidence and validation bonus.
    """
    type_base: dict[str, float] = {
        ItemType.DECISION:    0.85,
        ItemType.CONSTRAINT:  0.85,
        ItemType.PROCEDURE:   0.75,
        ItemType.FACT:        0.50,
        ItemType.DERIVED:     0.50,   # same as FACT — traceable computation
        ItemType.HYPOTHESIS:  0.45,
        ItemType.ASSUMPTION:  0.35,   # temporary, should not persist long
        ItemType.OBSERVATION: 0.40,
    }
    base = type_base.get(item_type, 0.50)
    val_bonus = 0.1 if validation_label == "verified" else (
        -0.1 if validation_label == "refuted" else 0.0
    )
    return round(min(1.0, max(0.0, base * confidence + val_bonus)), 6)


def compute_cvs(
    content_relevance: float,
    recency_score: float,
    novelty_score: float,
    weights: CVSWeights | None = None,
) -> float:
    """Contextual Value Score = α·relevance + β·recency + γ·novelty (CIMA A-5)."""
    w = weights or CVSWeights.default()
    return round(
        w.alpha * content_relevance
        + w.beta  * recency_score
        + w.gamma * novelty_score,
        6,
    )


def compute_recency_score(created_at_unix: float, now_unix: float) -> float:
    """Exponential decay: score = e^(-age_days / 7).

    Returns 1.0 for brand-new items, ~0.37 at 7 days, ~0 at 30 days.
    """
    age_days = max(0.0, (now_unix - created_at_unix) / 86400.0)
    return round(math.exp(-age_days / 7.0), 6)


def compute_novelty_score(item: CItem, existing_ids: set[str]) -> float:
    """1.0 if item not yet in context pool; 0.0 if already present."""
    return 0.0 if item.citem_id in existing_ids else 1.0


def compute_contextuality(item_type: str, phase: str, policy: PhasePolicy | None = None) -> float:
    """Dynamic contextuality Xᵢ(phase) — per-phase, per-type weight (CIMA A-2).

    Never performs I/O; uses the PhasePolicy lookup table.
    Falls back to 0.5 for unknown (phase, item_type) combinations.
    """
    p = policy or PhasePolicy.default()
    return p.get_contextuality(phase, item_type)

def compute_traceability_density(pack: "ContextPack") -> float:
    """Tᵈ = fraction of C_mem items with non-empty dependency_ids (CIMA A-9).

    C_mem = all C-Items in the ContextPack (they all come from memory).
    Level 1 trace: dependency_ids non-empty.
    Level 2 trace: summarized_by_node_id set (counted as traced).
    Returns 1.0 if the pack is empty (vacuous truth).
    """
    all_items = (
        pack.protected_items
        + pack.direct_evidence
        + pack.bridge_evidence
        + pack.global_summaries
        + pack.conflicts
    )
    n = len(all_items)
    if n == 0:
        return 1.0
    traced = sum(
        1 for it in all_items
        if it.dependency_ids or (it.summarized_by_node_id is not None)
    )
    return traced / n


def compute_bridge_score(item: CItem, seed_ids: set[str]) -> float:
    """Structural bridge score: fraction of dep_ids pointing into seed set (APP-D-08 Phase 1).

    Phase 1: dependency_ids counting only; no TEI calls.
    """
    if not item.dependency_ids:
        return 0.0
    hits = sum(1 for d in item.dependency_ids if d in seed_ids)
    return round(hits / len(item.dependency_ids), 6)


# ── Forget cycle predicates ───────────────────────────────────────────────────

def should_attenuate(
    item: CItem,
    age_days: float,
    params: ForgetParams | None = None,
) -> bool:
    """True if item should be attenuated this cycle (CIMA A-7).

    Conditions: active + age ≥ min_age + importance < threshold.
    Protected types (DECISION, CONSTRAINT, PROCEDURE) are never attenuated.
    """
    p = params or ForgetParams.default()
    protected = {ItemType.DECISION, ItemType.CONSTRAINT, ItemType.PROCEDURE}
    if item.item_type in protected:
        return False
    if item.scope_status != "active":
        return False
    if age_days < p.attenuation_age_days:
        return False
    return item.importance < p.attenuation_threshold


def can_purge(
    item: CItem,
    days_since_archived: float,
    params: ForgetParams | None = None,
) -> bool:
    """True if archived item qualifies for permanent deletion (CIMA A-7).

    Conditions: archived + days_since_archived ≥ alpha_purge_days + importance < min_importance.
    """
    p = params or ForgetParams.default()
    if item.scope_status != "archived":
        return False
    if days_since_archived < p.alpha_purge_days:
        return False
    return item.importance < p.min_importance_to_purge


# ── Promotion eligibility ─────────────────────────────────────────────────────

def is_promotion_eligible(
    item: CItem,
    reference_count: int,
    policy: PromotionPolicy | None = None,
) -> bool:
    """True if episodic item qualifies for global promotion."""
    p = policy or PromotionPolicy.default()
    if item.scope != "episodic":
        return False
    if item.scope_status != "active":
        return False
    # OBSERVATIONs are raw tool-call results — ephemeral by nature.
    # Never promote them to global: they pollute every future conversation.
    # Only FACT, HYPOTHESIS, DECISION, CONSTRAINT can become long-lived global knowledge.
    if item.item_type == ItemType.OBSERVATION:
        return False
    return reference_count >= p.min_references and item.importance >= p.min_importance


# ── Query classification ──────────────────────────────────────────────────────

_MULTI_HOP_KEYWORDS = frozenset({
    "relate", "connection", "between", "compare", "contrast", "link", "both",
    "all", "across", "together", "relationship",
})
_PROCEDURAL_RE = re.compile(
    # "run" excluded: too ambiguous as bare word (runner, running shoes, …).
    # "run" only qualifies when followed by a non-word char (end of sentence / space)
    # — captured by the trailing context, not as a standalone keyword.
    r'\b(?:how\s+to|steps|step\s+by\s+step|procedure|process|workflow'
    r'|guide|tutorial|instructions|do\s+i|execute|configure'
    r'|deploy|install|setup|set\s+up|build|compile|run\s+(?:the|a|my|this|it))\b',
    re.IGNORECASE,
)
_DIAGNOSTIC_RE = re.compile(
    r'\b(?:why|cause|reason|debug|error|issue|problem'
    r'|fail(?:ed|ing)?|broken|diagnose|root\s+cause|what\s+went\s+wrong)\b',
    re.IGNORECASE,
)

# Relative-clause chain pattern: "that|which|who|whose" introducing a subordinate clause.
# Two or more such clauses in a single query strongly suggest multi-hop chaining.
_REL_CLAUSE_RE = re.compile(r"\b(?:that|which|who|whose)\b", re.IGNORECASE)

# Quantitative / arithmetic override — these queries need precise lookup of a
# small number of numeric inputs, not broad multi-hop graph traversal.
# A match → LOCAL_PRECISE regardless of length or relational keywords.
_QUANTITATIVE_RE = re.compile(
    r'\b(?:'
    r'how\s+(?:many|much|long|far|fast|slow|big|small|heavy|old)'
    r'|calculat(?:e|ion|ing)'
    r'|calculer'
    r'|comput(?:e|ing)'
    r'|convert(?:ing)?'
    r'|round(?:ing|ed)?'
    r'|nearest'
    r'|distanc(?:e|es)'
    r'|speed|pace|rate|ratio'
    r'|km/?h|mph|m/?s'
    r'|kilom(?:eter|etre)s?'
    r'|meters?|miles?|yards?|feet|foot'
    r'|seconds?|minutes?|hours?'
    r'|percent(?:age)?|%'
    r'|average|mean|median|total|sum'
    r'|multiply|divide|subtract|add'
    r'|formula|equation'
    r')\b',
    re.IGNORECASE,
)


def classify_query(query: str) -> QueryType:
    """Rule-based query classification — pure, no LLM (APP-INV-24).

    Priority order (first match wins):
      1. LOCAL_PRECISE (quantitative override) — arithmetic, units, calculations;
         these have 2-3 numeric inputs and don't benefit from wide graph traversal.
         Checked FIRST so "how many km/h" and "calculate pace" never fall into PROCEDURAL.
      2. PROCEDURAL  — step-by-step / how-to queries (excluding quantitative)
      3. DIAGNOSTIC  — root-cause / error queries
      4. MULTI_HOP   — ≥2 relational keywords AND ≥1 relative-clause chain;
         raw length is NOT a criterion (a long question can still be simple)
      5. LOCAL_PRECISE — short, specific
      6. GLOBAL_SYNTHETIC — everything else
    """
    q = query.lower()
    words = set(q.split())

    # Quantitative override FIRST: numeric computation is never procedural/diagnostic.
    if _QUANTITATIVE_RE.search(q):
        return QueryType.LOCAL_PRECISE

    if _PROCEDURAL_RE.search(q):
        return QueryType.PROCEDURAL

    if _DIAGNOSTIC_RE.search(q):
        return QueryType.DIAGNOSTIC

    multi_hop_hits = len(words & _MULTI_HOP_KEYWORDS)
    rel_clause_hits = len(_REL_CLAUSE_RE.findall(q))

    # MULTI_HOP: three independent signals (first match wins).
    # 1. Dense comparative keyword set (≥3 hits) — "compare/contrast/both/across/all"
    #    queries are multi-hop even without relative clauses.
    # 2. Mixed signal: ≥2 relational keywords + ≥1 relative clause.
    # 3. Strong chaining: ≥2 relative clauses alone.
    # Raw length is NOT a criterion.
    if multi_hop_hits >= 3:
        return QueryType.MULTI_HOP
    if multi_hop_hits >= 2 and rel_clause_hits >= 1:
        return QueryType.MULTI_HOP
    if rel_clause_hits >= 2:
        return QueryType.MULTI_HOP

    # Heuristic: short, specific queries → LOCAL_PRECISE; longer → GLOBAL_SYNTHETIC
    token_count = len(query.split())
    if token_count <= 8:
        return QueryType.LOCAL_PRECISE

    return QueryType.GLOBAL_SYNTHETIC


# ── Retrieval plan factory ────────────────────────────────────────────────────

_QUERY_TYPE_PLANS: dict[str, dict[str, Any]] = {
    QueryType.LOCAL_PRECISE: dict(
        recall_top_k=20, rerank_top_n=10,
        geometric_expand=False, geometric_seeds_k=0,
        coverage_threshold=0.7,  estimated_hops=1,
    ),
    QueryType.GLOBAL_SYNTHETIC: dict(
        recall_top_k=40, rerank_top_n=20,
        geometric_expand=True,  geometric_seeds_k=10,
        coverage_threshold=0.6,  estimated_hops=1,
    ),
    QueryType.MULTI_HOP: dict(
        recall_top_k=40, rerank_top_n=20,
        geometric_expand=True,  geometric_seeds_k=15,
        coverage_threshold=0.65, estimated_hops=2,  # refined by MultiHopAnalyzer
    ),
    QueryType.PROCEDURAL: dict(
        recall_top_k=20, rerank_top_n=10,
        geometric_expand=True,  geometric_seeds_k=5,
        coverage_threshold=0.7,  estimated_hops=1,
    ),
    QueryType.DIAGNOSTIC: dict(
        recall_top_k=30, rerank_top_n=15,
        geometric_expand=True,  geometric_seeds_k=10,
        coverage_threshold=0.65, estimated_hops=1,
    ),
}


def make_retrieval_plan(query_type: str) -> RetrievalPlan:
    """Return the canonical RetrievalPlan for *query_type* (APP-INV-24)."""
    params = _QUERY_TYPE_PLANS.get(query_type, _QUERY_TYPE_PLANS[QueryType.LOCAL_PRECISE])
    return RetrievalPlan(query_type=QueryType(query_type), **params)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredCItem]],
    k: int = 60,
) -> list[ScoredCItem]:
    """Merge multiple ranked lists with RRF (CIMA A-4, INV-22).

    rrf_score(d) = Σ 1 / (k + rank_i(d))
    """
    scores: dict[str, float] = defaultdict(float)
    items:  dict[str, ScoredCItem] = {}

    for ranked in ranked_lists:
        for rank, scored in enumerate(ranked, start=1):
            cid = scored.citem.citem_id
            scores[cid] += 1.0 / (k + rank)
            if cid not in items:
                items[cid] = scored

    merged = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: -x[1]):
        entry = items[cid]
        merged.append(ScoredCItem(
            citem=entry.citem,
            score=round(rrf_score, 8),
            provenance=entry.provenance,
            dense_score=entry.dense_score,    # preserve cosine from Qdrant
            rerank_score=entry.rerank_score,  # 0.0 until CrossEncoder runs
        ))

    return merged


# ── Q3 (flagged) local relevance gate ────────────────────────────────────────

def filter_q3_by_local_relevance(
    q3_items: list[ScoredCItem],
    query_facets: QueryFacets,
    anchors: list[ScoredCItem],
    *,
    min_facet_overlap: int = 1,
    anchor_sim_threshold: float = 0.15,
) -> tuple[list[ScoredCItem], int]:
    """Return only q3 (flagged) items that are locally relevant.

    An item passes if at least one condition holds:
    - ≥ min_facet_overlap query facets appear in its content, OR
    - Jaccard similarity ≥ anchor_sim_threshold to any of the top-3 anchors
      (proxy for "this conflict may alter local resolution").

    Both conditions are cheap (text only, no embedding calls).
    Returns (relevant_items, total_before_filter).
    """
    total = len(q3_items)
    if not q3_items:
        return [], total

    has_facets = bool(query_facets.terms or query_facets.numbers or query_facets.phrases)
    top_anchors = anchors[:3]

    relevant: list[ScoredCItem] = []
    for sc in q3_items:
        # Signal 1: overlap with query facets
        if has_facets:
            covered = _covered_query_facets(sc, query_facets)
            if len(covered) >= min_facet_overlap:
                relevant.append(sc)
                continue

        # Signal 2: Jaccard proximity to top reranked anchors (requires disambiguation)
        for anchor in top_anchors:
            if _jaccard_text(_candidate_text(sc), _candidate_text(anchor)) >= anchor_sim_threshold:
                relevant.append(sc)
                break

    return relevant, total


# ── Global OBSERVATION relevance gate ────────────────────────────────────────

def is_relevant_global_obs_content(
    content: str,
    query_facets: QueryFacets,
    *,
    min_facet_overlap: int = 1,
) -> bool:
    """True if a global OBSERVATION content is locally relevant.

    Pass-through when no query facets are available (defensive fallback).
    Requires ≥ min_facet_overlap query facets to appear in the text.
    """
    if not query_facets.terms and not query_facets.numbers and not query_facets.phrases:
        return True  # no facets to gate on — pass through

    text = content.lower()
    covered: set[str] = set()
    for term in query_facets.terms:
        if term in text:
            covered.add(f"t:{term}")
    for num in query_facets.numbers:
        if num in text:
            covered.add(f"n:{num}")
    for phrase in query_facets.phrases:
        if phrase in text:
            covered.add(f"p:{phrase}")
    return len(covered) >= min_facet_overlap


def is_relevant_global_obs_candidate(
    sc: ScoredCItem,
    query_facets: QueryFacets,
    *,
    min_facet_overlap: int = 1,
) -> bool:
    """Wrapper around is_relevant_global_obs_content for ScoredCItem."""
    return is_relevant_global_obs_content(
        _candidate_text(sc),
        query_facets,
        min_facet_overlap=min_facet_overlap,
    )


# ── Greedy token budget selection ─────────────────────────────────────────────

# Context slot caps (fraction of available_for_content) — default / GLOBAL_SYNTHETIC profile
_SLOT_CAPS: dict[str, float] = {
    "direct_evidence":  0.50,
    "bridge_evidence":  0.15,
    "global_summaries": 0.25,
    "conflicts":        0.10,
}

# Packing profiles by query type.
# LOCAL_PRECISE / PROCEDURAL: narrow — few items, tight direct-evidence cap.
# MULTI_HOP: medium — slightly more bridge room, less global.
# DIAGNOSTIC: medium — similar to multi-hop.
# GLOBAL_SYNTHETIC: current generous profile (same as _SLOT_CAPS).
_SLOT_CAPS_BY_QUERY_TYPE: dict[str, dict[str, float]] = {
    QueryType.LOCAL_PRECISE: {
        "direct_evidence":  0.40,
        "bridge_evidence":  0.10,
        "global_summaries": 0.15,
        "conflicts":        0.10,
    },
    QueryType.PROCEDURAL: {
        "direct_evidence":  0.40,
        "bridge_evidence":  0.10,
        "global_summaries": 0.15,
        "conflicts":        0.10,
    },
    QueryType.DIAGNOSTIC: {
        "direct_evidence":  0.45,
        "bridge_evidence":  0.12,
        "global_summaries": 0.20,
        "conflicts":        0.10,
    },
    QueryType.MULTI_HOP: {
        "direct_evidence":  0.45,
        "bridge_evidence":  0.15,
        "global_summaries": 0.20,
        "conflicts":        0.10,
    },
    QueryType.GLOBAL_SYNTHETIC: {
        "direct_evidence":  0.50,
        "bridge_evidence":  0.15,
        "global_summaries": 0.25,
        "conflicts":        0.10,
    },
}


def slot_caps_for_query_type(query_type: str) -> dict[str, float]:
    """Return slot cap fractions for the given query type (packing profile)."""
    return _SLOT_CAPS_BY_QUERY_TYPE.get(query_type, _SLOT_CAPS)

_PROTECTED_TYPES = frozenset({
    ItemType.DECISION,
    ItemType.CONSTRAINT,
    ItemType.PROCEDURE,
})

# ── ABS: Anchor-Bridge Selector ───────────────────────────────────────────────
# Three-slot evidence selection optimised for multi-hop ≤ 2.
#
# Slot 1 (Anchor):  max(rerank_score)         — chunk that directly answers
# Slot 2 (Bridge):  max(α·dense_norm + (1-α)·rerank_norm)
#                   — chunk with high semantic proximity but medium direct relevance
#                     (the multi-hop connector); excluded from Slot 1 by construction
# Slots 3+ (Fill):  greedy by CVS density (sc.score after apply_cvs_density_scoring)
#                   — adaptive to phase, recency, and novelty; fills remaining budget
_ABS_ALPHA = 0.5  # balance between dense and rerank signal for bridge detection


def extract_query_facets(query: str) -> QueryFacets:
    quoted = []
    for m in re.finditer(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', query):
        phrase = next((g for g in m.groups() if g), None)
        if phrase:
            quoted.append(phrase.strip().lower())

    raw_tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+", query.lower())
    numbers = {t for t in raw_tokens if any(ch.isdigit() for ch in t)}
    terms = {
        t for t in raw_tokens
        if len(t) > 2 and t not in _STOPWORDS and t not in numbers
    }

    return QueryFacets(
        terms=terms,
        numbers=numbers,
        phrases=tuple(dict.fromkeys(quoted)),
    )


def _is_redundant_direct(
        candidate: ScoredCItem,
        selected_direct: list[ScoredCItem],
        threshold: float = 0.92,
) -> bool:
    for existing in selected_direct:
        if _similarity(candidate, existing) >= threshold:
            return True
    return False

def _marginal_facet_gain(
        candidate: ScoredCItem,
        selected_direct: list[ScoredCItem],
        query_facets: QueryFacets,
) -> float:
    if not query_facets.terms and not query_facets.numbers and not query_facets.phrases:
        return 1.0

    already: set[str] = set()
    for sc in selected_direct:
        already |= _covered_query_facets(sc, query_facets)

    candidate_cov = _covered_query_facets(candidate, query_facets)
    new_cov = candidate_cov - already

    total = len(query_facets.terms) + len(query_facets.numbers) + len(query_facets.phrases)
    if total <= 0:
        return 1.0
    return len(new_cov) / total


def _fallback_key(sc: ScoredCItem) -> tuple[float, float, float, str]:
    return (
        sc.value_density if sc.value_density is not None else float("-inf"),
        sc.cvs_score if sc.cvs_score is not None else float("-inf"),
        sc.rerank_score if sc.rerank_score is not None else float("-inf"),
        sc.citem.citem_id,
    )

def _abs_select(
    pool: list[ScoredCItem],
    *,
    direct_cap: int,
    bridge_cap: int,
    query_facets: QueryFacets | None,
    direct_strategy: DirectEvidenceStrategy,
    bridge_policy: BridgePolicy,
) -> tuple[list[ScoredCItem], list[ScoredCItem], dict[str, int]]:
    """ABS selector:
    - anchors por rerank_score
    - bridges por bridge_score
    - fallback por value_density/cvs_score

    Returns (direct_items, bridge_items, bridge_stats) where bridge_stats contains:
      candidates, selected, dropped_exact, dropped_near_identical
    """
    _empty_stats: dict[str, int] = {
        "candidates": 0, "selected": 0,
        "dropped_exact": 0, "dropped_near_identical": 0,
    }

    if not pool or direct_cap <= 0:
        return [], [], _empty_stats

    # Normalize query_facets so closures below never see None
    _qf: QueryFacets = query_facets if query_facets is not None else QueryFacets()

    # Fallback completo cuando bridge lane no está habilitado
    if (
        direct_strategy != DirectEvidenceStrategy.ANCHOR_BRIDGE_INTERLEAVED
        or bridge_policy is None
        or not bridge_policy.enabled
    ):
        direct_items: list[ScoredCItem] = []
        used_direct = 0

        for sc in sorted(pool, key=_fallback_key, reverse=True):
            tc = _candidate_tokens(sc)
            if used_direct + tc > direct_cap:
                continue
            # usa query_facets también en fallback para no repetir anchors vacíos
            if direct_items:
                gain = _marginal_facet_gain(sc, direct_items, _qf)
                if gain <= 0.0 and _is_redundant_direct(sc, direct_items):
                    continue
            direct_items.append(sc)
            used_direct += tc

        return direct_items, [], _empty_stats

    direct_items: list[ScoredCItem] = []
    bridge_items: list[ScoredCItem] = []
    seen_ids: set[str] = set()

    used_direct = 0
    used_bridge = 0

    # Bridge observability counters
    _bridge_dropped_exact = 0
    _bridge_dropped_near_identical = 0

    anchor_lane = sorted(
        [
            sc for sc in pool
            if sc.rerank_score is not None
            and sc.rerank_score >= bridge_policy.anchor_floor
        ],
        key=lambda sc: (
            sc.rerank_score if sc.rerank_score is not None else float("-inf"),
            sc.dense_score if sc.dense_score is not None else float("-inf"),
            sc.citem.citem_id,
        ),
        reverse=True,
    )

    bridge_lane = sorted(
        [
            sc for sc in pool
            if sc.bridge_score is not None
            and sc.bridge_score >= bridge_policy.bridge_floor
        ],
        key=lambda sc: (
            sc.bridge_score if sc.bridge_score is not None else float("-inf"),
            sc.rerank_score if sc.rerank_score is not None else float("-inf"),
            sc.citem.citem_id,
        ),
        reverse=True,
    )

    i_anchor = 0
    i_bridge = 0
    lane = "anchor"
    stalled_rounds = 0
    low_bridge_run = 0

    def _pick_anchor() -> bool:
        nonlocal i_anchor, used_direct
        while i_anchor < len(anchor_lane):
            scitem = anchor_lane[i_anchor]
            i_anchor += 1
            cid = scitem.citem.citem_id
            if cid in seen_ids:
                continue

            tc = _candidate_tokens(scitem)
            if used_direct + tc > direct_cap:
                continue

            if direct_items:
                gain = _marginal_facet_gain(scitem, direct_items, _qf)
                if gain <= 0.0 and _is_redundant_direct(scitem, direct_items):
                    continue

            direct_items.append(scitem)
            used_direct += tc
            seen_ids.add(cid)
            return True
        return False

    def _pick_bridge() -> bool:
        nonlocal i_bridge, used_bridge, low_bridge_run
        nonlocal _bridge_dropped_exact, _bridge_dropped_near_identical
        while i_bridge < len(bridge_lane):
            sc = bridge_lane[i_bridge]
            i_bridge += 1
            item = sc.citem
            cid = item.citem_id
            if cid in seen_ids:
                # Exact dedup: already selected as anchor (or other slot)
                _bridge_dropped_exact += 1
                continue

            tc = _candidate_tokens(sc)
            if used_bridge + tc > bridge_cap:
                continue

            if sc.bridge_score is None or sc.bridge_score < bridge_policy.bridge_floor:
                low_bridge_run += 1
                if low_bridge_run >= bridge_policy.max_consecutive_low_bridge:
                    return False
                continue

            # Soft dedup: bridge-to-bridge only (NOT against direct_items).
            # A bridge similar to an anchor is kept — that's its job.
            if is_duplicate_bridge(sc, bridge_items, bridge_policy.max_bridge_redundancy):
                _bridge_dropped_near_identical += 1
                continue

            bridge_items.append(sc)
            used_bridge += tc
            seen_ids.add(cid)
            low_bridge_run = 0
            return True
        return False

    # Intercalado anchor/bridge
    while True:
        progressed = False

        if lane == "anchor":
            progressed = _pick_anchor()
            lane = "bridge"
            if not progressed:
                progressed = _pick_bridge()
                lane = "anchor"
        else:
            progressed = _pick_bridge()
            lane = "anchor"
            if not progressed:
                progressed = _pick_anchor()
                lane = "bridge"

        if progressed:
            stalled_rounds = 0
            continue

        stalled_rounds += 1
        if stalled_rounds >= 2:
            break

    # Fill final SOLO para direct_evidence por value_density
    if used_direct < direct_cap:
        fallback_pool = [
            sc for sc in pool
            if sc.citem.citem_id not in seen_ids
        ]
        for sc in sorted(fallback_pool, key=_fallback_key, reverse=True):
            item = sc.citem
            tc = _candidate_tokens(sc)
            if used_direct + tc > direct_cap:
                continue

            if direct_items:
                gain = _marginal_facet_gain(sc, direct_items, _qf)
                if gain <= 0.0 and _is_redundant_direct(sc, direct_items):
                    continue

            direct_items.append(sc)
            used_direct += tc
            seen_ids.add(item.citem_id)

    _bridge_stats: dict[str, int] = {
        "candidates": len(bridge_lane),
        "selected": len(bridge_items),
        "dropped_exact": _bridge_dropped_exact,
        "dropped_near_identical": _bridge_dropped_near_identical,
    }
    return direct_items, bridge_items, _bridge_stats

def _provenance_name(sc: ScoredCItem) -> str:
    prov = getattr(sc, "provenance", None)
    if prov is None:
        return ""
    return getattr(prov, "name", str(prov)).lower()

def is_conflict_candidate(sc: ScoredCItem) -> bool:
    pname = _provenance_name(sc)
    if "flag" in pname or "conflict" in pname:
        return True
    item_type = getattr(sc.citem, "item_type", None)
    if item_type is not None:
        tname = getattr(item_type, "name", str(item_type)).lower()
        if "conflict" in tname:
            return True
    return False


def _candidate_embedding(sc: ScoredCItem) -> list[float] | None:
    emb = getattr(sc.citem, "embedding", None)
    if isinstance(emb, list) and emb:
        return emb
    return None

def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den_a = math.sqrt(sum(x * x for x in a))
    den_b = math.sqrt(sum(y * y for y in b))
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return num / (den_a * den_b)

def _token_set(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())
        if len(t) > 2 and t not in _STOPWORDS
    }

def _jaccard_text(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _candidate_text(sc: ScoredCItem) -> str:
    for attr in ("content", "text", "summary", "raw_text"):
        value = getattr(sc.citem, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""

def _similarity(a: ScoredCItem, b: ScoredCItem) -> float:
    ea = _candidate_embedding(a)
    eb = _candidate_embedding(b)
    if ea is not None and eb is not None and len(ea) == len(eb):
        return _cosine(ea, eb)
    return _jaccard_text(_candidate_text(a), _candidate_text(b))


def is_duplicate_bridge(
    candidate: ScoredCItem,
    selected_bridges: list[ScoredCItem],
    threshold: float,
) -> bool:
    if not selected_bridges:
        return False

    for existing in selected_bridges:
        if _similarity(candidate, existing) >= threshold:
            return True
    return False

def is_geometric_candidate(sc: ScoredCItem) -> bool:
    return "geometric" in _provenance_name(sc)

def _covered_query_facets(sc: ScoredCItem, query_facets: QueryFacets) -> set[str]:
    text = _candidate_text(sc).lower()
    covered: set[str] = set()

    for term in query_facets.terms:
        if term in text:
            covered.add(f"t:{term}")

    for num in query_facets.numbers:
        if num in text:
            covered.add(f"n:{num}")

    for phrase in query_facets.phrases:
        if phrase in text:
            covered.add(f'p:{phrase}')

    return covered

def select_direct_evidence_interleaved(
    candidates: list[ScoredCItem],
    token_budget: int,
    bridge_policy: BridgePolicy,
    query_facets: QueryFacets | None = None,
) -> tuple[list[ScoredCItem], list[ScoredCItem], int, set[str]]:
    if token_budget <= 0 or not candidates:
        return [], [], 0, set()

    selected_direct: list[ScoredCItem] = []
    selected_bridges: list[ScoredCItem] = []
    used_tokens = 0
    seen_ids: set[str] = set()

    direct_pool = [
        sc for sc in candidates
        if not is_conflict_candidate(sc) and not is_geometric_candidate(sc)
    ]

    anchor_lane = sorted(
        [
            sc for sc in direct_pool
            if sc.rerank_score is not None and sc.rerank_score >= bridge_policy.anchor_floor
        ],
        key=lambda sc: (
            sc.rerank_score if sc.rerank_score is not None else float("-inf"),
            sc.dense_score if sc.dense_score is not None else float("-inf"),
            sc.citem.citem_id,
        ),
        reverse=True,
    )

    bridge_lane = sorted(
        [
            sc for sc in direct_pool
            if sc.bridge_score is not None and sc.bridge_score >= bridge_policy.bridge_floor
        ],
        key=lambda sc: (
            sc.bridge_score if sc.bridge_score is not None else float("-inf"),
            sc.rerank_score if sc.rerank_score is not None else float("-inf"),
            sc.citem.citem_id,
        ),
        reverse=True,
    )

    i_anchor = 0
    i_bridge = 0
    lane = "anchor"
    low_bridge_run = 0

    def _candidate_tokens(sc: ScoredCItem) -> int:
        for attr in ("token_count", "tokens", "n_tokens"):
            value = getattr(sc.citem, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        text = _candidate_text(sc)
        return max(1, len(re.findall(r"\w+", text)))

    def _fits(sc: ScoredCItem) -> bool:
        return used_tokens + _candidate_tokens(sc) <= token_budget

    def _pick_next_anchor() -> ScoredCItem | None:
        nonlocal i_anchor
        while i_anchor < len(anchor_lane):
            sc = anchor_lane[i_anchor]
            i_anchor += 1
            cid = sc.citem.citem_id
            if cid in seen_ids or not _fits(sc):
                continue

            # query_facets se usa aquí de forma real
            if query_facets is not None and selected_direct:
                gain = _marginal_facet_gain(sc, selected_direct, query_facets)
                if gain <= 0.0 and _is_redundant_direct(sc, selected_direct):
                    continue

            return sc
        return None

    def _pick_next_bridge() -> ScoredCItem | None:
        nonlocal i_bridge, low_bridge_run
        while i_bridge < len(bridge_lane):
            sc = bridge_lane[i_bridge]
            i_bridge += 1
            cid = sc.citem.citem_id
            if cid in seen_ids or not _fits(sc):
                continue

            if sc.bridge_score is None or sc.bridge_score < bridge_policy.bridge_floor:
                low_bridge_run += 1
                if low_bridge_run >= bridge_policy.max_consecutive_low_bridge:
                    return None
                continue

            if is_duplicate_bridge(sc, selected_bridges, bridge_policy.max_bridge_redundancy):
                continue

            low_bridge_run = 0
            return sc
        return None

    while used_tokens < token_budget:
        picked: ScoredCItem | None = None

        if lane == "anchor":
            picked = _pick_next_anchor()
            if picked is None and bridge_policy.enabled:
                picked = _pick_next_bridge()
            lane = "bridge"
        else:
            if bridge_policy.enabled:
                picked = _pick_next_bridge()
            if picked is None:
                picked = _pick_next_anchor()
            lane = "anchor"

        if picked is None:
            break

        seen_ids.add(picked.citem.citem_id)
        used_tokens += _candidate_tokens(picked)

        # bridge real: solo si bridge_policy enabled y el item estaba en lane bridge
        is_bridge_pick = (
            bridge_policy.enabled
            and picked.bridge_score is not None
            and picked.bridge_score >= bridge_policy.bridge_floor
            and picked in bridge_lane
            and picked not in selected_bridges
            and len(selected_bridges) < len(selected_direct) + 1
        )

        if is_bridge_pick:
            selected_bridges.append(picked)
        else:
            selected_direct.append(picked)

    return selected_direct, selected_bridges, used_tokens, seen_ids

def _policy_multiplier(phase_policy: Any, phase: str, key: str, default: float = 1.0) -> float:
    if phase_policy is None:
        return default

    if isinstance(phase_policy, dict):
        phase_cfg = phase_policy.get(phase)
        if isinstance(phase_cfg, dict) and key in phase_cfg:
            return float(phase_cfg[key])
        if key in phase_policy:
            return float(phase_policy[key])
        return default

    phase_cfg = getattr(phase_policy, phase, None)
    if phase_cfg is not None and hasattr(phase_cfg, key):
        return float(getattr(phase_cfg, key))
    if hasattr(phase_policy, key):
        return float(getattr(phase_policy, key))
    return default

def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("-inf")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac

def _minmax_norm(v: float | None, lo: float, hi: float) -> float:
    if v is None:
        return 0.0
    if hi <= lo:
        return 1.0
    return (v - lo) / (hi - lo)

def compute_mean_sim_to_pool(
    vectors: dict[str, list[float]],
    citem_ids: list[str],
) -> dict[str, float]:
    """O(n²) pairwise cosine → per-item mean similarity to pool.

    Returns {citem_id: mean_sim} for IDs present in *vectors*.
    Items not in *vectors* are omitted.
    """
    import numpy as np

    ids_with_vecs = [cid for cid in citem_ids if cid in vectors]
    if len(ids_with_vecs) < 2:
        return {}

    mat = np.array([vectors[cid] for cid in ids_with_vecs], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat_normed = mat / norms
    sim_matrix = mat_normed @ mat_normed.T
    np.fill_diagonal(sim_matrix, 0.0)
    n = len(ids_with_vecs)
    mean_sims = sim_matrix.sum(axis=1) / max(n - 1, 1)
    return {cid: float(mean_sims[i]) for i, cid in enumerate(ids_with_vecs)}


# Bridge scoring weights — validated in rag-pressure-benchmark (Task B AUC 0.703)
_W_DENSE = 0.4
_W_RERANK = 0.4
_W_CENTRALITY = 0.2


def annotate_bridge_scores(
        candidates: list[ScoredCItem],
        *,
        mean_sim_map: dict[str, float] | None = None,
        w_dense: float = _W_DENSE,
        w_rerank: float = _W_RERANK,
        w_centrality: float = _W_CENTRALITY,
) -> None:
    """Annotate bridge_score using the 3-signal formula from pressure benchmark.

    bridge_score = w_dense·dense_norm + w_rerank·rerank_norm + w_centrality·mean_sim_norm

    Falls back to 2-signal (0.5·dense + 0.5·rerank) when mean_sim_map is empty.
    """
    dense_vals = [sc.dense_score for sc in candidates if sc.dense_score is not None]
    rerank_vals = [sc.rerank_score for sc in candidates if sc.rerank_score is not None]

    if not dense_vals or not rerank_vals:
        for sc in candidates:
            sc.bridge_score = None
        return

    d_lo, d_hi = min(dense_vals), max(dense_vals)
    r_lo, r_hi = min(rerank_vals), max(rerank_vals)

    has_centrality = bool(mean_sim_map)
    if has_centrality:
        sim_vals = [mean_sim_map[sc.citem.citem_id]
                    for sc in candidates
                    if sc.citem.citem_id in mean_sim_map]
        if sim_vals:
            s_lo, s_hi = min(sim_vals), max(sim_vals)
        else:
            has_centrality = False

    for sc in candidates:
        if sc.dense_score is None or sc.rerank_score is None:
            sc.bridge_score = None
            continue

        d = _minmax_norm(sc.dense_score, d_lo, d_hi)
        r = _minmax_norm(sc.rerank_score, r_lo, r_hi)

        if has_centrality and sc.citem.citem_id in mean_sim_map:
            s = _minmax_norm(mean_sim_map[sc.citem.citem_id], s_lo, s_hi)
            sc.bridge_score = w_dense * d + w_rerank * r + w_centrality * s
        else:
            sc.bridge_score = 0.5 * d + 0.5 * r

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _candidate_tokens(sc: ScoredCItem) -> int:
    for attr in ("token_count", "tokens", "n_tokens"):
        value = getattr(sc.citem, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    text = _candidate_text(sc)
    return max(1, len(re.findall(r"\w+", text)))


def apply_cvs_density_scoring(
        candidates: list[ScoredCItem],
        phase: str,
        cvs_weights,
        phase_policy,
) -> list[ScoredCItem]:
    if not candidates:
        return candidates

    raw_relevance: list[float] = []
    for sc in candidates:
        rel = (
            sc.rerank_score
            if sc.rerank_score is not None
            else sc.rrf_score
            if sc.rrf_score is not None
            else sc.dense_score
            if sc.dense_score is not None
            else sc.score
        )
        raw_relevance.append(float(rel))

    rel_lo = min(raw_relevance)
    rel_hi = max(raw_relevance)

    recency_mul = _policy_multiplier(phase_policy, phase, "recency_multiplier", 1.0)
    novelty_mul = _policy_multiplier(phase_policy, phase, "novelty_multiplier", 1.0)

    for sc, rel in zip(candidates, raw_relevance):
        content_relevance = _minmax_norm(rel, rel_lo, rel_hi)

        base_recency = float(getattr(sc.citem, "recency_score", 0.0))
        base_novelty = float(getattr(sc.citem, "novelty_score", 0.0))

        recency_score = _clamp01(base_recency * recency_mul)
        novelty_score = _clamp01(base_novelty * novelty_mul)

        sc.cvs_score = compute_cvs(
            content_relevance=content_relevance,
            recency_score=recency_score,
            novelty_score=novelty_score,
            weights=cvs_weights,
        )

        tokens = max(1, _candidate_tokens(sc))
        sc.value_density = sc.cvs_score / tokens

    candidates.sort(
        key=lambda sc: (
            sc.value_density if sc.value_density is not None else float("-inf"),
            sc.cvs_score if sc.cvs_score is not None else float("-inf"),
            sc.citem.citem_id,
        ),
        reverse=True,
    )
    return candidates

def greedy_select(
    candidates: list[ScoredCItem],
    budget: ContextBudget,
    active_procedure_id: str | None = None,
    *,
    query_facets: QueryFacets | None = None,
    direct_strategy: DirectEvidenceStrategy = DirectEvidenceStrategy.CVS_DENSITY,
    bridge_policy: BridgePolicy | None = None,
    slot_caps_override: dict[str, float] | None = None,
) -> tuple[ContextPack, BudgetReport]:
    """Fill ContextPack slots respecting token caps (CIMA A-8, APP-INV-22).

    D-08: active PROCEDURE is forced to PROTECTED slot (A-8.4 conformance).

    Budget accounting:
    1. First pass: collect all protected items (no cap) and measure their tokens.
    2. Slot caps for capped slots are computed over (available - protected_tokens).
       slot_caps_override (from slot_caps_for_query_type) allows narrower profiles.
    3. Second pass: conflicts + global summaries + evidence.
       Global OBSERVATION candidates are gated by is_relevant_global_obs_candidate()
       when query_facets are available — irrelevant global observations are dropped.
    4. Evidence uses either:
       - CVS_DENSITY fallback, or
       - ANCHOR_BRIDGE_INTERLEAVED with:
         * direct_evidence from rerank anchors
         * bridge_evidence from bridge_score (bridge-to-bridge soft dedup only)

    Budget is a ceiling, not an obligation. Underfill is explicitly allowed.
    """
    available = budget.available_for_content

    if bridge_policy is None:
        bridge_policy = BridgePolicy.disabled()

    _qf = query_facets or QueryFacets()
    _slot_caps = slot_caps_override if slot_caps_override is not None else _SLOT_CAPS

    pack = ContextPack()
    seen: set[str] = set()
    used: dict[str, int] = {
        "protected": 0,
        "direct_evidence": 0,
        "bridge_evidence": 0,
        "global_summaries": 0,
        "conflicts": 0,
    }

    # ── First pass: protected items (no cap) ───────────────────────────────
    non_protected: list[ScoredCItem] = []
    for scored in candidates:
        scored_item: ScoredCItem = scored
        item = scored.citem
        if item.citem_id in seen:
            continue

        tokens = _candidate_tokens(scored_item)
        is_active_proc = bool(active_procedure_id and item.citem_id == active_procedure_id)

        if is_active_proc or item.item_type in _PROTECTED_TYPES:
            pack.protected_items.append(item)
            used["protected"] += tokens
            seen.add(item.citem_id)
        else:
            non_protected.append(scored)

    remaining = max(0, available - used["protected"])
    caps = {slot: int(remaining * frac) for slot, frac in _slot_caps.items()}

    # ── Second pass: conflicts + global summaries + evidence pool ──────────
    def fits(slot: str, tokens: int) -> bool:
        return used[slot] + tokens <= caps.get(slot, 0)

    def add(slot_name: str, lst: list[CItem], scitem: ScoredCItem) -> None:
        lst.append(scitem.citem)
        used[slot_name] += _candidate_tokens(scitem)
        seen.add(scitem.citem.citem_id)

    evidence_pool: list[ScoredCItem] = []
    _global_obs_candidates = 0
    _global_obs_admitted = 0
    _q3_injected = 0

    for scored in non_protected:
        item = scored.citem
        tokens = _candidate_tokens(scored)

        if item.conflict_status == "flagged":
            if fits("conflicts", tokens):
                add("conflicts", pack.conflicts, scored)
                _q3_injected += 1
            continue

        if item.scope == "global" and item.item_type == ItemType.OBSERVATION:
            _global_obs_candidates += 1
            # Gate: require local relevance before occupying global_summaries budget.
            # Pass-through when no query facets are available (defensive fallback).
            if not is_relevant_global_obs_candidate(scored, _qf):
                continue  # drop: irrelevant global observation
            if fits("global_summaries", tokens):
                add("global_summaries", pack.global_summaries, scored)
                _global_obs_admitted += 1
            continue

        evidence_pool.append(scored)

    # ── ABS / fallback selection ───────────────────────────────────────────
    direct_items, bridge_items, bridge_stats = _abs_select(
        pool=evidence_pool,
        direct_cap=caps["direct_evidence"],
        bridge_cap=caps["bridge_evidence"],
        query_facets=query_facets,
        direct_strategy=direct_strategy,
        bridge_policy=bridge_policy,
    )

    for item in direct_items:
        add("direct_evidence", pack.direct_evidence, item)

    for item in bridge_items:
        add("bridge_evidence", pack.bridge_evidence, item)

    total_used = sum(used.values())
    pack.tokens_used = total_used

    # ── Observability ──────────────────────────────────────────────────────
    underfill = total_used < max(1, int(0.05 * available)) and len(candidates) > 0
    if underfill:
        log.debug(
            "abs_underfill: used=%d available=%d items_selected=%d",
            total_used, available, len(seen),
        )
    log.debug(
        "greedy_select: slot_caps=%s bridge_candidates=%d bridge_selected=%d "
        "bridge_dropped_exact=%d bridge_dropped_near_identical=%d "
        "global_obs_candidates=%d global_obs_admitted=%d q3_injected=%d",
        {k: caps.get(k) for k in _slot_caps},
        bridge_stats["candidates"],
        bridge_stats["selected"],
        bridge_stats["dropped_exact"],
        bridge_stats["dropped_near_identical"],
        _global_obs_candidates,
        _global_obs_admitted,
        _q3_injected,
    )

    report = BudgetReport(
        total_available=available,
        tokens_used=total_used,
        items_selected=len(seen),
    )
    return pack, report


# ── Coverage evaluation ───────────────────────────────────────────────────────

_CONCEPT_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "do", "does", "did", "will", "would", "could", "should",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "with", "from", "that", "this", "it", "its",
})


def _extract_concepts(text: str) -> set[str]:
    """Simple word-level concept extraction (lowercase, filtered)."""
    words = text.lower().split()
    return {w.strip(".,;:?!\"'") for w in words if w not in _CONCEPT_STOPWORDS and len(w) > 3}


def filter_pack_by_objectives(
    pack: "ContextPack",
    global_objective: str,
    local_objective: str,
) -> "ContextPack":
    """Drop global_summaries that serve neither the global nor the local objective.

    SCOPE: Only global_summaries are filtered. direct_evidence and bridge_evidence
    are already ranked by Qdrant + CrossEncoder rerank against the current query —
    filtering them with word-overlap would degrade precision (different vocabulary)
    and, critically, would discard the only full copy of tool-result content
    (history entries are truncated via trim_tool_results).

    global_summaries come from fetch_pyramid_tops() with NO per-query scoring,
    so word-overlap is the right gate to prevent cross-domain pyramid tops from
    consuming summary budget in unrelated conversations.

    Always kept regardless of overlap:
    - direct_evidence  (scored by retrieval pipeline — do not second-guess)
    - bridge_evidence  (geometric expansion — dependency-linked, trust the graph)
    - protected_items  (DECISION, CONSTRAINT, PROCEDURE — structural knowledge)
    - conflicts        (must surface unconditionally)

    If both objectives are empty the pack is returned unchanged (safety fallback).
    """
    global_concepts = _extract_concepts(global_objective)
    local_concepts  = _extract_concepts(local_objective)
    combined        = global_concepts | local_concepts
    if not combined:
        return pack  # nothing to filter against — pass through

    def _relevant(item: "CItem") -> bool:
        return bool(_extract_concepts(item.content) & combined)

    pack.global_summaries = [c for c in pack.global_summaries if _relevant(c)]
    # direct_evidence, bridge_evidence, protected_items, conflicts: untouched
    return pack


def evaluate_coverage_text(
    query: str,
    text: str,
    threshold: float = 0.6,
) -> CoverageReport:
    """Evaluate coverage against pre-rendered text (e.g. ContextView.text).

    Same word-overlap logic as evaluate_coverage() but operates on a flat
    string instead of a ContextPack.  Used by the orchestrator drift monitors
    where only the ContextView is available, avoiding an extra retrieval call.
    """
    query_concepts = _extract_concepts(query)
    if not query_concepts:
        return CoverageReport(coverage_score=1.0)
    text_concepts = _extract_concepts(text)
    covered = query_concepts & text_concepts
    score = round(len(covered) / len(query_concepts), 4)
    missing = list(query_concepts - covered)
    return CoverageReport(
        coverage_score=score,
        missing_concepts=missing,
        retry_recommended=score < threshold,
    )


def adaptive_pre_rerank_k(
    rrf_scores: list[float],
    base_rerank_n: int,
    *,
    floor_ratio: float = 0.3,
    min_n: int = 5,
) -> int:
    """Trim the number of candidates sent to the CrossEncoder reranker.

    Detects items with negligible RRF scores (below floor_ratio of the
    top score) and excludes them.  On iGPU the CrossEncoder is the
    dominant latency source; sending 15 instead of 40 items saves ~60%
    rerank time without losing quality (low-RRF items rarely survive
    the reranker anyway).

    Returns a k in [min_n, base_rerank_n].
    """
    if len(rrf_scores) <= min_n:
        return len(rrf_scores)

    desc = sorted(rrf_scores, reverse=True)
    if desc[0] <= 0:
        return min(base_rerank_n, len(rrf_scores))

    floor = desc[0] * floor_ratio
    above = sum(1 for s in desc if s >= floor)
    k = max(min_n, above)
    return min(k, base_rerank_n, len(rrf_scores))


def adaptive_rerank_k(
    scores: list[float],
    base_top_n: int,
    *,
    cliff_ratio: float = 0.4,
    min_k: int = 5,
) -> int:
    """Compute adaptive top_n from rerank score distribution.

    Detects the "score cliff" — the point where the gap between consecutive
    scores exceeds cliff_ratio × (max - min).  Items past the cliff are
    low-relevance padding that would dilute the context.

    Returns a k in [min_k, base_top_n].  If no cliff is detected, returns
    base_top_n unchanged (safe fallback).
    """
    if len(scores) < 2:
        return max(min_k, min(len(scores), base_top_n))

    desc = sorted(scores, reverse=True)
    span = desc[0] - desc[-1]
    if span <= 0:
        return base_top_n

    threshold = span * cliff_ratio

    for i in range(len(desc) - 1):
        gap = desc[i] - desc[i + 1]
        if gap >= threshold:
            k = i + 1
            return max(min_k, min(k, base_top_n))

    return base_top_n


def adaptive_retry_k(
    base_recall_k: int,
    coverage_score: float,
    coverage_threshold: float,
) -> int:
    """Compute adaptive recall_top_k for coverage retry.

    Scales the retry factor by the coverage gap magnitude:
      gap = 0.1 → factor ~1.5x  (minor gap, light expansion)
      gap = 0.3 → factor ~2.5x  (large gap, aggressive expansion)
      gap ≥ 0.5 → factor  3.0x  (cap)
    """
    gap = max(0.0, coverage_threshold - coverage_score)
    factor = min(3.0, 1.0 + gap * 5.0)
    return round(base_recall_k * factor)


def evaluate_coverage(
    query: str,
    pack: ContextPack,
    threshold: float = 0.6,
) -> CoverageReport:
    """Evaluate how well the context pack covers the query concepts (INV-22).

    Returns CoverageReport with coverage_score, missing_concepts, retry_recommended.
    """
    query_concepts = _extract_concepts(query)
    if not query_concepts:
        return CoverageReport(coverage_score=1.0)

    covered = set()
    for item in pack.all_items():
        item_words = _extract_concepts(item.content)
        covered |= query_concepts & item_words

    score = round(len(covered) / len(query_concepts), 4)
    missing = list(query_concepts - covered)

    return CoverageReport(
        coverage_score=score,
        missing_concepts=missing,
        retry_recommended=score < threshold,
    )


# ── ContextPack → ContextView ─────────────────────────────────────────────────

def context_pack_to_view(pack: ContextPack) -> ContextView:
    """Serialise ContextPack to an LLM-injectable text plus structured item refs.

    The textual representation stays backwards-compatible for the runtime; the
    structured item list is used exclusively by the demonstrator for snapshots
    and lineage.
    """
    sections: list[str] = []
    items_meta: list[dict[str, Any]] = []
    marker_seq = 0

    def _marker() -> str:
        nonlocal marker_seq
        marker_seq += 1
        return f"S{marker_seq}"

    def _append_item_meta(header: str, item: CItem, *, ref_kind: str) -> None:
        meta = {
            "marker": _marker(),
            "ref_kind": ref_kind,
            "ref_id": item.citem_id,
            "section": header,
            "item_type": str(item.item_type),
            "actor": item.actor,
            "importance": round(float(item.importance), 6),
            "confidence": round(float(item.confidence), 6),
            "dependency_ids": list(item.dependency_ids),
            "content": item.content,
        }
        item_resolution_mode = str(getattr(item, "citem_resolution_mode", "") or "")
        if item_resolution_mode:
            meta["item_resolution_mode"] = item_resolution_mode
        item_resolution_scope = str(getattr(item, "citem_resolution_scope", "") or "")
        if item_resolution_scope:
            meta["item_resolution_scope"] = item_resolution_scope
        items_meta.append(meta)

    def _fmt_items(header: str, items: list[CItem], *, ref_kind: str = "citem") -> None:
        if not items:
            return
        lines = [f"## {header}"]
        for item in items:
            _append_item_meta(header, item, ref_kind=ref_kind)
            marker = items_meta[-1]["marker"]
            dep_note = f" [deps: {', '.join(item.dependency_ids[:3])}]" if item.dependency_ids else ""
            conf_note = f" [conf={item.confidence:.2f}]" if item.confidence < 1.0 else ""
            actor = item.actor or "agent"
            actor_tag = {"agent": "SELF", "user": "USER"}.get(actor, actor.upper()[:6])
            score_tag = f"{item.importance:.2f}"
            lines.append(f"- [{marker}][{actor_tag}|{score_tag}][{item.item_type}] {item.content}{dep_note}{conf_note}")
        sections.append("\n".join(lines))

    def _fmt_conflicts(items: list[CItem]) -> None:
        if not items:
            return
        lines = ["## CONFLICTS_DETECTED"]
        lines.append(
            "The following memory items contain possibly contradictory information. "
            "You must mention both versions and ask the user for clarification."
        )
        for item in items:
            _append_item_meta("Conflicts", item, ref_kind="citem")
            marker = items_meta[-1]["marker"]
            conf_note = f" [conf={item.confidence:.2f}]" if item.confidence < 1.0 else ""
            lines.append(f"- [{marker}][{item.item_type}] {item.content}{conf_note}")
        sections.append("\n".join(lines))

    def _fmt_summaries_as_context(items: list[CItem]) -> None:
        # Summaries selected into the active ContextView are citable only as
        # summary nodes.  Their lineage is resolved level-by-level by CIMA
        # (summary -> direct children -> L0 C-items -> evidence substrate), not
        # by flattening every descendant into the prompt.
        if not items:
            return
        lines = ["## Summary Context"]
        for item in items:
            _append_item_meta("Summary Context", item, ref_kind="summary")
            marker = items_meta[-1]["marker"]
            lines.append(f"- [{marker}][SUMMARY][{item.item_type}] {item.content}")
        sections.append("\n".join(lines))

    _fmt_items("Protected Context", pack.protected_items)
    _fmt_items("Direct Evidence", pack.direct_evidence)
    _fmt_items("Bridge Evidence", pack.bridge_evidence)
    _fmt_summaries_as_context(pack.global_summaries)
    _fmt_conflicts(pack.conflicts)

    text = "\n\n".join(sections)
    return ContextView(
        text=text,
        tokens_used=pack.tokens_used,
        coverage_score=pack.coverage_score,
        citem_ids=[item.citem_id for item in pack.all_items()],
        items=items_meta,
    )


# ── Graph context annotation ──────────────────────────────────────────────────

def format_citem_with_graph(
    item: CItem,
    pool: dict[str, CItem],
    max_depth: int = 1,
) -> str:
    """Annotate a C-Item with dependency chain from *pool* (SP-INV-02, zero LLM calls).

    Only includes deps present in pool (no extra fetches).
    """
    if max_depth == 0 or not item.dependency_ids:
        return f"[{item.item_type}] {item.content}"

    dep_annotations: list[str] = []
    for dep_id in item.dependency_ids[:3]:  # max 3 deps shown
        dep = pool.get(dep_id)
        if dep:
            dep_annotations.append(f"  ← [{dep.item_type}] {dep.content[:80]}…")

    if dep_annotations:
        deps_text = "\n".join(dep_annotations)
        return f"[{item.item_type}] {item.content}\n{deps_text}"

    return f"[{item.item_type}] {item.content}"


# ── CItemFilter factory helpers ───────────────────────────────────────────────

def episodic_active_filter(conversation_id: str) -> CItemFilter:
    return CItemFilter(
        scope="episodic",
        scope_status="active",
        conversation_id=conversation_id,
        conflict_status_in=("none", "resolved"),
    )


def global_active_filter() -> CItemFilter:
    return CItemFilter(
        scope="global",
        scope_status="active",
        conflict_status_in=("none", "resolved"),
    )


def episodic_flagged_filter(conversation_id: str) -> CItemFilter:
    """Filter for active episodic items with conflict_status=flagged.

    Used as a dedicated third recall query so flagged items can populate
    the conflicts slot in greedy_select() (bug-fix: conflicts slot dead by design).
    """
    return CItemFilter(
        scope="episodic",
        scope_status="active",
        conversation_id=conversation_id,
        conflict_status_in=("flagged",),
    )
