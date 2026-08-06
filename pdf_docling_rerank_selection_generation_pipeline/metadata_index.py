from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - fallback for minimally provisioned smoke tests
    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.avgdl = sum(len(doc) for doc in corpus) / max(1, len(corpus))
            self.doc_freq: Counter[str] = Counter()
            for doc in corpus:
                self.doc_freq.update(set(doc))

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            scores: list[float] = []
            n_docs = max(1, len(self.corpus))
            for doc in self.corpus:
                counts = Counter(doc)
                score = 0.0
                doc_len = max(1, len(doc))
                for token in query_tokens:
                    if token not in counts:
                        continue
                    df = self.doc_freq.get(token, 0)
                    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                    tf = counts[token]
                    denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                    score += idf * tf * (self.k1 + 1) / denom
                scores.append(score)
            return scores


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
URL_RE = re.compile(r"https?://\S+")
METHOD_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-²][A-Za-z0-9]+)+|[A-Z]{2,}[A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[a-z][A-Z]{2,}[A-Za-z0-9-]*)\b"
)
MIXED_METHOD_RE = re.compile(
    r"\b(?:[0-9]+[A-Za-z][A-Za-z0-9.-]*|[A-Za-z]+[0-9][A-Za-z0-9.-]*|[A-Z]?[a-z][A-Z]{1,}[A-Za-z0-9.-]*)\b"
)
VENUE_RE = re.compile(r"\b(ACL|NAACL|EMNLP|CVPR|ICCV|ECCV|ICLR|ICML|NeurIPS)\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(20[0-9]{2})\b")
PRIMARY_VENUE_YEAR_RE = re.compile(
    r"\b(ACL|NAACL|EMNLP|CVPR|ICCV|ECCV|ICLR|ICML|NeurIPS)\s+(20[0-9]{2})\b"
    r"|\b(20[0-9]{2})\s+(ACL|NAACL|EMNLP|CVPR|ICCV|ECCV|ICLR|ICML|NeurIPS)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "the",
    "to",
    "via",
    "what",
    "which",
    "with",
}
NON_METHOD_TERMS = {
    "aime",
    "alpacaeval",
    "ap",
    "arkitscenes",
    "bench2drive",
    "cifar10",
    "coco",
    "dfid",
    "fid",
    "geneval",
    "gpu",
    "hypersim",
    "imagenet",
    "ipc",
    "kit",
    "kitti",
    "llava1",
    "llm",
    "modelnet40",
    "naturalq",
    "nuscenes",
    "objectron",
    "pope",
    "qwen",
    "rgbd",
    "rtdetrv2r50",
    "studentt",
    "sun",
    "tinyimagenet",
    "vlm",
    "vlmbased",
}
HYBRID_SCORE_WEIGHTS = {
    "title_bm25": 3.2,
    "abstract_bm25": 1.0,
    "full_bm25": 0.8,
    "alias_bm25": 4.5,
    "venue_match_boost": 12.0,
    "venue_mismatch_penalty": -8.0,
    "year_match_boost": 6.0,
    "year_mismatch_penalty": -4.0,
    "title_contains_method_boost": 90.0,
    "abstract_contains_method_boost": 30.0,
    "raw_contains_method_boost": 18.0,
    "title_exact_in_query_boost": 120.0,
}

_RETRIEVER_CACHE: dict[tuple[int, int, str], "_BaseRetriever"] = {}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def sanitize_metadata_text(text: str) -> str:
    return URL_RE.sub("[URL_REMOVED]", text or "")


def compact(text: str) -> str:
    normalized = (text or "").replace("²", "2").replace("³", "3")
    return re.sub(r"[^a-z0-9]+", "", normalized.lower())


def title_words(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def acronym(words: list[str]) -> str:
    return "".join(word[0] for word in words if word)


def window_acronyms(words: list[str], min_len: int = 2, max_len: int = 6) -> set[str]:
    aliases: set[str] = set()
    for size in range(min_len, min(max_len, len(words)) + 1):
        for start in range(0, len(words) - size + 1):
            value = acronym(words[start : start + size])
            if 2 <= len(value) <= 12:
                aliases.add(value.lower())
    return aliases


def extract_upper_aliases(text: str) -> set[str]:
    aliases: set[str] = set()
    for match in METHOD_RE.finditer(text or ""):
        value = match.group(0).strip("-")
        if 2 <= len(value) <= 30:
            aliases.add(compact(value))
    return aliases


def extract_query_terms(question: str) -> set[str]:
    terms = {compact(match.group(0)) for match in METHOD_RE.finditer(question or "") if len(compact(match.group(0))) >= 2}
    for match in MIXED_METHOD_RE.finditer(question or ""):
        value = compact(match.group(0))
        if len(value) >= 3:
            terms.add(value)
    for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", question or ""):
        terms.add(compact(quoted))
    for match in re.finditer(r"\b(?:the|a|an)\s+([A-Z][A-Za-z0-9².-]{2,}(?:\s+[A-Z][A-Za-z0-9².-]{2,}){0,4})\s+paper\b", question or ""):
        terms.add(compact(match.group(1)))
    return {term for term in terms if term and term not in NON_METHOD_TERMS}


def extract_query_terms_unfiltered(question: str) -> set[str]:
    """All method-ish/dataset-ish terms, WITHOUT the NON_METHOD_TERMS filter.

    The filtered variant keeps generic dataset/benchmark names out of the strong
    alias boosts; the unfiltered set is used for the weak substring-hit component
    so that e.g. `bench2drive` (in `NON_METHOD_TERMS`) can still surface ORION,
    whose abstract contains the term.
    """
    terms: set[str] = set()
    for match in METHOD_RE.finditer(question or ""):
        value = compact(match.group(0))
        if 2 <= len(value) <= 30:
            terms.add(value)
    for match in MIXED_METHOD_RE.finditer(question or ""):
        value = compact(match.group(0))
        if len(value) >= 3:
            terms.add(value)
    for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", question or ""):
        value = compact(quoted)
        if 2 <= len(value) <= 30:
            terms.add(value)
    return {term for term in terms if term}


def extract_query_mentions(question: str) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    for pattern in (METHOD_RE, MIXED_METHOD_RE):
        for match in pattern.finditer(question or ""):
            text = match.group(0).strip()
            key = compact(text)
            if len(key) < 3 or key in NON_METHOD_TERMS or key in seen:
                continue
            seen.add(key)
            mentions.append(text)
    for quoted in re.findall(r"['\"]([^'\"]{2,80})['\"]", question or ""):
        key = compact(quoted)
        if len(key) >= 3 and key not in NON_METHOD_TERMS and key not in seen:
            seen.add(key)
            mentions.append(quoted)
    return mentions[:16]


def venue_year_hints(question: str) -> tuple[set[str], set[str]]:
    venues = {match.group(1).upper() for match in VENUE_RE.finditer(question or "")}
    venues = {"NeurIPS" if venue == "NEURIPS" else venue for venue in venues}
    years = {match.group(1) for match in YEAR_RE.finditer(question or "")}
    return venues, years


def primary_venue_year_hints(question: str) -> tuple[set[str], set[str]]:
    """Detect a venue+year that constrains the candidate pool itself.

    Only adjacent `VENUE YEAR` / `YEAR VENUE` mentions count (e.g. "NAACL 2025
    papers", "a 2025 NeurIPS method").  A parenthesized citation year such as
    "(Planning-oriented Autonomous Driving, CVPR2023)" has no separating space
    and is intentionally ignored, so it cannot expand the pool filter.
    """
    venues: set[str] = set()
    years: set[str] = set()
    for match in PRIMARY_VENUE_YEAR_RE.finditer(question or ""):
        if match.group(1):
            venue = match.group(1).upper()
            venues.add("NeurIPS" if venue == "NEURIPS" else venue)
            years.add(match.group(2))
        else:
            venue = match.group(4).upper()
            venues.add("NeurIPS" if venue == "NEURIPS" else venue)
            years.add(match.group(3))
    return venues, years


def build_metadata_records(metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in metadata_rows:
        title = sanitize_metadata_text(str(row.get("title") or ""))
        abstract = sanitize_metadata_text(str(row.get("abstract") or ""))
        record = dict(row)
        record["paper_id"] = str(row.get("paper_id") or "")
        record["title"] = title
        record["abstract"] = abstract
        record["venue"] = row.get("venue", "")
        record["year"] = row.get("year", "")
        record["online_links"] = []
        record["_title_tokens"] = tokenize(title)
        record["_abstract_tokens"] = tokenize(abstract)
        records.append(record)
    return records


class _BaseRetriever:
    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class SimpleBM25Retriever(_BaseRetriever):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.index = BM25Okapi([record["_title_tokens"] + record["_abstract_tokens"] for record in records])

    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        tokens = tokenize(question)
        if not tokens:
            return []
        scores = self.index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [_candidate_from_record(self.records[idx], rank, float(score), "bm25_simple", {"bm25": float(score)}) for rank, (idx, score) in enumerate(ranked, start=1)]


class HybridAliasRetriever(_BaseRetriever):
    def __init__(self, records: list[dict[str, Any]], term_substring_boost: bool = False) -> None:
        self.records = records
        self.term_substring_boost = term_substring_boost
        self.features: list[dict[str, Any]] = []
        title_corpus: list[list[str]] = []
        abstract_corpus: list[list[str]] = []
        full_corpus: list[list[str]] = []
        alias_corpus: list[list[str]] = []
        for record in records:
            title = str(record.get("title") or "")
            abstract = str(record.get("abstract") or "")
            authors = " ".join(str(author) for author in record.get("authors") or [])
            title_tokens = tokenize(title)
            abstract_tokens = tokenize(abstract)
            words = title_words(title)
            aliases = set()
            if len(words) >= 2:
                aliases.add(acronym(words).lower())
                aliases.update(window_acronyms(words))
            aliases.update(extract_upper_aliases(title))
            aliases.update(extract_upper_aliases(abstract[:1600]))
            aliases = {alias for alias in aliases if 2 <= len(alias) <= 30}
            self.features.append(
                {
                    "title_compact": compact(title),
                    "abstract_compact": compact(abstract[:2400]),
                    "raw_compact": compact(f"{title} {abstract}"),
                    "aliases": aliases,
                    "venue": str(record.get("venue") or ""),
                    "year": str(record.get("year") or ""),
                }
            )
            title_corpus.append(title_tokens + list(aliases) * 3)
            abstract_corpus.append(abstract_tokens)
            full_corpus.append(tokenize(f"{title} {abstract} {authors} {record.get('venue', '')} {record.get('year', '')}") + list(aliases) * 2)
            alias_corpus.append(list(aliases) or ["noalias"])
        self.title_index = BM25Okapi(title_corpus)
        self.abstract_index = BM25Okapi(abstract_corpus)
        self.full_index = BM25Okapi(full_corpus)
        self.alias_index = BM25Okapi(alias_corpus)
        self.alias_df = Counter(alias for features in self.features for alias in features["aliases"])

    def retrieve(self, question: str, top_k: int) -> list[dict[str, Any]]:
        tokens = tokenize(question)
        if not tokens:
            return []
        query_terms = extract_query_terms(question)
        substring_terms = extract_query_terms_unfiltered(question) if self.term_substring_boost else set()
        query_compact = compact(question)
        venues, years = venue_year_hints(question)
        title_scores = self.title_index.get_scores(tokens)
        abstract_scores = self.abstract_index.get_scores(tokens)
        full_scores = self.full_index.get_scores(tokens)
        alias_scores = self.alias_index.get_scores(list(query_terms) or tokens)
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for idx, record in enumerate(self.records):
            features = self.features[idx]
            components: dict[str, Any] = {
                "title_bm25": float(title_scores[idx]),
                "abstract_bm25": float(abstract_scores[idx]),
                "full_bm25": float(full_scores[idx]),
                "alias_bm25": float(alias_scores[idx]),
                "alias_exact_boost": 0.0,
                "method_substring_boost": 0.0,
                "title_exact_in_query_boost": 0.0,
                "term_substring_boost": 0.0,
                "venue_year_adjustment": 0.0,
                "matched_aliases": [],
                "matched_substrings": [],
                "query_terms": sorted(query_terms),
            }
            score = (
                HYBRID_SCORE_WEIGHTS["title_bm25"] * components["title_bm25"]
                + HYBRID_SCORE_WEIGHTS["abstract_bm25"] * components["abstract_bm25"]
                + HYBRID_SCORE_WEIGHTS["full_bm25"] * components["full_bm25"]
                + HYBRID_SCORE_WEIGHTS["alias_bm25"] * components["alias_bm25"]
            )
            aliases: set[str] = features["aliases"]
            match_terms = query_terms & aliases
            components["matched_aliases"] = sorted(match_terms)
            for term in match_terms:
                df = max(1, self.alias_df.get(term, 1))
                boost = 4.0 if len(term) <= 3 and df > 80 else max(10.0 if len(term) <= 3 else 18.0, (75.0 if len(term) <= 3 else 95.0) / math.sqrt(df))
                components["alias_exact_boost"] += boost
                score += boost
            for term in query_terms:
                if len(term) >= 4 and term in features["title_compact"]:
                    score += HYBRID_SCORE_WEIGHTS["title_contains_method_boost"]
                    components["method_substring_boost"] += HYBRID_SCORE_WEIGHTS["title_contains_method_boost"]
                elif len(term) >= 5 and term in features["abstract_compact"]:
                    score += HYBRID_SCORE_WEIGHTS["abstract_contains_method_boost"]
                    components["method_substring_boost"] += HYBRID_SCORE_WEIGHTS["abstract_contains_method_boost"]
                elif len(term) >= 5 and term in features["raw_compact"]:
                    score += HYBRID_SCORE_WEIGHTS["raw_contains_method_boost"]
                    components["method_substring_boost"] += HYBRID_SCORE_WEIGHTS["raw_contains_method_boost"]
            if features["title_compact"] and len(features["title_compact"]) >= 12 and features["title_compact"] in query_compact:
                score += HYBRID_SCORE_WEIGHTS["title_exact_in_query_boost"]
                components["title_exact_in_query_boost"] += HYBRID_SCORE_WEIGHTS["title_exact_in_query_boost"]
            if substring_terms:
                matched_substrings: list[str] = []
                for term in substring_terms:
                    if len(term) >= 4 and term in features["title_compact"]:
                        matched_substrings.append(term)
                        score += 14.0
                        components["term_substring_boost"] += 14.0
                    elif len(term) >= 5 and term in features["abstract_compact"]:
                        matched_substrings.append(term)
                        score += 5.0
                        components["term_substring_boost"] += 5.0
                components["matched_substrings"] = sorted(set(matched_substrings))
            if venues and features["venue"] in venues:
                score += HYBRID_SCORE_WEIGHTS["venue_match_boost"]
                components["venue_year_adjustment"] += HYBRID_SCORE_WEIGHTS["venue_match_boost"]
            if years and features["year"] in years:
                score += HYBRID_SCORE_WEIGHTS["year_match_boost"]
                components["venue_year_adjustment"] += HYBRID_SCORE_WEIGHTS["year_match_boost"]
            if venues and features["venue"] not in venues and "across all venues" not in question.lower():
                score += HYBRID_SCORE_WEIGHTS["venue_mismatch_penalty"]
                components["venue_year_adjustment"] += HYBRID_SCORE_WEIGHTS["venue_mismatch_penalty"]
            if years and features["year"] not in years:
                score += HYBRID_SCORE_WEIGHTS["year_mismatch_penalty"]
                components["venue_year_adjustment"] += HYBRID_SCORE_WEIGHTS["year_mismatch_penalty"]
            components["weighted_total"] = float(score)
            scored.append((idx, float(score), components))
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            _candidate_from_record(self.records[idx], rank, score, "hybrid_alias", components)
            for rank, (idx, score, components) in enumerate(ranked, start=1)
        ]


def _candidate_from_record(record: dict[str, Any], rank: int, score: float, method: str, components: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": round(score, 6),
        "bm25_score": float(score),
        "hybrid_score": float(score) if method == "hybrid_alias" else None,
        "retrieval_method": method,
        "retrieval_score_components": components,
        "paper_id": record.get("paper_id", ""),
        "title": record.get("title", ""),
        "abstract": record.get("abstract", ""),
        "authors": record.get("authors", []),
        "venue": record.get("venue", ""),
        "year": record.get("year", ""),
        "pdf_url": record.get("pdf_url"),
        "source_url": record.get("source_url"),
        "arxiv_id": record.get("arxiv_id"),
        "doi": record.get("doi"),
        "openreview_id": record.get("openreview_id"),
        "anthology_id": record.get("anthology_id"),
        "matched_aliases": components.get("matched_aliases", []),
        "online_links": [],
    }


def _get_retriever(
    metadata_records: list[dict[str, Any]],
    method: str,
    term_substring_boost: bool = False,
) -> _BaseRetriever:
    normalized_method = method if method in {"bm25_simple", "hybrid_alias"} else "hybrid_alias"
    key = (id(metadata_records), len(metadata_records), normalized_method, term_substring_boost)
    cached = _RETRIEVER_CACHE.get(key)
    if cached is not None:
        return cached
    retriever: _BaseRetriever
    if normalized_method == "hybrid_alias":
        retriever = HybridAliasRetriever(metadata_records, term_substring_boost=term_substring_boost)
    else:
        retriever = SimpleBM25Retriever(metadata_records)
    _RETRIEVER_CACHE[key] = retriever
    return retriever


_VENUE_SUBSET_CACHE: dict[tuple[str, str, str, bool], _BaseRetriever] = {}


def _get_venue_subset_retriever(
    records: list[dict[str, Any]],
    venue: str,
    year: str,
    method: str,
    term_substring_boost: bool,
) -> _BaseRetriever:
    key = (venue, year, method, term_substring_boost)
    cached = _VENUE_SUBSET_CACHE.get(key)
    if cached is not None:
        return cached
    subset = [record for record in records if str(record.get("venue") or "") == venue and str(record.get("year") or "") == year]
    retriever = _get_retriever(subset, method, term_substring_boost=term_substring_boost)
    _VENUE_SUBSET_CACHE[key] = retriever
    return retriever


def retrieve_candidates(
    question: str,
    metadata_records: list[dict[str, Any]],
    top_k: int = 12,
    method: str = "hybrid_alias",
    enable_query_decomposition: bool = False,
    subquery_top_k: int = 4,
    term_substring_boost: bool = False,
    venue_year_route_top_k: int = 0,
) -> list[dict[str, Any]]:
    if not tokenize(question):
        return []
    retriever = _get_retriever(metadata_records, method, term_substring_boost=term_substring_boost)
    candidates = retriever.retrieve(question, top_k)
    if enable_query_decomposition and method in {"hybrid_alias", "hybrid_alias_decomposed"}:
        candidates = _merge_decomposed_candidates(retriever, question, candidates, top_k, subquery_top_k)
    if venue_year_route_top_k > 0:
        candidates = _merge_venue_year_candidates(
            question,
            candidates,
            metadata_records,
            top_k,
            venue_year_route_top_k,
            method,
            term_substring_boost,
        )
    if not candidates and metadata_records:
        first = metadata_records[0]
        candidates.append(_candidate_from_record(first, 1, 0.0, method, {"fallback": True, "weighted_total": 0.0}))
    return candidates


def _merge_venue_year_candidates(
    question: str,
    base_candidates: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    top_k: int,
    venue_year_top_k: int,
    method: str,
    term_substring_boost: bool,
) -> list[dict[str, Any]]:
    """Union the base pool with a venue-year-restricted retrieval when the
    question constrains the pool to a venue+year (e.g. "NAACL 2025 papers").

    The venue-restricted hybrid runs with its own larger budget inside the small
    venue-year universe; results not already in the base pool are appended so the
    final candidate pool can exceed `top_k` for those queries.  This is the
    measured recall lever for q_020/q_022/q_023-style questions.
    """
    venues, years = primary_venue_year_hints(question)
    if not venues or not years:
        return base_candidates
    merged: dict[str, dict[str, Any]] = {str(c.get("paper_id") or ""): c for c in base_candidates if c.get("paper_id")}
    next_rank = len(base_candidates) + 1
    for venue in sorted(venues):
        for year in sorted(years):
            sub_retriever = _get_venue_subset_retriever(
                metadata_records, venue, year, method, term_substring_boost
            )
            sub_candidates = sub_retriever.retrieve(question, venue_year_top_k)
            for candidate in sub_candidates:
                paper_id = str(candidate.get("paper_id") or "")
                if not paper_id:
                    continue
                if paper_id in merged:
                    existing = merged[paper_id]
                    components = existing.get("retrieval_score_components")
                    if isinstance(components, dict):
                        components["venue_year_route"] = {
                            "venue": venue,
                            "year": year,
                            "rank": int(candidate.get("rank") or 0),
                        }
                    continue
                copied = dict(candidate)
                copied["rank"] = next_rank
                copied["retrieval_venue_year_route"] = {"venue": venue, "year": year}
                copied["retrieval_method"] = "hybrid_alias_venue_year_route"
                next_rank += 1
                merged[paper_id] = copied
    return [merged[paper_id] for paper_id in merged]


def _merge_decomposed_candidates(
    retriever: _BaseRetriever,
    question: str,
    base_candidates: list[dict[str, Any]],
    top_k: int,
    subquery_top_k: int,
) -> list[dict[str, Any]]:
    mentions = extract_query_mentions(question)
    if not mentions:
        return base_candidates
    merged: dict[str, dict[str, Any]] = {}
    order = 0

    def add(candidate: dict[str, Any], source: str, rank: int) -> None:
        nonlocal order
        paper_id = str(candidate.get("paper_id") or "")
        if not paper_id:
            return
        order += 1
        base_score = float(candidate.get("hybrid_score") or candidate.get("score") or candidate.get("bm25_score") or 0.0)
        if source == "full_query":
            merge_score = base_score + max(0, top_k + 1 - rank) * 2.0
        else:
            merge_score = base_score * 0.35 + max(0, subquery_top_k + 1 - rank) * 24.0
        existing = merged.get(paper_id)
        if existing is None:
            copied = dict(candidate)
            copied["retrieval_method"] = "hybrid_alias_decomposed"
            copied["_merge_score"] = merge_score
            copied["_first_seen"] = order
            copied["retrieval_decomposition_sources"] = [{"source": source, "rank": rank, "score": base_score}]
            merged[paper_id] = copied
            return
        existing["_merge_score"] = float(existing.get("_merge_score") or 0.0) + merge_score
        existing.setdefault("retrieval_decomposition_sources", []).append({"source": source, "rank": rank, "score": base_score})

    for rank, candidate in enumerate(base_candidates, start=1):
        add(candidate, "full_query", rank)
    for mention in mentions:
        for rank, candidate in enumerate(retriever.retrieve(mention, max(1, subquery_top_k)), start=1):
            add(candidate, mention, rank)

    ranked = sorted(merged.values(), key=lambda item: (-float(item.get("_merge_score") or 0.0), int(item.get("_first_seen") or 0)))[:top_k]
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
        candidate["retrieval_rank"] = rank
        components = candidate.get("retrieval_score_components")
        if isinstance(components, dict):
            components["decomposition_enabled"] = True
            components["decomposition_mentions"] = mentions
            components["decomposition_merge_score"] = float(candidate.get("_merge_score") or 0.0)
    for candidate in ranked:
        candidate.pop("_merge_score", None)
        candidate.pop("_first_seen", None)
    return ranked
