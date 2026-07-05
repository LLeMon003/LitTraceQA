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
    "cifar10",
    "coco",
    "fid",
    "gpu",
    "imagenet",
    "llava1",
    "llm",
    "pope",
    "qwen",
    "rgbd",
    "vlm",
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


def venue_year_hints(question: str) -> tuple[set[str], set[str]]:
    venues = {match.group(1).upper() for match in VENUE_RE.finditer(question or "")}
    venues = {"NeurIPS" if venue == "NEURIPS" else venue for venue in venues}
    years = {match.group(1) for match in YEAR_RE.finditer(question or "")}
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
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
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
                "venue_year_adjustment": 0.0,
                "matched_aliases": [],
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


def _get_retriever(metadata_records: list[dict[str, Any]], method: str) -> _BaseRetriever:
    normalized_method = method if method in {"bm25_simple", "hybrid_alias"} else "hybrid_alias"
    key = (id(metadata_records), len(metadata_records), normalized_method)
    cached = _RETRIEVER_CACHE.get(key)
    if cached is not None:
        return cached
    retriever: _BaseRetriever = HybridAliasRetriever(metadata_records) if normalized_method == "hybrid_alias" else SimpleBM25Retriever(metadata_records)
    _RETRIEVER_CACHE[key] = retriever
    return retriever


def retrieve_candidates(
    question: str,
    metadata_records: list[dict[str, Any]],
    top_k: int = 12,
    method: str = "hybrid_alias",
) -> list[dict[str, Any]]:
    if not tokenize(question):
        return []
    retriever = _get_retriever(metadata_records, method)
    candidates = retriever.retrieve(question, top_k)
    if not candidates and metadata_records:
        first = metadata_records[0]
        candidates.append(_candidate_from_record(first, 1, 0.0, method, {"fallback": True, "weighted_total": 0.0}))
    return candidates
