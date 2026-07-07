from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ARTIFACT_VERSION = "v5_eval_grounded_minimal_symbolic"
DEFAULT_PARSER_EXTRACTION_MODE = "text_first_symbolic_transcription"
ARTIFACT_VERSION = DEFAULT_ARTIFACT_VERSION


@dataclass(frozen=True)
class PipelineConfig:
    env_path: Path
    env_exists: bool
    parser_provider: str
    parser_api_key: str | None
    parser_base_url: str
    parser_model: str
    parser_temperature: float
    parser_max_tokens: int
    parser_timeout_seconds: float
    parser_extraction_mode: str
    parser_max_records_per_call: int
    parser_max_continuations_per_page: int
    parser_require_completeness: bool
    parser_allow_partial_page: bool
    parser_retry_on_json_failure: int
    parser_retry_on_schema_failure: int
    parser_allow_region_split: bool
    parser_region_split_on_repeated_failure: bool
    symbolic_artifact_version: str
    answer_provider: str
    answer_api_key: str | None
    answer_base_url: str
    answer_model: str
    answer_temperature: float
    answer_max_tokens: int
    answer_timeout_seconds: float
    render_dpi: int
    render_format: str
    render_max_pages_per_paper: int
    structured_cache_policy: str
    vlm2_context_mode: str
    vlm2_include_parse_confidence: bool
    vlm2_evidence_total_budget: int
    vlm2_primary_evidence_min: int
    vlm2_support_text_min: int
    vlm2_context_types_enabled: bool
    vlm2_context_type_budget_per_type: int
    retrieval_method: str
    retrieval_enable_topic_expansion: bool
    retrieval_enable_query_decomposition: bool
    retrieval_subquery_top_k: int
    task_family_budget_enabled: bool
    single_paper_top_k_papers: int
    single_paper_page_routing_top_pages_per_candidate: int
    multi_paper_top_k_papers: int
    multi_paper_page_routing_top_pages_per_candidate: int
    pdf_openreview_policy: str
    page_routing_enabled: bool
    page_routing_source: str
    page_routing_method: str
    page_routing_top_pages_per_candidate: int
    page_routing_top_pages_global: int
    page_routing_max_pages_global: int
    page_routing_parse_batch_size: int
    page_routing_enable_progressive_expansion: bool
    page_routing_expansion_step_global: int
    pdf_native_text_cache_enabled: bool
    pdf_native_text_extraction_method: str
    pdf_native_text_min_chars_per_page: int
    page_routing_fallback_on_empty_text: bool
    page_routing_empty_text_global_parse_first_n_pages: int
    page_routing_min_selected_records: int
    page_routing_require_primary_type_match: bool


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_pipeline_config(env_path: str | Path = ".env") -> PipelineConfig:
    path = Path(env_path)
    env = _read_env(path)

    def get(name: str, default: str = "") -> str:
        return os.environ.get(name) or env.get(name) or default

    def get_bool(name: str, default: bool) -> bool:
        value = get(name, "true" if default else "false").strip().lower()
        return value in {"1", "true", "yes", "y", "on"}

    shared_key = get("SILICONFLOW_API_KEY")
    shared_base = get("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    parser_key = get("PARSER_API_KEY") or shared_key
    parser_base = get("PARSER_BASE_URL") or shared_base
    answer_key = get("ANSWER_API_KEY") or shared_key
    answer_base = get("ANSWER_BASE_URL") or shared_base
    top_pages_global_value = get("PAGE_ROUTING_TOP_PAGES_GLOBAL", "").strip()
    return PipelineConfig(
        env_path=path,
        env_exists=path.exists(),
        parser_provider=get("PARSER_PROVIDER", "siliconflow"),
        parser_api_key=parser_key,
        parser_base_url=parser_base,
        parser_model=get("PARSER_MODEL", DEFAULT_VLM_MODEL) or DEFAULT_VLM_MODEL,
        parser_temperature=float(get("PARSER_TEMPERATURE", "0") or "0"),
        parser_max_tokens=int(get("PARSER_MAX_TOKENS", "6144") or "6144"),
        parser_timeout_seconds=float(get("PARSER_TIMEOUT_SECONDS", "180") or "180"),
        parser_extraction_mode=get("PARSER_EXTRACTION_MODE", DEFAULT_PARSER_EXTRACTION_MODE) or DEFAULT_PARSER_EXTRACTION_MODE,
        parser_max_records_per_call=int(get("PARSER_MAX_RECORDS_PER_CALL", "16") or "16"),
        parser_max_continuations_per_page=int(get("PARSER_MAX_CONTINUATIONS_PER_PAGE", "4") or "4"),
        parser_require_completeness=get_bool("PARSER_REQUIRE_COMPLETENESS", False),
        parser_allow_partial_page=get_bool("PARSER_ALLOW_PARTIAL_PAGE", True),
        parser_retry_on_json_failure=int(get("PARSER_RETRY_ON_JSON_FAILURE", "1") or "1"),
        parser_retry_on_schema_failure=int(get("PARSER_RETRY_ON_SCHEMA_FAILURE", "1") or "1"),
        parser_allow_region_split=get_bool("PARSER_ALLOW_REGION_SPLIT", False),
        parser_region_split_on_repeated_failure=get_bool("PARSER_REGION_SPLIT_ON_REPEATED_FAILURE", False),
        symbolic_artifact_version=get("SYMBOLIC_ARTIFACT_VERSION", DEFAULT_ARTIFACT_VERSION) or DEFAULT_ARTIFACT_VERSION,
        answer_provider=get("ANSWER_PROVIDER", "siliconflow"),
        answer_api_key=answer_key,
        answer_base_url=answer_base,
        answer_model=get("ANSWER_MODEL", DEFAULT_VLM_MODEL) or DEFAULT_VLM_MODEL,
        answer_temperature=float(get("ANSWER_TEMPERATURE", "0") or "0"),
        answer_max_tokens=int(get("ANSWER_MAX_TOKENS", "4096") or "4096"),
        answer_timeout_seconds=float(get("ANSWER_TIMEOUT_SECONDS", "180") or "180"),
        render_dpi=int(get("PDF_RENDER_DPI", "160") or "160"),
        render_format=(get("PDF_RENDER_FORMAT", "jpg") or "jpg").lower(),
        render_max_pages_per_paper=int(get("PDF_RENDER_MAX_PAGES_PER_PAPER", "0") or "0"),
        structured_cache_policy=get("STRUCTURED_CACHE_POLICY", "reuse_complete_only") or "reuse_complete_only",
        vlm2_context_mode=get("VLM2_CONTEXT_MODE", "text_only") or "text_only",
        vlm2_include_parse_confidence=get_bool("VLM2_INCLUDE_PARSE_CONFIDENCE", True),
        vlm2_evidence_total_budget=int(get("VLM2_EVIDENCE_TOTAL_BUDGET", "24") or "24"),
        vlm2_primary_evidence_min=int(get("VLM2_PRIMARY_EVIDENCE_MIN", "6") or "6"),
        vlm2_support_text_min=int(get("VLM2_SUPPORT_TEXT_MIN", "4") or "4"),
        vlm2_context_types_enabled=get_bool("VLM2_CONTEXT_TYPES_ENABLED", True),
        vlm2_context_type_budget_per_type=int(get("VLM2_CONTEXT_TYPE_BUDGET_PER_TYPE", "3") or "3"),
        retrieval_method=get("RETRIEVAL_METHOD", "hybrid_alias") or "hybrid_alias",
        retrieval_enable_topic_expansion=get_bool("RETRIEVAL_ENABLE_TOPIC_EXPANSION", False),
        retrieval_enable_query_decomposition=get_bool("RETRIEVAL_ENABLE_QUERY_DECOMPOSITION", True),
        retrieval_subquery_top_k=int(get("RETRIEVAL_SUBQUERY_TOP_K", "4") or "4"),
        task_family_budget_enabled=get_bool("TASK_FAMILY_BUDGET_ENABLED", True),
        single_paper_top_k_papers=int(get("SINGLE_PAPER_TOP_K_PAPERS", "5") or "5"),
        single_paper_page_routing_top_pages_per_candidate=int(get("SINGLE_PAPER_PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE", "5") or "5"),
        multi_paper_top_k_papers=int(get("MULTI_PAPER_TOP_K_PAPERS", "12") or "12"),
        multi_paper_page_routing_top_pages_per_candidate=int(get("MULTI_PAPER_PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE", "3") or "3"),
        pdf_openreview_policy=get("PDF_OPENREVIEW_POLICY", "proceedings_first_skip_direct_openreview") or "proceedings_first_skip_direct_openreview",
        page_routing_enabled=get_bool("PAGE_ROUTING_ENABLED", True),
        page_routing_source=get("PAGE_ROUTING_SOURCE", "native_text") or "native_text",
        page_routing_method=get("PAGE_ROUTING_METHOD", "global_native_text_bm25_rules") or "global_native_text_bm25_rules",
        page_routing_top_pages_per_candidate=int(get("PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE", "2") or "2"),
        page_routing_top_pages_global=int(top_pages_global_value) if top_pages_global_value else 0,
        page_routing_max_pages_global=int(get("PAGE_ROUTING_MAX_PAGES_GLOBAL", "16") or "16"),
        page_routing_parse_batch_size=int(get("PAGE_ROUTING_PARSE_BATCH_SIZE", get("PAGE_ROUTING_MAX_PAGES_GLOBAL", "16")) or "16"),
        page_routing_enable_progressive_expansion=get_bool("PAGE_ROUTING_ENABLE_PROGRESSIVE_EXPANSION", True),
        page_routing_expansion_step_global=int(get("PAGE_ROUTING_EXPANSION_STEP_GLOBAL", "4") or "4"),
        pdf_native_text_cache_enabled=get_bool("PDF_NATIVE_TEXT_CACHE_ENABLED", True),
        pdf_native_text_extraction_method=get("PDF_NATIVE_TEXT_EXTRACTION_METHOD", "pymupdf_text") or "pymupdf_text",
        pdf_native_text_min_chars_per_page=int(get("PDF_NATIVE_TEXT_MIN_CHARS_PER_PAGE", "40") or "40"),
        page_routing_fallback_on_empty_text=get_bool("PAGE_ROUTING_FALLBACK_ON_EMPTY_TEXT", True),
        page_routing_empty_text_global_parse_first_n_pages=int(get("PAGE_ROUTING_EMPTY_TEXT_GLOBAL_PARSE_FIRST_N_PAGES", "4") or "4"),
        page_routing_min_selected_records=int(get("PAGE_ROUTING_MIN_SELECTED_RECORDS", "8") or "8"),
        page_routing_require_primary_type_match=get_bool("PAGE_ROUTING_REQUIRE_PRIMARY_TYPE_MATCH", True),
    )


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
    return slug.strip("._-") or "model"


def mask_api_key(key: str) -> str:
    if not key:
        return "<missing>"
    if len(key) <= 8:
        return f"{key[:2]}****"
    return f"{key[:3]}****{key[-4:]}"


def is_api_key_configured(key: str | None) -> bool:
    if key is None:
        return False
    value = key.strip()
    if not value:
        return False
    lowered = value.lower()
    placeholders = {
        "...",
        "your_key",
        "your_api_key",
        "sk-xxx",
        "replace_me",
        "put_your_new_key_here",
        "put-your-key-here",
        "put_your_key_here",
        "api_key",
        "none",
        "null",
    }
    if lowered in placeholders:
        return False
    if "your" in lowered and "key" in lowered:
        return False
    if "replace" in lowered or "placeholder" in lowered:
        return False
    if lowered.startswith("sk-") and set(lowered[3:]) <= {"x", "*"}:
        return False
    return len(value) >= 12


__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_ARTIFACT_VERSION",
    "DEFAULT_PARSER_EXTRACTION_MODE",
    "PipelineConfig",
    "is_api_key_configured",
    "load_pipeline_config",
    "mask_api_key",
    "model_slug",
]
