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
    symbolic_cache_root: str
    answer_provider: str
    answer_api_key: str | None
    answer_base_url: str
    answer_model: str
    metadata_only_retrieval_eval_model: str
    answer_temperature: float
    answer_max_tokens: int
    answer_timeout_seconds: float
    generation_parse_max_retries: int
    generation_retry_on_429: bool
    generation_429_max_retries: int
    generation_429_initial_backoff_seconds: float
    generation_429_backoff_multiplier: float
    generation_429_max_backoff_seconds: float
    generation_cooldown_after_429_seconds: float
    render_dpi: int
    render_format: str
    render_max_pages_per_paper: int
    structured_cache_policy: str
    vlm2_context_mode: str
    vlm2_context_selection_mode: str
    vlm2_max_context_records: int
    vlm2_max_context_chars: int
    vlm2_include_parse_confidence: bool
    vlm2_evidence_total_budget: int
    vlm2_primary_evidence_min: int
    vlm2_support_text_min: int
    vlm2_context_types_enabled: bool
    vlm2_context_type_budget_per_type: int
    section_relevance_backend: str
    single_paper_section_relevance_top_k: int
    multi_paper_section_relevance_top_k: int
    section_relevance_chunk_max_tokens: int
    section_relevance_chunk_overlap_tokens: int
    section_relevance_text_max_chars: int
    section_relevance_record_top_k: int
    section_relevance_unit_mode: str
    section_relevance_unit_target_tokens: int
    section_relevance_unit_max_tokens: int
    section_relevance_unit_overlap_records: int
    section_relevance_object_units_enabled: bool
    section_relevance_object_neighbor_records: int
    section_relevance_aggregation_top_k: int
    section_relevance_section_bonus_weight: float
    section_relevance_object_section_bonus_weight: float
    section_relevance_section_bonus_max: float
    single_paper_retrieval_unit_top_k: int
    multi_paper_retrieval_unit_top_k: int
    section_relevance_hybrid_bm25_weight: float
    section_relevance_hybrid_e5_weight: float
    section_relevance_e5_model: str
    section_relevance_pooling: str
    section_relevance_log_mean_exp_lambda: float
    section_relevance_query_section_bonus_enabled: bool
    section_relevance_query_section_bonus: float
    section_relevance_primary_type_prior_enabled: bool
    section_relevance_primary_type_prior_max_bonus: float
    llmrerank_model: str
    llmrerank_input_mode: str
    llmrerank_batch_size: int
    llmrerank_request_concurrency: int
    llmrerank_request_timeout_seconds: float
    llmrerank_max_retries: int
    llmrerank_failure_fallback: str
    llmrerank_section_chunk_max_tokens: int
    llmrerank_section_chunk_overlap_tokens: int
    llmrerank_section_pooling: str
    llmrerank_log_mean_exp_lambda: float
    llmrerank_top_k_mean_chunks: int
    llmrerank_context_top_k_chunks: int
    llmrerank_max_images_per_section: int
    llmrerank_instruction_version: str
    llmrerank_unit_prefilter_enabled: bool
    llmrerank_unit_prefilter_top_k: int
    llmrerank_unit_prefilter_per_section: int
    llmrerank_unit_prefilter_primary_top_k: int
    llmrerank_prefilter_fallback_weight: float
    llmrerank_deterministic_locator_only_enabled: bool
    section_relevance_apply_section_type_bonus: bool
    multi_paper_hyde_enabled: bool
    multi_paper_hyde_model: str
    multi_paper_hyde_original_weight: float
    multi_paper_hyde_claim_weight: float
    multi_paper_hyde_max_claims: int
    multi_paper_hyde_cache_enabled: bool
    multi_paper_hyde_temperature: float
    multi_paper_hyde_max_tokens: int
    multi_paper_hyde_timeout_seconds: float
    multi_paper_hyde_unit_max_tokens: int
    multi_paper_hyde_top_hits_per_claim: int
    symbolic_evidence_standardization: bool
    symbolic_source_type_hints: bool
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
    page_ranking_bonus_enabled: bool
    page_routing_task_family_strategy: bool
    page_routing_single_strategy: str
    page_routing_multi_strategy: str
    page_routing_single_top1_min_pages: int
    page_ranking_structural_evidence_weight: float
    page_ranking_multi_text_span_hybrid_enabled: bool
    page_ranking_multi_text_span_hybrid_alpha: float
    page_ranking_multi_text_span_hybrid_gamma: float
    page_ranking_multi_text_span_hybrid_chunk_max_chars: int
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
    docling_do_ocr: bool


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
        symbolic_cache_root=get("SYMBOLIC_CACHE_ROOT", ""),
        answer_provider=get("ANSWER_PROVIDER", "siliconflow"),
        answer_api_key=answer_key,
        answer_base_url=answer_base,
        answer_model=get("ANSWER_MODEL", DEFAULT_VLM_MODEL) or DEFAULT_VLM_MODEL,
        metadata_only_retrieval_eval_model=get("METADATA_ONLY_RETRIEVAL_EVAL_MODEL", "") or get("ANSWER_MODEL", DEFAULT_VLM_MODEL) or DEFAULT_VLM_MODEL,
        answer_temperature=float(get("ANSWER_TEMPERATURE", "0") or "0"),
        answer_max_tokens=int(get("ANSWER_MAX_TOKENS", "4096") or "4096"),
        answer_timeout_seconds=float(get("ANSWER_TIMEOUT_SECONDS", "180") or "180"),
        generation_parse_max_retries=int(get("GENERATION_PARSE_MAX_RETRIES", "2") or "2"),
        generation_retry_on_429=get_bool("GENERATION_RETRY_ON_429", True),
        generation_429_max_retries=int(get("GENERATION_429_MAX_RETRIES", "6") or "6"),
        generation_429_initial_backoff_seconds=float(get("GENERATION_429_INITIAL_BACKOFF_SECONDS", "20") or "20"),
        generation_429_backoff_multiplier=float(get("GENERATION_429_BACKOFF_MULTIPLIER", "2") or "2"),
        generation_429_max_backoff_seconds=float(get("GENERATION_429_MAX_BACKOFF_SECONDS", "300") or "300"),
        generation_cooldown_after_429_seconds=float(get("GENERATION_COOLDOWN_AFTER_429_SECONDS", "120") or "120"),
        render_dpi=int(get("PDF_RENDER_DPI", "160") or "160"),
        render_format=(get("PDF_RENDER_FORMAT", "jpg") or "jpg").lower(),
        render_max_pages_per_paper=int(get("PDF_RENDER_MAX_PAGES_PER_PAPER", "0") or "0"),
        structured_cache_policy=get("STRUCTURED_CACHE_POLICY", "reuse_complete_only") or "reuse_complete_only",
        vlm2_context_mode=get("VLM2_CONTEXT_MODE", "text_only") or "text_only",
        vlm2_context_selection_mode=get("VLM2_CONTEXT_SELECTION_MODE", "page_all_symbolic") or "page_all_symbolic",
        vlm2_max_context_records=int(get("VLM2_MAX_CONTEXT_RECORDS", "0") or "0"),
        vlm2_max_context_chars=int(get("VLM2_MAX_CONTEXT_CHARS", "0") or "0"),
        vlm2_include_parse_confidence=get_bool("VLM2_INCLUDE_PARSE_CONFIDENCE", True),
        vlm2_evidence_total_budget=int(get("VLM2_EVIDENCE_TOTAL_BUDGET", "24") or "24"),
        vlm2_primary_evidence_min=int(get("VLM2_PRIMARY_EVIDENCE_MIN", "6") or "6"),
        vlm2_support_text_min=int(get("VLM2_SUPPORT_TEXT_MIN", "4") or "4"),
        vlm2_context_types_enabled=get_bool("VLM2_CONTEXT_TYPES_ENABLED", True),
        vlm2_context_type_budget_per_type=int(get("VLM2_CONTEXT_TYPE_BUDGET_PER_TYPE", "3") or "3"),
        section_relevance_backend=get("SECTION_RELEVANCE_BACKEND", "bm25") or "bm25",
        single_paper_section_relevance_top_k=int(get("SINGLE_PAPER_SECTION_RELEVANCE_TOP_K", "5") or "5"),
        multi_paper_section_relevance_top_k=int(get("MULTI_PAPER_SECTION_RELEVANCE_TOP_K", "12") or "12"),
        section_relevance_chunk_max_tokens=int(get("SECTION_RELEVANCE_CHUNK_MAX_TOKENS", "448") or "448"),
        section_relevance_chunk_overlap_tokens=int(get("SECTION_RELEVANCE_CHUNK_OVERLAP_TOKENS", "0") or "0"),
        section_relevance_text_max_chars=int(get("SECTION_RELEVANCE_TEXT_MAX_CHARS", "6000") or "6000"),
        section_relevance_record_top_k=int(get("SECTION_RELEVANCE_RECORD_TOP_K", "0") or "0"),
        section_relevance_unit_mode=get("SECTION_RELEVANCE_UNIT_MODE", "token_chunks") or "token_chunks",
        section_relevance_unit_target_tokens=int(get("SECTION_RELEVANCE_UNIT_TARGET_TOKENS", "1280") or "1280"),
        section_relevance_unit_max_tokens=int(get("SECTION_RELEVANCE_UNIT_MAX_TOKENS", "1536") or "1536"),
        section_relevance_unit_overlap_records=int(get("SECTION_RELEVANCE_UNIT_OVERLAP_RECORDS", "1") or "1"),
        section_relevance_object_units_enabled=get_bool("SECTION_RELEVANCE_OBJECT_UNITS_ENABLED", True),
        section_relevance_object_neighbor_records=int(get("SECTION_RELEVANCE_OBJECT_NEIGHBOR_RECORDS", "1") or "1"),
        section_relevance_aggregation_top_k=int(get("SECTION_RELEVANCE_AGGREGATION_TOP_K", "3") or "3"),
        section_relevance_section_bonus_weight=float(get("SECTION_RELEVANCE_SECTION_BONUS_WEIGHT", "0.10") or "0.10"),
        section_relevance_object_section_bonus_weight=float(get("SECTION_RELEVANCE_OBJECT_SECTION_BONUS_WEIGHT", "0.15") or "0.15"),
        section_relevance_section_bonus_max=float(get("SECTION_RELEVANCE_SECTION_BONUS_MAX", "0.10") or "0.10"),
        single_paper_retrieval_unit_top_k=int(get("SINGLE_PAPER_RETRIEVAL_UNIT_TOP_K", "12") or "12"),
        multi_paper_retrieval_unit_top_k=int(get("MULTI_PAPER_RETRIEVAL_UNIT_TOP_K", "36") or "36"),
        section_relevance_hybrid_bm25_weight=float(get("SECTION_RELEVANCE_HYBRID_BM25_WEIGHT", "0.4") or "0.4"),
        section_relevance_hybrid_e5_weight=float(get("SECTION_RELEVANCE_HYBRID_E5_WEIGHT", "0.6") or "0.6"),
        section_relevance_e5_model=get("SECTION_RELEVANCE_E5_MODEL", "intfloat/e5-base-v2") or "intfloat/e5-base-v2",
        section_relevance_pooling=get("SECTION_RELEVANCE_POOLING", "log_mean_exp") or "log_mean_exp",
        section_relevance_log_mean_exp_lambda=float(get("SECTION_RELEVANCE_LOG_MEAN_EXP_LAMBDA", "3") or "3"),
        section_relevance_query_section_bonus_enabled=get_bool("SECTION_RELEVANCE_QUERY_SECTION_BONUS_ENABLED", True),
        section_relevance_query_section_bonus=float(get("SECTION_RELEVANCE_QUERY_SECTION_BONUS", "0.2") or "0.2"),
        section_relevance_primary_type_prior_enabled=get_bool("SECTION_RELEVANCE_PRIMARY_TYPE_PRIOR_ENABLED", True),
        section_relevance_primary_type_prior_max_bonus=float(get("SECTION_RELEVANCE_PRIMARY_TYPE_PRIOR_MAX_BONUS", "0.05") or "0.05"),
        llmrerank_model=get("LLMRERANK_MODEL", "Qwen/Qwen3-VL-Reranker-8B") or "Qwen/Qwen3-VL-Reranker-8B",
        llmrerank_input_mode=get("LLMRERANK_INPUT_MODE", "text_with_object_images") or "text_with_object_images",
        llmrerank_batch_size=int(get("LLMRERANK_BATCH_SIZE", "8") or "8"),
        llmrerank_request_concurrency=int(get("LLMRERANK_REQUEST_CONCURRENCY", "1") or "1"),
        llmrerank_request_timeout_seconds=float(get("LLMRERANK_REQUEST_TIMEOUT_SECONDS", "120") or "120"),
        llmrerank_max_retries=int(get("LLMRERANK_MAX_RETRIES", "3") or "3"),
        llmrerank_failure_fallback=get("LLMRERANK_FAILURE_FALLBACK", "none") or "none",
        llmrerank_section_chunk_max_tokens=int(get("LLMRERANK_SECTION_CHUNK_MAX_TOKENS", "6144") or "6144"),
        llmrerank_section_chunk_overlap_tokens=int(get("LLMRERANK_SECTION_CHUNK_OVERLAP_TOKENS", "128") or "128"),
        llmrerank_section_pooling=get("LLMRERANK_SECTION_POOLING", "log_mean_exp") or "log_mean_exp",
        llmrerank_log_mean_exp_lambda=float(get("LLMRERANK_LOG_MEAN_EXP_LAMBDA", "5") or "5"),
        llmrerank_top_k_mean_chunks=int(get("LLMRERANK_TOP_K_MEAN_CHUNKS", "3") or "3"),
        llmrerank_context_top_k_chunks=int(get("LLMRERANK_CONTEXT_TOP_K_CHUNKS", "0") or "0"),
        llmrerank_max_images_per_section=int(get("LLMRERANK_MAX_IMAGES_PER_SECTION", "4") or "4"),
        llmrerank_instruction_version=get("LLMRERANK_INSTRUCTION_VERSION", "v1") or "v1",
        llmrerank_unit_prefilter_enabled=get_bool("LLMRERANK_UNIT_PREFILTER_ENABLED", False),
        llmrerank_unit_prefilter_top_k=int(get("LLMRERANK_UNIT_PREFILTER_TOP_K", "64") or "64"),
        llmrerank_unit_prefilter_per_section=int(get("LLMRERANK_UNIT_PREFILTER_PER_SECTION", "3") or "3"),
        llmrerank_unit_prefilter_primary_top_k=int(get("LLMRERANK_UNIT_PREFILTER_PRIMARY_TOP_K", "32") or "32"),
        llmrerank_prefilter_fallback_weight=float(get("LLMRERANK_PREFILTER_FALLBACK_WEIGHT", "0.25") or "0.25"),
        llmrerank_deterministic_locator_only_enabled=get_bool("LLMRERANK_DETERMINISTIC_LOCATOR_ONLY_ENABLED", False),
        section_relevance_apply_section_type_bonus=get_bool("SECTION_RELEVANCE_APPLY_SECTION_TYPE_BONUS", False),
        multi_paper_hyde_enabled=get_bool("MULTI_PAPER_HYDE_ENABLED", False),
        multi_paper_hyde_model=get("MULTI_PAPER_HYDE_MODEL", "deepseek-ai/DeepSeek-V4-Flash") or "deepseek-ai/DeepSeek-V4-Flash",
        multi_paper_hyde_original_weight=float(get("MULTI_PAPER_HYDE_ORIGINAL_WEIGHT", "0.7") or "0.7"),
        multi_paper_hyde_claim_weight=float(get("MULTI_PAPER_HYDE_CLAIM_WEIGHT", "0.3") or "0.3"),
        multi_paper_hyde_max_claims=int(get("MULTI_PAPER_HYDE_MAX_CLAIMS", "4") or "4"),
        multi_paper_hyde_cache_enabled=get_bool("MULTI_PAPER_HYDE_CACHE_ENABLED", True),
        multi_paper_hyde_temperature=float(get("MULTI_PAPER_HYDE_TEMPERATURE", "0") or "0"),
        multi_paper_hyde_max_tokens=int(get("MULTI_PAPER_HYDE_MAX_TOKENS", "512") or "512"),
        multi_paper_hyde_timeout_seconds=float(get("MULTI_PAPER_HYDE_TIMEOUT_SECONDS", "120") or "120"),
        multi_paper_hyde_unit_max_tokens=int(get("MULTI_PAPER_HYDE_UNIT_MAX_TOKENS", "448") or "448"),
        multi_paper_hyde_top_hits_per_claim=int(get("MULTI_PAPER_HYDE_TOP_HITS_PER_CLAIM", "20") or "20"),
        symbolic_evidence_standardization=get_bool("SYMBOLIC_EVIDENCE_STANDARDIZATION", True),
        symbolic_source_type_hints=get_bool("SYMBOLIC_SOURCE_TYPE_HINTS", False),
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
        page_ranking_bonus_enabled=get_bool("PAGE_RANKING_BONUS_ENABLED", True),
        page_routing_task_family_strategy=get_bool("PAGE_ROUTING_TASK_FAMILY_STRATEGY", True),
        page_routing_single_strategy=get("PAGE_ROUTING_SINGLE_STRATEGY", "top1_candidate_quota") or "top1_candidate_quota",
        page_routing_multi_strategy=get("PAGE_ROUTING_MULTI_STRATEGY", "global_ranked_pages") or "global_ranked_pages",
        page_routing_single_top1_min_pages=int(get("PAGE_ROUTING_SINGLE_TOP1_MIN_PAGES", "0") or "0"),
        page_ranking_structural_evidence_weight=float(get("PAGE_RANKING_STRUCTURAL_EVIDENCE_WEIGHT", "0") or "0"),
        page_ranking_multi_text_span_hybrid_enabled=get_bool("PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_ENABLED", True),
        page_ranking_multi_text_span_hybrid_alpha=float(get("PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_ALPHA", "0.75") or "0.75"),
        page_ranking_multi_text_span_hybrid_gamma=float(get("PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_GAMMA", "4") or "4"),
        page_ranking_multi_text_span_hybrid_chunk_max_chars=int(get("PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_CHUNK_MAX_CHARS", "700") or "700"),
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
        docling_do_ocr=get_bool("DOCLING_DO_OCR", True),
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
