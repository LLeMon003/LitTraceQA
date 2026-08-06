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
    generation_request_retries: int
    generation_retry_on_429: bool
    generation_429_max_retries: int
    generation_429_initial_backoff_seconds: float
    generation_429_backoff_multiplier: float
    generation_429_max_backoff_seconds: float
    generation_cooldown_after_429_seconds: float
    evidence_hierarchy_card_mode: str
    evidence_hierarchy_max_claims: int
    evidence_hierarchy_max_cards: int
    evidence_hierarchy_l1_max_chars: int
    evidence_hierarchy_l3_paper_chars: int
    evidence_hierarchy_card_max_tokens: int
    evidence_hierarchy_card_source_chars: int
    evidence_hierarchy_verification_mode: str
    evidence_hierarchy_max_images: int
    evidence_hierarchy_micro_index_chars: int
    evidence_hierarchy_micro_text_chars: int
    evidence_hierarchy_keyed_micro_index_chars: int
    evidence_hierarchy_keyed_micro_text_chars: int
    evidence_hierarchy_keyed_micro_order: str
    evidence_hierarchy_posthoc_refinement_enabled: bool
    evidence_hierarchy_visual_cards_enabled: bool
    evidence_hierarchy_visual_cards_max_per_query: int
    evidence_hierarchy_visual_cards_max_per_paper: int
    evidence_hierarchy_visual_cards_max_tokens: int
    evidence_hierarchy_visual_verify_max_tokens: int
    evidence_triple_mode: str
    evidence_triple_text_model: str
    evidence_triple_visual_model: str
    evidence_triple_source_chars: int
    evidence_triple_batch_source_chars: int
    evidence_triple_text_max_tokens: int
    evidence_triple_text_max_windows: int
    evidence_triple_timeout_seconds: float
    evidence_triple_cache_enabled: bool
    evidence_triple_visual_max_tokens: int
    evidence_triple_visual_max_per_query: int
    evidence_triple_sufficiency_enabled: bool
    evidence_triple_sufficiency_model: str
    evidence_triple_sufficiency_max_tokens: int
    render_dpi: int
    render_format: str
    render_max_pages_per_paper: int
    structured_cache_policy: str
    transcription_backend: str
    vlm2_context_mode: str
    vlm2_context_selection_mode: str
    vlm2_include_parse_confidence: bool
    single_paper_evidence_package_budget: int
    multi_paper_evidence_package_budget: int
    single_paper_evidence_package_min: int
    multi_paper_evidence_package_min: int
    multi_paper_min_distinct_papers: int
    evidence_package_adaptive_stop: bool
    multi_paper_modality_packages_per_paper: int
    multi_paper_supporting_text_packages_per_paper: int
    evidence_package_max_context_chars: int
    evidence_package_rrf_k: int
    evidence_package_candidate_pool_per_route: int
    evidence_package_max_per_page: int
    evidence_package_page_text_anchors_per_page: int
    section_relevance_backend: str
    section_relevance_unit_mode: str
    section_relevance_unit_target_tokens: int
    section_relevance_unit_max_tokens: int
    section_relevance_unit_overlap_records: int
    section_relevance_object_units_enabled: bool
    section_relevance_object_neighbor_records: int
    llmrerank_model: str
    llmrerank_input_mode: str
    llmrerank_batch_size: int
    llmrerank_request_concurrency: int
    llmrerank_request_timeout_seconds: float
    llmrerank_max_retries: int
    llmrerank_failure_fallback: str
    llmrerank_max_images_per_section: int
    llmrerank_instruction_version: str
    llmrerank_query_mode: str
    llmrerank_include_paper_identity: bool
    retriever_pool_budget: int
    multi_paper_hyde_enabled: bool
    multi_paper_hyde_model: str
    multi_paper_hyde_max_claims: int
    multi_paper_hyde_cache_enabled: bool
    multi_paper_hyde_temperature: float
    multi_paper_hyde_max_tokens: int
    multi_paper_hyde_timeout_seconds: float
    paper_conditioned_claims_enabled: bool
    paper_conditioned_claims_model: str
    paper_conditioned_claims_max_papers: int
    paper_conditioned_claims_cache_enabled: bool
    paper_conditioned_claims_temperature: float
    paper_conditioned_claims_max_tokens: int
    paper_conditioned_claims_timeout_seconds: float
    paper_local_bm25_route_mode: str
    symbolic_evidence_standardization: bool
    symbolic_source_type_hints: bool
    retrieval_method: str
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
        generation_request_retries=int(get("GENERATION_REQUEST_RETRIES", "1") or "1"),
        generation_retry_on_429=get_bool("GENERATION_RETRY_ON_429", True),
        generation_429_max_retries=int(get("GENERATION_429_MAX_RETRIES", "6") or "6"),
        generation_429_initial_backoff_seconds=float(get("GENERATION_429_INITIAL_BACKOFF_SECONDS", "20") or "20"),
        generation_429_backoff_multiplier=float(get("GENERATION_429_BACKOFF_MULTIPLIER", "2") or "2"),
        generation_429_max_backoff_seconds=float(get("GENERATION_429_MAX_BACKOFF_SECONDS", "300") or "300"),
        generation_cooldown_after_429_seconds=float(get("GENERATION_COOLDOWN_AFTER_429_SECONDS", "120") or "120"),
        evidence_hierarchy_card_mode=get("EVIDENCE_HIERARCHY_CARD_MODE", "verified_llm") or "verified_llm",
        evidence_hierarchy_max_claims=int(get("EVIDENCE_HIERARCHY_MAX_CLAIMS", "6") or "6"),
        evidence_hierarchy_max_cards=int(get("EVIDENCE_HIERARCHY_MAX_CARDS", "24") or "24"),
        evidence_hierarchy_l1_max_chars=int(get("EVIDENCE_HIERARCHY_L1_MAX_CHARS", "420") or "420"),
        evidence_hierarchy_l3_paper_chars=int(get("EVIDENCE_HIERARCHY_L3_PAPER_CHARS", "360") or "360"),
        evidence_hierarchy_card_max_tokens=int(get("EVIDENCE_HIERARCHY_CARD_MAX_TOKENS", "4096") or "4096"),
        evidence_hierarchy_card_source_chars=int(get("EVIDENCE_HIERARCHY_CARD_SOURCE_CHARS", "85000") or "85000"),
        evidence_hierarchy_verification_mode=get("EVIDENCE_HIERARCHY_VERIFICATION_MODE", "extractive") or "extractive",
        evidence_hierarchy_max_images=int(get("EVIDENCE_HIERARCHY_MAX_IMAGES", "8") or "8"),
        evidence_hierarchy_micro_index_chars=int(get("EVIDENCE_HIERARCHY_MICRO_INDEX_CHARS", "26000") or "26000"),
        evidence_hierarchy_micro_text_chars=int(get("EVIDENCE_HIERARCHY_MICRO_TEXT_CHARS", "240") or "240"),
        evidence_hierarchy_keyed_micro_index_chars=int(get("EVIDENCE_HIERARCHY_KEYED_MICRO_INDEX_CHARS", "0") or "0"),
        evidence_hierarchy_keyed_micro_text_chars=int(get("EVIDENCE_HIERARCHY_KEYED_MICRO_TEXT_CHARS", "180") or "180"),
        evidence_hierarchy_keyed_micro_order=get("EVIDENCE_HIERARCHY_KEYED_MICRO_ORDER", "selection") or "selection",
        evidence_hierarchy_posthoc_refinement_enabled=get_bool("EVIDENCE_HIERARCHY_POSTHOC_REFINEMENT_ENABLED", False),
        evidence_hierarchy_visual_cards_enabled=get_bool("EVIDENCE_HIERARCHY_VISUAL_CARDS_ENABLED", False),
        evidence_hierarchy_visual_cards_max_per_query=int(get("EVIDENCE_HIERARCHY_VISUAL_CARDS_MAX_PER_QUERY", "8") or "8"),
        evidence_hierarchy_visual_cards_max_per_paper=int(get("EVIDENCE_HIERARCHY_VISUAL_CARDS_MAX_PER_PAPER", "1") or "1"),
        evidence_hierarchy_visual_cards_max_tokens=int(get("EVIDENCE_HIERARCHY_VISUAL_CARDS_MAX_TOKENS", "320") or "320"),
        evidence_hierarchy_visual_verify_max_tokens=int(get("EVIDENCE_HIERARCHY_VISUAL_VERIFY_MAX_TOKENS", "120") or "120"),
        evidence_triple_mode=get("EVIDENCE_TRIPLE_MODE", "extractive") or "extractive",
        evidence_triple_text_model=get("EVIDENCE_TRIPLE_TEXT_MODEL", DEFAULT_VLM_MODEL) or DEFAULT_VLM_MODEL,
        evidence_triple_visual_model=get("EVIDENCE_TRIPLE_VISUAL_MODEL", DEFAULT_VLM_MODEL) or DEFAULT_VLM_MODEL,
        evidence_triple_source_chars=int(get("EVIDENCE_TRIPLE_SOURCE_CHARS", "30000") or "30000"),
        evidence_triple_batch_source_chars=int(get("EVIDENCE_TRIPLE_BATCH_SOURCE_CHARS", "5000") or "5000"),
        evidence_triple_text_max_tokens=int(get("EVIDENCE_TRIPLE_TEXT_MAX_TOKENS", "512") or "512"),
        evidence_triple_text_max_windows=int(get("EVIDENCE_TRIPLE_TEXT_MAX_WINDOWS", "8") or "8"),
        evidence_triple_timeout_seconds=float(get("EVIDENCE_TRIPLE_TIMEOUT_SECONDS", "45") or "45"),
        evidence_triple_cache_enabled=get_bool("EVIDENCE_TRIPLE_CACHE_ENABLED", True),
        evidence_triple_visual_max_tokens=int(get("EVIDENCE_TRIPLE_VISUAL_MAX_TOKENS", "320") or "320"),
        evidence_triple_visual_max_per_query=int(get("EVIDENCE_TRIPLE_VISUAL_MAX_PER_QUERY", "4") or "4"),
        evidence_triple_sufficiency_enabled=get_bool("EVIDENCE_TRIPLE_SUFFICIENCY_ENABLED", False),
        evidence_triple_sufficiency_model=get("EVIDENCE_TRIPLE_SUFFICIENCY_MODEL", DEFAULT_VLM_MODEL) or DEFAULT_VLM_MODEL,
        evidence_triple_sufficiency_max_tokens=int(get("EVIDENCE_TRIPLE_SUFFICIENCY_MAX_TOKENS", "320") or "320"),
        render_dpi=int(get("PDF_RENDER_DPI", "160") or "160"),
        render_format=(get("PDF_RENDER_FORMAT", "jpg") or "jpg").lower(),
        render_max_pages_per_paper=int(get("PDF_RENDER_MAX_PAGES_PER_PAPER", "0") or "0"),
        structured_cache_policy=get("STRUCTURED_CACHE_POLICY", "reuse_complete_only") or "reuse_complete_only",
        transcription_backend=get("LITTRACEQA_TRANSCRIPTION_BACKEND", "docling") or "docling",
        vlm2_context_mode=get("VLM2_CONTEXT_MODE", "text_only") or "text_only",
        vlm2_context_selection_mode=get("VLM2_CONTEXT_SELECTION_MODE", "page_all_symbolic") or "page_all_symbolic",
        vlm2_include_parse_confidence=get_bool("VLM2_INCLUDE_PARSE_CONFIDENCE", True),
        single_paper_evidence_package_budget=int(get("SINGLE_PAPER_EVIDENCE_PACKAGE_BUDGET", "12") or "12"),
        multi_paper_evidence_package_budget=int(get("MULTI_PAPER_EVIDENCE_PACKAGE_BUDGET", "36") or "36"),
        single_paper_evidence_package_min=int(get("SINGLE_PAPER_EVIDENCE_PACKAGE_MIN", "4") or "4"),
        multi_paper_evidence_package_min=int(get("MULTI_PAPER_EVIDENCE_PACKAGE_MIN", "4") or "4"),
        multi_paper_min_distinct_papers=int(get("MULTI_PAPER_MIN_DISTINCT_PAPERS", "4") or "4"),
        evidence_package_adaptive_stop=get_bool("EVIDENCE_PACKAGE_ADAPTIVE_STOP", True),
        multi_paper_modality_packages_per_paper=int(get("MULTI_PAPER_MODALITY_PACKAGES_PER_PAPER", "2") or "2"),
        multi_paper_supporting_text_packages_per_paper=int(get("MULTI_PAPER_SUPPORTING_TEXT_PACKAGES_PER_PAPER", "0") or "0"),
        evidence_package_max_context_chars=int(get("EVIDENCE_PACKAGE_MAX_CONTEXT_CHARS", "80000") or "80000"),
        evidence_package_rrf_k=int(get("EVIDENCE_PACKAGE_RRF_K", "60") or "60"),
        evidence_package_candidate_pool_per_route=int(get("EVIDENCE_PACKAGE_CANDIDATE_POOL_PER_ROUTE", "0") or "0"),
        evidence_package_max_per_page=int(get("EVIDENCE_PACKAGE_MAX_PER_PAGE", "2") or "2"),
        evidence_package_page_text_anchors_per_page=int(get("EVIDENCE_PACKAGE_PAGE_TEXT_ANCHORS_PER_PAGE", "0") or "0"),
        section_relevance_backend=get("SECTION_RELEVANCE_BACKEND", "bm25") or "bm25",
        section_relevance_unit_mode=get("SECTION_RELEVANCE_UNIT_MODE", "token_chunks") or "token_chunks",
        section_relevance_unit_target_tokens=int(get("SECTION_RELEVANCE_UNIT_TARGET_TOKENS", "1280") or "1280"),
        section_relevance_unit_max_tokens=int(get("SECTION_RELEVANCE_UNIT_MAX_TOKENS", "1536") or "1536"),
        section_relevance_unit_overlap_records=int(get("SECTION_RELEVANCE_UNIT_OVERLAP_RECORDS", "1") or "1"),
        section_relevance_object_units_enabled=get_bool("SECTION_RELEVANCE_OBJECT_UNITS_ENABLED", True),
        section_relevance_object_neighbor_records=int(get("SECTION_RELEVANCE_OBJECT_NEIGHBOR_RECORDS", "1") or "1"),
        llmrerank_model=get("LLMRERANK_MODEL", "Qwen/Qwen3-VL-Reranker-8B") or "Qwen/Qwen3-VL-Reranker-8B",
        llmrerank_input_mode=get("LLMRERANK_INPUT_MODE", "text_with_object_images") or "text_with_object_images",
        llmrerank_batch_size=int(get("LLMRERANK_BATCH_SIZE", "8") or "8"),
        llmrerank_request_concurrency=int(get("LLMRERANK_REQUEST_CONCURRENCY", "1") or "1"),
        llmrerank_request_timeout_seconds=float(get("LLMRERANK_REQUEST_TIMEOUT_SECONDS", "120") or "120"),
        llmrerank_max_retries=int(get("LLMRERANK_MAX_RETRIES", "3") or "3"),
        llmrerank_failure_fallback=get("LLMRERANK_FAILURE_FALLBACK", "none") or "none",
        llmrerank_max_images_per_section=int(get("LLMRERANK_MAX_IMAGES_PER_SECTION", "4") or "4"),
        llmrerank_instruction_version=get("LLMRERANK_INSTRUCTION_VERSION", "v1") or "v1",
        llmrerank_query_mode=get("LLMRERANK_QUERY_MODE", "original") or "original",
        llmrerank_include_paper_identity=get_bool("LLMRERANK_INCLUDE_PAPER_IDENTITY", False),
        retriever_pool_budget=int(get("RETRIEVER_POOL_BUDGET", "0") or "0"),
        multi_paper_hyde_enabled=get_bool("MULTI_PAPER_HYDE_ENABLED", False),
        multi_paper_hyde_model=get("MULTI_PAPER_HYDE_MODEL", "deepseek-ai/DeepSeek-V4-Flash") or "deepseek-ai/DeepSeek-V4-Flash",
        multi_paper_hyde_max_claims=int(get("MULTI_PAPER_HYDE_MAX_CLAIMS", "4") or "4"),
        multi_paper_hyde_cache_enabled=get_bool("MULTI_PAPER_HYDE_CACHE_ENABLED", True),
        multi_paper_hyde_temperature=float(get("MULTI_PAPER_HYDE_TEMPERATURE", "0") or "0"),
        multi_paper_hyde_max_tokens=int(get("MULTI_PAPER_HYDE_MAX_TOKENS", "512") or "512"),
        multi_paper_hyde_timeout_seconds=float(get("MULTI_PAPER_HYDE_TIMEOUT_SECONDS", "120") or "120"),
        paper_conditioned_claims_enabled=get_bool("PAPER_CONDITIONED_CLAIMS_ENABLED", False),
        paper_conditioned_claims_model=get("PAPER_CONDITIONED_CLAIMS_MODEL", "deepseek-ai/DeepSeek-V4-Flash") or "deepseek-ai/DeepSeek-V4-Flash",
        paper_conditioned_claims_max_papers=int(get("PAPER_CONDITIONED_CLAIMS_MAX_PAPERS", "12") or "12"),
        paper_conditioned_claims_cache_enabled=get_bool("PAPER_CONDITIONED_CLAIMS_CACHE_ENABLED", True),
        paper_conditioned_claims_temperature=float(get("PAPER_CONDITIONED_CLAIMS_TEMPERATURE", "0") or "0"),
        paper_conditioned_claims_max_tokens=int(get("PAPER_CONDITIONED_CLAIMS_MAX_TOKENS", "1800") or "1800"),
        paper_conditioned_claims_timeout_seconds=float(get("PAPER_CONDITIONED_CLAIMS_TIMEOUT_SECONDS", "120") or "120"),
        paper_local_bm25_route_mode=get("PAPER_LOCAL_BM25_ROUTE_MODE", "disabled") or "disabled",
        symbolic_evidence_standardization=get_bool("SYMBOLIC_EVIDENCE_STANDARDIZATION", True),
        symbolic_source_type_hints=get_bool("SYMBOLIC_SOURCE_TYPE_HINTS", False),
        retrieval_method=get("RETRIEVAL_METHOD", "hybrid_alias") or "hybrid_alias",
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
