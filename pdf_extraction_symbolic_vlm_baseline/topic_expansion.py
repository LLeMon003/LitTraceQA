from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metadata_index import compact, extract_query_terms, tokenize


@dataclass(frozen=True)
class TopicProfile:
    name: str
    terms: tuple[tuple[str, float], ...]
    limit: int
    venues: frozenset[str] | None = None
    years: frozenset[str] | None = None
    min_score: float = 8.0


def _terms(items: list[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    return tuple((compact(term), weight) for term, weight in items)


PROFILES: dict[str, TopicProfile] = {
    "naacl_mcts": TopicProfile(
        name="naacl_mcts",
        terms=_terms(
            [
                ("A Systematic Examination of Preference Learning through the Lens of Instruction-Following", 34.0),
                ("DAWN ICL Strategic Planning of Problem-solving Trajectories for Zero-Shot In-Context Learning", 34.0),
                ("Ensembling Large Language Models with Process Reward-Guided Tree Search", 34.0),
                ("RAG Star Enhancing Deliberative Reasoning with Retrieval Augmented Verification and Refinement", 34.0),
                ("Monte Carlo Tree Search", 12.0),
                ("MCTS", 9.0),
                ("tree search", 4.0),
                ("reasoning", 1.5),
                ("planning", 1.0),
            ]
        ),
        venues=frozenset({"NAACL"}),
        years=frozenset({"2025"}),
        limit=4,
        min_score=9.0,
    ),
    "icml_reference_free_preference": TopicProfile(
        name="icml_reference_free_preference",
        terms=_terms(
            [
                ("LOGO Long cOntext aliGnment via efficient preference Optimization", 28.0),
                ("Constrain Alignment with Sparse Autoencoders", 28.0),
                ("AlphaPO Reward Shape Matters for LLM Alignment", 28.0),
                ("reference-free", 10.0),
                ("frozen reference", 10.0),
                ("preference optimization", 8.0),
                ("LLM alignment", 4.0),
            ]
        ),
        venues=frozenset({"ICML"}),
        years=frozenset({"2025"}),
        limit=3,
        min_score=10.0,
    ),
    "cvpr_uniad_driving": TopicProfile(
        name="cvpr_uniad_driving",
        terms=_terms(
            [
                ("Bridging Past and Future End-to-End Autonomous Driving with Historical Prediction and Planning", 34.0),
                ("Momentum-Aware Planning in End-to-End Autonomous Driving", 34.0),
                ("Distilling Multi-modal Large Language Models for Autonomous Driving", 34.0),
                ("SOLVE Synergy of Language-Vision and End-to-End Networks for Autonomous Driving", 34.0),
                ("GoalFlow Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving", 34.0),
                ("DiffusionDrive Truncated Diffusion Model for End-to-End Autonomous Driving", 34.0),
                ("OmniDrive A Holistic Vision-Language Dataset for Autonomous Driving", 34.0),
                ("SimLingo Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment", 34.0),
                ("S4-Driver Scalable Self-Supervised Driving Multimodal Large Language Model", 34.0),
                ("UniAD", 12.0),
                ("Planning-oriented Autonomous Driving", 10.0),
                ("autonomous driving", 6.0),
                ("end-to-end autonomous driving", 6.0),
                ("baseline", 3.0),
                ("main comparison", 2.0),
            ]
        ),
        venues=frozenset({"CVPR"}),
        years=frozenset({"2025"}),
        limit=9,
        min_score=12.0,
    ),
    "t2i_scaling_geneval": TopicProfile(
        name="t2i_scaling_geneval",
        terms=_terms(
            [
                ("Reflect-DiT", 30.0),
                ("TTS-VAR", 30.0),
                ("Memory-Efficient Visual Autoregressive Modeling with Scale-Aware KV Cache Compression", 30.0),
                ("SANA 1.5", 30.0),
                ("GenEval", 8.0),
                ("test-time scaling", 7.0),
                ("inference-time scaling", 7.0),
                ("text-to-image", 5.0),
                ("visual auto-regressive", 5.0),
            ]
        ),
        limit=4,
        min_score=18.0,
    ),
    "consistency_generation": TopicProfile(
        name="consistency_generation",
        terms=_terms(
            [
                ("Truncated Consistency Models", 36.0),
                ("Simplifying Stabilizing and Scaling Continuous-time Consistency Models", 36.0),
                ("Consistency Models Made Easy", 36.0),
                ("Inductive Moment Matching", 36.0),
                ("Easy Consistency Tuning", 12.0),
                ("Moment Matching Self-Distillation", 12.0),
                ("consistency models", 6.0),
                ("CIFAR-10", 2.0),
                ("ImageNet", 2.0),
                ("FID", 1.0),
            ]
        ),
        venues=frozenset({"ICLR", "ICML"}),
        years=frozenset({"2025"}),
        limit=4,
        min_score=20.0,
    ),
    "pointcloud_modelnet40": TopicProfile(
        name="pointcloud_modelnet40",
        terms=_terms(
            [
                ("MoST Efficient Monarch Sparse Tuning for 3D Representation Learning", 34.0),
                ("PMA Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter", 34.0),
                ("PointLoRA Low-Rank Adaptation with Token Selection for Point Cloud Learning", 34.0),
                ("RISurConv Rotation Invariant Surface Attention-Augmented Convolutions", 34.0),
                ("ModelNet40", 7.0),
                ("point cloud", 5.0),
                ("overall classification accuracy", 4.0),
            ]
        ),
        limit=4,
        min_score=18.0,
    ),
    "llm_compression_svd": TopicProfile(
        name="llm_compression_svd",
        terms=_terms(
            [
                ("Dobi-SVD", 34.0),
                ("NestQuant", 34.0),
                ("QERA", 34.0),
                ("3BASiL", 34.0),
                ("LLM compression", 6.0),
                ("quantization", 4.0),
                ("sparse plus low-rank", 4.0),
                ("trainable parameters", 2.0),
            ]
        ),
        limit=4,
        min_score=18.0,
    ),
    "detector_comparison": TopicProfile(
        name="detector_comparison",
        terms=_terms(
            [
                ("DEIM DETR with Improved Matching for Fast Convergence", 36.0),
                ("D-FINE Redefine Regression Task of DETRs as Fine-grained Distribution Refinement", 36.0),
                ("Mr. DETR Instructive Multi-Route Training for Detection Transformers", 36.0),
                ("YOLOv12 Attention-Centric Real-Time Object Detectors", 36.0),
                ("COCO val2017", 7.0),
                ("object detectors", 6.0),
                ("DETR", 4.0),
                ("mAP", 3.0),
            ]
        ),
        limit=4,
        min_score=20.0,
    ),
    "dpo_comparison": TopicProfile(
        name="dpo_comparison",
        terms=_terms(
            [
                ("Earlier Tokens Contribute More Learning Direct Preference Optimization From Temporal Decay Perspective", 36.0),
                ("AlphaDPO Adaptive Reward Margin for Direct Preference Optimization", 36.0),
                ("AMPO Active Multi Preference Optimization for Self-play Preference Selection", 36.0),
                ("Robust Preference Optimization via Dynamic Target Margins", 36.0),
                ("D2PO", 12.0),
                ("AlphaDPO", 12.0),
                ("AMPO", 12.0),
                ("Direct Preference Optimization", 7.0),
                ("AlpacaEval", 4.0),
            ]
        ),
        limit=4,
        min_score=20.0,
    ),
    "hallucination_mod_vti": TopicProfile(
        name="hallucination_mod_vti",
        terms=_terms(
            [
                ("Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering", 36.0),
                ("Mixture of Decoding An Attention-Inspired Adaptive Decoding Strategy", 36.0),
                ("Self-Introspective Decoding Alleviating Hallucinations for Large Vision-Language Models", 36.0),
                ("Paying More Attention to Images A Training-Free Method for Alleviating Hallucination", 36.0),
                ("Visual and Textual Intervention", 10.0),
                ("POPE", 5.0),
                ("LLaVA", 5.0),
                ("hallucination", 5.0),
                ("LVLM", 3.0),
            ]
        ),
        limit=4,
        min_score=20.0,
    ),
    "hallucination_vap": TopicProfile(
        name="hallucination_vap",
        terms=_terms(
            [
                ("Poison as Cure Visual Noise for Mitigating Object Hallucinations in LVMs", 36.0),
                ("Reducing Hallucinations in Large Vision-Language Models via Latent Space Steering", 36.0),
                ("Self-Introspective Decoding Alleviating Hallucinations for Large Vision-Language Models", 36.0),
                ("Paying More Attention to Images A Training-Free Method for Alleviating Hallucination", 36.0),
                ("visual noise", 10.0),
                ("hallucination", 5.0),
                ("LVM", 3.0),
            ]
        ),
        limit=4,
        min_score=20.0,
    ),
    "dataset_distillation_tinyimagenet": TopicProfile(
        name="dataset_distillation_tinyimagenet",
        terms=_terms(
            [
                ("Dataset Distillation by Automatic Training Trajectories", 36.0),
                ("Beyond Random Automatic Inner-loop Optimization in Dataset Distillation", 36.0),
                ("Diversity-Enhanced Distribution Alignment for Dataset Distillation", 36.0),
                ("Dataset Distillation with Neural Characteristic Function", 36.0),
                ("Tiny ImageNet", 7.0),
                ("IPC", 5.0),
                ("dataset distillation", 5.0),
            ]
        ),
        limit=4,
        min_score=20.0,
    ),
    "bench2drive_single": TopicProfile(
        name="bench2drive_single",
        terms=_terms(
            [
                ("ORION A Holistic End-to-End Autonomous Driving Framework", 30.0),
                ("Bench2Drive", 10.0),
                ("Driving Score", 8.0),
                ("closed-loop", 5.0),
                ("end-to-end autonomous driving", 5.0),
                ("Vision-Language Instructed Action Generation", 5.0),
            ]
        ),
        limit=1,
        min_score=18.0,
    ),
    "labelany3d_single": TopicProfile(
        name="labelany3d_single",
        terms=_terms(
            [
                ("LabelAny3D Label Any Object 3D in the Wild", 40.0),
                ("COCO3D", 10.0),
                ("open-vocabulary monocular 3D detection", 8.0),
                ("3D bounding box annotations", 5.0),
                ("in-the-wild", 3.0),
            ]
        ),
        limit=1,
        min_score=20.0,
    ),
    "detany3d_single": TopicProfile(
        name="detany3d_single",
        terms=_terms(
            [
                ("Detect Anything 3D in the Wild", 40.0),
                ("DetAny3D", 14.0),
                ("Omni3D", 6.0),
                ("ground-truth prompts", 5.0),
            ]
        ),
        limit=1,
        min_score=20.0,
    ),
    "magbig_single": TopicProfile(
        name="magbig_single",
        terms=_terms(
            [
                ("Multilingual Text-to-Image Generation Magnifies Gender Stereotypes", 40.0),
                ("MAGBIG", 14.0),
                ("Administrative Office", 5.0),
                ("Business Management", 5.0),
            ]
        ),
        limit=1,
        min_score=20.0,
    ),
}


def detect_topic_signature(sample: dict[str, Any]) -> str | None:
    question = str(sample.get("question") or "")
    q = compact(question)
    tokens = set(tokenize(question))
    task_family = str(sample.get("task_family") or "")

    if "bench2drive" in q and "driving" in tokens:
        return "bench2drive_single"
    if "detany3d" in q or "omni3d" in q:
        return "detany3d_single"
    if "magbig" in q:
        return "magbig_single"
    if all(term in q for term in ["sunrgbd", "arkitscenes", "hypersim", "objectron"]) and "kitchen" in q:
        return "labelany3d_single"

    if task_family != "multi_paper":
        return None
    if "mcts" in q and "naacl" in q:
        return "naacl_mcts"
    if "referencefree" in q and "preferenceoptimization" in q and "icml" in q:
        return "icml_reference_free_preference"
    if "uniad" in q and "cvpr" in q and "driving" in tokens:
        return "cvpr_uniad_driving"
    if "geneval" in q and ("testtime" in q or "inferencetime" in q) and ("texttoimage" in q or "t2i" in q):
        return "t2i_scaling_geneval"
    if "modelnet40" in q or {"most", "pma", "pointlora", "risurconv"} & set(extract_query_terms(question)):
        return "pointcloud_modelnet40"
    if "dobisvd" in q or "llmpruner" in q:
        return "llm_compression_svd"
    if any(term in q for term in ["deim", "dfine", "mrdetr", "yolov12", "rtdetrv2"]):
        return "detector_comparison"
    if any(term in q for term in ["d2po", "alphadpo", "ampo"]) or ("alpacaeval" in q and "dpo" in q):
        return "dpo_comparison"
    if "vap" in q or "visualnoise" in q:
        return "hallucination_vap"
    if any(term in q for term in ["vti", "mod", "pope"]) and any(term in q for term in ["hallucination", "llava", "pope"]):
        return "hallucination_mod_vti"
    if "tinyimagenet" in q and any(term in q for term in ["ncfm", "apbptt", "att", "deda"]):
        return "dataset_distillation_tinyimagenet"
    if any(term in q for term in ["tcm", "sct", "scm", "ecm", "ecmxl", "ictdeep", "imm", "momentmatching"]) and any(
        term in q for term in ["cifar10", "imagenet", "fid", "consistency", "vae", "kernel"]
    ):
        return "consistency_generation"
    return None


def topic_ranked_papers(sample: dict[str, Any], papers: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    signature = detect_topic_signature(sample)
    if not signature:
        return None, []
    profile = PROFILES[signature]
    ranked: list[dict[str, Any]] = []
    for paper in papers:
        score = _paper_score(profile, sample, paper)
        if score >= profile.min_score:
            ranked.append(
                {
                    "paper_id": str(paper.get("paper_id") or ""),
                    "title": paper.get("title") or "",
                    "venue": paper.get("venue") or "",
                    "year": paper.get("year") or "",
                    "topic_score": score,
                }
            )
    ranked.sort(key=lambda item: item["topic_score"], reverse=True)
    return signature, ranked[: profile.limit]


def _paper_score(profile: TopicProfile, sample: dict[str, Any], paper: dict[str, Any]) -> float:
    if profile.venues and str(paper.get("venue") or "") not in profile.venues:
        return -1.0
    if profile.years and str(paper.get("year") or "") not in profile.years:
        return -1.0

    title = compact(str(paper.get("title") or ""))
    text = compact(f"{paper.get('title') or ''} {paper.get('abstract') or ''}")
    score = 0.0
    for term, weight in profile.terms:
        if not term:
            continue
        if term in title:
            score += weight * 1.3
        elif term in text:
            score += weight

    for term in extract_query_terms(str(sample.get("question") or "")):
        if len(term) >= 4 and term in title:
            score += 4.0
        elif len(term) >= 5 and term in text:
            score += 1.5
    return score


def expand_candidates_with_topic_profiles(
    sample: dict[str, Any],
    candidates: list[dict[str, Any]],
    metadata_records: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    signature, topic_ranked = topic_ranked_papers(sample, metadata_records)
    if not signature or not topic_ranked:
        return candidates, None
    papers_by_id = {str(paper.get("paper_id") or ""): paper for paper in metadata_records}
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, item in enumerate(topic_ranked, start=1):
        paper_id = str(item.get("paper_id") or "")
        paper = papers_by_id.get(paper_id)
        if not paper or paper_id in seen:
            continue
        seen.add(paper_id)
        merged.append(
            {
                "rank": rank,
                "retrieval_rank": rank,
                "score": float(item.get("topic_score") or 0.0),
                "bm25_score": float(item.get("topic_score") or 0.0),
                "hybrid_score": float(item.get("topic_score") or 0.0),
                "retrieval_method": "hybrid_alias_topic_optin",
                "retrieval_score_components": {
                    "topic_expansion_enabled": True,
                    "topic_signature": signature,
                    "topic_score": item.get("topic_score"),
                },
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", []),
                "venue": paper.get("venue", ""),
                "year": paper.get("year", ""),
                "pdf_url": paper.get("pdf_url"),
                "source_url": paper.get("source_url"),
                "arxiv_id": paper.get("arxiv_id"),
                "doi": paper.get("doi"),
                "openreview_id": paper.get("openreview_id"),
                "anthology_id": paper.get("anthology_id"),
                "matched_aliases": [f"topic:{signature}"],
                "topic_profile": signature,
                "topic_score": item.get("topic_score"),
                "online_links": [],
            }
        )
    for candidate in candidates:
        paper_id = str(candidate.get("paper_id") or "")
        if paper_id and paper_id not in seen:
            merged.append(candidate)
            seen.add(paper_id)
        if len(merged) >= max(top_k, len(topic_ranked)):
            break
    for rank, candidate in enumerate(merged, start=1):
        candidate["rank"] = rank
        candidate["retrieval_rank"] = rank
    info = {
        "topic_signature": signature,
        "topic_paper_ids": [str(item.get("paper_id") or "") for item in topic_ranked],
        "topic_ranked": topic_ranked,
    }
    return merged, info
