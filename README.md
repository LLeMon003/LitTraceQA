# Log 000｜LitTraceQA 实验日志索引

本仓库用于记录 LitTraceQA 任务上的 baseline 设计、代码实现与实验运行日志。当前阶段的核心目标不是直接追求最高分，而是先建立一个可复现、可扩展、边界清晰的实验框架：在无法稳定访问在线 PDF 或全文内容的前提下，先完成一个 **PDF-free / metadata-only baseline**，验证数据读取、候选论文检索、LLM 调用、预测格式生成和本地评估流程。

后续如果加入 PDF-native API input、raw PDF 缓存解析、full-text retrieval、table/figure/equation-aware reasoning 或 neuro-symbolic verification 等方案，应以新的实验日志继续追加，而不是直接混入当前 baseline。为了保留未来方案的结构余量，本文档使用 `# Log XXX｜...` 作为最高层级日志标题，具体实验内容使用二级及以下标题展开。

## 当前实验路线总览

| 日志编号    | 方案名称                                             |     当前状态 | 是否访问 PDF / URL | 主要作用                                         |
| ------- | ------------------------------------------------ | -------: | -------------: | -------------------------------------------- |
| Log 001 | PDF-Free Baseline：`metadata_only_title_abstract` |      已实现 |              否 | 只用 title + abstract 完成候选检索与 LLM 回答           |
| Log 002 | PDF OCR Context VLM Baseline：`pdf_ocr_context_vlm` |   Smoke 已接通 |       读取本地 PDF | 用 DeepSeek-OCR 生成结构化 OCR contexts，再交给 VLM 回答 |

## 仓库当前结构

```text
LitTraceQA/
├── metadata_only_baseline/
│   ├── __init__.py
│   ├── config.py
│   ├── data_io.py
│   ├── evaluate_local.py
│   ├── link_utils.py
│   ├── llm_client.py
│   ├── metadata_index.py
│   ├── parser.py
│   ├── pdf_access_probe.py
│   ├── pdf_downloader.py
│   ├── prompt_builder.py
│   └── run_baseline.py
├── pdf_ocr_vlm_baseline/
│   ├── __init__.py
│   ├── answer_client.py
│   ├── config.py
│   ├── context_index.py
│   ├── deepseek_ocr_converter.py
│   ├── metadata_index.py
│   ├── ocr_context_prompt_builder.py
│   ├── parser.py
│   ├── pdf_cache.py
│   ├── pipeline_capability_probe.py
│   └── run_pdf_ocr_context_vlm_baseline.py
├── official_dev/
│   ├── data/
│   ├── docs/
│   ├── schema/
│   ├── scripts/
│   ├── CITATION.cff
│   ├── LICENSE.md
│   └── README.md
├── .env.example
├── .gitignore
└── README.md
```

当前仓库中，`official_dev/` 是 LitTraceQA 官方开发集，必须视为不可修改原始数据；`metadata_only_baseline/` 和 `pdf_ocr_vlm_baseline/` 是本仓库新增的 baseline 代码；`outputs/`、`raw_pdfs/`、`processed_pdfs/` 和 `indexes/` 被 `.gitignore` 排除，用于本地实验输出、PDF 缓存和中间处理结果，不进入版本控制。

# Log 001｜PDF-Free Baseline：metadata_only_title_abstract

## 实验目标

当前 baseline 的目标是在 **完全不访问 PDF、URL、DOI、arXiv、OpenReview 或任何在线全文内容** 的前提下，建立一个最小可运行系统。该系统只使用 `official_dev/data/paper_metadata.jsonl` 中的 `title` 与 `abstract` 字段完成以下流程：

1. 从官方输入文件读取问题；
2. 基于问题文本对论文 metadata 做候选检索；
3. 将 top-k 候选论文的 title、abstract、venue、year 和 paper_id 发送给 LLM；
4. 要求 LLM 生成符合 LitTraceQA 提交格式的 JSON；
5. 对 LLM 输出进行解析、归一化和 fallback；
6. 生成 `predictions.jsonl`；
7. 调用官方本地 evaluator 做开发集评估。

该方案的意义在于建立一个稳定的工程起点。由于不访问全文，当前 baseline 无法可靠定位 table、figure、equation、citation context 或 page-level evidence。因此它不是强 evidence-grounded baseline，而是一个 **数据管线与格式验证 baseline**。

## 当前 baseline 边界

当前实验严格遵守以下边界：

* 不下载 PDF；
* 不打开 DOI；
* 不访问 arXiv 页面；
* 不访问 OpenReview 页面；
* 不读取网页；
* 不解析全文；
* 不抽取 table、figure、equation 或 citation context；
* 不声称使用了 page-level evidence；
* 不将 `pdf_access_probe` 或 `pdf_downloader` 的结果混入正式预测。

发送给 LLM 的 prompt 是英文，以降低模型输出格式不稳定的风险；README、实验日志和本地说明使用中文。

## 核心代码逻辑

### 1. 配置读取：`config.py`

`config.py` 负责读取 `.env` 或系统环境变量中的 SiliconFlow 配置。默认配置为：

```env
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

程序会检查 `SILICONFLOW_API_KEY` 是否为空值或占位符，并且只输出 mask 后的 key，不输出完整 API key。

### 2. 数据读取：`data_io.py`

`data_io.py` 提供 JSONL 读写工具，并负责在 `official_dev/data/` 或 `official_dev/` 下查找官方文件。当前 baseline 主要读取：

```text
official_dev/data/validation_inputs.jsonl
official_dev/data/paper_metadata.jsonl
official_dev/data/validation.jsonl
```

其中：

* `validation_inputs.jsonl`：输入问题，不包含 gold answer；
* `paper_metadata.jsonl`：可检索论文池；
* `validation.jsonl`：本地评估用 gold file。

### 3. 候选论文检索：`metadata_index.py`

`metadata_index.py` 当前使用 BM25 做稀疏检索。检索字段只包括：

```text
title + abstract
```

处理逻辑如下：

1. 对 question 做 lowercase tokenization；
2. 对每篇论文的 title 和 abstract 做 tokenization；
3. 用 `rank_bm25.BM25Okapi` 建立候选排序；
4. 返回 top-k 候选；
5. 每个候选包含 `paper_id`、`title`、`abstract`、`venue`、`year` 和 BM25 score；
6. 在线链接字段被置空，避免 metadata-only baseline 意外使用 URL。

当前检索方式非常轻量，适合作为初始 baseline，但存在明显上限：

* 没有 dense retrieval；
* 没有 reranker；
* 没有 entity-aware matching；
* 没有 citation graph；
* 没有全文 evidence；
* BM25 index 尚未做持久化或复用优化。

### 4. Prompt 构造：`prompt_builder.py`

`prompt_builder.py` 构造两段消息：

1. system prompt：声明模型是 metadata-only baseline，只能使用候选论文 title 和 abstract；
2. user prompt：包含 query、answer_types、table_schema 和 candidate_papers。

prompt 明确要求：

* 只输出合法 JSON；
* 不输出 markdown；
* 不声称访问过 PDF 或网页；
* 只能使用候选中的 `paper_id`；
* evidence grounding 置信度应保持保守；
* 如果需要 table answer，必须使用 `table_schema` 中的列名。

输出目标结构包括：

```json
{
  "query_id": "",
  "gold_papers": [{"paper_id": ""}],
  "evidence": [],
  "answer": {},
  "confidence": {
    "paper_retrieval": 0.0,
    "evidence_grounding": 0.0,
    "answer": 0.0
  },
  "notes": {
    "baseline_type": "metadata_only_title_abstract",
    "used_online_access": false,
    "accessed_links": [],
    "limitations": "This baseline used only paper titles and abstracts, without PDF or full-text access."
  }
}
```

正式提交文件中，`parser.py` 会进一步归一化为 evaluator 需要的字段。

### 5. LLM 调用：`llm_client.py`

`llm_client.py` 使用 OpenAI-compatible `/chat/completions` 接口调用 SiliconFlow。当前实现使用 Python 标准库 `urllib.request`，不依赖 OpenAI SDK。

调用参数来自 `.env`：

```env
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V4-Flash
SILICONFLOW_TEMPERATURE=0
SILICONFLOW_MAX_TOKENS=3000
SILICONFLOW_TIMEOUT_SECONDS=120
```

当前默认重试次数为 2 次。API 调用失败后，主流程会记录 error，并生成 fallback prediction。

### 6. 输出解析与 fallback：`parser.py`

`parser.py` 负责从 LLM 回复中提取 JSON，并将输出归一化为官方预测格式。

主要逻辑包括：

* 支持从 `json fenced block` 中提取 JSON；
* 如果模型输出包含额外文本，尝试定位第一个完整 JSON object；
* 根据输入中的 `answer_types` 过滤答案字段；
* 如果模型返回非候选 paper_id，默认替换为 top-1 候选；
* 如果无法解析 JSON，则 fallback 到 top-1 candidate；
* 如果输出结构非法，则再次 fallback。

fallback prediction 的基本结构为：

```json
{
  "query_id": "...",
  "gold_papers": [{"paper_id": "top1_candidate"}],
  "evidence": [],
  "answer": {}
}
```

这保证了即使 API 输出不稳定，`predictions.jsonl` 也尽可能保持可评估。

### 7. 主流程：`run_baseline.py`

`run_baseline.py` 是当前 metadata-only baseline 的主入口。

流程如下：

1. 读取参数；
2. 创建输出目录；
3. 读取 `validation_inputs.jsonl`；
4. 读取 `paper_metadata.jsonl`；
5. 构建 metadata records；
6. 对每个 query 做 BM25 top-k 检索；
7. 保存候选论文；
8. 构造 prompt preview；
9. 如果是 dry run，则停止在 prompt preview；
10. 如果不是 dry run，则调用 LLM；
11. 解析 LLM 输出；
12. 生成 prediction；
13. 记录 raw response 和 errors；
14. 生成 run report。

当前 baseline 类型固定为：

```text
metadata_only_title_abstract
```

## 环境准备

建议使用独立 conda 环境：

```bash
conda activate littraceqa
```


```bash
pip install -r requirements.txt
```

其中 `python-dotenv` 是可选依赖；如果未安装，`config.py` 会回退到内部 `.env` 解析逻辑。

## `.env` 配置

在 workspace 根目录创建 `.env`：

```env
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

程序会检查 key 是否为占位符，并且不会输出完整 API key。

## Dry Run

Dry run 不调用 API，只生成候选论文和 prompt preview。建议先运行该模式检查数据路径、检索结果和 prompt 格式。

```bash
python -m metadata_only_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 8 \
  --max-queries 2 \
  --dry-run
```

预期输出：

```text
outputs/api_baseline/candidate_papers.jsonl
outputs/api_baseline/prompt_previews.jsonl
outputs/api_baseline/errors.jsonl
outputs/api_baseline/run_report.md
```

Dry run 不会生成正式 API response，也不会生成最终 `predictions.jsonl`。

## 2-Query API Smoke Test

确认 `.env` 中 API key 已补齐后，运行 2 条样本测试：

```bash
python -m metadata_only_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 8 \
  --max-queries 2
```

该实验用于确认：

* `.env` 可被正确读取；
* API endpoint 可访问；
* 模型能返回 JSON-like 内容；
* parser 能抽取并归一化输出；
* fallback 机制可用；
* `predictions.jsonl` 可被写入。

## Full Validation

完整运行 public validation split：

```bash
python -m metadata_only_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 12
```

当前默认 `top-k=12`。由于该 baseline 只使用 metadata，增大 top-k 可能提高 gold paper recall，但也可能增加 prompt 噪声。后续可记录不同 top-k 的对比实验。

## Resume 运行

如果中途 API 调用中断，可以使用：

```bash
python -m metadata_only_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 12 \
  --resume
```

`--resume` 会读取已有 `predictions.jsonl` 中的 `query_id`，跳过已完成样本。

## Local Evaluation

生成 `predictions.jsonl` 后，运行本地 evaluator：

```bash
python -m metadata_only_baseline.evaluate_local \
  --official-dir official_dev \
  --pred outputs/api_baseline/predictions.jsonl
```

`evaluate_local.py` 会自动寻找：

```text
official_dev/scripts/evaluate.py
official_dev/data/validation.jsonl
```

如果 evaluator 参数可安全确认，则自动执行：

```bash
python official_dev/scripts/evaluate.py \
  --gold official_dev/data/validation.jsonl \
  --pred outputs/api_baseline/predictions.jsonl
```

## 当前输出文件

正式 baseline 输出目录默认为：

```text
outputs/api_baseline_bm25/
```

主要文件包括：

```text
predictions.jsonl          # 官方评估输入
raw_llm_responses.jsonl    # 原始 API 返回
candidate_papers.jsonl     # 每个 query 的 BM25 top-k 候选
prompt_previews.jsonl      # 每个 query 的 prompt
errors.jsonl               # parse failure、API failure、noncandidate replacement 等错误
run_report.md              # 本次运行摘要
```

这些文件被 `.gitignore` 排除，不进入 GitHub 仓库。

## 当前实验记录模板

### Full Validation｜top-k=10

```text
运行状态：Success
运行命令：python -m metadata_only_baseline.run_baseline --official-dir official_dev --output-dir outputs/api_baseline --top-k 12
- baseline type: `metadata_only_title_abstract`
- processed query count: 55
- successful API call count: 55
- parse failure count: 0
- fallback prediction count: 0
- model: `deepseek-ai/DeepSeek-V4-Flash`
- base url: `https://api.siliconflow.cn/v1`
- top_k: 10
```

### Official Evaluation

```text
paper_precision_macro: 0.6818
paper_recall_macro:    0.5273
paper_f1_macro:        0.5629

evidence_f1_macro:     0.0
multiple_choice_accuracy: 0.0488
freeform_exact_match:  0.0
table_row_f1_macro:    0.2712
table_cell_accuracy_macro: 0.0530
table_cell_accuracy_micro: 0.0370
```

## 当前方案预期局限

该方案的主要局限是任务信息不足，而不是实现错误。

LitTraceQA 要求系统检索相关论文、定位粗粒度 evidence，并生成指定格式答案。但当前 baseline 只读取 title 和 abstract，因此：

* 对 `text_span` 类型问题可能只能做弱推断；
* 对 `table` 类型问题通常缺少真实表格内容；
* 对 `figure` 类型问题缺少图像与 caption；
* 对 `equation_algorithm` 类型问题缺少公式和算法正文；
* 对 `citation_context` 类型问题缺少引用上下文；
* evidence 字段大概率为空或低质量；
* answer 可能依赖 abstract 中是否直接出现目标信息；
* 多论文问题容易受到 BM25 候选召回限制。

因此，当前 baseline 的实验价值主要是：

1. 作为代码管线 sanity check；
2. 作为 PDF-free 下限 baseline；
3. 作为未来 PDF-native / full-text baseline 的对照组；
4. 作为后续 neuro-symbolic evidence verification 的输入格式参考。

# Log 002｜PDF OCR Context VLM Baseline：pdf_ocr_context_vlm

## 实验目标

当前 baseline 的目标是在 **不让 answer generation model 原生读取 PDF，也不访问在线 PDF URL** 的前提下，建立一个 PDF-aware 的最小可运行系统。该系统以本地缓存 PDF 为输入，先调用 `deepseek-ai/DeepSeek-OCR` 将 PDF 页面转换为结构化 OCR artifacts，如果 answer model supports_image_input=True：
  selected page images / visual contexts 会作为 image input 附加给 answer model。

如果 answer model supports_image_input=False：
  answer model 只读取 OCR text contexts 和 visual metadata；
  不允许声称模型检查了图片内容。 
利用answer generation model生成 LitTraceQA 预测。

当前流程使用的数据来源包括：

1. `official_dev/data/validation_inputs.jsonl` 中的问题；
2. `official_dev/data/paper_metadata.jsonl` 中的 title、abstract、paper_id 等 metadata；
3. `raw_pdfs/pdf/<paper_id>.pdf` 中的本地 PDF 缓存；
4. `processed_pdfs/deepseek_ocr/<paper_id>/` 中由 OCR 生成的结构化中间结果。

该方案的意义不是直接替代 metadata-only baseline，而是验证 PDF 处理链路是否可复现、可审计、可逐步增强。与 Log 001 不同，本 baseline 允许读取本地 PDF，但必须先经过 OCR input conversion 模块，不能把 PDF 文件直接交给最终 answer model，也不能让 answer model 在线打开论文链接。

## 当前 baseline 边界

当前实验严格遵守以下边界：

* 不修改 `official_dev/` 中的任何官方原始数据；
* 不让最终 answer generation model 直接读取 PDF；
* answer generation model不访问 PDF 的在线 URL，不打开 DOI、arXiv、OpenReview 或会议网页；
* baseline evaluation 的推理阶段只读取本地 PDF 缓存。
* 如果启用 on-demand PDF cache，系统允许在进入 OCR 前由 downloader 下载缺失 PDF；该下载行为必须记录到 pdf_availability.jsonl 和 run_report.md。
* 不伪造 OCR chunks、bbox、figure、table 或 equation locator；
* 不把 prompt echo、schema echo 或占位 OCR 输出当成真实 OCR；
* 当 OCR 没有产出可信 chunks 时，不生成正式 prediction。
* 默认复用 OCR artifacts，当显式传入 --ocr-overwrite 或 --fresh-run 时才重新生成。每次 run 创建独立 run_id，并记录 OCR artifact hash、OCR model、prompt version、API provider。

当retrieve的结果在本地没有对应的pdf时，爬虫会下载对应pdf。当前 baseline 允许使用 `raw_pdfs/pdf/<paper_id>.pdf` 中已经存在的本地 PDF 文件。PDF 页面会被渲染为图片，再以 base64 image input 的形式发送给 DeepSeek-OCR。DeepSeek-OCR 的输出必须落盘为结构化 artifacts 后，才能进入 context selection 和 answer generation 阶段。

## 核心代码逻辑

### 1. 配置读取：`config.py`

`config.py` 负责读取 `.env`、系统环境变量以及 Log 001 中复用的 SiliconFlow 基础配置。当前 PDF baseline 的核心配置为：

```env
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

OCR_MODEL=deepseek-ai/DeepSeek-OCR
OCR_PROVIDER=siliconflow
OCR_API_KEY=
OCR_BASE_URL=

ANSWER_MODEL=Qwen/Qwen3-VL-8B-Instruct
ANSWER_PROVIDER=siliconflow
ANSWER_API_KEY=
ANSWER_BASE_URL=
```

其中：

* `OCR_API_KEY` 为空时复用 `SILICONFLOW_API_KEY`；
* `OCR_BASE_URL` 为空时复用 `SILICONFLOW_BASE_URL`；
* `ANSWER_API_KEY` 为空时复用 `SILICONFLOW_API_KEY`；
* `ANSWER_BASE_URL` 为空时复用 `SILICONFLOW_BASE_URL`；
* `ANSWER_MODEL` 是首选 answer generation model；


程序会检查 API key 是否为空值或占位符，并且只输出 mask 后的 key，不输出完整 API key。

### 2. PDF 缓存与候选论文：`pdf_cache.py` 与 `metadata_index.py`

候选论文仍由 metadata BM25 检索产生。检索阶段读取：

```text
official_dev/data/validation_inputs.jsonl
official_dev/data/paper_metadata.jsonl
```

检索字段仍然只使用：

```text
title + abstract
```

`pdf_cache.py` 负责检查候选论文是否已有本地 PDF：

```text
raw_pdfs/pdf/<paper_id>.pdf
```

当前正式逻辑优先使用本地缓存。若候选 PDF 不存在，会使用原有爬虫下载并缓存。同时保证爬虫原有的一次性下载功能仍不变。

### 3. DeepSeek-OCR 转换：`deepseek_ocr_converter.py`

`deepseek_ocr_converter.py` 是当前 PDF baseline 的 input conversion 核心模块。它负责：

1. 用 PyMuPDF 打开本地 PDF；
2. 将PDF渲染为 JPEG 页面图片；
3. 将页面图片转为 base64 data URL；
4. 通过 OpenAI-compatible `/chat/completions` 调用 `deepseek-ai/DeepSeek-OCR`；
5. 要求 OCR model 返回 page-aware JSON；
6. 将结果写入 `processed_pdfs/deepseek_ocr/<paper_id>/`。

目标 OCR artifact 包括：

```text
document.json          # paper-level OCR 状态与失败原因
pages.jsonl            # page-level OCR text 与 raw content
chunks.jsonl           # text chunks，含 page、chunk_id、type、bbox、reading_order
visual_contexts.jsonl  # figure/table visual contexts，含 visual_id、caption、bbox、image_path
raw_ocr_responses.jsonl # OCR API 原始返回，用于调试 prompt echo、schema echo 或解析失败
document.md            # OCR text 的 markdown 汇总
page_images/           # 渲染后的页面图片
```

期望 OCR 输出的结构化 JSONL 目标如下。

`document.json` 是 paper-level manifest：

```json
{
  "paper_id": "acl2025_00005",
  "pdf_path": "raw_pdfs/pdf/acl2025_00005.pdf",
  "status": "ok",
  "parser": "deepseek-ai/DeepSeek-OCR",
  "provider": "siliconflow",
  "prompt_version": "deepseek_ocr_page_json_v2",
  "created_at": "2026-07-04T14:14:17Z",
  "max_pages": 1,
  "page_count_processed": 1,
  "page_count_succeeded": 1,
  "page_failures": [],
  "artifact_hashes": {
    "pages.jsonl": "...",
    "chunks.jsonl": "...",
    "visual_contexts.jsonl": "...",
    "raw_ocr_responses.jsonl": "..."
  }
}
```

`pages.jsonl` 每行对应一个 PDF page：

```json
{
  "paper_id": "acl2025_00005",
  "page": 1,
  "image_path": "processed_pdfs/deepseek_ocr/acl2025_00005/page_images/page_001.jpg",
  "width": 1191,
  "height": 1684,
  "text": "actual page text in reading order",
  "raw_content": "raw OCR model content for this page"
}
```

`chunks.jsonl` 每行对应一个可检索文本单元：

```json
{
  "paper_id": "acl2025_00005",
  "page": 1,
  "chunk_id": "p001_c001",
  "type": "paragraph",
  "text": "actual visible text block",
  "bbox": [0.12, 0.18, 0.88, 0.27],
  "reading_order": 1,
  "table_id": null,
  "figure_id": null,
  "equation_id": null,
  "parser_confidence": null
}
```

`visual_contexts.jsonl` 每行对应一个 figure/table 等视觉对象：

```json
{
  "paper_id": "acl2025_00005",
  "page": 1,
  "visual_id": "fig_p001_001",
  "source_type": "figure",
  "caption": "actual visible caption text",
  "bbox": [0.10, 0.42, 0.90, 0.74],
  "image_path": "processed_pdfs/deepseek_ocr/acl2025_00005/page_images/page_001.jpg"
}
```

`raw_ocr_responses.jsonl` 每行保留 OCR API 原始响应：

```json
{
  "paper_id": "acl2025_00005",
  "page": 1,
  "image_path": "processed_pdfs/deepseek_ocr/acl2025_00005/page_images/page_001.jpg",
  "status": "received",
  "parse_warning": "",
  "raw_content": "original model response before quality gate"
}
```

当前实现对 OCR 输出设置了质量闸门。如果 DeepSeek-OCR 返回的是 prompt echo、schema echo、占位文本或重复 instruction，而不是页面真实转录，则该页会被标记为失败，不会写入可用 chunks。典型失败记录会进入：

```json
{
  "page": 1,
  "error": "OCR response looked like a prompt/schema echo, not page transcription."
}
```

这条规则是为了保证后续 answer generation 不会基于伪 OCR 内容生成污染 prediction。

### 4. Context selection：`context_index.py`

当前 baseline 不再使用独立 embedding model。`context_index.py` 直接读取 OCR artifacts：

```text
processed_pdfs/deepseek_ocr/<paper_id>/chunks.jsonl
processed_pdfs/deepseek_ocr/<paper_id>/visual_contexts.jsonl
```

selection 逻辑如下：

1. 对 query 做轻量 tokenization；
2. 对 OCR chunk text 做 tokenization；
3. 根据 query token overlap 排序；
4. 对 table、figure_caption、equation 类型 chunk 加轻量 boost；
5. 返回 top-n text contexts；
6. 对 visual context caption 做同样的轻量 overlap 排序；
7. 返回 top-n visual contexts。

当前 selection method 固定记录为：

```text
ocr_chunk_lexical_without_embedding
```

该方法不是强 retrieval baseline，只是为了在没有独立 embedding 模型的情况下，把 OCR 产出的结构化上下文以可审计方式送入 answer model。

### 5. Prompt 构造：`ocr_context_prompt_builder.py`

`ocr_context_prompt_builder.py` 构造两段消息：

1. system prompt：声明模型是 OCR-context baseline，只能使用候选 metadata 和 selected OCR contexts；
2. user prompt：包含 query、answer_types、table_schema、candidate_papers、selected_text_contexts 和 selected_visual_contexts。

prompt 明确要求：

* 只输出合法 JSON；
* 不输出 markdown；
* 只能使用候选中的 `paper_id`；
* 使用 OCR context 时保留 paper_id、page、chunk_id 等 locator；
* 不发明 bbox、table_id、figure_id 或 equation_id；
* 当前 baseline 不使用原生 PDF input。

输出目标结构包括：

```json
{
  "query_id": "",
  "gold_papers": [{"paper_id": ""}],
  "evidence": [
    {
      "paper_id": "",
      "source_type": "text_span",
      "locator": {"page": 1, "chunk_id": "p001_c001"}
    }
  ],
  "answer": {},
  "confidence": {
    "paper_retrieval": 0.0,
    "context_grounding": 0.0,
    "answer": 0.0
  },
  "notes": {
    "baseline_type": "pdf_ocr_context_vlm",
    "ocr_model": "deepseek-ai/DeepSeek-OCR",
    "context_selection": "ocr_chunk_lexical_without_embedding",
    "limitations": ""
  }
}
```

正式提交文件中，`parser.py` 会移除内部字段，例如 `chunk_id` 和 `visual_id`，只保留 evaluator 需要的字段。

### 6. Answer model 调用：`answer_client.py`

`answer_client.py` 使用 OpenAI-compatible `/chat/completions` 接口调用 answer generation model。当前 `.env` 中首选：

```env
ANSWER_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

调用前会先进行 capability probe：

1. 用 `ANSWER_MODEL` 发送一个最小 chat generation 请求；
2. 如果返回有效 content，则使用 `ANSWER_MODEL`；


当前 smoke test 中：

```text
ANSWER_MODEL=Qwen/Qwen3-VL-8B-Instruct
can_generate=True
answer_model_source=configured_vlm_answer_model_generation_capable
```

因此当前逻辑不会 fallback 到 base generation model。

### 7. 主流程：`run_pdf_ocr_context_vlm_baseline.py`

`run_pdf_ocr_context_vlm_baseline.py` 是当前 PDF OCR Context VLM baseline 的主入口。

流程如下：

1. 读取参数；
2. 创建输出目录；
3. 读取 `validation_inputs.jsonl`；
4. 读取 `paper_metadata.jsonl`；
5. 构建 metadata records；
6. 对每个 query 做 metadata BM25 top-k 检索；
7. 检查候选论文的本地 PDF 是否存在；
8. 对可用 PDF 调用 DeepSeek-OCR；
9. 将 OCR artifacts 写入 `processed_pdfs/deepseek_ocr/<paper_id>/`；
10. 从 OCR chunks 和 visual captions 中选择 contexts；
11. 生成 prompt preview；
12. 如果是 dry run 或 skip-generation，则停止；
13. 如果没有 selected OCR contexts，则记录 error 并跳过 prediction；
14. 如果存在 selected OCR contexts，则调用 answer model；
15. 解析并归一化 answer model 输出；
16. 生成 prediction；
17. 记录 raw response、errors 和 run report。

当前 baseline 类型固定为：

```text
pdf_ocr_context_vlm
```

## 环境准备

当前工作环境为：

```bash
conda activate littraceqa
```

安装依赖：

```bash
pip install -r requirements.txt
```

当前方案通过 API 调用 OCR 与 answer generation，不需要本地 `torch` 或 `transformers`。本地仍需要：

* `PyMuPDF`：PDF 页面渲染；
* `Pillow`：页面图片压缩与 base64 编码；
* `requests` / `urllib`：API 与文件访问；
* `rank-bm25`：metadata 候选检索。

## `.env` 配置

在 workspace 根目录创建 `.env`：

```env
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

OCR_MODEL=deepseek-ai/DeepSeek-OCR
OCR_PROVIDER=siliconflow
OCR_API_KEY=
OCR_BASE_URL=

ANSWER_PROVIDER=siliconflow
ANSWER_API_KEY=
ANSWER_BASE_URL=
ANSWER_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

程序会检查 key 是否为占位符，并且不会输出完整 API key。

## Capability Probe

Capability probe 用于确认 metadata、PDF cache、OCR API 配置和 answer generation model 是否可用。

```bash
python -m pdf_ocr_vlm_baseline.pipeline_capability_probe \
  --official-dir official_dev \
  --pdf-output-dir raw_pdfs \
  --processed-output-dir processed_pdfs/deepseek_ocr \
  --index-output-dir indexes/ocr_chunk_lexical \
  --output-dir outputs/pdf_ocr_context_vlm_full_from_retrieval \
  --max-papers 1 \
  --env-path .env
```

本次 probe 结果：

```text
metadata_readable: true
pdf_available: true
ocr_available: true
answer_model: Qwen/Qwen3-VL-8B-Instruct
answer_generation_available: true
answer_supports_image_input: true
answer_model_source: configured_vlm_answer_model_generation_capable
```

## Full Run｜from retrieval

本次从 retrieval 开始完整跑 public validation inputs，限制每篇候选 PDF 只 OCR 第 1 页，避免在 OCR 输出协议尚未稳定前产生过高 API 成本。

```bash
OCR_MAX_PAGES_PER_PAPER=1 python -m pdf_ocr_vlm_baseline.run_pdf_ocr_context_vlm_baseline \
  --official-dir official_dev \
  --output-dir outputs/pdf_ocr_context_vlm_full_from_retrieval \
  --pdf-output-dir raw_pdfs \
  --processed-output-dir processed_pdfs/deepseek_ocr \
  --index-output-dir indexes/ocr_chunk_lexical \
  --top-k-papers 4 \
  --top-n-text-contexts 12 \
  --top-n-visual-contexts 4 \
  --pdf-sleep-seconds 0 \
  --pdf-timeout-seconds 30 \
  --pdf-max-retries 0 \
  --env-path .env
```

本次输出目录：

```text
outputs/pdf_ocr_context_vlm_full_from_retrieval/
```

主要输出文件：

```text
candidate_papers.jsonl         # 55 行，每个 query 的 metadata top-k 候选
pdf_availability.jsonl         # 55 行，每个 query 的候选 PDF cache / download 状态
ocr_artifacts.jsonl            # 68 行，所有进入 OCR 阶段的候选 PDF 状态
selected_contexts.jsonl        # 55 行，每个 query 的 selected OCR contexts
errors.jsonl                   # 55 行，context retrieval unavailable 等错误
run_report.md                  # 本次 run 摘要
```

由于没有任何可信 OCR chunks，本次未生成可评估的：

```text
predictions.jsonl
```

## 当前实验记录

### Full Run｜top-k-papers=4｜max-pages=1

```text
运行状态：Blocked by OCR output quality gate
run_id: 20260704T141417Z
baseline type: pdf_ocr_context_vlm
processed query count: 55
top_k_papers: 4
top_n_text_contexts: 12
top_n_visual_contexts: 4

PDF cache:
- existing PDFs: 61
- newly downloaded PDFs: 7
- failed PDF downloads: 152

OCR:
- OCR model: deepseek-ai/DeepSeek-OCR
- OCR provider: siliconflow
- OCR prompt version: deepseek_ocr_page_json_v2
- OCR success paper count: 0
- OCR failed paper count: 68
- OCR failure distribution:
  - ocr_failed: 66
  - ocr_unavailable: 2
  - prompt/schema echo quality-gate failures: 66

Context / generation:
- selected text contexts: 0
- selected visual contexts: 0
- answer generation model: Qwen/Qwen3-VL-8B-Instruct
- answer model source: configured_vlm_answer_model_generation_capable
- answer model supports image input: True
- attached answer images: 0
- successful answer API calls: 0
- fallback predictions: 0
- predictions generated: False
```

### PDF Download Failure Detail

本次 `failed_pdf_downloads=152` 来自候选论文缺失本地 PDF 后的 on-demand download 尝试。下载失败被记录在：

```text
outputs/pdf_ocr_context_vlm_full_from_retrieval/pdf_availability.jsonl
```

状态分布：

```text
existing: 61
downloaded: 7
download_failed: 152
```

错误分布：

```text
direct_url 200 not a pdf: 152
direct_pdf 403 Forbidden: 139
openreview 403 Forbidden: 123
openreview 429 Too Many Requests: 29
direct_pdf 429 Too Many Requests: 13
```

其中 `direct_url 200 not a pdf` 通常表示 OpenReview forum 或 challenge redirect 返回 HTML 页面，而不是 PDF 文件；`403` 和 `429` 主要来自 OpenReview PDF endpoint 的访问限制或限流。

### OCR Raw Response Detail

原始 OCR response 现在保存在：

```text
processed_pdfs/deepseek_ocr/<paper_id>/raw_ocr_responses.jsonl
```

示例：

```text
processed_pdfs/deepseek_ocr/acl2025_00005/raw_ocr_responses.jsonl
```

当前观察到的原始 response 不是页面转录，而是 instruction echo：

```text
raw_len: 17912
raw_preview: Do not use markdown. - Do not use any other markdown syntax. - Do not use any other formatting...
```

因此质量闸门将其判定为：

```text
OCR response looked like a prompt/schema echo, not page transcription.
```

## Local Evaluation

本次 run 没有生成 `predictions.jsonl`，因此不运行官方 evaluator。此前单条污染 prediction 的 evaluation 只用于验证 evaluator 调用链，不作为有效 baseline 分数。

## 当前方案预期局限

该方案当前的主要瓶颈是 OCR 输出结构质量，而不是 answer generation。

当前 capability probe 已确认：

* `Qwen/Qwen3-VL-8B-Instruct` 可以完成 chat generation；
* 本地 PDF cache 可被读取；
* PDF 页面可被渲染并发送给 OCR API。

但 DeepSeek-OCR 作为结构化 OCR 模块的输出不理想。当前通过 OpenAI-compatible `/chat/completions` 传入 page image 和结构化 JSON 要求时，模型返回内容多次表现为 prompt echo、schema echo 或重复 instruction，而不是页面真实转录。可能原因包括：

* `deepseek-ai/DeepSeek-OCR` 在当前 SiliconFlow 接口上并不按普通 vision chat completion 的方式提供稳定 OCR；
* 当前 `/chat/completions` request schema 可能不是该 OCR 模型期望的专用调用协议；
* 结构化 JSON prompt 可能诱导模型复读 schema 或 instruction，而不是执行页面识别；
* OCR 模型可能更适合专用 endpoint、专用模板、或非 chat completion 的 processor 输出；
* 对 bbox、figure、table、equation 等结构化字段的一次性要求超过当前接口可稳定返回的能力。

因此：

* `chunks.jsonl` 可能无法产生可信 text chunks；
* `visual_contexts.jsonl` 可能无法产生可信 figure/table locator；
* `raw_ocr_responses.jsonl` 中可见大量 instruction echo；
* context selection 可能为空；
* answer generation 会被跳过；
* full validation 会出现大量 missing predictions；
* 当前不能把 OCR smoke 的污染输出作为正式 baseline 分数。

基于本次结果，后续将暂停使用 OCR model 作为结构化信息抽取模型。下一阶段不再把 DeepSeek-OCR 视为可靠的 `chunks / figures / bbox / table / equation` 生成器，而应转向其他 PDF 处理路线，例如：

1. 使用专门 PDF parser / layout parser 先提取文本与版面；
2. 使用 VLM answer model 直接读取 selected page images，但不声称有 OCR bbox grounding；
3. 使用更简单的 page-level markdown/text OCR，再由单独 parser 做后处理；
4. 等待或确认 DeepSeek-OCR 的专用 endpoint / 专用调用协议后，再恢复结构化 OCR 实验。
