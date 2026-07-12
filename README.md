# Log 000｜LitTraceQA 实验日志索引

本仓库用于记录 LitTraceQA 任务上的 baseline 设计、代码实现与实验运行日志。当前阶段的核心目标不是直接追求最高分，而是先建立一个可复现、可扩展、边界清晰的实验框架：在无法稳定访问在线 PDF 或全文内容的前提下，先完成一个 **PDF-free / metadata-only baseline**，验证数据读取、候选论文检索、LLM 调用、预测格式生成和本地评估流程。

后续如果加入 PDF-native API input、raw PDF 缓存解析、full-text retrieval、table/figure/equation-aware reasoning 或 neuro-symbolic verification 等方案，应以新的实验日志继续追加，而不是直接混入当前 baseline。为了保留未来方案的结构余量，本文档使用 `# Log XXX｜...` 作为最高层级日志标题，具体实验内容使用二级及以下标题展开。

## 当前实验路线总览

| 日志编号    | 方案名称                                             |     当前状态 | 是否访问 PDF / URL | 主要作用                                         |
| ------- | ------------------------------------------------ | -------: | -------------: | -------------------------------------------- |
| Log 001 | PDF-Free Baseline：`metadata_only_title_abstract` |      已实现 |              否 | 只用 title + abstract 完成候选检索与 LLM 回答           |
| Log 002 | PDF OCR Context VLM Baseline：`pdf_ocr_context_vlm` |   Smoke 已接通 |       读取本地 PDF | 用 DeepSeek-OCR 生成结构化 OCR contexts，再交给 VLM 回答 |
| Log 003 | PDF VLM Symbolic VLM Baseline：`pdf_vlm_symbolic_vlm` | v5 hybrid page routing ready | 读取本地 PDF | task-family budget + multi-paper hybrid span page routing + VLM-1 symbolic records + VLM-2 回答 |

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

# Log 003｜PDF VLM Symbolic VLM Baseline：pdf_vlm_symbolic_vlm

## 当前代码版本快照｜v5 hybrid page routing

当前 `pdf_vlm_symbolic_vlm_baseline` 已从固定 `top_k=5 / top_p=25` 策略，更新为按官方 `task_family` 分配 retrieval 和 page-routing budget，并在 multi-paper page ranking 中加入 hybrid text-span page score。该 score 只影响送入 VLM-1 前的 page selection，不进入 VLM prompt，也不作为官方 evidence 提交。
当前逻辑的数据结构链同步整理在 `Expected_examples/task_family_single_query_data_flow.md`。

当前默认配置：

```env
TASK_FAMILY_BUDGET_ENABLED=true
SINGLE_PAPER_TOP_K_PAPERS=5
SINGLE_PAPER_PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE=5
MULTI_PAPER_TOP_K_PAPERS=12
MULTI_PAPER_PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE=3
PAGE_ROUTING_TASK_FAMILY_STRATEGY=true
PAGE_ROUTING_SINGLE_STRATEGY=top1_candidate_quota
PAGE_ROUTING_MULTI_STRATEGY=global_ranked_pages
PAGE_RANKING_STRUCTURAL_EVIDENCE_WEIGHT=0
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_ENABLED=true
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_ALPHA=0.75
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_GAMMA=4
PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_CHUNK_MAX_CHARS=700
SYMBOLIC_ARTIFACT_VERSION=v5_eval_grounded_minimal_symbolic
VLM2_CONTEXT_MODE=text_only
VLM2_INCLUDE_PARSE_CONFIDENCE=false
```

预算语义：

```text
hidden_source_single_paper / other non-multi task_family
→ effective_top_k_papers = 5
→ effective_top_p_pages = 5 × actual_candidate_count = 25

multi_paper
→ effective_top_k_papers = 12
→ effective_top_p_pages = 3 × actual_candidate_count = 36
```

这里的 `top_p` 是 query-level global page ranking 后的页面预算，不是每篇 paper 固定页数。所有候选 PDF 的 native text pages 会进入同一个 query-level page pool，由 `global_native_text_bm25_rules` 排序后选择前 `top_p` 页给 VLM-1。

multi-paper 默认启用 hybrid span score：

```text
chunk_score = alpha * normalized_BM25(query, chunk)
            + (1 - alpha) * local_TF-IDF_cosine(query, chunk)

page_span_score = log_mean_exp(chunk_score, gamma)
final_page_score = page_span_score + normalized_current_policy_page_score
```

其中 `PAGE_RANKING_MULTI_TEXT_SPAN_HYBRID_CHUNK_MAX_CHARS=700` 控制 page native text 切成 paragraph/text chunks 时的最大字符长度。该策略不增加 VLM 调用次数，不增加 page budget，只改变 multi-paper page ranking 的排序。

当前 run report 会额外记录：

```text
task_family_budget_enabled
single_paper_budget
multi_paper_budget
task_family_budget_usage
effective_top_k_distribution
effective_top_p_distribution
effective_page_routing_top_pages_per_candidate_distribution
page_ranking_multi_text_span_hybrid_enabled
page_ranking_multi_text_span_hybrid_alpha
page_ranking_multi_text_span_hybrid_gamma
```

当前 VLM 边界仍保持不变：

* VLM-1 只读取 selected rendered page images；
* VLM-1 输出 evaluator-grounded minimal symbolic records；
* `record_id`、`source_type`、`reading_order`、`locator` 由系统层生成；
* VLM-2 不读取 full page image；
* VLM-2 只接收 selected symbolic evidence；
* retrieval score、page ranking score、selector score、bbox、parser confidence 不送入 VLM-1 或 VLM-2；
* native PDF text 只用于 page routing，不作为最终 answer evidence 直接提交。

当前推荐全量 v5 命令：

```bash
conda activate littraceqa

/usr/bin/time -p python -m pdf_vlm_symbolic_vlm_baseline.run_pdf_vlm_symbolic_vlm_baseline \
  --official-dir official_dev \
  --output-dir outputs/pdf_vlm_symbolic_vlm_v5_task_family_budget_full \
  --pdf-output-dir raw_pdfs \
  --processed-output-dir processed_pdfs/vlm_symbolic \
  --top-k-papers 5 \
  --top-n-records 24 \
  --top-n-visual-records 6 \
  --page-routing-enabled \
  --page-routing-parse-batch-size 16 \
  --show-progress \
  --max-parser-json-failures 6 \
  --env-path .env
```

注意：`--top-k-papers 5` 在 `TASK_FAMILY_BUDGET_ENABLED=true` 时只是 fallback。实际 top-k 由 `task_family` 决定。

## 实验目标

本阶段目标是在放弃 OCR 方案后，建立一个 **PDF page image -> symbolic layer -> answer VLM** 的双层 VLM baseline。该方案不让 VLM-2 读取完整 PDF page image，而是先由 VLM-1 将 selected rendered PDF pages 转为可审计 symbolic records，再由 selector 选取 compact symbolic evidence 送入 VLM-2。

当前方案的核心路线为：

```text
task-family budgeted metadata hybrid retrieval
→ ensure local PDFs
→ native PyMuPDF text scan for query-level global page ranking
→ multi-paper hybrid span scoring over native text chunks
→ select top-p pages where top_p = effective_pages_per_candidate * effective_top_k
→ render selected PDF pages
→ VLM-1 parse selected page images into symbolic records
→ symbolic validator / page-level cache / context selector
→ VLM-2 answer from selected symbolic evidence only
→ parser normalization
→ official evaluator
```

该 baseline 的实验意义是验证：

* 是否可以用 native text routing 避免全 PDF page image VLM parsing；
* 是否可以把 VLM-1 的 page image understanding 结果沉淀为可复用 symbolic cache；
* 是否可以阻止 retrieval score、selector score、parser confidence、bbox 等非官方评估字段进入 VLM-2；
* 是否可以用 system-derived locator 将 symbolic records grounding 到官方 evaluator 需要的 coarse evidence locator。

## 当前 baseline 边界

本实验严格遵守以下边界：

* 不使用 OCR；
* 不让 VLM-2 读取 full page image；
* VLM-2 只接收 selected symbolic evidence；
* retrieval score、page ranking score、selector score、parser confidence 不送入任何 VLM；
* bbox 不作为官方 evidence grounding 目标；
* native PDF text 只用于 page routing，不作为最终 answer evidence 直接提交；
* OpenReview PDF 采用 proceedings-first 策略，默认跳过 direct OpenReview 访问；
* partial / failed parser artifacts 必须显式落盘，不伪装成 complete cache。

## 关键代码逻辑

### 1. Metadata retrieval

当前使用 `hybrid_alias` retrieval，融合：

```text
title BM25
abstract BM25
full metadata BM25
alias / method-name matching
venue / year hint
title exact / substring boost
```

默认关闭 task-specific topic expansion：

```env
RETRIEVAL_ENABLE_TOPIC_EXPANSION=false
```

本次 run 的 retrieval 记录：

```text
retrieval method: hybrid_alias
top_k_papers: 5
retrieval_enable_topic_expansion: False
```

### 2. PDF 获取策略

当前不在 metadata 阶段直接跳过 OpenReview papers，而是优先尝试官方 proceedings mirror：

```text
ICLR   → proceedings.iclr.cc
ICML   → proceedings.mlr.press
NeurIPS → papers.nips.cc
```

本次 run 使用：

```text
PDF_OPENREVIEW_POLICY=proceedings_first_skip_direct_openreview
```

PDF source distribution：

```text
cache: 23
proceedings.icml_pmlr: 4
proceedings.neurips: 2
proceedings.iclr: 1
```

direct OpenReview：

```text
direct_openreview_skipped_count: 12
direct_openreview_attempted_count: 0
```

### 3. Query-level global page routing

当前 page routing 不再是 per-paper top pages，而是 query-level global top-p pages。

当前 v5 task-family budget 版本的配置语义：

```text
top_p = effective_page_routing_top_pages_per_candidate * actual_candidate_count

single_paper: 5 * 5 = 25
multi_paper: 3 * 12 = 36
```

当 `TASK_FAMILY_BUDGET_ENABLED=false` 时，系统回退到通用 `PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE * actual_candidate_count`。

本次 run：

```text
PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE=5
top_k_papers=5
global_top_p_pages_initial=25
```

也就是说，每个 query 理论上初始选取 25 个全局 ranked pages，而不是每篇固定解析 5 页。`PAGE_ROUTING_PARSE_BATCH_SIZE=16` 只控制调度 batch，不作为 top-p 语义截断。

本次统计：

```text
total_candidate_pages_before_routing: 604
top_p_pages_selected_total: 175
global_parse_batch_size: 16
global_parse_batches: 14
vlm1_pages_selected_after_global_routing: 175
average_global_selected_pages_per_query: 25.0
```

### 4. VLM-1 symbolic parser

当前 VLM-1 是 document-to-symbol converter。它读取 selected page image，输出 minimal symbolic records。

在本次实验后，VLM-1 输入/输出已进一步收缩到 evaluator-grounded minimal schema：VLM-1 不再被要求生成 bbox、confidence、record_id、source_type、reading_order 或 locator。VLM-1 只需输出：

```json
{
  "kind": "text_span | table | figure | equation_algorithm | citation_context | header_footer | unknown",
  "text": "...",
  "label": "Table 4 | Figure 2 | Equation 3 | null"
}
```

系统层负责：

* 分配 `record_id`；
* 归一化 `source_type`；
* 按数组顺序分配 `reading_order`；
* 从 `label/text` echo 出 official evaluator 需要的 locator，例如：
  * `Table 4` → `{"page": 6, "table_id": "Table 4"}`;
  * `Figure 2` → `{"page": 3, "figure_id": "Figure 2"}`;
  * `Equation (3)` → `{"page": 6, "equation_id": "Equation 3"}`;
  * `[24]` → `{"page": 12, "citation_id": 24}`。

### 5. VLM-2 answer prompt

VLM-2 只接收 answer-facing fields：

```json
{
  "paper_id": "...",
  "page": 6,
  "source_type": "table",
  "locator": {"page": 6, "table_id": "Table 4"},
  "text": "..."
}
```

不送入 VLM-2：

```text
global_record_id
record_type
record_id
label
image_path
score
bbox_1000
vlm_parse_confidence
```

本次 run report 记录：

```text
vlm2_prompt_context_fields:
['locator', 'page', 'paper_id', 'source_type', 'text']

fields_removed_from_vlm2_prompt:
['global_record_id', 'record_type', 'record_id', 'label', 'image_path', 'score', 'bbox_1000', 'vlm_parse_confidence']

vlm2_context_mode: text_only
vlm2_attached_image_count: 0
```

## 环境准备

当前工作环境：

```bash
conda activate littraceqa
```

核心 `.env` 配置应包括：

```env
PARSER_MODEL=Qwen/Qwen3-VL-8B-Instruct
ANSWER_MODEL=Qwen/Qwen3-VL-8B-Instruct

PARSER_EXTRACTION_MODE=text_first_symbolic_transcription
PARSER_MAX_TOKENS=6144
PARSER_MAX_RECORDS_PER_CALL=16
PARSER_MAX_CONTINUATIONS_PER_PAGE=4

RETRIEVAL_METHOD=hybrid_alias
RETRIEVAL_ENABLE_TOPIC_EXPANSION=false
PDF_OPENREVIEW_POLICY=proceedings_first_skip_direct_openreview

PAGE_ROUTING_ENABLED=true
PAGE_ROUTING_SOURCE=native_text
PAGE_ROUTING_METHOD=global_native_text_bm25_rules
PAGE_ROUTING_TOP_PAGES_PER_CANDIDATE=5
PAGE_ROUTING_TOP_PAGES_GLOBAL=
PAGE_ROUTING_PARSE_BATCH_SIZE=16
PAGE_ROUTING_ENABLE_PROGRESSIVE_EXPANSION=true

VLM2_CONTEXT_MODE=text_only
VLM2_INCLUDE_PARSE_CONFIDENCE=false
```

注意：本次输出目录名使用 `v5_eval_grounded_full`，但 run report 中实际读取到：

```text
symbolic artifact version: v4_global_query_page_routing
```

这说明运行时 `.env` 仍未切换到：

```env
SYMBOLIC_ARTIFACT_VERSION=v5_eval_grounded_minimal_symbolic
```

因此本次应记录为 **v5 code path / v4 artifact version mismatch** 的 partial failed run，不能视为干净的 v5 cache-compatible 实验。

## Full Run｜v5 evaluator-grounded minimal symbolic

运行命令：

```bash
conda activate littraceqa

/usr/bin/time -p python -m pdf_vlm_symbolic_vlm_baseline.run_pdf_vlm_symbolic_vlm_baseline \
  --official-dir official_dev \
  --output-dir outputs/pdf_vlm_symbolic_vlm_v5_eval_grounded_full \
  --pdf-output-dir raw_pdfs \
  --processed-output-dir processed_pdfs/vlm_symbolic \
  --top-k-papers 5 \
  --top-n-records 24 \
  --top-n-visual-records 6 \
  --page-routing-enabled \
  --page-routing-parse-batch-size 16 \
  --show-progress \
  --max-parser-json-failures 3 \
  --env-path .env
```

输出目录：

```text
outputs/pdf_vlm_symbolic_vlm_v5_eval_grounded_full/
```

主要输出文件：

```text
candidate_papers.jsonl
pdf_availability.jsonl
native_page_text.jsonl
global_page_pool.jsonl
global_page_ranking.jsonl
global_page_parse_plan.jsonl
page_rendering_artifacts.jsonl
parser_artifacts.jsonl
raw_vlm_parser_responses.jsonl
symbolic_records.runtime.jsonl
symbolic_records.debug.jsonl
selected_symbolic_contexts.prompt.jsonl
selected_symbolic_contexts.debug.jsonl
raw_vlm_answer_responses.jsonl
internal_predictions.jsonl
predictions.jsonl
errors.jsonl
run_report.md
```

## 当前实验记录

### Full Run｜top-k-papers=5｜global top-p=25｜parse-batch=16

```text
运行状态：failed
interrupted reason: Parser JSON/page failure threshold reached: 3 >= 3
processed query count: 6
candidate_papers rows: 7
predictions rows: 6
errors rows: 7

baseline type: pdf_vlm_symbolic_vlm
parser model: Qwen/Qwen3-VL-8B-Instruct
answer model: Qwen/Qwen3-VL-8B-Instruct
parser extraction mode: text_first_symbolic_transcription
symbolic artifact version observed by run: v4_global_query_page_routing
structured cache policy: reuse_complete_only

retrieval:
- method: hybrid_alias
- topic expansion: false
- top_k_papers: 5

page routing:
- source: native_text
- method: global_native_text_bm25_rules
- page_routing_top_pages_per_candidate: 5
- global_top_p_pages_initial: 25
- top_p_pages_selected_total: 175
- global_parse_batch_size: 16
- global_parse_batches: 14
- total_candidate_pages_before_routing: 604
- native_text_scanned_papers: 30
- native_text_scanned_pages: 604

PDF:
- existing PDFs: 23
- newly downloaded PDFs: 7
- failed PDF downloads: 5
- proceedings candidate attempts: 8
- proceedings match success count: 7
- direct OpenReview skipped count: 12
- direct OpenReview attempted count: 0

VLM-1:
- parser API calls: 118
- total parser calls: 118
- rendered papers: 50
- rendered pages: 171
- page_level_cache_hits: 62
- page_level_cache_misses: 106
- complete pages: 161
- partial pages: 3
- failed pages: 2
- pages needing continuation: 5
- parser JSON/page failures: 3
- records accepted total: 2008
- records deduplicated total: 71

VLM-2:
- context mode: text_only
- attached image count: 0
- answer API calls: 6
- successful predictions: 4
- parse failures: 2
- fallback predictions: 2
```

### Failure Detail

本次 run 在处理到第 7 个 query 的过程中触发：

```text
Parser JSON/page failure threshold reached: 3 >= 3
```

错误文件：

```text
outputs/pdf_vlm_symbolic_vlm_v5_eval_grounded_full/errors.jsonl
```

主要失败类型是 VLM-1 页面输出不可解析 JSON。结合前一轮运行观察，常见原因包括：

* VLM-1 completion 触顶，`finish_reason=length`；
* 输出 JSON 没有完整闭合；
* 大表格或复杂页面导致单页 structured transcription 过长；
* 当前仍是串行 page parser，失败前已经消耗较多 API 调用。

当前 baseline 选择在 parser JSON/page failures 达到阈值后停止，而不是伪造 page artifacts。

### Local Evaluation

评估命令：

```bash
python -m pdf_vlm_symbolic_vlm_baseline.evaluate_local \
  --official-dir official_dev \
  --pred outputs/pdf_vlm_symbolic_vlm_v5_eval_grounded_full/predictions.jsonl
```

本次只生成 6 条 prediction，官方 validation 共 55 条，因此 evaluator 报告大量 missing predictions。该结果只作为 partial failed run 的诊断参考，不能作为完整 baseline 分数。

评估结果：

```json
{
  "paper_precision_macro": 0.09090909090909091,
  "paper_recall_macro": 0.09090909090909091,
  "paper_f1_macro": 0.09090909090909091,
  "evidence_precision_macro": 0.006060606060606061,
  "evidence_recall_macro": 0.01818181818181818,
  "evidence_f1_macro": 0.00909090909090909,
  "multiple_choice_accuracy": 0.0,
  "freeform_exact_match": 0.0,
  "table_row_f1_macro": 0.0,
  "table_cell_accuracy_macro": 0.0,
  "table_cell_accuracy_micro": 0.0,
  "missing_prediction_count": 49,
  "extra_prediction_count": 0
}
```

## 当前方案预期局限

本次实验暴露的主要瓶颈是 VLM-1 parser 的稳定性和效率，而不是 retrieval 或 VLM-2。

当前限制：

* 每个 query 约 25 个 top-p pages，VLM-1 串行解析成本高；
* 大表格、reference、appendix 或复杂双栏页面容易导致 VLM-1 输出超长；
* 即使取消 bbox/confidence 等无用字段，复杂页面仍可能超过 JSON 稳定输出能力；
* 当前 `PARSER_MAX_RECORDS_PER_CALL=16` 对复杂页面可能偏大；
* `PARSER_MAX_TOKENS=6144` 不是根本解法，继续放大会增加成本和不稳定性；
* 本次 `.env` 的 artifact version 未切到 v5，存在 cache compatibility 记录不干净的问题；
* 当前 run 因 `max-parser-json-failures=3` 提前停止，没有完成 55 条全量验证。

## 后续优化方向

下一阶段应优先做不改变实验语义的工程优化：

1. 将 `.env` 修正为：

```env
SYMBOLIC_ARTIFACT_VERSION=v5_eval_grounded_minimal_symbolic
```

2. 针对 VLM-1 parser 增加 length failure 的小批量 retry：

```text
PARSER_RETRY_SMALLER_BATCH_ON_LENGTH=true
PARSER_SMALL_BATCH_RECORDS_PER_CALL=8
```

3. 引入 page-level parser 并发：

```text
PARSER_CONCURRENCY=6
```

4. 压缩 VLM-1 图像输入：

```text
PARSER_IMAGE_MAX_SIDE=1536
PARSER_IMAGE_QUALITY=82
PARSER_IMAGE_DETAIL=auto
```

5. 保留当前 evaluator-grounded minimal symbolic schema，不恢复 bbox/confidence 作为 VLM 输出目标。

6. 在 1-query 和 5-query smoke test 中先验证：

```text
parser JSON/page failure count
average parser calls per page
records accepted total
selected evidence locator quality
official evaluator partial score
```

只有当 v5 artifact version、parser retry 和进度/并发机制稳定后，再启动新的 full validation run。

# Log 003｜pdf_vlm_symbolic_vlm_baseline v5, VLM-2 Rerun, Metadata-Only V2

## 当前目标

本阶段围绕 `pdf_vlm_symbolic_vlm_baseline` 做了三类实验：

1. 完成 v5 symbolic pipeline 的全量运行与评估。
2. 只重跑 VLM-2 answer stage，修复 answer contract、freeform、multiple-choice、table schema 等输出问题。
3. 新增 metadata-only v2：`top-k metadata candidates -> VLM-2 paper selection -> prediction.gold_papers`，其中 evidence 和 answer 仍为空。

当前 pipeline 的完整数据流已整理在：

```text
Expected_examples/task_family_single_query_data_flow.md
Expected_examples/task_family_multi_paper_data_flow.md
```

## 当前主流程

```text
validation_inputs.jsonl
+ sanitized options/schema from validation.jsonl
+ paper_metadata.jsonl
-> hybrid metadata retrieval
-> optional multi-paper query decomposition
-> optional topic-profile expansion only when explicitly enabled
-> PDF cache / proceedings-first source resolution
-> native-text global page routing
-> multi-paper hybrid span page scoring when enabled
-> selected rendered page images
-> VLM-1 minimal symbolic parsing
-> processed_pdfs durable symbolic store
-> symbolic context selector
-> VLM-2 answer generation from selected symbolic evidence
-> parser normalization
-> official predictions.jsonl
```

VLM-2 不接收 full page image、native PDF、URL、local file path、retrieval score、page score、selector score、bbox、parser confidence 或 internal record id。VLM-2 只接收 selected symbolic evidence 的 answer-facing projection。

当前 baseline 从 `pdf_vlm_symbolic_vlm_opt_baseline` 移植的有效优化为：

```text
multi-paper page ranking:
native-text chunks -> BM25 + local TF-IDF cosine hybrid span score
hybrid span score + normalized current page policy score -> global page rank
```

已明确删除/不保留的 opt 实验项包括 evidence-block / neighbor page-only selector、paper-prior alpha、page-score beta、adaptive paper prior 和 multi-paper per-paper cap。

## 关键数据结构

### candidate_papers.jsonl

记录 metadata retrieval 的候选论文与审计分数：

```json
{
  "query_id": "q_001",
  "task_family": "hidden_source_single_paper",
  "task_family_bucket": "single_paper",
  "effective_top_k_papers": 5,
  "effective_top_p_pages": 25,
  "topic_expansion": null,
  "candidates": [
    {
      "rank": 1,
      "paper_id": "acl2025_00005",
      "title": "...",
      "abstract": "...",
      "retrieval_method": "hybrid_alias",
      "retrieval_score_components": {
        "title_bm25": 13.83,
        "abstract_bm25": 67.07,
        "full_bm25": 72.72,
        "method_substring_boost": 90.0
      }
    }
  ]
}
```

这些 score 只用于审计，不送入任何 VLM。

### symbolic_records.runtime.jsonl

记录 VLM-1 后的 evaluator-grounded minimal symbolic records：

```json
{
  "paper_id": "acl2025_00005",
  "page": 6,
  "record_id": "p006_r0003",
  "record_type": "table",
  "source_type": "table",
  "label": "Table 4",
  "text": "Dataset Length Eval. Metrics ... Absolute Δ 18.99 18.45 14.70 ...",
  "locator": {"page": 6, "table_id": "Table 4"},
  "page_status": "complete",
  "figure_crop_path": null
}
```

### processed_pdfs durable symbolic store

当前结构化数据统一存入：

```text
processed_pdfs/vlm_symbolic_runs/<run_or_cache_name>/<parser_model_slug>/<paper_id>/
  artifact_status.json
  symbolic_records.runtime.jsonl
  symbolic_records.debug.jsonl
  symbolic_index.json
  page_records/
  page_status/
  page_images/
  page_XXX/figure_crops/
```

`processed_pdfs` 的职责是保存可复用、尽可能完整的 processed symbolic information。`outputs/` 主要保存 run-level audit、prompt、raw response 和 predictions。

### selected_symbolic_contexts.prompt.jsonl

这是 VLM-2 真正看到的 symbolic evidence：

```json
{
  "query_id": "q_001",
  "selected_evidence": [
    {
      "paper_id": "acl2025_00005",
      "page": 6,
      "source_type": "table",
      "label": "Table 4",
      "grounding_label": {"type": "table_id", "value": "Table 4"},
      "text": "Dataset Length Eval. Metrics ..."
    }
  ],
  "has_partial_artifacts": false,
  "attached_image_refs": []
}
```

prompt projection 会移除 retrieval score、selector score、bbox、parser confidence、record id、local path 和 full page image。

### predictions.jsonl

官方 prediction：

```json
{
  "query_id": "q_001",
  "gold_papers": [{"paper_id": "acl2025_00005"}],
  "evidence": [
    {
      "paper_id": "acl2025_00005",
      "source_type": "table",
      "locator": {"page": 6, "table_id": "Table 4"}
    }
  ],
  "answer": {
    "freeform": {"text": "14.70"},
    "multiple_choice": {"gold": "C"}
  }
}
```

## Answer Contract 修复

最初 VLM-2 的 `multiple_choice` 和 `table` 输出很差，主要原因是 options/schema 不在 `validation_inputs.jsonl`，而在 `validation.jsonl` 的 answer 容器中。

当前修复策略：

```text
validation_inputs.jsonl:
  query_id, task_family, primary_evidence_type, question, answer_types

validation.jsonl:
  only read answer.multiple_choice.options
  only read answer.table.schema
  never read answer.*.gold, gold_papers, evidence
```

同时修复了：

```text
bare string freeform -> {"text": "..."}
missing freeform + valid MC key -> fill freeform from option text
dynamic required_answer_fields
dynamic required_answer_shape
```

## VLM-2 Rerun 结果

VLM-2-only rerun 使用已有 retrieval、page routing、VLM-1 symbolic cache 和 selected symbolic contexts，只重跑 answer generation。

### 8B Instruct rerun

在修复 answer contract、freeform/table 输出后，8B rerun 曾得到：

```json
{
  "paper_precision_macro": 0.8045454545454546,
  "paper_recall_macro": 0.5545454545454546,
  "paper_f1_macro": 0.6074025974025974,
  "evidence_f1_macro": 0.33146005509641874,
  "multiple_choice_accuracy": 0.4878048780487805,
  "freeform_exact_match": 0.15384615384615385,
  "table_row_f1_macro": 0.31233766233766236,
  "table_cell_accuracy_micro": 0.2222222222222222
}
```

### 32B Instruct rerun

当前 `.env` 使用：

```env
ANSWER_MODEL=Qwen/Qwen3-VL-32B-Instruct
ANSWER_MAX_TOKENS=4096
ANSWER_TEMPERATURE=0
```

32B VLM-2-only rerun：

```json
{
  "paper_precision_macro": 0.7881818181818182,
  "paper_recall_macro": 0.55,
  "paper_f1_macro": 0.5978787878787879,
  "evidence_precision_macro": 0.3310822510822511,
  "evidence_recall_macro": 0.33181818181818185,
  "evidence_f1_macro": 0.2970431588613407,
  "multiple_choice_accuracy": 0.5609756097560976,
  "freeform_exact_match": 0.038461538461538464,
  "table_row_f1_macro": 0.4065656565656566,
  "table_cell_accuracy_macro": 0.13383838383838384,
  "table_cell_accuracy_micro": 0.18518518518518517
}
```

32B 的输出结构完整性更好：

```text
free_empty: 0
mc_empty: 0
table_empty: 0
fallback_predictions: 0
```

但 freeform exact 和 evidence F1 不一定优于 8B。这说明模型规模不是当前唯一瓶颈，selected symbolic context 的质量影响很大。

## Metadata-Only V2

metadata-only v2 不是直接提交 retrieval top-k。它的流程是：

```text
top-k metadata candidates
-> VLM-2 sees title/abstract/authors/venue/year only
-> VLM-2 selects prediction.gold_papers
-> evidence = []
-> answer fields = empty values
```

输出目录：

```text
outputs/pdf_vlm_symbolic_vlm_baseline_metadata_only_v2/
  candidate_papers.jsonl
  metadata_selection_prompts.jsonl
  raw_vlm_metadata_selection.jsonl
  predictions.jsonl
  metadata_only_report.md
```

官方评估：

```json
{
  "paper_precision_macro": 0.7445454545454546,
  "paper_recall_macro": 0.5636363636363636,
  "paper_f1_macro": 0.6009090909090908,
  "evidence_precision_macro": 0.0,
  "evidence_recall_macro": 0.0,
  "evidence_f1_macro": 0.0,
  "multiple_choice_accuracy": 0.0,
  "freeform_exact_match": 0.0,
  "table_row_f1_macro": 0.0,
  "table_cell_accuracy_micro": 0.0
}
```

该结果说明 metadata-only paper selection 可作为 retrieval/paper-selection 诊断基线，但不能反映 evidence 或 answer 能力。

## Retrieval 结论

当前 generic `hybrid_alias` 是无 topic hint 的通用检索器。`pdf_vlm_symbolic_vlm_baseline` 内置了一个 opt-in 的 topic-profile expansion，用于在所有官方 metadata 上做显式 topic 标注和打分；它不依赖其他 baseline 的代码。

当前实现中：

```env
RETRIEVAL_ENABLE_TOPIC_EXPANSION=false
```

默认关闭 topic profile。显式开启时可作为上限/消融实验：

```bash
RETRIEVAL_ENABLE_TOPIC_EXPANSION=true ...
```

Topic profile 属于 task/dev-set-oriented retrieval hint，不应混入默认 generic baseline。

## 当前核心瓶颈

当前最重要的观察是：**纯 symbolic 层对上下文质量影响过大。**

现有结构中，VLM-2 主要依赖 selected symbolic records 作为上下文。如果上游任一环节失真，VLM-2 就无法恢复信息：

```text
metadata retrieval miss -> wrong papers enter downstream
page routing miss -> VLM-1 never sees key page
VLM-1 transcription loss -> symbolic record 缺失关键值
symbolic validation/normalization -> 信息可能被简化或重排
symbolic context selector miss -> VLM-2 看不到正确 record
VLM-2 over symbolic text -> 对复杂表格/figure/citation 的语义恢复能力受限
```

换句话说，symbolic layer 带来了审计性和可控性，但当它成为 VLM-2 的主要上下文来源时，会放大 page routing、transcription 和 selection 的误差。

## 下一阶段 Baseline 方向

下一轮 baseline 应研究：

```text
VLM-1 + symbolic layer = hints / anchors / provenance
not the primary context source
```

也就是说，不再让 VLM-2 只依赖 selected symbolic text。新的设计应考虑：

```text
metadata candidates
+ routed pages or cropped evidence context
+ VLM-1 symbolic records as structured hints
+ page/source/label/table_id/figure_id/equation_id anchors
-> VLM-2 answer generation
```

预期目标：

1. 保留 symbolic layer 的 auditability。
2. 降低 VLM-1 transcription loss 对答案的硬性影响。
3. 让 symbolic records 主要承担定位、类型、标签和 provenance 提示。
4. 允许 VLM-2 利用更接近原始证据的上下文完成最终推理。
5. 继续禁止将 retrieval score、selector score、bbox、parser confidence、gold answer 或 gold evidence 送入 VLM。
