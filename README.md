# Log 000｜LitTraceQA 实验日志索引

本仓库用于记录 LitTraceQA 任务上的 baseline 设计、代码实现与实验运行日志。当前阶段的核心目标不是直接追求最高分，而是先建立一个可复现、可扩展、边界清晰的实验框架：在无法稳定访问在线 PDF 或全文内容的前提下，先完成一个 **PDF-free / metadata-only baseline**，验证数据读取、候选论文检索、LLM 调用、预测格式生成和本地评估流程。

后续如果加入 PDF-native API input、raw PDF 缓存解析、full-text retrieval、table/figure/equation-aware reasoning 或 neuro-symbolic verification 等方案，应以新的实验日志继续追加，而不是直接混入当前 baseline。为了保留未来方案的结构余量，本文档使用 `# Log XXX｜...` 作为最高层级日志标题，具体实验内容使用二级及以下标题展开。

## 当前实验路线总览

| 日志编号    | 方案名称                                             |     当前状态 | 是否访问 PDF / URL | 主要作用                                         |
| ------- | ------------------------------------------------ | -------: | -------------: | -------------------------------------------- |
| Log 001 | PDF-Free Baseline：`metadata_only_title_abstract` |      已实现 |              否 | 只用 title + abstract 完成候选检索与 LLM 回答           |         |
| Log 002 | PDF-Native API Baseline                          |      未实现 |       计划访问 PDF | 未来路线：API 原生 PDF input                        |

## 仓库当前结构

```text
LitTraceQA/
├── littrace_baseline/
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

当前仓库中，`official_dev/` 是 LitTraceQA 官方开发集；`littrace_baseline/` 是本仓库新增的 baseline 代码；`outputs/` 和 `raw_pdfs/` 被 `.gitignore` 排除，用于本地实验输出和未来 PDF 缓存，不进入版本控制。

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
python -m littrace_baseline.run_baseline \
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
python -m littrace_baseline.run_baseline \
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
python -m littrace_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 12
```

当前默认 `top-k=12`。由于该 baseline 只使用 metadata，增大 top-k 可能提高 gold paper recall，但也可能增加 prompt 噪声。后续可记录不同 top-k 的对比实验。

## Resume 运行

如果中途 API 调用中断，可以使用：

```bash
python -m littrace_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 12 \
  --resume
```

`--resume` 会读取已有 `predictions.jsonl` 中的 `query_id`，跳过已完成样本。

## Local Evaluation

生成 `predictions.jsonl` 后，运行本地 evaluator：

```bash
python -m littrace_baseline.evaluate_local \
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
运行命令：python -m littrace_baseline.run_baseline --official-dir official_dev --output-dir outputs/api_baseline --top-k 12
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

