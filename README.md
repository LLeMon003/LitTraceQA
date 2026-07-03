# LitTraceQA Metadata-Only Baseline

当前 baseline 类型是 `metadata_only_title_abstract`。它只使用
`paper_metadata.jsonl` 中的 `title` 和 `abstract` 做候选论文检索和 LLM
回答，不访问 PDF、URL、DOI、arXiv 或 OpenReview。

发送给 LLM API 的 prompt 是英文；本地报告和说明可以使用中文。

## 环境

未来工作环境：

```bash
conda activate littraceqa
```

## `.env`

workspace 根目录需要 `.env`：

```env
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V4-Flash
```

程序会检查 key 是否为占位符，并且不会输出完整 API key。

## Dry Run

不调用 API，只生成候选和 prompt preview：

```bash
python -m littrace_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 8 \
  --max-queries 2 \
  --dry-run
```

## 2-Query API Smoke Test

确认 `.env` 中 key 已补齐后运行：

```bash
python -m littrace_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 8 \
  --max-queries 2
```

## Full Validation

```bash
python -m littrace_baseline.run_baseline \
  --official-dir official_dev \
  --output-dir outputs/api_baseline \
  --top-k 12
```

## Local Evaluation

```bash
python -m littrace_baseline.evaluate_local \
  --official-dir official_dev \
  --pred outputs/api_baseline/predictions.jsonl
```

## PDF-Access Probe

该 probe 与 metadata-only baseline 解耦，不修改 `predictions.jsonl`，不参与
official evaluator。

```bash
python -m littrace_baseline.pdf_access_probe \
  --official-dir official_dev \
  --output-dir outputs/pdf_access_probe \
  --max-papers 2 \
  --env-path .env
```

## 输出

- `outputs/api_baseline/predictions.jsonl`
- `outputs/api_baseline/raw_llm_responses.jsonl`
- `outputs/api_baseline/candidate_papers.jsonl`
- `outputs/api_baseline/prompt_previews.jsonl`
- `outputs/api_baseline/errors.jsonl`
- `outputs/api_baseline/run_report.md`
- `outputs/pdf_access_probe/pdf_access_probe_report.md`
- `outputs/pdf_access_probe/pdf_access_probe_raw_responses.jsonl`
- `outputs/pdf_access_probe/pdf_access_probe_errors.jsonl`

## 未来扩展

`littrace_baseline/link_utils.py` 只为未来 baseline 预留在线链接抽取能力。
即使后续确认 PDF 可以被代码下载、解析，或可以被 API 访问，也应作为独立
`pdf_access_baseline` 或 `fulltext_baseline` 实现，不应混入当前
metadata-only baseline。
