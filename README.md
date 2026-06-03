# RAG-based NLP Literature Q&A System

A Retrieval-Augmented Generation (RAG) system that answers NLP research questions by retrieving relevant papers from a corpus of 100+ papers and generating answers with citations.

**Live Demo**: [Kaggle Gradio](https://28ca182464f8b699e6.gradio.live) (GPU required)

## Architecture

```
User Question → BGE Embedding → ChromaDB Vector Search (top-10)
→ BGE Reranker (top-3) → Qwen2.5-0.5B LLM → Answer with Citations
```

## Knowledge Base

- 100+ papers from Semantic Scholar API
- Covers: machine translation, LLM alignment (RLHF), text summarization, multilingual NLP, cross-lingual transfer
- Time range: 2016-2026

## Experiments

### Chunk Size Impact on Retrieval

| Chunk Size | Chunks | Top-1 Score | Top-5 Avg |
|---|---|---|---|
| 256 | 159 | **0.5860** | **0.5775** |
| 512 | 100 | 0.5604 | 0.5550 |
| 1024 | 100 | 0.5604 | 0.5550 |

Chunk size 256 gives the highest retrieval precision. 512 is used in production as the best trade-off between precision and index size.

### Reranker Impact

| Query | Top-1 w/o Reranker | Top-1 w/ Reranker |
|---|---|---|
| RLHF alignment | InfAlign (0.53) | Accelerated Preference Optimization (0.97) |
| Low-resource MT | Low-Resource NLP (0.61) | Cross-Lingual Transfer (reranked) |

The reranker (BGE-reranker-base) effectively re-orders initial retrieval results, boosting the most relevant papers to the top.

## Key Findings

- RAG prevents hallucination — all answers are grounded in real retrieved papers
- Retrieval quality depends on corpus coverage (100 papers is the main bottleneck)
- Chunk size 256 maximizes precision but increases index size by 60%
- Reranker adds significant value when initial retrieval is noisy

## Usage

```bash
pip install -r requirements.txt
# Download index files from Kaggle output
python app.py
```

## Tech Stack

- **Embedding**: BAAI/bge-small-zh (512-dim)
- **Vector DB**: ChromaDB
- **Reranker**: BAAI/bge-reranker-base
- **LLM**: Qwen2.5-0.5B-Instruct
- **Framework**: LlamaIndex
- **Data**: Semantic Scholar API

## Files

- `train.ipynb` — Kaggle notebook (data crawling, indexing, RAG pipeline)
- `app.py` — Gradio demo
- `requirements.txt` — Python dependencies
