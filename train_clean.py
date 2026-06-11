"""
============================================================
项目二：基于 RAG 的 NLP 文献智能问答系统
============================================================
在 Kaggle Notebook 中按 Cell 顺序运行（T4 GPU, Internet On）
"""

# ============================================================
# Cell 1: 安装依赖
# ============================================================
!pip install -q chromadb sentence-transformers llama-index llama-index-vector-stores-chroma llama-index-embeddings-huggingface datasets gradio --upgrade

# ============================================================
# Cell 2: Semantic Scholar API 爬取论文
# ============================================================
import requests, json, time

def search_papers(keyword, limit=40):
    papers = []
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keyword,
        "limit": min(limit, 100),
        "fields": "title,abstract,year,authors,url,venue"
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        papers = resp.json().get("data", [])
    time.sleep(1.2)
    return papers

# 用短关键词，分批搜索
keywords = [
    "machine translation",
    "language model fine tuning",
    "retrieval augmented generation",
    "low resource machine translation",
    "text summarization abstractive",
    "multilingual NLP cross lingual",
    "RLHF language model alignment",
    "named entity recognition",
    "question answering NLP",
    "knowledge distillation NLP",
]

all_papers = []
for kw in keywords:
    papers = search_papers(kw, limit=40)
    all_papers.extend(papers)
    print(f"'{kw}': {len(papers)} 篇")

# 去重
seen = set()
unique_papers = []
for p in all_papers:
    pid = p.get("paperId", "")
    if pid and pid not in seen and p.get("abstract"):
        seen.add(pid)
        unique_papers.append(p)

print(f"\n去重后: {len(unique_papers)} 篇论文")

with open('/kaggle/working/nlp_papers.json', 'w', encoding='utf-8') as f:
    json.dump(unique_papers, f, ensure_ascii=False, indent=2)

print("已保存到 nlp_papers.json")

# ============================================================
# Cell 3: 构建向量索引（BGE + ChromaDB）
# ============================================================
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

with open('/kaggle/working/nlp_papers.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

print(f"加载 {len(papers)} 篇论文")

documents = []
for p in papers:
    text = (
        f"Title: {p['title']}\n"
        f"Abstract: {p.get('abstract', '')}\n"
        f"Year: {p.get('year', 'N/A')}\n"
        f"Venue: {p.get('venue', 'N/A')}"
    )
    doc = Document(
        text=text,
        metadata={
            "title": p['title'],
            "year": str(p.get('year', '')),
            "venue": p.get('venue', ''),
            "url": p.get('url', ''),
        }
    )
    documents.append(doc)

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = splitter.get_nodes_from_documents(documents)
print(f"{len(documents)} 篇论文 → {len(nodes)} 个 chunks")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh", device="cuda")

chroma_client = chromadb.PersistentClient(path="/kaggle/working/chroma_db")
chroma_collection = chroma_client.get_or_create_collection("nlp_papers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)

print(f"索引构建完成！向量维度: 512")
print("索引已持久化到 /kaggle/working/chroma_db")

# ============================================================
# Cell 4: 检索测试
# ============================================================
retriever = index.as_retriever(similarity_top_k=5)

test_queries = [
    "How to handle rare words in neural machine translation?",
    "How does RLHF improve LLM alignment?",
    "What are the main challenges in multilingual NLP?",
    "How to evaluate text summarization quality?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"查询: {query}")
    results = retriever.retrieve(query)
    for i, node in enumerate(results):
        print(f"  Top-{i+1} (score: {node.score:.4f})")
        print(f"  标题: {node.metadata.get('title', 'N/A')[:80]}")
        print(f"  年份: {node.metadata.get('year', 'N/A')}")

# ============================================================
# Cell 5: 加载 LLM + RAG 生成
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def ask_nlp_question(query):
    results = retriever.retrieve(query)

    context_parts = []
    for i, node in enumerate(results):
        title = node.metadata.get('title', 'N/A')
        year = node.metadata.get('year', 'N/A')
        context_parts.append(f"[{i+1}] {title} ({year})\n{node.text[:300]}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are an NLP research assistant. Answer the question based on the retrieved papers below.
If the papers don't provide enough information, say so honestly. Do not fabricate.

Retrieved papers:
{context}

Question: {query}

Answer:"""

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response, results

# 测试
test_q = "How does RLHF improve LLM alignment?"
answer, sources = ask_nlp_question(test_q)
print(f"Q: {test_q}")
print(f"A: {answer}")
print(f"\n引用论文:")
for i, node in enumerate(sources):
    print(f"  [{i+1}] {node.metadata.get('title', 'N/A')[:60]} ({node.metadata.get('year', '')})")

# ============================================================
# Cell 6: Gradio 演示
# ============================================================
import gradio as gr

def rag_query(query):
    if not query.strip():
        return "Please enter a question."
    answer, sources = ask_nlp_question(query)

    refs = "\n\n**References:**\n"
    for i, node in enumerate(sources):
        title = node.metadata.get('title', 'N/A')
        year = node.metadata.get('year', '')
        refs += f"\n[{i+1}] {title} ({year})"

    return answer + refs

demo = gr.Interface(
    fn=rag_query,
    inputs=gr.Textbox(lines=3, placeholder="Ask an NLP question...", label="Question"),
    outputs=gr.Textbox(lines=15, label="Answer with References"),
    title="NLP Literature Q&A (RAG)",
    description="Ask questions about NLP research. Powered by 100+ papers + BGE embeddings + ChromaDB + Reranker + Qwen2.5",
    examples=[
        ["How does RLHF improve LLM alignment?"],
        ["What are the main challenges in multilingual NLP?"],
        ["How to evaluate text summarization quality?"],
        ["What are challenges in low-resource machine translation?"],
    ],
)

demo.launch(share=True)
# 记下输出里的 .gradio.live 链接！

# ============================================================
# Cell 7: Chunk Size 消融实验
# ============================================================
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb, json

with open('/kaggle/working/nlp_papers.json', 'r', encoding='utf-8') as f:
    papers = json.load(f)

documents = []
for p in papers:
    text = f"Title: {p['title']}\nAbstract: {p.get('abstract', '')}"
    doc = Document(text=text, metadata={"title": p['title'], "year": str(p.get('year', ''))})
    documents.append(doc)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh", device="cuda")
chroma_client = chromadb.PersistentClient(path="/kaggle/working/chroma_db_exp")

results_summary = {}
test_query = "How to improve low-resource machine translation?"

for chunk_size in [256, 512, 1024]:
    print(f"\n--- Chunk Size = {chunk_size} ---")

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size // 8)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"  Chunks: {len(nodes)}")

    collection_name = f"nlp_cs_{chunk_size}"
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    idx = VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)

    ret = idx.as_retriever(similarity_top_k=5)
    results = ret.retrieve(test_query)

    scores = [r.score for r in results]
    titles = [r.metadata.get('title', '')[:50] for r in results]
    print(f"  Top-1: {scores[0]:.4f} | {titles[0]}")
    print(f"  Top-5 Avg: {sum(scores)/len(scores):.4f}")
    results_summary[chunk_size] = scores

print("\n" + "="*50)
print("Chunk Size 对比表:")
print("| Chunk Size | Chunks | Top-1 Score | Top-5 Avg |")
print("|---|---|---|---|")
for cs, scores in results_summary.items():
    print(f"| {cs} | - | {scores[0]:.4f} | {sum(scores)/len(scores):.4f} |")

# ============================================================
# Cell 8: Reranker 对比实验
# ============================================================
from llama_index.core.postprocessor import SentenceTransformerRerank

retriever = index.as_retriever(similarity_top_k=10)

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=3,
)

test_queries = [
    "How does RLHF improve LLM alignment?",
    "What are challenges in low-resource machine translation?",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"查询: {q}")

    results_no = retriever.retrieve(q)
    print("\n--- 无 Reranker (Top-3 of 10) ---")
    for i, node in enumerate(results_no[:3]):
        print(f"  [{i+1}] score={node.score:.4f} | {node.metadata.get('title','')[:60]}")

    results_rerank = reranker.postprocess_nodes(results_no, query_str=q)
    print("\n--- 有 Reranker (Top-3 of 10) ---")
    for i, node in enumerate(results_rerank[:3]):
        print(f"  [{i+1}] score={node.score:.4f} | {node.metadata.get('title','')[:60]}")
