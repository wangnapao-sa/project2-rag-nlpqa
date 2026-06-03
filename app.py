import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
import chromadb
import json

# Load index
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh", device="cuda")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("nlp_papers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

retriever = index.as_retriever(similarity_top_k=10)
reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)

# Load LLM
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def ask(query):
    if not query.strip():
        return "Please enter a question."

    results = retriever.retrieve(query)
    results = reranker.postprocess_nodes(results, query_str=query)

    context_parts = []
    for i, node in enumerate(results):
        title = node.metadata.get('title', 'N/A')
        year = node.metadata.get('year', 'N/A')
        context_parts.append(f"[{i+1}] {title} ({year})\n{node.text[:300]}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are an NLP research assistant. Answer the question based on the retrieved papers.
If papers don't provide enough information, say so honestly. Do not fabricate.

Retrieved papers:
{context}

Question: {query}

Answer:"""

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=True,
            temperature=0.3, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    refs = "\n\n**References:**\n"
    for i, node in enumerate(results):
        refs += f"\n[{i+1}] {node.metadata.get('title','')} ({node.metadata.get('year','')})"

    return response + refs

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(lines=3, placeholder="Ask an NLP question...", label="Question"),
    outputs=gr.Textbox(lines=15, label="Answer with References"),
    title="NLP Literature Q&A (RAG)",
    description="RAG system: 100+ papers + BGE embeddings + ChromaDB + Reranker + Qwen2.5",
    examples=[
        ["How does RLHF improve LLM alignment?"],
        ["What are challenges in low-resource machine translation?"],
        ["How to evaluate text summarization quality?"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
