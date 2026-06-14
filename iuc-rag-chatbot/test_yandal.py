# test_yandal.py (ana dizinde)
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pickle, torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from config import VECTORDB_DIR
import rag_engine

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
vectorstore = FAISS.load_local(VECTORDB_DIR, embedding_model, allow_dangerous_deserialization=True)
with open(os.path.join(VECTORDB_DIR, "bm25.pkl"), "rb") as f:
    bm25 = pickle.load(f)
with open(os.path.join(VECTORDB_DIR, "chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

rag_engine.get_reranker()
llm = OllamaLLM(model="gemma3:4b", temperature=0.1)

result = rag_engine.ask("Kayıt yenileme nasıl yapılır?", vectorstore, bm25, chunks, llm)
print("CEVAP:\n", result["answer"])
print("\nKAYNAKLAR:", result["sources"])
print("\n=== RETRIEVE EDILEN CHUNK ID'LER ===")
for c in result["chunks"]:
    print(f"  - {c['metadata']['chunk_id']}")
print("\n=== HYBRID_SEARCH TOP-10 (rerank ONCESI) ===")
from query_rewriter import rewrite_query
rq = rewrite_query("Kayıt yenileme nasıl yapılır?")
top10 = rag_engine.hybrid_search(rq, vectorstore, bm25, chunks, k=10)
for c in top10:
    print(f"  - {c['metadata']['chunk_id']}")

print("\n=== sss_manuel.html_chunk_1 ICERIGI ===")
for c in chunks:
    if c["metadata"]["chunk_id"] == "sss_manuel.html_chunk_1":
        print(c["content"])
        break

print("\n=== RERANK SKORLARI (ceza haric) ===")
from rag_engine import get_reranker, is_calendar_source
reranker = get_reranker()
pairs = [[rq, c["content"]] for c in top10]
scores = reranker.predict(pairs)
for score, c in zip(scores, top10):
    src = c["metadata"]["source"]
    is_cal = is_calendar_source(src)
    print(f"  {score:.3f} {'[TAKVIM-CEZA]' if is_cal else '':14s} {c['metadata']['chunk_id']}")