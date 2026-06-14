import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import pickle
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from config import VECTORDB_DIR
import rag_engine

QUESTION = "Yandal programına başvuru şartları nelerdir?"
RUNS = 3

def load():
    print("Indeksler yukleniyor...\n")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    vectorstore = FAISS.load_local(VECTORDB_DIR, embedding_model, allow_dangerous_deserialization=True)
    with open(os.path.join(VECTORDB_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(VECTORDB_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return vectorstore, bm25, chunks

def show_top_chunks(label, query, vectorstore, bm25, chunks):
    print(f"\n--- {label} ---")
    top = rag_engine.hybrid_search(query, vectorstore, bm25, chunks, k=10)
    for i, c in enumerate(top[:5]):
        src = c["metadata"]["source"]
        preview = c["content"][:80].replace("\n", " ")
        print(f"  {i+1}. [{src}] {preview}...")

def main():
    vectorstore, bm25, chunks = load()
    rag_engine.get_reranker()

    rewritten = rag_engine.SYSTEM_PROMPT  # placeholder, not used
    from query_rewriter import rewrite_query
    rq = rewrite_query(QUESTION)
    print(f"Sorgu: '{QUESTION}' -> Rewritten: '{rq}'")

    # 1) BONUS'LU hybrid_search sonucu (mevcut hal)
    show_top_chunks("BONUS'LU (mevcut hybrid_search)", rq, vectorstore, bm25, chunks)

    # 2) BONUS'SUZ versiyon - priority_sources'u boşalt
    original_func = rag_engine.hybrid_search
    def hybrid_search_no_bonus(query, vectorstore, bm25, chunks, k=10, alpha=0.4):
        import rank_bm25
        faiss_results = vectorstore.similarity_search_with_score(query, k=10)
        faiss_scores = {}
        for doc, score in faiss_results:
            chunk_id = doc.metadata.get("chunk_id", "")
            faiss_scores[chunk_id] = (1 - score, doc)

        tokenized_query = query.lower().split()
        bm25_scores_raw = bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_normalized = bm25_scores_raw / max_bm25

        final_scores = {}
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["metadata"]["chunk_id"]
            faiss_score = faiss_scores.get(chunk_id, (0, None))[0]
            bm25_score = float(bm25_normalized[i])
            final_scores[chunk_id] = (alpha * faiss_score + (1 - alpha) * bm25_score, chunk)

        sorted_results = sorted(final_scores.items(), key=lambda x: x[1][0], reverse=True)
        return [item[1][1] for item in sorted_results[:k]]

    rag_engine.hybrid_search = hybrid_search_no_bonus
    show_top_chunks("BONUS'SUZ (priority_sources kaldirildi)", rq, vectorstore, bm25, chunks)
    rag_engine.hybrid_search = original_func

    # 3) Tam pipeline ile NON-DETERMINISM testi (bonus'lu, normal akis)
    print(f"\n\n=== NON-DETERMINISM TESTI ({RUNS} kez ayni soru) ===")
    llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
    for run in range(1, RUNS + 1):
        result = rag_engine.ask(QUESTION, vectorstore, bm25, chunks, llm)
        print(f"\n--- Run {run} ---")
        print(f"Cevap: {result['answer'][:200]}")
        print(f"Kaynaklar: {result['sources']}")

if __name__ == "__main__":
    main()