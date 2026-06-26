import sys
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json

# Klasor yollarini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared import get_llm
from rag_engine import ask, load_indexes

app = FastAPI(
    title="İÜC RAG Chatbot API",
    description="İstanbul Üniversitesi-Cerrahpaşa Akademik RAG Chatbot API Servisi",
    version="1.0.0"
)

# Global index degiskenleri
vectorstore, bm25, chunks = None, None, None

@app.on_event("startup")
def startup_event():
    global vectorstore, bm25, chunks
    try:
        print("Uygulama baslatiliyor, indeksler yukleniyor...")
        vectorstore, bm25, chunks = load_indexes()
        print("İndeksler basariyla yuklendi ve API hazir!")
    except Exception as e:
        print(f"HATA: İndeksler yuklenirken sorun olustu: {e}")
        # Hata olsa bile API ayakta kalsin, endpoint'lerde hata dondururuz.

class QueryRequest(BaseModel):
    query: str
    model_choice: str = "gemma3:4b"
    temperature: float = 0.1
    chat_history: List[Dict[str, str]] = []

@app.get("/health")
def health_check():
    if vectorstore is None or bm25 is None or chunks is None:
        return {"status": "unhealthy", "message": "İndeksler yuklenemedi. Pipeline'in calistigindan emin olun."}
    return {"status": "ok", "message": "İÜC RAG API ayakta ve hazir!"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    if vectorstore is None or bm25 is None or chunks is None:
        raise HTTPException(
            status_code=503, 
            detail="Arama indeksleri yuklenemedi. Lutfen pipeline.py'i calistirip yerel veritabani dizinini kontrol edin."
        )
    
    try:
        start_time = time.time()
        
        # RAG motorunu cagir (Groq/Ollama Hibrit)
        llm = get_llm(temperature=request.temperature)
        
        # RAG motorunu cagir
        result = ask(
            query=request.query,
            vectorstore=vectorstore,
            bm25=bm25,
            chunks=chunks,
            llm=llm,
            chat_history=request.chat_history
        )
        
        elapsed = time.time() - start_time
        
        # Clean chunks serialization for JSON safety
        safe_chunks = []
        if "chunks" in result:
            for c in result["chunks"]:
                safe_chunks.append({
                    "content": c.get("content", ""),
                    "metadata": c.get("metadata", {})
                })
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": safe_chunks,
            "elapsed": elapsed,
            "engine": result.get("engine", "Bilinmeyen Motor")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorgu islenirken hata olustu: {str(e)}")

@app.post("/ask_stream")
def ask_question_stream(request: QueryRequest):
    if vectorstore is None or bm25 is None or chunks is None:
        raise HTTPException(
            status_code=503, 
            detail="Arama indeksleri yuklenemedi. Lutfen pipeline.py'i calistirip yerel veritabani dizinini kontrol edin."
        )
    
    try:
        llm = get_llm(temperature=request.temperature)
        
        generator, sources, top_chunks = ask(
            query=request.query,
            vectorstore=vectorstore,
            bm25=bm25,
            chunks=chunks,
            llm=llm,
            chat_history=request.chat_history,
            stream=True
        )
        
        safe_chunks = []
        for c in top_chunks:
            safe_chunks.append({
                "content": c.get("content", ""),
                "metadata": c.get("metadata", {})
            })
        
        def sse_generator():
            meta = {
                "type": "meta",
                "sources": sources,
                "chunks": safe_chunks,
                "engine": "Hibrit Motor (Stream)"
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            
            for chunk_text in generator:
                if chunk_text:
                    payload = {"type": "chunk", "content": chunk_text}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(sse_generator(), media_type="text/event-stream")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming sorgusu islenirken hata olustu: {str(e)}")

