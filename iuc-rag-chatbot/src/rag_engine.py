import pickle
import os
import datetime
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from config import VECTORDB_DIR, DEVICE
from shared import get_display_name

# NOT: SOURCE_DISPLAY_NAMES / get_display_name burada app.py ile birebir
# kopya halindeydi; artik shared.py'den import ediliyor (tek kaynak).
# Ayrica burada hic kullanilmayan CALENDAR_FILE_PATTERNS/is_calendar_source
# ciftiyle academic_calendar.py'deki gercekte calisan TAKVIM_PDFS listesi
# birbirinden bagimsizdi; takvim dosyasi tespiti artik tek yerde
# (shared.is_calendar_source) toplandi, bu dosyada ayrica tanimlanmiyor.

SYSTEM_PROMPT = """Sen İstanbul Üniversitesi-Cerrahpaşa'nın resmi akademik asistanısın.
Sana verilen BAĞLAM BELGELERİ'ni kullanarak öğrencilerin sorularını yanıtla.

KESİN KURALLAR:
1. Türkçe dilinde, kısa, net ve anlaşılır yanıt ver.
2. YALNIZCA BAĞLAM BELGELERİNDE BULUNAN BİLGİLERE dayan. Bağlamda cevap yoksa KESİNLİKLE uydurma, "Bu konuda bilgim bulunmamaktadır, lütfen öğrenci işleri ile iletişime geçin." de.
3. Sorulmayan konuları, karşılaştırmaları veya ilgisiz maddeleri cevaba EKLEME.
4. Kaynak belirtirken, bağlamın en başındaki [Belge: ...] etiketini referans al.
5. ASLA önceki sohbet geçmişindeki takvimleri veya tarihleri yeni cevaba kopyalama.
6. Eğer cevabı SADECE sana verilen SİSTEM BİLGİSİ'ne (örneğin bugünün tarihi) dayanarak veriyorsan ve BAĞLAM BELGELERİ tamamen ilgisizse, cevabının sonuna tam olarak şu etiketi ekle: <KAYNAK_YOK>
"""

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print("Re-ranking modeli yükleniyor...")
        _reranker = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            device=DEVICE
        )
        print(f"Re-ranking modeli hazır! (device: {DEVICE})")
    return _reranker

def load_indexes():
    print("İndeksler yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": DEVICE}
    )
    vectorstore = FAISS.load_local(
        VECTORDB_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    with open(os.path.join(VECTORDB_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(VECTORDB_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    print("İndeksler yüklendi!")
    return vectorstore, bm25, chunks

def hybrid_search(query, vectorstore, bm25, chunks, k=15, alpha=0.4):
    faiss_k = max(k * 2, 20)  # Bug fix: FAISS havuzunu genişlet (eskiden sabit 10'du)
    faiss_results = vectorstore.similarity_search_with_score(query, k=faiss_k)
    faiss_scores = {}
    for doc, score in faiss_results:
        chunk_id = doc.metadata.get("chunk_id", "")
        faiss_scores[chunk_id] = (1 - score, doc)

    tokenized_query = query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_normalized = bm25_scores_raw / max_bm25

    priority_sources = [
        ("sss_manuel", 0.08),
        ("411.1y_iuc-onlisans", 0.05),
        ("iu-cerrahpasa-onlisans-ve-lisans-yonetmeligi-web", 0.05),
        ("411.3y_iuc-cift-anadal", 0.05),
        ("411.21y_iuc-on-lisans-ve-lisans", 0.05),
        ("411.4y_iuc-yandal", 0.03),
        ("411.14y_iuc-lisans-staj", 0.03),
        ("411.15y_iuc-hastalik", 0.03),
    ]

    final_scores = {}
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["metadata"]["chunk_id"]
        faiss_score = faiss_scores.get(chunk_id, (0, None))[0]
        bm25_score = float(bm25_normalized[i])

        bonus = 0
        for ps, ps_bonus in priority_sources:
            if ps in chunk["metadata"]["source"]:
                bonus = ps_bonus
                break

        final_scores[chunk_id] = (
            alpha * faiss_score + (1 - alpha) * bm25_score + bonus,
            chunk
        )

    sorted_results = sorted(final_scores.items(), key=lambda x: x[1][0], reverse=True)
    top_chunks = [item[1][1] for item in sorted_results[:k]]
    return top_chunks

def rerank(query, chunks, top_k=5):
    reranker = get_reranker()
    pairs = [[query, chunk["content"]] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]

def build_context(chunks):
    context_parts = []
    for chunk in chunks:
        source = chunk["metadata"].get("source", "Bilinmiyor")
        display_source = get_display_name(source)
        content = chunk["content"]
        context_parts.append(f"[Kaynak: {display_source}]\n{content}")
    return "\n\n---\n\n".join(context_parts)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def semantic_router(query):
    query_lower = query.lower()
    
    takvim_keywords = [
        "ne zaman", "hangi tarih", "tarihi nedir", "başlıyor", "bitiyor", 
        "akademik takvim", "vize tarihleri", "final tarihleri", "bütünleme tarihleri"
    ]
    
    genel_keywords = [
        "merhaba", "selam", "nasılsın", "sen kimsin", "adın ne", "bugün ayın kaçı", "bugünün tarihi"
    ]
    
    if any(kw in query_lower for kw in genel_keywords):
        return False
        
    if any(kw in query_lower for kw in takvim_keywords):
        return True
        
    return False

def ask(query, vectorstore, bm25, chunks, llm, chat_history=None):
    from query_rewriter import rewrite_query
    from academic_calendar import answer_calendar_query

    # 🧠 Akıllı Yönlendirme
    is_calendar = semantic_router(query)
    
    if is_calendar:
        calendar_answer = answer_calendar_query(query, chat_history)
        return {
            "answer": calendar_answer,
            "sources": ["Akademik Takvim PDF"],
            "chunks": []
        }

    # Normal RAG akışı (Multi-Query Retrieval)
    queries_to_search = rewrite_query(query)
    if not isinstance(queries_to_search, list):
        queries_to_search = [queries_to_search]
        
    all_retrieved_chunks = []
    seen_chunk_ids = set()
    
    for q in queries_to_search:
        results = hybrid_search(q, vectorstore, bm25, chunks, k=15)
        for chunk in results:
            c_id = chunk["metadata"]["chunk_id"]
            if c_id not in seen_chunk_ids:
                seen_chunk_ids.add(c_id)
                all_retrieved_chunks.append(chunk)
                
    # Havuzdaki tüm chunk'ları tek seferde kullanıcının Orijinal Sorusu'na göre sırala
    top_chunks = rerank(query, all_retrieved_chunks, top_k=5)
    context = build_context(top_chunks)

    history_text = ""
    if chat_history:
        for turn in chat_history[-1:]:
            safe_assistant = turn['assistant']
            
            # ZEHİR PANZEHİRİ: Eğer önceki cevap takvimse, LLM'den gizle!
            if "📅" in safe_assistant or "Akademik Takvim" in safe_assistant:
                safe_assistant = "[Öğrenciye akademik takvim bilgisi iletildi. Bir sonraki cevabında kendi normal ve resmi formatına dön.]"
                
            history_text += f"Kullanıcı: {turn['user']}\nAsistan: {safe_assistant}\n\n"

    # Eğitimde kullandığımız şablona (### Soru: ve ### Yanıt:) uyumlu hale getirildi
    current_date_str = datetime.date.today().strftime("%d %B %Y")
    
    prompt = f"""{SYSTEM_PROMPT}

[SİSTEM BİLGİSİ: Bugünün tarihi {current_date_str}]

{f'ÖNCEKİ KONUŞMA:{chr(10)}{history_text}' if history_text else ''}
BAĞLAM BELGELERİ:
{context}

### Soru:
{query}

### Yanıt:
"""

    response_text = llm.invoke(prompt)
    
    # Kaynak halüsinasyonunu engelleme (Örn: Sadece tarih sorulduğunda alakasız PDF kaynak göstermemesi için)
    if "<KAYNAK_YOK>" in response_text:
        response_text = response_text.replace("<KAYNAK_YOK>", "").strip()
        sources = []
    else:
        # NOT: Eskiden list(set(...)) kullanilarak kaynaklar dedup ediliyordu,
        # ancak set sirasiz oldugu icin en alakali (rerank sirasinda ust siraya
        # cikan) kaynagin UI'da ilk gosterilmesi garanti degildi. dict.fromkeys
        # ile sira korunarak dedup yapiliyor.
        sources = list(dict.fromkeys(c["metadata"].get("source", "") for c in top_chunks))

    return {
        "answer": response_text.strip(),
        "sources": sources,
        "chunks": top_chunks
    }

if __name__ == "__main__":
    # Bu modul artik sadece app.py uzerinden Ollama ile kullaniliyor.
    # Hizli bir CLI testi icin Ollama gerektirir.
    from langchain_ollama import OllamaLLM

    vectorstore, bm25, chunks = load_indexes()
    get_reranker()

    llm = OllamaLLM(model="gemma3:4b", temperature=0.1)

    print("\nİÜC Akademik Asistan hazır! (Çıkmak için 'quit' yazın)\n")
    while True:
        query = input("Sorunuz: ").strip()
        if query.lower() in ["quit", "exit", "çıkış"]:
            break
        if not query:
            continue

        print("\nYanıt aranıyor...")
        result = ask(query, vectorstore, bm25, chunks, llm)

        print(f"\n{'='*50}")
        print(f"YANIT:\n{result['answer']}")
        print(f"\nKAYNAKLAR:")
        for s in result['sources']:
            print(f"  - {s}")
        print(f"{'='*50}\n")
