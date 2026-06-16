import pickle
import os
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
Sana verilen bağlam belgelerini kullanarak öğrencilerin sorularını yanıtla.
Yanıtların her zaman:
- Türkçe olmalı
- ASLA sohbet geçmişindeki önceki cevaplarını (özellikle takvim, tarih veya formatları) kopyalama ve tekrar etme. Her yeni soruya sıfırdan ve sadece yeni bağlama göre cevap ver.
- Yalnızca verilen belgelere dayanmalı
- Kısa, net ve anlaşılır olmalı
- Kaynak belirtmeli (hangi yönetmelik/yönerge)
- SADECE sorulan soruya cevap ver. Sorulmayan ek konuları, karşılaştırmaları veya ilgisiz maddeleri cevaba EKLEME.
- Kaynak gösterirken sadece gerçek belge/dosya adlarını kullan. Context içindeki madde başlıklarını veya numaralarını "kaynak" olarak gösterme.
Eğer bilgi belgelerinde yoksa "Bu konuda bilgim bulunmamaktadır, lütfen öğrenci işleri ile iletişime geçin." de.
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

def ask(query, vectorstore, bm25, chunks, llm, chat_history=None):
    from query_rewriter import rewrite_query
    from academic_calendar import answer_calendar_query

    # Akademik takvim sorusuysa direkt modülü kullan (chat_history eklendi!)
    takvim_keywords = ["vize", "final", "bütünleme", "kayıt yenileme tarihi",
                       "sınav tarihi", "akademik takvim", "ne zaman başlıyor",
                       "büt ne zaman", "tatil ne zaman", "vizeler"]
    # Yönetmelik-tipi sorular takvime yönlendirilmesin
    yonetmelik_keywords = ["kimler girebilir", "kimler başvurabilir", "şartları",
                           "koşulları", "nereden", "nasıl yapılır", "nedir",
                           "ne gerekir", "hakkı var", "kaç kez"]
    query_lower = query.lower()
    is_calendar = any(kw in query_lower for kw in takvim_keywords)
    is_regulation = any(kw in query_lower for kw in yonetmelik_keywords)
    if is_calendar and not is_regulation:
        calendar_answer = answer_calendar_query(query, chat_history)
        return {
            "answer": calendar_answer,
            "sources": ["Akademik Takvim PDF"],
            "chunks": []
        }

    # Normal RAG akışı
    rewritten_query = rewrite_query(query)
    top_chunks = hybrid_search(rewritten_query, vectorstore, bm25, chunks, k=15)
    top_chunks = rerank(rewritten_query, top_chunks, top_k=5)
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
    prompt = f"""{SYSTEM_PROMPT}

{f'ÖNCEKİ KONUŞMA:{chr(10)}{history_text}' if history_text else ''}
BAĞLAM BELGELERİ:
{context}

### Soru:
{query}

### Yanıt:
"""

    response = llm.invoke(prompt)
    # NOT: Eskiden list(set(...)) kullanilarak kaynaklar dedup ediliyordu,
    # ancak set sirasiz oldugu icin en alakali (rerank sirasinda ust siraya
    # cikan) kaynagin UI'da ilk gosterilmesi garanti degildi. dict.fromkeys
    # ile sira korunarak dedup yapiliyor.
    sources = list(dict.fromkeys(c["metadata"].get("source", "") for c in top_chunks))

    return {
        "answer": response.strip(),
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
