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
Sana verilen bağlam belgelerini kullanarak öğrencilerin sorularını yanıtla.
Yanıtların her zaman:
- Türkçe olmalı
- ASLA sohbet geçmişindeki önceki cevaplarını (özellikle takvim, tarih veya formatları) kopyalama ve tekrar etme. Her yeni soruya sıfırdan ve sadece yeni bağlama göre cevap ver.
- Yalnızca verilen belgelere dayanmalı
- Kısa, net ve anlaşılır olmalı
- Kaynak belirtmeli (hangi yönetmelik/yönerge)
- SADECE sorulan soruya cevap ver. Sorulmayan ek konuları, karşılaştırmaları veya ilgisiz maddeleri cevaba EKLEME.
- Kaynak gösterirken sadece gerçek belge/dosya adlarını kullan. Context içindeki madde başlıklarını veya numaralarını "kaynak" olarak gösterme.
- Eğer cevabı SADECE sana verilen SİSTEM BİLGİSİ'ne (örneğin bugünün tarihi) dayanarak veriyorsan ve BAĞLAM BELGELERİ tamamen ilgisizse, cevabının sonuna tam olarak şu etiketi ekle: <KAYNAK_YOK>
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

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def semantic_router(query, embedding_model):
    # Intent 1: Akademik Takvim
    takvim_ornekleri = [
        "Vizeler ne zaman?",
        "Bahar dönemi ne zaman başlıyor?",
        "Tatil hangi gün?",
        "Bütünleme sınav tarihleri nedir?",
        "Ders seçim haftası ne zaman?",
        "Kayıt yenileme ne zaman"
    ]
    
    # Intent 2: Yönetmelik ve SSS
    yonetmelik_ornekleri = [
        "Kopya çekmenin cezası nedir?",
        "Kayıt dondurma şartları nelerdir?",
        "Kimler mazeret sınavına girebilir?",
        "Ders geçme notu AGNO",
        "Öğrenci belgesi nereden alınır?",
        "Disiplin cezası",
        "Yatay geçiş nasıl yapılır"
    ]

    # Intent 3: Genel/Sohbet (Tarih, Merhaba vs.)
    genel_ornekler = [
        "Bugün ayın kaçı?",
        "Bugünün tarihi ne?",
        "Merhaba",
        "Sen kimsin?",
        "Nasılsın?",
        "Şu an hangi yıldayız?"
    ]
    
    query_vec = embedding_model.embed_query(query)
    takvim_vecs = embedding_model.embed_documents(takvim_ornekleri)
    yonetmelik_vecs = embedding_model.embed_documents(yonetmelik_ornekleri)
    genel_vecs = embedding_model.embed_documents(genel_ornekler)
    
    takvim_scores = [cosine_similarity(query_vec, v) for v in takvim_vecs]
    yonetmelik_scores = [cosine_similarity(query_vec, v) for v in yonetmelik_vecs]
    genel_scores = [cosine_similarity(query_vec, v) for v in genel_vecs]
    
    max_takvim = max(takvim_scores)
    max_yonetmelik = max(yonetmelik_scores)
    max_genel = max(genel_scores)
    
    # Eğer takvim örneklerine olan anlamsal yakınlık, hem yönetmelik hem de genel sohbet örneklerinden fazlaysa
    # ve belirli bir eşiği aşıyorsa (alakasız soruları engellemek için), takvime yönlendir.
    if max_takvim > max_yonetmelik and max_takvim > max_genel and max_takvim > 0.4:
        return True
    return False

def ask(query, vectorstore, bm25, chunks, llm, chat_history=None):
    from query_rewriter import rewrite_query
    from academic_calendar import answer_calendar_query

    # 🧠 Semantic Router (Akıllı Yönlendirme)
    # Eski sistemdeki ilkel kelime eşleştirmeleri tamamen silindi.
    # Artık cümlenin anlamsal matematiksel karşılığına göre niyet analizi yapılıyor.
    is_calendar = semantic_router(query, vectorstore.embeddings)
    
    if is_calendar:
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
