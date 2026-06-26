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

KURALLAR:
1. Türkçe cevap ver.
2. SADECE VE SADECE sorunun net cevabını ver. Belgedeki fıkraları, uzun cümleleri KOPYALAMA.
3. Cevabının en altına mutlaka yeni bir satır açarak "Kaynak: [Belge Adı] - Madde X" formatında referansını yaz. Eger Madde yoksa sadece belge adini yaz.
4. Rakamları ve yüzdeleri metindeki gibi yaz ("yüzde 70" yerine "%70").

ÖRNEK 1:
Kullanıcı: Derslere devam zorunluluğu yüzde kaçtır?
Bağlam: [Kaynak: Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği] Zorunlu ve isteğe bağlı yabancı dil hazırlık programlarında ... toplam ders saatinin en az %80 ine katılmış olmak zorunludur. Madde 11.
Yanıt: Derslere devam zorunluluğu %80'dir.
Kaynak: Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği - Madde 11

ÖRNEK 2:
Kullanıcı: Yaz okulunda en fazla kaç kredi alabilirim?
Bağlam: [Kaynak: Yaz Okulu Duyurusu] Yaz okulunda bir öğrenci en fazla 10 ulusal kredi değerinde ders alabilir.
Yanıt: Yaz okulunda en fazla 10 ulusal kredi alabilirsiniz.
Kaynak: Yaz Okulu Duyurusu

Eğer bilgi belgelerde yoksa "Bu konuda bilgim bulunmamaktadır." de.
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

def get_topic_filters(query):
    query_lower = query.lower()
    filters = []
    if "çap" in query_lower or "çift anadal" in query_lower or "cift anadal" in query_lower:
        filters.append("cift-anadal")
    if "yandal" in query_lower or "yan dal" in query_lower:
        filters.append("yandal")
    if "staj" in query_lower:
        filters.append("staj")
    if "yatay geçiş" in query_lower or "yatay gecis" in query_lower:
        filters.append("yatay-gecis")
    if "yaz okulu" in query_lower:
        filters.append("yaz okulu")
    if "muafiyet" in query_lower or "intibak" in query_lower:
        filters.append("muafiyet")
    if "disiplin" in query_lower or "kopya" in query_lower:
        filters.append("disiplin")
    if "hastalık" in query_lower or "rapor" in query_lower:
        filters.append("hastalik")
    return filters

def hybrid_search(query, vectorstore, bm25, chunks, k=15, alpha=0.3, topic_filters=None):
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

    final_scores = {}
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["metadata"]["chunk_id"]
        faiss_score = faiss_scores.get(chunk_id, (0, None))[0]
        bm25_score = float(bm25_normalized[i])

        bonus = 0
        source_lower = chunk["metadata"].get("source", "").lower()
        if "yönetmelik" in source_lower or "yonetmelik" in source_lower:
            bonus += 0.05
        elif "yönerge" in source_lower or "yonerge" in source_lower:
            bonus += 0.03
        elif "sss" in source_lower or "sorulan" in source_lower:
            bonus += 0.08

        if topic_filters:
            for tf in topic_filters:
                if tf in source_lower:
                    bonus += 0.5  # Massive boost for exact document match
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
    
    # 1. Açıkça takvim kelimesi geçiyorsa direkt takvime gönder
    if "takvim" in query_lower:
        return True
        
    # 2. Genel sohbetse RAG'e gönderme
    genel_keywords = ["merhaba", "selam", "nasılsın", "sen kimsin", "adın ne", "bugün ayın kaçı", "bugünün tarihi"]
    if any(kw in query_lower for kw in genel_keywords):
        return False
        
    # 3. Yönetmelik ve şart bildiren güçlü kelimeler varsa RAG'e gönder (takvime DEĞİL)
    yonetmelik_keywords = [
        "şart", "koşul", "nasıl", "nedir", "kaç gün", "ne kadar", "kimler",
        "başvurulmalı", "gerekir", "zorunlu", "yapılır", "verilir",
        "tamamlanmalı", "agno", "not ortalaması", "kredi",
        "yatay geçiş", "muafiyet", "kayıt dondur", "harç",
        "diploma", "sertifika", "itiraz", "dilekçe", "disiplin", "kopya",
        "çap", "çift anadal", "yandal", "yüzde"
    ]
    if any(kw in query_lower for kw in yonetmelik_keywords):
        return False
        
    # 4. Zaman belirten kelime + Etkinlik kelimesi kombinasyonu
    zaman_sorulari = ["ne zaman", "hangi tarih", "tarihi nedir", "hangi gün", "başlıyor", "bitiyor", "başlangıç", "bitiş"]
    etkinlikler = ["vize", "final", "bütünleme", "sınav", "kayıt", "ders", "okul", "dönem", "yarıyıl"]
    
    is_time_question = any(z in query_lower for z in zaman_sorulari)
    is_event = any(e in query_lower for e in etkinlikler)
    
    if is_time_question and is_event:
        return True
        
    # 5. Sadece vizeler/finaller kelimesi geçiyorsa takvime bakma ihtimali çok yüksek
    if any(kw in query_lower for kw in ["vizeler", "finaller", "bütler", "bütünlemeler"]):
        return True

    return False

def ask(query, vectorstore, bm25, chunks, llm, chat_history=None, stream=False):
    from query_rewriter import rewrite_query
    from academic_calendar import format_calendar_context

    # 🧠 Akıllı Yönlendirme
    is_calendar = semantic_router(query)
    
    if is_calendar:
        context_str = format_calendar_context()
        context = f"AŞAĞIDAKİ JSON VERİSİ İSTANBUL ÜNİVERSİTESİ-CERRAHPAŞA AKADEMİK TAKVİMİDİR. Soruyu SADECE bu JSON verisine bakarak Türkçe ve doğal bir cümleyle cevapla:\n\n{context_str}"
        top_chunks = []
        sources = ["Akademik Takvim JSON"]
    else:
        # Normal RAG akışı (Multi-Query Retrieval)
        queries_to_search = rewrite_query(query)
        if not isinstance(queries_to_search, list):
            queries_to_search = [queries_to_search]
            
        all_retrieved_chunks = []
        seen_chunk_ids = set()
        
        topic_filters = get_topic_filters(query)

        for q in queries_to_search:
            results = hybrid_search(q, vectorstore, bm25, chunks, k=15, topic_filters=topic_filters)
            for chunk in results:
                c_id = chunk["metadata"]["chunk_id"]
                if c_id not in seen_chunk_ids:
                    seen_chunk_ids.add(c_id)
                    all_retrieved_chunks.append(chunk)
                    
        # Havuzdaki tüm chunk'ları tek seferde kullanıcının Orijinal Sorusu'na göre sırala
        top_chunks = rerank(query, all_retrieved_chunks, top_k=5)
        context = build_context(top_chunks)
        sources = list(dict.fromkeys(c["metadata"].get("source", "") for c in top_chunks))

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

    if stream:
        def generate():
            if hasattr(llm, 'stream'):
                for chunk in llm.stream(prompt):
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            else:
                res = llm.invoke(prompt)
                yield res.content if hasattr(res, "content") else str(res)
        return generate(), sources, top_chunks

    response_obj = llm.invoke(prompt)
    response_text = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
    engine_used = getattr(response_obj, "engine", "Bilinmeyen Motor")
    
    # Kaynak halüsinasyonunu engelleme (Örn: Sadece tarih sorulduğunda alakasız PDF kaynak göstermemesi için)
    if "<KAYNAK_YOK>" in response_text:
        response_text = response_text.replace("<KAYNAK_YOK>", "").strip()
        sources = []

    return {
        "answer": response_text.strip(),
        "sources": sources,
        "chunks": top_chunks,
        "engine": engine_used
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
