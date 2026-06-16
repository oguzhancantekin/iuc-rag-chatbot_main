import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM

REWRITE_PROMPT = """Sen bir üniversite bilgi sistemi için sorgu optimizasyon asistanısın.
Kullanıcının sorusunu, akademik yönetmelik ve yönergelerde arama yapmak için daha iyi bir sorguya dönüştür.

Kurallar:
- Soruyu daha açık ve net hale getir
- Eksik bağlamı tamamla
- Kısaltmaları aç (örn: ÇAP → Çift Anadal Programı)
- Yönetmelik terminolojisini kullan
- Sadece yeniden yazılmış soruyu döndür, açıklama yapma

Örnekler:
Kullanıcı: hoca onay vermezse?
Yeniden yazılmış: Danışman öğrencinin ders kaydını onaylamazsa ne olur?

Kullanıcı: çap şartları
Yeniden yazılmış: Çift Anadal Programına başvuru şartları nelerdir?

Kullanıcı: not ortalaması kaç olmalı onur için
Yeniden yazılmış: Onur öğrencisi olmak için gerekli minimum ağırlıklı genel not ortalaması (AGNO) nedir?

Kullanıcı: {query}
Yeniden yazılmış:"""

_rewrite_llm = None

def get_rewrite_llm():
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = OllamaLLM(model="gemma3:4b", temperature=0.0)
    return _rewrite_llm

def rewrite_query(query):
    # DEVRE DISI: gemma3:4b rewriter testi sonucu Content Acc %58->%54,
    # Recall@5 %76->%70'e dustu. Model sorgu anlamini bozuyor
    # (ornek: "sinavi"->"siniri", "universite"->"ogretim uyesi").
    # Daha iyi bir rewriter modeli bulunana kadar kapatildi.
    return query
    try:
        # 12 kelimeden uzun sorgularda rewriting yapma (zaten yeterince açık)
        if len(query.split()) > 12:
            return query
        llm = get_rewrite_llm()
        prompt = REWRITE_PROMPT.format(query=query)
        rewritten = llm.invoke(prompt).strip()
        if len(rewritten) > 200 or len(rewritten) < 5:
            return query
        print(f"Sorgu yeniden yazildi: '{query}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"Query rewrite hatası: {e}")
        return query

if __name__ == "__main__":
    test_queries = [
        "hoca onay vermezse?",
        "çap şartları",
        "not ortalaması kaç olmalı onur için",
        "yaz okulu kredi limiti",
        "kayıt dondurma ne kadar sürer",
        "Onur öğrencisi olmak için not ortalaması kaç olmalı?",
        "Derslere devam zorunluluğu yüzde kaçtır?"
    ]

    print("Query Rewriting Test\n" + "="*40)
    for q in test_queries:
        rewritten = rewrite_query(q)
        print(f"Orijinal : {q}")
        print(f"Yeniden  : {rewritten}\n")
