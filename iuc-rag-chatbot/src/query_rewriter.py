import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM

REWRITE_PROMPT = """Sen bir üniversite bilgi sistemi için sorgu optimizasyon asistanısın.
Kullanıcının sorusunu, akademik yönetmelik ve yönergelerde arama yapmak için 3 farklı ve kapsamlı arama sorgusuna dönüştür.

Kurallar:
1. İlk sorgu, kullanıcının sorusunun en resmi ve net hali olmalı.
2. İkinci sorgu, soruyu farklı eşanlamlı kelimeler veya alternatif terimlerle sormalı.
3. Üçüncü sorgu, sorunun bağlamını daha geniş veya dar tutarak farklı bir açıdan ele almalı.
4. Kısaltmaları (Örn: ÇAP, AGNO) açmalısın.
5. SADECE geçerli bir JSON dizisi (listesi) döndür, başka hiçbir metin veya açıklama ekleme.

Örnek Çıktı Formatı:
[
  "Çift Anadal Programına başvuru şartları nelerdir?",
  "İkinci anadal yapmak için gerekli not ortalaması nedir?",
  "ÇAP başvuru koşulları ve gerekli belgeler"
]

Kullanıcı Sorusu: {query}
JSON Çıktısı:"""

_rewrite_llm = None

def get_rewrite_llm():
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = OllamaLLM(model="gemma3:4b", temperature=0.0)
    return _rewrite_llm

def rewrite_query(query):
    # Eger sorgu cok uzunsa (zaten yeterince detaysa) sadece kendini dondur
    if len(query.split()) > 15:
        return [query]
        
    try:
        llm = get_rewrite_llm()
        prompt = REWRITE_PROMPT.format(query=query)
        rewritten_text = llm.invoke(prompt).strip()
        
        # Markdown kod bloklarini temizle (eger LLM ```json ... ``` kullanirsa)
        if rewritten_text.startswith("```json"):
            rewritten_text = rewritten_text[7:]
        if rewritten_text.startswith("```"):
            rewritten_text = rewritten_text[3:]
        if rewritten_text.endswith("```"):
            rewritten_text = rewritten_text[:-3]
            
        rewritten_text = rewritten_text.strip()
        
        import json
        queries = json.loads(rewritten_text)
        
        if isinstance(queries, list) and len(queries) > 0:
            print(f"Çoklu Sorgu Üretildi: {queries}")
            return queries
        else:
            return [query]
    except Exception as e:
        print(f"Query rewrite hatası (JSON ayıklanamadı): {e}")
        return [query]

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
