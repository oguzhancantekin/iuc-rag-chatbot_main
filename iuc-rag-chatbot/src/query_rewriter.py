import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared import get_llm

REWRITE_PROMPT = """Sen bir üniversite bilgi sistemi için uzman bir akademik asistansın.
Amacımız, kullanıcının sorusunu "Yönetmelik ve Yönergeler" veri tabanında (Vektör Arama) bulabilmek için HyDE (Hypothetical Document Embeddings) yöntemi uygulamaktır.
Kullanıcının sorusunu al ve SADECE 3 elemanlı bir JSON dizisi (listesi) döndür. Başka hiçbir açıklama yazma.

JSON Dizisinin Elemanları:
1. Orijinal Sorunun Resmi Hali: Sorunun daha akademik ve resmi kelimelerle yazılmış net hali.
2. Kısa Yönetmelik Maddesi (HyDE): Eğer sen bir resmi üniversite yönetmeliği olsaydın, bu sorunun cevabını içeren o yönetmelik maddesi nasıl yazılırdı? Resmi ve ciddi bir tonda varsayımsal (hipotetik) kısa bir madde uydur.
3. Uzun Açıklayıcı Paragraf (HyDE): Sorunun cevabını detaylandıran, akademik bir dil ve terminoloji ile yazılmış varsayımsal bir rehber paragrafı uydur.

Örnek Çıktı Formatı:
[
  "Yaz okulunda en fazla kaç kredi değerinde ders alınabilir?",
  "Yaz okulunda alınabilecek derslerin toplamı 10 krediyi geçemez. Öğrenciler bir yaz döneminde en fazla 3 ders alabilirler.",
  "Üniversitemiz yaz öğretimi yönergesine göre, öğrencilerin yaz okulunda alabilecekleri derslerin kredi yükü, güz ve bahar yarıyıllarındaki başarı durumlarına bakılmaksızın sınırlandırılmıştır. İlgili akademik takvimde belirtilen kurallar çerçevesinde, öğrenciler toplamda en fazla 10 AKTS kredisini aşmayacak şekilde ders seçimi yapabilirler."
]

Kullanıcı Sorusu: {query}
JSON Çıktısı:"""

_rewrite_llm = None

def get_rewrite_llm():
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = get_llm(temperature=0.0, model_name="llama-3.3-70b-versatile")
    return _rewrite_llm

def rewrite_query(query):
    # Eger sorgu cok uzunsa (zaten yeterince detaysa) sadece kendini dondur
    if len(query.split()) > 15:
        return [query]
        
    try:
        llm = get_rewrite_llm()
        prompt = REWRITE_PROMPT.format(query=query)
        response = llm.invoke(prompt)
        rewritten_text = response.content if hasattr(response, "content") else str(response)
        rewritten_text = rewritten_text.strip()
        
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
