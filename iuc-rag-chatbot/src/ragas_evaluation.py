import os
import json
import time
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

def custom_ragas_evaluation():
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("HATA: langchain_groq kurulu degil. Lütfen 'pip install langchain_groq' calistirin.")
        return

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("HATA: .env dosyasında GROQ_API_KEY bulunamadı!")
        return

    print("Alternatif RAGAS (LLM-as-a-Judge) Değerlendirmesi Başlıyor...")
    llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.0)

    eval_file = os.path.join(BASE_DIR, "evaluation_results.json")
    if not os.path.exists(eval_file):
        print(f"HATA: {eval_file} bulunamadi.")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results_data = data.get("detailed_results", [])
    if not results_data:
        print("Detayli sonuclar bulunamadi.")
        return

    # Ilk 10 soruyu test edelim (API limitine takilmamak icin)
    test_results = results_data[:10]
    
    faithfulness_scores = []
    relevance_scores = []

    print(f"Toplam {len(test_results)} soru değerlendiriliyor...\n")

    for idx, item in enumerate(test_results):
        question = item["question"]
        answer = item["rag"]["answer"]
        
        # 1. Answer Relevance (Cevap Alakaliligi) Promptu
        relevance_prompt = f"""
        Aşağıdaki soruya verilen cevabın ne kadar alakalı olduğunu 0 ile 1 arasında puanla. 
        Sadece bir sayı döndür. (Örneğin: 0.8 veya 1.0)
        
        Soru: {question}
        Cevap: {answer}
        
        Puan:"""
        
        try:
            rel_response = llm.invoke(relevance_prompt).content.strip()
            # Extract number
            score = float(''.join(c for c in rel_response if c.isdigit() or c == '.'))
            score = min(max(score, 0.0), 1.0) # Clamp between 0 and 1
            relevance_scores.append(score)
        except:
            relevance_scores.append(0.0)

        print(f"[{idx+1}/{len(test_results)}] Soru değerlendirildi.")
        time.sleep(1) # Rate limit korumasi

    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

    print("\n" + "="*50)
    print("RAGAS (BENZETILMIS) AKADEMIK METRIKLER")
    print("="*50)
    print(f"Cevap Alakalılığı (Answer Relevance) : % {avg_relevance * 100:.1f}")
    print("*" * 50)
    print("Not: C++ derleme hatasını aşmak için bu metrikler doğrudan Groq LLM-as-a-judge yöntemiyle hesaplanmıştır.")

if __name__ == "__main__":
    custom_ragas_evaluation()
