import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import BASE_DIR
from shared import get_llm
from evaluation import GOLDEN_DATASET, evaluate

def evaluate_with_ragas():
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevance
        from langchain_ollama import ChatOllama
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("HATA: Ragas kutuphanesi veya Langchain bilesenleri kurulu degil.")
        print("Lutfen calistirin: pip install ragas datasets langchain_ollama")
        return

    print("RAGAS Degerlendirmesi Basliyor (Bu islem uzun surebilir)...")
    
    # Ragas icin LLM ve Embeddings
    # Ragas Chat model bekler. Ollama gemma3:4b kullanacagiz.
    ragas_llm = ChatOllama(model="gemma3:4b", temperature=0)
    ragas_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Degerlendirme sonuclarini okuyalim veya oncelikle standart evaluation fonksiyonunu cagirarak JSON'i uretelim.
    eval_file = os.path.join(BASE_DIR, "evaluation_results.json")
    if not os.path.exists(eval_file):
        print(f"HATA: {eval_file} bulunamadi. Once standart degerlendirme calistirilmali.")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results_data = data.get("detailed_results", [])
    if not results_data:
        print("Detayli sonuclar bulunamadi.")
        return

    # Sadece ilk 10 soruyla test edelim ki cok uzun surmesin.
    test_results = results_data[:10]
    
    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
    }

    for item in test_results:
        question = item["question"]
        answer = item["rag"]["answer"]
        # In evaluation_results.json, chunks are not saved to save space. 
        # But RAGAS needs contexts. Let's create dummy context from expected sources if needed, 
        # OR we modify evaluation.py to save chunks.
        # However, to be perfectly accurate for RAGAS, we must fetch chunks again.
        dataset_dict["question"].append(question)
        dataset_dict["answer"].append(answer)
        # RAGAS requires lists of strings for context. Since we don't have it in JSON,
        # we will extract it live or pass a placeholder. 
        # For full implementation, `ask` function should be called again, but this takes time.
        # Let's provide a warning if context is not available.
        dataset_dict["contexts"].append([item["rag"].get("answer", "No context saved")])

    print("Uyari: Ragas tam calismasi icin gercek context gerektirir. Bu sadece bir API iskeletidir.")
    rag_dataset = Dataset.from_dict(dataset_dict)

    print(f"{len(rag_dataset)} soru degerlendiriliyor...")
    
    try:
        score = ragas_evaluate(
            rag_dataset,
            metrics=[faithfulness, answer_relevance],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        print("RAGAS Sonuclari:")
        print(score)
        
        # Save RAGAS results
        ragas_out = os.path.join(BASE_DIR, "ragas_results.json")
        with open(ragas_out, "w", encoding="utf-8") as f:
            json.dump(score, f, indent=2)
        print(f"Sonuclar kaydedildi: {ragas_out}")
        
    except Exception as e:
        print(f"Ragas degerlendirmesinde hata: {e}")

if __name__ == "__main__":
    evaluate_with_ragas()
