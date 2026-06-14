import sys
import os
# GÜVENLİK KİLİDİ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import pickle
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from rag_engine import ask

from config import VECTORDB_DIR, BASE_DIR

GOLDEN_DATASET = [
    {"question": "Yaz okulunda en fazla kaç kredi alabilirim?",
     "keywords": ["kredi"], "expected_facts": ["10"]},

    {"question": "Çift anadal başvuru şartları nelerdir?",
     "keywords": ["anadal", "başvuru"], "expected_facts": ["3.00", "3,00", "%20"]},

    {"question": "Derslere devam zorunluluğu yüzde kaçtır?",
     "keywords": ["devam"], "expected_facts": ["%70", "%80"]},

    {"question": "Kayıt dondurma süresi en fazla ne kadardır?",
     "keywords": ["kayıt dondurma"], "expected_facts": ["yarı"]},

    {"question": "Onur öğrencisi olmak için not ortalaması kaç olmalı?",
     "keywords": ["onur"], "expected_facts": ["3.00", "3,00", "3.49", "3,49"]},

    {"question": "Yüksek onur öğrencisi kime denir?",
     "keywords": ["yüksek onur"], "expected_facts": ["3.50", "3,50"]},

    {"question": "Mazeret sınavına kimler girebilir?",
     "keywords": ["mazeret"], "expected_facts": []},

    {"question": "Yatay geçiş başvuruları ne zaman yapılır?",
     "keywords": ["yatay geçiş"], "expected_facts": []},

    {"question": "Staj defterini ne zaman teslim etmeliyim?",
     "keywords": ["staj"], "expected_facts": []},

    {"question": "Vize sınavları ne zaman başlıyor?",
     "keywords": ["vize"], "expected_facts": []},

    {"question": "Bütünleme sınavına kimler girebilir?",
     "keywords": ["bütünleme"], "expected_facts": []},

    {"question": "Tek ders sınavı nedir?",
     "keywords": ["tek ders"], "expected_facts": []},

    {"question": "Mezuniyet için minimum AGNO kaçtır?",
     "keywords": ["mezuniyet", "agno"], "expected_facts": ["2.00", "2,00"]},

    {"question": "Yandal programına başvuru şartları nelerdir?",
     "keywords": ["yandal"], "expected_facts": ["2.50", "2,50"]},

    {"question": "Staj zorunlu mu?",
     "keywords": ["staj"], "expected_facts": []},

    {"question": "Kayıt yenileme nasıl yapılır?",
     "keywords": ["kayıt yenileme"], "expected_facts": ["AKSİS", "aksis"]},

    {"question": "Ders muafiyeti nasıl alınır?",
     "keywords": ["muafiyet"], "expected_facts": ["dilekçe"]},

    {"question": "Not itirazı nasıl yapılır?",
     "keywords": ["itiraz"], "expected_facts": ["5", "beş"]},

    {"question": "Transkript nasıl alınır?",
     "keywords": ["transkript"], "expected_facts": ["AKSİS", "aksis", "EBS", "e-Devlet", "e-devlet"]},

    {"question": "Öğrenci belgesi nereden alınır?",
     "keywords": ["öğrenci belgesi"], "expected_facts": ["AKSİS", "aksis"]},

    {"question": "Çift anadal ile yandal farkı nedir?",
     "keywords": ["anadal", "yandal"], "expected_facts": ["diploma", "sertifika"]},

    {"question": "Yaz okulu dersleri ortalamayı etkiler mi?",
     "keywords": ["yaz okulu"], "expected_facts": ["agno", "AGNO"]},

    {"question": "Harç ödenmezse ne olur?",
     "keywords": ["harç"], "expected_facts": ["yenilenmez"]},

    {"question": "Üstten ders alabilir miyim?",
     "keywords": ["üstten"], "expected_facts": ["agno", "AGNO"]},
]

def evaluate():
    print("Sistem yükleniyor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": device}
    )
    vectorstore = FAISS.load_local(VECTORDB_DIR, embedding_model, allow_dangerous_deserialization=True)
    with open(os.path.join(VECTORDB_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(VECTORDB_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
    results = []
    topic_correct = 0
    content_correct = 0
    content_total = 0
    total = len(GOLDEN_DATASET)

    print(f"\n{'='*60}")
    print(f"Degerlendirme Basliyor - {total} soru")
    print(f"{'='*60}\n")

    for i, item in enumerate(GOLDEN_DATASET):
        print(f"Soru {i+1}/{total}: {item['question']}")
        start = time.time()
        result = ask(item["question"], vectorstore, bm25, chunks, llm)
        latency = time.time() - start

        answer_lower = result["answer"].lower()

        # Konu eşleşmesi: tüm keyword'ler geçmeli
        topic_hit = all(kw.lower() in answer_lower for kw in item["keywords"])
        if topic_hit:
            topic_correct += 1

        # İçerik eşleşmesi: expected_facts varsa, en az biri geçmeli
        content_hit = None
        if item["expected_facts"]:
            content_total += 1
            content_hit = any(fact.lower() in answer_lower for fact in item["expected_facts"])
            if content_hit:
                content_correct += 1

        status_topic = "OK" if topic_hit else "KONU_KACTI"
        status_content = "-" if content_hit is None else ("OK" if content_hit else "ICERIK_HATALI")

        print(f"Konu: {status_topic} | Icerik: {status_content} | Latency: {latency:.2f}s")
        print(f"Cevap: {result['answer']}\n")

        results.append({
            "question": item["question"],
            "keywords": item["keywords"],
            "expected_facts": item["expected_facts"],
            "answer": result["answer"],
            "sources": result["sources"],
            "latency": latency,
            "topic_hit": topic_hit,
            "content_hit": content_hit
        })

    topic_accuracy = topic_correct / total * 100
    content_accuracy = (content_correct / content_total * 100) if content_total > 0 else None
    avg_latency = sum(r["latency"] for r in results) / total

    print(f"\n{'='*60}")
    print(f"SONUCLAR")
    print(f"{'='*60}")
    print(f"Konu Dogrulugu (topic_accuracy): {topic_accuracy:.1f}% ({topic_correct}/{total})")
    if content_accuracy is not None:
        print(f"Icerik Dogrulugu (content_accuracy): {content_accuracy:.1f}% ({content_correct}/{content_total})")
    else:
        print("Icerik Dogrulugu: olculmedi (expected_facts bos)")
    print(f"Ortalama Latency: {avg_latency:.2f} saniye")
    print(f"{'='*60}\n")

    output_path = os.path.join(BASE_DIR, "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "topic_accuracy": topic_accuracy,
            "content_accuracy": content_accuracy,
            "avg_latency": avg_latency,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"Sonuclar '{output_path}' dosyasina kaydedildi.")

if __name__ == "__main__":
    evaluate()