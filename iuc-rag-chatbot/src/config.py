import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Alt dizinler
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PDF_DIR = os.path.join(RAW_DIR, "pdfs")
HTML_DIR = os.path.join(RAW_DIR, "html")
MD_DIR = os.path.join(RAW_DIR, "markdowns")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
VECTORDB_DIR = os.path.join(BASE_DIR, "vectordb")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

# MODEL YOLU (İşte buraya ekledik)
MODEL_PATH = os.path.join(MODELS_DIR, "llama3-iuc-finetuned")

# Embedding/reranker modelleri icin ortak cihaz secimi.
# pipeline.py, rag_engine.py, evaluation.py ve finetune.py bu degeri kullanir;
# her dosyada ayri ayri "cuda" if torch.cuda.is_available() else "cpu"
# tekrarini onler.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tüm dizinleri oluştur
for d in [PDF_DIR, HTML_DIR, MD_DIR, PROCESSED_DIR, VECTORDB_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"PDF_DIR: {PDF_DIR}")
    print(f"VECTORDB_DIR: {VECTORDB_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"DEVICE: {DEVICE}")
