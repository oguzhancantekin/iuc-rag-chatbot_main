import os
import json
import pickle
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import re
import hashlib

from config import PDF_DIR, HTML_DIR, MD_DIR, JSON_DIR, PROCESSED_DIR, VECTORDB_DIR, DEVICE
from shared import get_display_name

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VECTORDB_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r'Dok\.?No[:.]?\s*[\w.\-]+\s*(?:İlk\s*)?Yay\.?\s*Tar[:.]?\s*[\d.\-]+(?:\s*Revizyon\s*(?:No)?\s*Tar[:.]?\s*[\d.\-]+)?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'ELEKTRON[İI]K\s*N[ÜU]SHADIR\.?\s*BASILI\s*HAL[İI]\s*KONTROLS[ÜU]Z\s*KOPYADIR\.?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?()%\-çğıöşüÇĞİÖŞÜ]', ' ', text)
    return text.strip()

def extract_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return clean_text(text)
    except Exception as e:
        print(f"PDF okuma hatası {filepath}: {e}")
        return ""

def extract_html(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return clean_text(soup.get_text(separator="\n"))
    except Exception as e:
        print(f"HTML okuma hatası {filepath}: {e}")
        return ""

def extract_md(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            # MD dosyalarini saf metin olarak donduruyoruz.
            # RAG sistemi (RecursiveCharacterTextSplitter) basliklari (#) ayristirmada iyidir.
            return f.read().strip()
    except Exception as e:
        print(f"MD okuma hatası {filepath}: {e}")
        return ""


def load_all_documents():
    documents = []
    seen_hashes = {}  # content_hash -> filename (ilk gorulen)

    print("PDF'ler okunuyor...")
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(PDF_DIR, filename)
            text = extract_pdf(filepath)
            if len(text) > 100:
                content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    print(f"  [DUPLICATE] atlandi: {filename[:60]} (ayni icerik: {seen_hashes[content_hash][:60]})")
                    continue
                seen_hashes[content_hash] = filename
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "type": "pdf",
                        "filepath": filepath
                    }
                })
                print(f"  + {filename[:60]}")

    # NOT: Bu dongude eskiden ayni extract/hash/append blogu IKI KEZ
    # ust uste yazilmisti. Ilk blok "continue" ile devam ettiginde ikinci
    # blok hicbir zaman calismiyordu (zaten skip edilen dosyaya tekrar
    # bakiyordu); duplicate olmayan dosyalarda ise ayni is iki kez
    # tekrarlaniyordu. Tek, temiz bir dongu olarak birlestirildi.
    print(f"\nHTML sayfalar okunuyor...")
    for filename in os.listdir(HTML_DIR):
        if filename.endswith(".html"):
            filepath = os.path.join(HTML_DIR, filename)
            text = extract_html(filepath)
            if len(text) > 100:
                content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    print(f"  [DUPLICATE] atlandi: {filename[:60]} (ayni icerik: {seen_hashes[content_hash][:60]})")
                    continue
                seen_hashes[content_hash] = filename
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "type": "html",
                        "filepath": filepath
                    }
                })
                print(f"  + {filename[:60]}")

    print(f"\nMarkdown (MD) sayfalar okunuyor...")
    for filename in os.listdir(MD_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(MD_DIR, filename)
            text = extract_md(filepath)
            if len(text) > 50: # MD kisa olabilir
                content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    print(f"  [DUPLICATE] atlandi: {filename[:60]} (ayni icerik: {seen_hashes[content_hash][:60]})")
                    continue
                seen_hashes[content_hash] = filename
                documents.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "type": "md",
                        "filepath": filepath
                    }
                })
                print(f"  + {filename[:60]}")

    print(f"\nJSON sayfalar okunuyor...")
    for filename in os.listdir(JSON_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(JSON_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            if "soru" in item and "cevap" in item:
                                text = f"Soru: {item['soru']}\nCevap: {item['cevap']}"
                                content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                                if content_hash in seen_hashes:
                                    continue
                                seen_hashes[content_hash] = f"{filename}_{i}"
                                documents.append({
                                    "content": text,
                                    "metadata": {
                                        "source": filename,
                                        "type": "json",
                                        "filepath": filepath
                                    }
                                })
                        print(f"  + {filename[:60]}")
            except Exception as e:
                print(f"JSON okuma hatası {filepath}: {e}")

    print(f"\nToplam doküman: {len(documents)}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    seen_chunk_hashes = set()
    duplicate_count = 0
   

    for doc in documents:
        parts = splitter.split_text(doc["content"])
        for i, part in enumerate(parts):
            stripped = part.strip()
            if len(stripped) > 50:
                chunk_hash = hashlib.md5(stripped.encode("utf-8")).hexdigest()
                if chunk_hash in seen_chunk_hashes:
                    duplicate_count += 1
                    continue
                seen_chunk_hashes.add(chunk_hash)
                
                # Contextual Chunking: Parçanın başına belge adını ekle
                display_source = get_display_name(doc["metadata"]["source"])
                contextual_content = f"[Belge: {display_source}]\n{stripped}"
                
                chunks.append({
                    "content": contextual_content,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": f"{doc['metadata']['source']}_chunk_{i}"
                    }
                })
    
    print(f"Toplam chunk: {len(chunks)} (Duplicate olarak atlanan: {duplicate_count})")
    return chunks

def build_faiss_index(chunks):
    print("\nFAISS indeksi oluşturuluyor...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": DEVICE}
    )

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    vectorstore.save_local(VECTORDB_DIR)
    print(f"FAISS indeksi kaydedildi: {VECTORDB_DIR}")
    return vectorstore

def build_bm25_index(chunks):
    print("\nBM25 indeksi oluşturuluyor...")
    tokenized = [c["content"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    bm25_path = os.path.join(VECTORDB_DIR, "bm25.pkl")
    chunks_path = os.path.join(VECTORDB_DIR, "chunks.pkl")

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"BM25 indeksi kaydedildi.")
    return bm25, chunks

if __name__ == "__main__":
    print("=" * 50)
    print("IUC RAG Pipeline Başlıyor")
    print("=" * 50)

    documents = load_all_documents()
    chunks = chunk_documents(documents)

    with open(os.path.join(PROCESSED_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    build_faiss_index(chunks)
    build_bm25_index(chunks)

    print("\n" + "=" * 50)
    print("Pipeline tamamlandı!")
    print(f"Toplam chunk: {len(chunks)}")
    print("=" * 50)
