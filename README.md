# İÜC Yapay Zekâ Destekli RAG Chatbot Projesi

Bu proje, **İstanbul Üniversitesi - Cerrahpaşa (İÜC)** Bilgisayar Mühendisliği Bölümü bitirme projesi kapsamında geliştirilmiştir. Projenin amacı; öğrencilerin, akademik ve idari personelin üniversite yönetmelikleri, staj yönergeleri, akademik takvim ve sıkça sorulan sorulara hızlı, doğru ve kaynak referanslı yanıtlar bulmasını sağlayan bir **Retrieval-Augmented Generation (RAG)** tabanlı chatbot geliştirmektir.

---

## 🛠️ Kullanılan Teknolojiler

* **Programlama Dili:** Python 3.10+
* **Arama & İndeksleme:** FAISS (Dense Retrieval) + BM25 (Sparse Retrieval) + Cross-Encoder Re-ranker (Hibrit Arama)
* **LLM Altyapısı:** Ollama (Gemma 3, Llama 3 vb.)
* **Arayüz:** Streamlit
* **Metin İşleme:** PyMuPDF, BeautifulSoup4, LangChain

---

## 📂 Klasör Yapısı

```
iuc-rag-chatbot_main/
├── iuc-rag-chatbot/
│   ├── data/
│   │   ├── raw/                 # Kaynak PDF ve HTML dosyaları (Git'e yüklenir)
│   │   └── processed/           # İşlenmiş metin parçaları (Git ignore edilir)
│   ├── src/                     # Uygulama kaynak kodları
│   │   ├── app.py               # Streamlit Arayüzü
│   │   ├── pipeline.py          # Veri İşleme & İndeksleme Hattı
│   │   ├── rag_engine.py        # RAG Arama & Yanıt Motoru
│   │   ├── query_rewriter.py    # Sorgu İyileştirici
│   │   └── academic_calendar.py # Akademik Takvim Özel Mantığı
│   ├── vectordb/                # Yerel FAISS & BM25 Veritabanı (Git ignore edilir)
│   └── models/                  # Fine-tuned model çıktıları (Git ignore edilir)
├── .gitignore                   # Git yoksayma kuralları
├── requirements.txt             # Gerekli Python paketleri
└── README.md                    # Kurulum ve Çalıştırma Kılavuzu
```

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

### 1. Adım: Depoyu Klonlayın
```bash
git clone <repo-url>
cd iuc-rag-chatbot_main
```

### 2. Adım: Sanal Ortam Oluşturun ve Paketleri Yükleyin
```bash
# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı aktif etme (Windows)
.\venv\Scripts\activate

# Gerekli paketleri yükleme
pip install -r requirements.txt
```

### 3. Adım: Ollama Kurulumu ve Modeli İndirme
Chatbot'un çalışabilmesi için bilgisayarınızda yerel bir LLM çalışıyor olmalıdır.
1. [Ollama Resmi Web Sitesi'nden](https://ollama.com) Ollama'yı indirip kurun.
2. Arka planda Ollama uygulamasını çalıştırın.
3. Terminalden varsayılan modeli indirin:
   ```bash
   ollama pull gemma3:4b
   ```

### 4. Adım: Vektör Veritabanını Oluşturun (Pipeline Çalıştırma)
Projede bulunan PDF belgelerinden metin çıkarıp bunları FAISS ve BM25 veritabanına kaydetmek için veri işleme hattını bir defaya mahsus çalıştırın:
```bash
python iuc-rag-chatbot/src/pipeline.py
```
*Bu komut tamamlandığında `iuc-rag-chatbot/vectordb/` klasörü otomatik olarak oluşturulacaktır.*

### 5. Adım: Chatbot'u Başlatın
Uygulama arayüzünü çalıştırmak için:
```bash
streamlit run iuc-rag-chatbot/src/app.py
```
*Tarayıcınızda otomatik olarak açılan `http://localhost:8501` adresinden chatbot ile sohbet etmeye başlayabilirsiniz.*

---

## 📊 Test ve Değerlendirme

Sistemin erişim ve yanıt kalitesini test etmek için `evaluation.py` scriptini çalıştırabilirsiniz:
```bash
python iuc-rag-chatbot/src/evaluation.py
```
