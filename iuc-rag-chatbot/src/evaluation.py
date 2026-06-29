import sys
import os
import json
import time
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# Klasor yollarini ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import VECTORDB_DIR, DEVICE
from shared import get_llm
from rag_engine import ask, hybrid_search

# 50 Soruluk Genişletilmiş Golden Dataset
GOLDEN_DATASET = [
    # Önlisans ve Lisans Yönetmeliği Soruları
    {
        "question": "Derslere devam zorunluluğu yüzde kaçtır?",
        "keywords": ["devam", "zorunlu"],
        "expected_facts": ["%70", "%80", "70", "80"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Kayıt dondurma süresi en fazla ne kadardır?",
        "keywords": ["kayıt dondurma", "süre"],
        "expected_facts": ["yarı", "öğrenim süresinin"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Onur öğrencisi olmak için not ortalaması kaç olmalı?",
        "keywords": ["onur", "ortalama"],
        "expected_facts": ["3.00", "3,00", "3.49", "3,49"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Yüksek onur öğrencisi olmak için AGNO kaç olmalıdır?",
        "keywords": ["yüksek onur", "agno"],
        "expected_facts": ["3.50", "3,50"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Mazeret sınavına kimler girebilir?",
        "keywords": ["mazeret", "sınav"],
        "expected_facts": ["haklı ve geçerli", "yönetim kurulu"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Bütünleme sınavına kimler girebilir?",
        "keywords": ["bütünleme", "sınav"],
        "expected_facts": ["başarısız", "koşulu sağlayan", "girebilir"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Tek ders sınavı nedir ve kimler girebilir?",
        "keywords": ["tek ders", "sınav"],
        "expected_facts": ["mezuniyet", "tek dersi kalan"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Mezuniyet için minimum AGNO kaçtır?",
        "keywords": ["mezuniyet", "agno"],
        "expected_facts": ["2.00", "2,00"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Not itirazı nasıl ve kaç gün içinde yapılır?",
        "keywords": ["itiraz", "gün"],
        "expected_facts": ["5", "beş", "iş günü"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Harç ödenmezse kayıt yenilenir mi?",
        "keywords": ["harç", "kayıt yenileme"],
        "expected_facts": ["yenilenmez", "yapamaz"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Üstten ders alabilmek için AGNO kaç olmalıdır?",
        "keywords": ["üstten", "agno", "ders"],
        "expected_facts": ["3.00", "3,00"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Sınav sonuçlarına kaç gün içinde itiraz edilebilir?",
        "keywords": ["sınav", "itiraz", "gün"],
        "expected_facts": ["5", "beş", "iş günü"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Kayıt dondurmak için en geç ne zamana kadar başvurulmalıdır?",
        "keywords": ["kayıt dondurma", "başvuru"],
        "expected_facts": ["başlangıç", "ilk haftası", "ders ekleme"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Katkı payı ödemeyen öğrencinin kaydı silinir mi?",
        "keywords": ["katkı payı", "kayıt", "silinir"],
        "expected_facts": ["silinmez", "yenilenmez", "haklarından"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Ders kayıt onayını kim verir?",
        "keywords": ["danışman", "onay", "ders"],
        "expected_facts": ["danışman", "onay"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },

    # Çift Anadal (ÇAP) Soruları
    {
        "question": "Çift anadal başvuru şartları nelerdir?",
        "keywords": ["çift anadal", "şart"],
        "expected_facts": ["3.00", "3,00", "%20", "yüzde yirmi"],
        "expected_sources": ["f=411.3y_iuc-cift-anadal-programi-yonergesi.pdf", "f=411.3y_iuc-cift-anadal-programi-yonergesi-rev.01-(1).pdf"]
    },
    {
        "question": "ÇAP yapmak için sınıf başarısında yüzde kaçta olmak gerekir?",
        "keywords": ["başarı", "yüzde", "çap"],
        "expected_facts": ["%20", "20", "yüzde yirmi"],
        "expected_sources": ["f=411.3y_iuc-cift-anadal-programi-yonergesi.pdf", "f=411.3y_iuc-cift-anadal-programi-yonergesi-rev.01-(1).pdf"]
    },
    {
        "question": "Çift anadal programı ne zaman tamamlanmalıdır?",
        "keywords": ["tamamlama", "süre", "çap"],
        "expected_facts": ["normal", "ek süre", "yıl"],
        "expected_sources": ["f=411.3y_iuc-cift-anadal-programi-yonergesi.pdf", "f=411.3y_iuc-cift-anadal-programi-yonergesi-rev.01-(1).pdf"]
    },
    {
        "question": "Aynı anda birden fazla çift anadal programına kayıt yapılabilir mi?",
        "keywords": ["birden fazla", "kayıt", "çap"],
        "expected_facts": ["yapılamaz", "olamaz", "hayır"],
        "expected_sources": ["f=411.3y_iuc-cift-anadal-programi-yonergesi.pdf", "f=411.3y_iuc-cift-anadal-programi-yonergesi-rev.01-(1).pdf"]
    },
    {
        "question": "ÇAP ortalaması kaçın altına düşerse kaydı silinir?",
        "keywords": ["çap", "ortalama", "silinir"],
        "expected_facts": ["2.50", "2,50", "2.00", "2,00"],
        "expected_sources": ["f=411.3y_iuc-cift-anadal-programi-yonergesi.pdf", "f=411.3y_iuc-cift-anadal-programi-yonergesi-rev.01-(1).pdf"]
    },

    # Yandal Soruları
    {
        "question": "Yandal programına başvuru şartları nelerdir?",
        "keywords": ["yandal", "şart"],
        "expected_facts": ["2.50", "2,50"],
        "expected_sources": ["f=411.4y_iuc-yandal-programi-yonergesi.pdf"]
    },
    {
        "question": "Yandal programını başarıyla tamamlayana ne verilir?",
        "keywords": ["yandal", "tamamlama", "belge"],
        "expected_facts": ["sertifika", "yandal sertifikası"],
        "expected_sources": ["f=411.4y_iuc-yandal-programi-yonergesi.pdf"]
    },
    {
        "question": "Yandal ve çift anadal arasındaki temel fark nedir?",
        "keywords": ["diploma", "sertifika", "çap", "yandal"],
        "expected_facts": ["diploma", "sertifika"],
        "expected_sources": ["f=411.3y_iuc-cift-anadal-programi-yonergesi.pdf", "f=411.4y_iuc-yandal-programi-yonergesi.pdf"]
    },
    {
        "question": "Yandal öğrencisi anadaldan mezun olduğunda yandalı ne olur?",
        "keywords": ["yandal", "mezuniyet", "anadal"],
        "expected_facts": ["tamamlayabilir", "süre verilir"],
        "expected_sources": ["f=411.4y_iuc-yandal-programi-yonergesi.pdf"]
    },
    {
        "question": "Yandal için AGNO kaçın altına düşerse kayıt silinir?",
        "keywords": ["yandal", "agno", "kayıt", "silinir"],
        "expected_facts": ["2.00", "2,00", "altına"],
        "expected_sources": ["f=411.4y_iuc-yandal-programi-yonergesi.pdf"]
    },

    # Staj Soruları
    {
        "question": "Staj defterini ne zaman teslim etmeliyim?",
        "keywords": ["staj", "teslim", "defter"],
        "expected_facts": ["3", "üç", "hafta", "akademik dönem"],
        "expected_sources": ["staj_yonergesi.md", "staj_kurallari.md"]
    },
    {
        "question": "Staj zorunlu mudur?",
        "keywords": ["staj", "zorunlu"],
        "expected_facts": ["evet", "zorunludur", "müfredat"],
        "expected_sources": ["staj_yonergesi.md", "staj_kurallari.md"]
    },
    {
        "question": "Staj değerlendirme sonucu itirazı kaç gün içinde yapılır?",
        "keywords": ["staj", "itiraz", "gün"],
        "expected_facts": ["5", "beş", "iş günü"],
        "expected_sources": ["staj_yonergesi.md", "staj_kurallari.md"]
    },
    {
        "question": "Bir haftada en fazla kaç gün staj yapılabilir?",
        "keywords": ["staj", "gün", "hafta"],
        "expected_facts": ["6", "altı", "iş günü"],
        "expected_sources": ["staj_yonergesi.md", "staj_kurallari.md"]
    },
    {
        "question": "Staj süresi boyunca öğrencilerin sigortasını kim öder?",
        "keywords": ["sigorta", "staj", "öde"],
        "expected_facts": ["üniversite", "İÜC", "rektörlük"],
        "expected_sources": ["staj_yonergesi.md", "staj_kurallari.md"]
    },

    # Yatay Geçiş Soruları
    {
        "question": "Yatay geçiş başvuruları ne zaman ve nereye yapılır?",
        "keywords": ["yatay geçiş", "başvuru"],
        "expected_facts": ["akademik takvim", "aksis", "tarihlerinde"],
        "expected_sources": ["f=411.21y_iuc-on-lisans-ve-lisans-duzeyindeki-programlar-arasinda-kurum-ici-ve-kurumlar-arasi-yatay-gecis-esaslarina-iliskin-yonerge-(2).pdf"]
    },
    {
        "question": "Kurum içi yatay geçiş için AGNO en az kaç olmalıdır?",
        "keywords": ["kurum içi", "yatay geçiş", "agno"],
        "expected_facts": ["3.00", "3,00", "2.00", "2,00"],
        "expected_sources": ["f=411.21y_iuc-on-lisans-ve-lisans-duzeyindeki-programlar-arasinda-kurum-ici-ve-kurumlar-arasi-yatay-gecis-esaslarina-iliskin-yonerge-(2).pdf"]
    },
    {
        "question": "Merkezi yerleştirme puanıyla (Ek Madde 1) yatay geçiş yapılabilir mi?",
        "keywords": ["ek madde 1", "yatay geçiş", "puan"],
        "expected_facts": ["evet", "yapılabilir", "taban puan"],
        "expected_sources": ["f=411.21y_iuc-on-lisans-ve-lisans-duzeyindeki-programlar-arasinda-kurum-ici-ve-kurumlar-arasi-yatay-gecis-esaslarina-iliskin-yonerge-(2).pdf"]
    },
    {
        "question": "Yatay geçiş yapan öğrencinin önceki dersleri muaf edilir mi?",
        "keywords": ["muaf", "yatay geçiş", "ders"],
        "expected_facts": ["intibak", "muafiyet", "yönetim kurulu"],
        "expected_sources": ["f=411.13y_iuc-intibak-ve-muafiyet-islemleri-yonergesi.pdf", "f=411.21y_iuc-on-lisans-ve-lisans-duzeyindeki-programlar-arasinda-kurum-ici-ve-kurumlar-arasi-yatay-gecis-esaslarina-iliskin-yonerge-(2).pdf"]
    },

    # Yaz Okulu Soruları
    {
        "question": "Yaz okulunda en fazla kaç kredi değerinde ders alınabilir?",
        "keywords": ["yaz okulu", "kredi"],
        "expected_facts": ["10", "on"],
        "expected_sources": ["Yaz Okulu Duyurusu.pdf", "f=411.16y_iuc-onlisans-ve-lisans-program.-ogrenim-goren-ogrencilerin-baska-uni.-yaz-ogretiminde-ders-alabil.-da.-yonerge.pdf"]
    },
    {
        "question": "Yaz okulunda başka bir üniversiteden ders alınabilir mi?",
        "keywords": ["başka", "yaz okulu", "üniversite"],
        "expected_facts": ["evet", "alabilir", "senato", "taban puan"],
        "expected_sources": ["f=411.16y_iuc-onlisans-ve-lisans-program.-ogrenim-goren-ogrencilerin-baska-uni.-yaz-ogretiminde-ders-alabil.-da.-yonerge.pdf"]
    },
    {
        "question": "Yaz okulu notları AGNO'yu etkiler mi?",
        "keywords": ["yaz okulu", "agno", "etki"],
        "expected_facts": ["etkiler", "dahil edilir", "agno hesabına"],
        "expected_sources": ["f=411.16y_iuc-onlisans-ve-lisans-program.-ogrenim-goren-ogrencilerin-baska-uni.-yaz-ogretiminde-ders-alabil.-da.-yonerge.pdf"]
    },

    # Muafiyet ve İntibak Soruları
    {
        "question": "Muafiyet başvuruları ne zaman yapılır?",
        "keywords": ["muafiyet", "başvuru"],
        "expected_facts": ["kayıt", "ilk haftası", "tarihlerinde"],
        "expected_sources": ["f=411.13y_iuc-intibak-ve-muafiyet-islemleri-yonergesi.pdf"]
    },
    {
        "question": "Muaf olunan derslerin notu transkripte nasıl işlenir?",
        "keywords": ["muaf", "not", "transkript"],
        "expected_facts": ["M", "muaf", "harf notu"],
        "expected_sources": ["f=411.13y_iuc-intibak-ve-muafiyet-islemleri-yonergesi.pdf"]
    },

    # Genel SSS & Web Sayfaları
    {
        "question": "Kayıt yenileme işlemi nereden ve nasıl yapılır?",
        "keywords": ["kayıt yenileme", "nereden"],
        "expected_facts": ["AKSİS", "aksis"],
        "expected_sources": ["sss_manuel.html.html", "sss_manuel.html"]
    },
    {
        "question": "Öğrenci belgesi nereden temin edilebilir?",
        "keywords": ["öğrenci belgesi", "nereden"],
        "expected_facts": ["AKSİS", "aksis", "e-devlet", "e-Devlet"],
        "expected_sources": ["sss_manuel.html"]
    },
    {
        "question": "Transkript belgesi nereden alınır?",
        "keywords": ["transkript", "nereden"],
        "expected_facts": ["AKSİS", "aksis", "e-devlet", "e-Devlet"],
        "expected_sources": ["sss_manuel.html"]
    },
    {
        "question": "Akademik takvime nereden ulaşabilirim?",
        "keywords": ["akademik takvim", "ulaş"],
        "expected_facts": ["web sitesi", "oidb", "aksis"],
        "expected_sources": ["sss_manuel.html"]
    },
    {
        "question": "Ders muafiyeti için nereye dilekçe vermeliyim?",
        "keywords": ["muafiyet", "dilekçe", "nereye"],
        "expected_facts": ["bölüm", "dekanlık", "evrak kayıt"],
        "expected_sources": ["f=411.13y_iuc-intibak-ve-muafiyet-islemleri-yonergesi.pdf"]
    },
    {
        "question": "Disiplin cezası alan öğrenci ÇAP yapabilir mi?",
        "keywords": ["disiplin", "çap"],
        "expected_facts": ["alamamış", "olmaması gerekir"],
        "expected_sources": ["f=411.3y_iuc-cift-anadal-programi-yonergesi.pdf"]
    },
    {
        "question": "Sınavda kopya çeken öğrenciye hangi işlem yapılır?",
        "keywords": ["kopya", "ceza", "işlem"],
        "expected_facts": ["sıfır", "0", "almış sayılır"],
        "expected_sources": ["f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf", "f=411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi.pdf"]
    },
    {
        "question": "Öğrenci kimlik kartı kaybolursa ne yapılmalıdır?",
        "keywords": ["kimlik", "kayıp", "kart"],
        "expected_facts": ["kart", "gazete ilanı", "başvuru", "harç"],
        "expected_sources": ["sss_manuel.html"]
    },
    {
        "question": "Yaz okulu ücretleri neye göre belirlenir?",
        "keywords": ["yaz okulu", "ücret", "saat"],
        "expected_facts": ["saatlik", "kredi", "bakanlar kurulu"],
        "expected_sources": ["Yaz Okulu Duyurusu.pdf"]
    },
    {
        "question": "DGS ile kayıt yaptıranların intibak işlemleri neye göre yapılır?",
        "keywords": ["dgs", "intibak"],
        "expected_facts": ["muafiyet", "intibak komisyonu", "yönetim kurulu"],
        "expected_sources": ["f=411.13y_iuc-intibak-ve-muafiyet-islemleri-yonergesi.pdf", "f=2025-dgs-kayit-kilavuzu.pdf"]
    },
    {
        "question": "Yandal sertifikası diploma yerine geçer mi?",
        "keywords": ["yandal", "diploma", "geçer mi"],
        "expected_facts": ["geçmez", "diploma değildir", "sertifika"],
        "expected_sources": ["f=411.4y_iuc-yandal-programi-yonergesi.pdf"]
    }
]

def check_hit(answer: str, items_list: list) -> bool:
    answer_lower = answer.lower()
    return any(item.lower() in answer_lower for item in items_list)

def check_all_hits(answer: str, items_list: list) -> bool:
    answer_lower = answer.lower()
    return all(item.lower() in answer_lower for item in items_list)

def calculate_recall_and_mrr(chunks, expected_sources):
    """Recall@5 ve MRR metriklerini hesaplar"""
    if not expected_sources:
        return 1.0, 1.0  # Beklenen kaynak belirtilmediyse varsayılan
        
    recall_at_5 = 0.0
    first_rank = 0
    
    # Ilk 5 icinde arama (Recall@5)
    for idx, c in enumerate(chunks[:5]):
        source = c["metadata"].get("source", "")
        # Beklenen kaynaklardan biriyle eslesiyor mu?
        if any(expected.lower() in source.lower() for expected in expected_sources):
            recall_at_5 = 1.0
            break
            
    # Tum getirilenler icinde ilk eslesme sirasi (MRR)
    for idx, c in enumerate(chunks):
        source = c["metadata"].get("source", "")
        if any(expected.lower() in source.lower() for expected in expected_sources):
            first_rank = idx + 1
            break
            
    mrr = 1.0 / first_rank if first_rank > 0 else 0.0
    return recall_at_5, mrr

def evaluate():
    print("Sistem yukleniyor...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": DEVICE}
    )
    
    try:
        vectorstore = FAISS.load_local(VECTORDB_DIR, embedding_model, allow_dangerous_deserialization=True)
        with open(os.path.join(VECTORDB_DIR, "bm25.pkl"), "rb") as f:
            bm25 = pickle.load(f)
        with open(os.path.join(VECTORDB_DIR, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
    except Exception as e:
        print(f"HATA: İndeksler yuklenemedi. Pipeline'in calistigindan emin olun. (Hata: {e})")
        return

    # Baglantiyi hazirla (Groq/Ollama Hibrit)
    llm = get_llm(temperature=0.1)
    
    results = []
    
    rag_topic_hits = 0
    rag_content_hits = 0
    ragless_topic_hits = 0
    ragless_content_hits = 0
    
    total_recall = 0.0
    total_mrr = 0.0
    total_latency_rag = 0.0
    total_latency_ragless = 0.0
    
    content_eval_count = 0
    total = len(GOLDEN_DATASET)

    print(f"\n{'='*80}")
    print(f"AKADEMİK DEĞERLENDİRME BAŞLIYOR - {total} Soru")
    print(f"{'='*80}\n")

    for i, item in enumerate(GOLDEN_DATASET):
        print(f"[{i+1}/{total}] Soru: {item['question']}")
        
        # 1) RAG'li Sistemin Calistirilmasi ve Zamanlama
        start_rag = time.time()
        rag_res = ask(item["question"], vectorstore, bm25, chunks, llm)
        latency_rag = time.time() - start_rag
        total_latency_rag += latency_rag
        
        # 2) RAG'siz (Saf LLM) Calistirilmasi ve Zamanlama
        start_ragless = time.time()
        ragless_prompt = f"### Soru:\n{item['question']}\n\n### Yanıt:\n"
        try:
            ragless_answer = llm.invoke(ragless_prompt).strip()
        except Exception as e:
            ragless_answer = f"[LLM Hatasi: {e}]"
        latency_ragless = time.time() - start_ragless
        total_latency_ragless += latency_ragless
        
        # 3) RAG'li Isabet Kontrolleri
        rag_ans_lower = rag_res["answer"].lower()
        rag_topic_hit = check_all_hits(rag_ans_lower, item["keywords"])
        if rag_topic_hit:
            rag_topic_hits += 1
            
        rag_content_hit = None
        if item["expected_facts"]:
            rag_content_hit = check_hit(rag_ans_lower, item["expected_facts"])
            if rag_content_hit:
                rag_content_hits += 1
                
        # 4) RAG'siz Isabet Kontrolleri
        ragless_ans_lower = ragless_answer.lower()
        ragless_topic_hit = check_all_hits(ragless_ans_lower, item["keywords"])
        if ragless_topic_hit:
            ragless_topic_hits += 1
            
        ragless_content_hit = None
        if item["expected_facts"]:
            ragless_content_hit = check_hit(ragless_ans_lower, item["expected_facts"])
            if ragless_content_hit:
                ragless_content_hits += 1
        
        if item["expected_facts"]:
            content_eval_count += 1
            
        # 5) Erişim Başarısı Metrikleri (Recall@5 ve MRR)
        retrieved_chunks = rag_res.get("chunks", [])
        recall_5, mrr = calculate_recall_and_mrr(retrieved_chunks, item["expected_sources"])
        total_recall += recall_5
        total_mrr += mrr
        
        # Konsol geri bildirimi
        print(f"  -> RAG     | Konu: {'OK' if rag_topic_hit else 'FAIL'} | Icerik: {'OK' if rag_content_hit else 'FAIL'} | Latency: {latency_rag:.2f}s")
        print(f"  -> RAG'siz | Konu: {'OK' if ragless_topic_hit else 'FAIL'} | Icerik: {'OK' if ragless_content_hit else 'FAIL'} | Latency: {latency_ragless:.2f}s")
        print(f"  -> Arama   | Recall@5: {recall_5:.1f} | MRR: {mrr:.2f}")
        print("-" * 50)
        
        results.append({
            "question": item["question"],
            "expected_sources": item["expected_sources"],
            "expected_facts": item["expected_facts"],
            "rag": {
                "answer": rag_res["answer"],
                "topic_hit": rag_topic_hit,
                "content_hit": rag_content_hit,
                "latency": latency_rag,
                "recall_at_5": recall_5,
                "mrr": mrr
            },
            "ragless": {
                "answer": ragless_answer,
                "topic_hit": ragless_topic_hit,
                "content_hit": ragless_content_hit,
                "latency": latency_ragless
            }
        })
        
        # Groq (Free Tier) Dakika Basina Istek (RPM) limitini asmamak icin ufak bir bekleme:
        time.sleep(4)

    # Genel Oranlar
    rag_topic_acc = (rag_topic_hits / total) * 100
    rag_content_acc = (rag_content_hits / content_eval_count) * 100 if content_eval_count > 0 else 0
    ragless_topic_acc = (ragless_topic_hits / total) * 100
    ragless_content_acc = (ragless_content_hits / content_eval_count) * 100 if content_eval_count > 0 else 0
    
    avg_recall_5 = (total_recall / total) * 100
    avg_mrr = total_mrr / total
    
    avg_lat_rag = total_latency_rag / total
    avg_lat_ragless = total_latency_ragless / total

    # Sonuclari konsola tablo olarak bas
    print(f"\n{'='*80}")
    print(f"AKADEMİK PERFORMANS DEĞERLENDİRME TABLOSU")
    print(f"{'='*80}")
    print(f"| Metrik                         | RAG Sistemi | RAG'siz (Saf LLM) | Fark      |")
    print(f"|--------------------------------|-------------|-------------------|-----------|")
    print(f"| Konu Doğruluğu (Topic Acc)     | %{rag_topic_acc:5.1f}      | %{ragless_topic_acc:5.1f}          | %{rag_topic_acc-ragless_topic_acc:+5.1f}   |")
    print(f"| İçerik Doğruluğu (Fact Acc)    | %{rag_content_acc:5.1f}      | %{ragless_content_acc:5.1f}          | %{rag_content_acc-ragless_content_acc:+5.1f}   |")
    print(f"| Ortalama Yanıt Süresi (Latency)| {avg_lat_rag:5.2f} sn     | {avg_lat_ragless:5.2f} sn        | {avg_lat_rag-avg_lat_ragless:+5.2f} sn   |")
    print(f"|--------------------------------|-------------|-------------------|-----------|")
    print(f"| Arama Başarısı (Recall@5)      | %{avg_recall_5:5.1f}      | N/A               | N/A       |")
    print(f"| Sıralama Kalitesi (MRR)        | {avg_mrr:6.3f}      | N/A               | N/A       |")
    print(f"{'='*80}\n")

    # Sonuçları JSON dosyasına kaydet
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(BASE_DIR, "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "rag_topic_accuracy": rag_topic_acc,
                "rag_content_accuracy": rag_content_acc,
                "ragless_topic_accuracy": ragless_topic_acc,
                "ragless_content_accuracy": ragless_content_acc,
                "avg_recall_at_5": avg_recall_5,
                "avg_mrr": avg_mrr,
                "avg_latency_rag": avg_lat_rag,
                "avg_latency_ragless": avg_lat_ragless
            },
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"Detaylı değerlendirme analiz sonuçları '{output_path}' adresine başarıyla kaydedildi.")

if __name__ == "__main__":
    evaluate()
