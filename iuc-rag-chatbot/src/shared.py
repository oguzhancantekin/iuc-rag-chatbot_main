import re

# Kaynak dosya adlarini kullaniciya gosterilecek okunabilir isimlere cevirir.
# app.py ve rag_engine.py tarafindan ortak kullanilir (eskiden iki dosyada
# birebir kopyalanmisti).
SOURCE_DISPLAY_NAMES = {
    "411.1y_iuc-onlisans-ve-lisans-egitim-ogretim-yonetmeligi": "Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği",
    "iu-cerrahpasa-onlisans-ve-lisans-yonetmeligi-web": "Önlisans ve Lisans Yönetmeliği (Web)",
    "411.3y_iuc-cift-anadal-programi-yonergesi": "Çift Anadal Programı Yönergesi",
    "411.4y_iuc-yandal-programi-yonergesi": "Yandal Programı Yönergesi",
    "411.14y_iuc-lisans-staj-yonergesi": "Lisans Staj Yönergesi",
    "411.17y_iuc-onlisans-staj-yonergesi": "Önlisans Staj Yönergesi",
    "411.15y_iuc-hastalik-raporlari-yonergesi": "Hastalık Raporları Yönergesi",
    "411.21y_iuc-on-lisans-ve-lisans-duzeyindeki-programlar-arasinda": "Yatay Geçiş Esaslarına İlişkin Yönerge",
    "411.13y_iuc-intibak-ve-muafiyet-islemleri-yonergesi": "İntibak ve Muafiyet İşlemleri Yönergesi",
    "sss_manuel": "Sıkça Sorulan Sorular",
    "ogrenci.iuc.edu.tr_tr_content_sss_": "Sıkça Sorulan Sorular (Web)",
    "1.5.2547": "2547 Sayılı Yükseköğretim Kanunu",
    "Yaz Okulu Duyurusu": "Yaz Okulu Duyurusu",
    "2025-dgs-kayit-kilavuzu": "DGS Kayıt Kılavuzu",
    "411.6y_iuc-mufredat-guncel": "Müfredat Güncelleme ve İntibak Esasları",
    "cap-yonerge-senatoda-kabul-edilen": "Çift Anadal Programı Yönergesi (Senato)",
}


def get_display_name(source):
    """Kaynak dosya adini okunabilir bir isme cevirir. Eslesme yoksa, temizlenmis dosya adini dondurur."""
    cleaned = source.replace("f=", "").replace(".pdf", "").replace(".html", "")
    for key, display_name in SOURCE_DISPLAY_NAMES.items():
        if key in cleaned:
            return display_name

    uuid_pattern = r'^[0-9a-f]{8}[\s\-][0-9a-f]{4}[\s\-][0-9a-f]{4}[\s\-][0-9a-f]{4}[\s\-][0-9a-f]{12}$'
    if re.match(uuid_pattern, cleaned, re.IGNORECASE):
        return "Üniversite Belgesi"

    fallback = cleaned.split("_")[0].replace("-", " ")
    return fallback[:60]


# Akademik takvim PDF'lerini tespit etmek icin kullanilan pattern'ler.
# academic_calendar.py bu patternlere uyan PDF'leri data/raw/pdfs icinde
# otomatik bulur. Eskiden TAKVIM_PDFS adinda elle yazilmis, kirilgan bir
# dosya adi listesi vardi; ayrica rag_engine.py'de bu amaçla tanimlanmis
# ama hicbir yerde cagrilmayan (ve gercek dosya adlariyla da eslesmeyen)
# ayri bir CALENDAR_FILE_PATTERNS/is_calendar_source ciftine daha vardi.
# Asagidaki liste, gercekte kullanilan 4 takvim PDF'inin tum varyasyonlarini
# (ayrintili-takvim, ayrintili-akad.-takv, ozet-akademik-takvimi) kapsar.
CALENDAR_FILE_PATTERNS = [
    "akademik-takvim",
    "akademik_takvim",
    "ayrintili-akad",
    "ayrintili-takvim",
    "lisansustu-akademik",
    "ozet-akademik",
]


def is_calendar_source(source):
    source_lower = source.lower()
    return any(pattern in source_lower for pattern in CALENDAR_FILE_PATTERNS)


class RobustLLM:
    def __init__(self, temperature=0.0):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        self.temperature = temperature
        self.keys = []
        
        # Olası tüm GROQ_API_KEY varyasyonlarını topla
        base_key = os.getenv("GROQ_API_KEY")
        if base_key and base_key.strip():
            self.keys.append(base_key.strip())
            
        for i in range(2, 10):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k and k.strip():
                self.keys.append(k.strip())
                
        # Çift anahtarları temizle
        self.keys = list(dict.fromkeys(self.keys))
        
        # Uygulama boyunca dönen ortak bir indeks (class seviyesinde değil, instance seviyesinde tutuyoruz,
        # ancak kalıcı bir rotator için global bir pointer da eklenebilir. Basitlik için sırayla deneyecek).
        self.current_key_idx = 0

    def invoke(self, prompt):
        from langchain_ollama import OllamaLLM
        import time
        
        # Eger anahtar varsa Groq uzerinden donerek dene
        if self.keys:
            try:
                from langchain_groq import ChatGroq
            except ImportError:
                print("[Rotator] Uyari: langchain_groq kütüphanesi bulunamadı! Groq atlanıp Ollama'ya geçilecek.")
                self.keys = []
                
        if self.keys:
            attempts = len(self.keys)
            for attempt in range(attempts):
                current_key = self.keys[self.current_key_idx]
                try:
                    llm = ChatGroq(
                        api_key=current_key,
                        model_name="llama-3.1-8b-instant",
                        temperature=self.temperature
                    )
                    return llm.invoke(prompt)
                except Exception as e:
                    error_str = str(e).lower()
                    print(f"\n[Rotator] Groq API Key {self.current_key_idx + 1} HATA verdi! Detay: {error_str[:80]}...")
                    
                    # Hata rate limit ise veya token bitmis ise siradaki key'e gec
                    if "429" in error_str or "rate limit" in error_str or "tokens" in error_str or "too many requests" in error_str:
                        self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
                        print(f"[Rotator] -> Limit Doldu! Siradaki API Anahtarina geciliyor (Anahtar No: {self.current_key_idx + 1})...")
                        time.sleep(1) # Cok hizli istekleri onlemek icin kucuk bekleme
                        continue
                    else:
                        # Bilinmeyen bir sunucu hatasi ise donguyu kirip Ollama'ya dus
                        print("[Rotator] -> Rate limit disi bir hata! Ollama (Yerel) motora gecilecek.")
                        break

        # Tum anahtarlar tukendi veya hic anahtar yok veya internet yok.
        print("\n[Fallback] Hibrit Sistem Devrede: Groq devre disi, OLLAMA (gemma3:4b) motoruna yonlendiriliyor...")
        try:
            ollama = OllamaLLM(model="gemma3:4b", temperature=self.temperature)
            return ollama.invoke(prompt)
        except Exception as e:
            # Eger bilgisayarda Ollama acik degilse veya coktu ise kullaniciyi bilgilendir.
            class DummyResponse:
                def __init__(self, content):
                    self.content = content
            return DummyResponse(f"⚠️ **Sistem Hatası:** Bulut API sunucuları (Groq) limitlere ulaştı ve yerel destek motoru (Ollama) başlatılamadı.\n\nLütfen bir süre bekleyin veya API anahtarlarınızı güncelleyin. (Ollama Hatası: {str(e)[:100]})")

def get_llm(temperature=0.0):
    return RobustLLM(temperature=temperature)
