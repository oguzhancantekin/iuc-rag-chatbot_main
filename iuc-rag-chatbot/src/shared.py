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
