import sys
import os
# GÜVENLİK KİLİDİ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import fitz
from config import PDF_DIR

# 2025-2026 akademik takvim PDF'leri
TAKVIM_PDFS = [
    "f=411.1.1f-on-lisans-lisans-ayrintili-takvim.pdf",
    "f=ilan-411.1.16-on-lisans-lisans-ozet-akademik-takvimi.pdf",
    "f=411.1.16f-206-2027-egitim-ogretim-yili-on-lisans-ve-lisans-ozet-akademik-takvimi.pdf",
    "f=411.1.1f-2026-2027-egi.-ogr.-yili-on-lis.-ve-lis.-ayrintili-akad.-takv.-formu.pdf",
]

def extract_calendar_text():
    texts = []
    for pdf_name in TAKVIM_PDFS:
        filepath = os.path.join(PDF_DIR, pdf_name)
        if os.path.exists(filepath):
            try:
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                texts.append({"file": pdf_name, "text": text})
            except Exception as e:
                pass
    return texts

def search_calendar(query, texts, chat_history=None):
    query_lower = query.lower()
    
    # 1. Aşama: Doğrudan sorudan dönem tespiti
    requested_donem = "bahar" if "bahar" in query_lower else "güz" if "güz" in query_lower else "yaz" if "yaz" in query_lower else None

    # 2. Aşama: SOHBET GEÇMİŞİ BİLİNCİ
    if not requested_donem and chat_history:
        for turn in reversed(chat_history):
            user_msg = turn.get("user", "").lower()
            ast_msg = turn.get("assistant", "").lower()
            if "bahar" in user_msg or "bahar" in ast_msg:
                requested_donem = "bahar"
                break
            elif "güz" in user_msg or "güz" in ast_msg:
                requested_donem = "güz"
                break
            elif "yaz" in user_msg or "yaz okulu" in ast_msg:
                requested_donem = "yaz"
                break

    keywords = {
        "vize": ["ara sınav", "vize", "midterm", "yıl içi etkinlikleri", "yıl içi sınav"],
        "final": ["final", "yarıyıl sonu sınav", "yıl sonu sınav", "bitirme sınav"],
        "bütünleme": ["bütünleme", "büt"],
        "kayıt": ["kayıt yenileme", "ders kaydı", "kayıt tarihi"],
        "başlangıç": ["eğitim başlangıç", "öğretime başlama", "yarıyıl başlangıç", "derslerin başlaması"],
        "tatil": ["tatil", "bayram", "resmi tatil", "ara tatil"],
    }

    # 🧠 YENİ NESİL ZIRHLAR (Filtreler)
    is_asking_hazirlik = "hazırlık" in query_lower
    hazirlik_words = ["hazırlık", "yabancı dil", "konuşma", "yazılı", "muafiyet", "seviye tespit"]

    is_asking_admin = any(w in query_lower for w in ["ilan", "sistem", "not", "giril"])
    admin_words = ["ilan", "otomasyon", "tanımlanma", "yedi (7)", "giril", "notunun", "teslim", "öğrenci işleri", "başarı durumu", "mazeret"]

    matched_keyword = None
    for key, variants in keywords.items():
        if any(v in query_lower for v in variants):
            matched_keyword = key
            break

    results = []
    aylar = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]

    for item in texts:
        lines = item["text"].split("\n")
        for i, line in enumerate(lines):
            line_strip = line.strip()
            line_lower = line_strip.lower()
            if not line_strip:
                continue

            # Anahtar kelime yakalama
            if matched_keyword and any(v in line_lower for v in keywords[matched_keyword]):
                
                context = line_strip
                
                # ADIM 1: Önce eksik tarihleri birleştir (Cümleyi tam haline getir)
                if not any(m in line_lower for m in aylar):
                    if i > 0 and any(m in lines[i-1].lower() for m in aylar):
                        context = lines[i-1].strip() + " — " + context
                    elif i < len(lines)-1 and any(m in lines[i+1].lower() for m in aylar):
                        context = context + " — " + lines[i+1].strip()
                
                context_lower = context.lower()

                # ADIM 2: ŞİMDİ FİLTRELE (Bağlam tamamen oluştuktan sonra!)
                # 🚫 İdari filtre
                if not is_asking_admin and any(admin_word in context_lower for admin_word in admin_words):
                    continue
                # 🚫 Hazırlık sınıfı filtresi (Öğrenci açıkça sormadıysa yoksay)
                if not is_asking_hazirlik and any(hw in context_lower for hw in hazirlik_words):
                    continue
                
                # ADIM 3: Dönemi Kesinleştir
                line_donem = None
                if "bahar" in context_lower:
                    line_donem = "bahar"
                elif "güz" in context_lower:
                    line_donem = "güz"
                elif "yaz" in context_lower:
                    line_donem = "yaz"
                else:
                    if any(m in context_lower for m in ["eylül", "ekim", "kasım", "aralık", "ocak"]):
                        line_donem = "güz"
                    elif any(m in context_lower for m in ["mart", "nisan", "mayıs", "haziran", "temmuz"]):
                        line_donem = "bahar"
                    elif "ağustos" in context_lower:
                        line_donem = "yaz"
                    elif "şubat" in context_lower:
                        if any(w in context_lower for w in ["bütünleme", "final", "bitirme", "yarıyıl sonu"]):
                            line_donem = "güz"
                        else:
                            line_donem = "bahar"

                if requested_donem and line_donem and requested_donem != line_donem:
                    continue

                if 10 < len(context) < 130:
                    display_donem = line_donem if line_donem else (requested_donem if requested_donem else "AKADEMİK")
                    clean_res = f"[{display_donem.upper()} DÖNEMİ] {context}"
                    results.append(clean_res.replace("\n", " "))

    unique_res = list(dict.fromkeys(results))
    return unique_res[:4]

def answer_calendar_query(query, chat_history=None):
    texts = extract_calendar_text()
    if not texts:
        return "Akademik takvim veritabanında bulunamadı."

    results = search_calendar(query, texts, chat_history)
    if not results:
        return "Bu konuyla ilgili akademik takvimde net bir tarih bulunamadı. (Sadece öğrencileri ilgilendiren ana sınav ve kayıt tarihleri listelenmektedir.) Lütfen dönemi belirterek tekrar sorunuz."

    response = "📅 **Akademik Takvim Sonuçları:**\n\n"
    for r in results:
        response += f"🔹 {r}\n"
        
    response += f"\n*(Kaynak: Akademik Takvim PDF)*"
    return response