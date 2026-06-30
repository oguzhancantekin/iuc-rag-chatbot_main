import streamlit as st
import pickle
import os
import sys
import time
import json
import requests
import base64
from datetime import datetime
import streamlit.components.v1 as components

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import VECTORDB_DIR, BASE_DIR
from shared import get_display_name

# NOT: SOURCE_DISPLAY_NAMES ve get_display_name() burada rag_engine.py ile
# birebir kopya halindeydi (iki dosyada ayni sozluk, ayni fonksiyon).
# Artik shared.py'den import ediliyor; yeni bir kaynak dosyasi eklendiginde
# tek yerde guncelleme yeterli.

API_URL = "http://localhost:8000"
FEEDBACK_FILE = os.path.join(BASE_DIR, "data", "feedback.json")

st.set_page_config(
    page_title="İÜC Akademik Asistan",
    page_icon="🏛️",
    layout="wide"
)

# Logo yolunu belirle
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.png")

# Watermark için logoyu base64'e çevir
base64_logo = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as image_file:
        base64_logo = base64.b64encode(image_file.read()).decode("utf-8")

# ── Dark Mode State & CSS Injection (FOUC Önleyici En Üst Blok) ──
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

if dark:
    bg_primary = "#050814"
    bg_card = "rgba(15, 25, 45, 0.4)"
    text_color = "#e0e0e0"
    border_color = "rgba(212, 175, 55, 0.25)"
    user_bubble_bg = "linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(212, 175, 55, 0.05) 100%)"
    user_bubble_text = "#f8f0d8"
    assistant_bubble_bg = "linear-gradient(135deg, rgba(20, 40, 80, 0.4) 0%, rgba(10, 20, 50, 0.2) 100%)"
    assistant_bubble_text = "#dbe4f0"
    stat_bg = "rgba(15, 25, 45, 0.5)"
    input_bg = "rgba(10, 15, 25, 0.95)"
    input_text = "#ffffff"
    watermark_opacity = "0.08"
else:
    bg_primary = "#f4f6f9"
    bg_card = "rgba(255, 255, 255, 0.7)"
    text_color = "#111111"
    border_color = "rgba(15, 32, 75, 0.2)"
    user_bubble_bg = "linear-gradient(135deg, rgba(212, 175, 55, 0.2) 0%, rgba(212, 175, 55, 0.05) 100%)"
    user_bubble_text = "#111111"
    assistant_bubble_bg = "linear-gradient(135deg, rgba(15, 32, 75, 0.08) 0%, rgba(15, 32, 75, 0.03) 100%)"
    assistant_bubble_text = "#111111"
    stat_bg = "rgba(255, 255, 255, 0.6)"
    input_bg = "rgba(255, 255, 255, 0.95)"
    input_text = "#111111"
    watermark_opacity = "0.15"

st.markdown(f"""
<style>
    /* Filigran (Yeni HTML Metodu) */
    .bg-watermark {{
        position: absolute;
        top: 50vh;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 50%;
        max-width: 500px;
        opacity: {watermark_opacity};
        z-index: 0;
        pointer-events: none;
    }}
    
    /* Global Themes */
    {"" if not dark else '''
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"], .stApp, [role="dialog"], [data-testid="stDialog"], [data-testid="stModal"] { background-color: #050814 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: rgba(8, 12, 24, 0.8) !important; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp div, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [role="dialog"] p, [role="dialog"] span, [role="dialog"] label, [role="dialog"] div { color: #e0e0e0 !important; }
    '''}
    {"" if dark else '''
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"], .stApp, [role="dialog"], [data-testid="stDialog"], [data-testid="stModal"] { background-color: #f4f6f9 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: rgba(230, 235, 240, 0.9) !important; border-right: 1px solid rgba(0, 0, 0, 0.05); }
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp div, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [role="dialog"] p, [role="dialog"] span, [role="dialog"] label, [role="dialog"] div { color: #111111 !important; }
    '''}
    
    /* Sayfanın Kaymasını (Scroll) Önlemek İçin Üst Boşluğu Silme */
    .block-container {{
        padding-top: 3.5rem !important;
        padding-bottom: 0rem !important;
    }}

    /* Toggle Switch Rengi (SARI KUTU KESİN ÇÖZÜM) */
    div[data-testid="stToggle"] {{
        background-color: {"rgba(255, 255, 255, 0.05)" if dark else "#D4AF37"} !important;
        padding: 10px 15px !important;
        border-radius: 12px !important;
        border: 2px solid {"rgba(212, 175, 55, 0.5)" if dark else "#000000"} !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }}
    div[data-testid="stToggle"] > label {{ color: {"rgba(255, 255, 255, 0.9)" if dark else "#000000"} !important; font-weight: 600 !important; }}
    
    /* Streamlit Rerun Fading (Silikleşme/Titreme) İptali - ÇOK AGRESIF KONTROL */
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stHeader"],
    [data-testid="stSidebar"],
    .block-container,
    .stApp,
    div[class*="st-emotion-cache"] {{
        opacity: 1 !important;
        transition: none !important;
        filter: none !important;
    }}
    
    /* Gerçek Zamanlı Akıcı Düşünme Animasyonu */
    @keyframes pulse {{
        0% {{ opacity: 0.5; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.5; }}
    }}
    .thinking-realtime {{
        color: #888;
        font-style: italic;
        padding: 15px;
        font-size: 0.95em;
        animation: pulse 1.5s infinite ease-in-out;
    }}
    
    /* ====== SELECTBOX (MODEL SEÇİMİ) VE DROPDOWN ====== */
    .stSelectbox > div[data-baseweb="select"] {{ background-color: {input_bg} !important; border: 1px solid {border_color} !important; border-radius: 8px !important; }}
    .stSelectbox > div[data-baseweb="select"] * {{ color: {input_text} !important; background-color: transparent !important; }}
    div[data-baseweb="popover"], ul[role="listbox"] {{ background-color: {input_bg} !important; border: 1px solid {border_color} !important; border-radius: 8px !important; }}
    li[role="option"] {{ background-color: {input_bg} !important; color: {input_text} !important; }}
    li[role="option"]:hover {{ background-color: rgba(212, 175, 55, 0.2) !important; }}
    
    /* ====== CHAT INPUT (ARAMA YERİ) ====== */
    [data-testid="stChatInput"] {{ background-color: transparent !important; }}
    [data-testid="stChatInput"] * {{ background-color: transparent !important; }}
    [data-testid="stChatInput"] > div {{ background-color: {input_bg} !important; border: 1px solid {border_color} !important; border-radius: 12px !important; overflow: hidden !important; }}
    [data-testid="stChatInput"] textarea {{ color: {input_text} !important; }}
    [data-testid="stChatInput"] textarea::placeholder {{ color: {text_color} !important; opacity: 0.8 !important; }}

    /* ====== TEXT INPUT & TEXT AREA (FORM GİRDİLERİ) ====== */
    [data-testid="stTextInput"] div[data-baseweb="base-input"], [data-testid="stTextArea"] div[data-baseweb="base-input"],
    [data-testid="stTextInput"] div[data-baseweb="input"], [data-testid="stTextArea"] div[data-baseweb="textarea"] {{ background-color: {input_bg} !important; border-color: {border_color} !important; }}
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {{ color: {input_text} !important; background-color: {input_bg} !important; border-radius: 8px !important; }}
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder {{ color: {text_color} !important; opacity: 0.6 !important; }}

    /* ====== BEĞEN/BEĞENMEME VE KENAR ÇUBUĞU BUTONLARI (KIND="SECONDARY") ====== */
    div[data-testid="stButton"] button[kind="secondary"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: 1px solid rgba(150,150,150,0.6) !important;
        border-radius: 20px !important;
        color: {text_color} !important;
        box-shadow: none !important;
        padding: 4px 12px !important;
        min-height: 0px !important;
        height: auto !important;
        transition: transform 0.2s ease, background-color 0.2s ease, border-color 0.2s ease !important;
    }}
    div[data-testid="stButton"] button[kind="secondary"] * {{
        color: {text_color} !important;
    }}
    div[data-testid="stButton"] button[kind="secondary"]:hover {{
        background: rgba(150,150,150,0.1) !important;
        background-color: rgba(150,150,150,0.1) !important;
        border-color: rgba(212, 175, 55, 0.7) !important;
        transform: translateY(-3px) !important;
    }}

    /* ====== ANA BUTONLAR (ÖNERİLEN SORULAR - PILL DESIGN) ====== */
    div[data-testid="stButton"] button[kind="primary"], 
    div.stDownloadButton > button {{
        background: {"rgba(25, 35, 55, 0.6)" if dark else "rgba(255, 255, 255, 0.8)"} !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid {"rgba(212, 175, 55, 0.4)" if dark else "rgba(15, 32, 75, 0.2)"} !important;
        color: {text_color} !important;
        border-radius: 30px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    
    /* Örnek Soru Butonlarını Kusursuz Eşit Boyuta Getirme (GARANTİLİ YÖNTEM) */
    div[data-testid="stButton"] button[kind="primary"] {{
        height: 65px !important;
        min-height: 65px !important;
        max-height: 65px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: normal !important; 
        overflow: hidden !important;
        padding: 5px 15px !important;
        line-height: 1.2 !important;
        word-break: break-word !important;
        margin-bottom: 5px !important;
        font-size: 0.9rem !important;
    }}
    div[data-testid="stButton"] button[kind="primary"] p {{
        margin: 0 !important;
        text-align: center !important;
        display: -webkit-box !important;
        -webkit-line-clamp: 2 !important;
        -webkit-box-orient: vertical !important;
        overflow: hidden !important;
    }}
    div[data-testid="stButton"] button[kind="primary"] div[data-testid="stMarkdownContainer"] {{
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    /* Sol Menüdeki Butonların (Örn: Sohbeti Temizle) Boyutunu Normale Döndür */
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {{
        height: 45px !important;
        min-height: 45px !important;
        max-height: 45px !important;
        padding: 8px 16px !important;
    }}
    
    div[data-testid="stButton"] button[kind="primary"] *, 
    div.stDownloadButton > button * {{ color: {text_color} !important; }}
    
    div[data-testid="stButton"] button[kind="primary"]:hover, 
    div.stDownloadButton > button:hover {{
        background: {"rgba(212, 175, 55, 0.15)" if dark else "rgba(15, 32, 75, 0.05)"} !important;
        border: 1px solid {"rgba(212, 175, 55, 0.8)" if dark else "rgba(15, 32, 75, 0.5)"} !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1) !important;
        transform: translateY(-3px) !important;
    }}
    div[data-testid="stButton"] button[kind="primary"]:active, 
    div.stDownloadButton > button:active {{
        transform: scale(0.97) !important;
    }}
    
    /* ====== PREMIUM HERO SECTION (KARŞILAMA EKRANI) ====== */
    .hero-section {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1rem 1rem 1rem;
        text-align: center;
        animation: fadeIn 1s ease-out;
    }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .hero-logo {{
        width: 100px;
        margin-bottom: 1rem;
        filter: {"drop-shadow(0 0 15px rgba(212, 175, 55, 0.3))" if dark else "drop-shadow(0 0 15px rgba(15, 32, 75, 0.2))"};
    }}
    /* Sol Menüdeki Logo Parlaması */
    [data-testid="stSidebar"] img {{
        filter: {"drop-shadow(0 0 15px rgba(212, 175, 55, 0.3))" if dark else "drop-shadow(0 0 15px rgba(0, 71, 171, 0.3))"};
        transition: filter 0.3s ease;
    }}
    [data-testid="stSidebar"] img:hover {{
        filter: {"drop-shadow(0 0 25px rgba(212, 175, 55, 0.6))" if dark else "drop-shadow(0 0 25px rgba(0, 71, 171, 0.7))"};
    }}
    /* Streamlit varsayılan tam ekran (fullscreen) butonunu logodan gizle */
    [data-testid="stSidebar"] [data-testid="StyledFullScreenButton"],
    [data-testid="stSidebar"] button[title="View fullscreen"] {{
        display: none !important;
    }}
    .hero-title {{
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: {"linear-gradient(135deg, #D4AF37 0%, #F3E5AB 100%)" if dark else "linear-gradient(135deg, #0F204B 0%, #2A437C 100%)"};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 !important;
        letter-spacing: -1px;
        height: 90px; /* Yazının tek veya iki satır olmasına karşı sabit alan */
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .hero-subtitle {{
        font-size: 1.1rem;
        color: {text_color};
        opacity: 0.8;
        max-width: 650px;
        margin: 1rem auto 1.5rem auto;
        line-height: 1.5;
    }}
    
    /* Chat & Stats */
    .stat-card {{ background: {stat_bg}; border: 1px solid {border_color}; border-left: 4px solid #D4AF37; border-radius: 16px; padding: 1.2rem; text-align: center; margin-bottom: 1rem; backdrop-filter: blur(10px); }}
    .stat-value {{ font-size: 1.8rem; font-weight: 800; color: #D4AF37; }}
    .stat-label {{ font-size: 0.85rem; opacity: 0.9; font-weight: 600; text-transform: uppercase; color: {text_color}; }}
    
    .user-bubble {{ background: {user_bubble_bg}; border: 1px solid {border_color}; padding: 18px 24px; border-radius: 24px 24px 4px 24px; color: {user_bubble_text} !important; margin-left: auto; max-width: 85%; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); font-size: 1.05rem; }}
    .assistant-bubble {{ background: {assistant_bubble_bg}; border: 1px solid {border_color}; padding: 18px 24px; border-radius: 24px 24px 24px 4px; color: {assistant_bubble_text} !important; line-height: 1.7; margin-right: auto; max-width: 90%; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); font-size: 1.05rem; }}
    [data-testid="stChatMessage"] {{ background-color: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }}
    
    /* ====== HOVER DROPDOWN MENÜ (Kaydırma Önleyici CSS Aktif) ====== */
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarUserContent"] {{
        overflow: visible !important;
    }}
    [data-testid="stSidebar"] {{
        z-index: 999999 !important;
    }}
    .hover-dropdown {{
        position: relative;
        display: inline-block;
        width: 100%;
        margin-bottom: 1rem;
    }}
    .hover-dropdown-btn {{
        background-color: transparent;
        color: {text_color};
        padding: 14px 16px;
        font-size: 1.05rem;
        font-weight: 600;
        border: 1px solid {border_color};
        border-left: 4px solid #D4AF37;
        border-radius: 8px;
        cursor: pointer;
        width: 100%;
        text-align: left;
        transition: 0.3s;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .hover-dropdown-btn:hover {{
        background: {"rgba(212, 175, 55, 0.1)" if dark else "rgba(15, 32, 75, 0.05)"};
    }}
    .hover-dropdown-content {{
        display: none;
        position: absolute;
        top: 0;
        left: calc(100% + 5px); /* Menü temiz bir şekilde sağda, 5px estetik boşlukla açılır */
        width: 220px; 
        background: {bg_primary};
        box-shadow: 8px 8px 24px rgba(0,0,0,0.3);
        z-index: 9999999 !important;
        border: 1px solid {border_color};
        border-radius: 8px;
        margin-left: 0px; 
        overflow: visible; 
    }}
    /* GÖRÜNMEZ KÖPRÜ: Görsel olarak butonla menü arasında boşluk olsa da, farenin menüyü kaybetmemesi için görünmez bir alan. Streamlit resizer'ın üzerine biner. */
    .hover-dropdown-content::before {{
        content: "";
        position: absolute;
        top: -20px;
        left: -40px; /* 40 piksel geriye uzanır, hem boşluğu hem çubuğu kaplar */
        width: 40px;
        height: calc(100% + 40px);
        background: transparent;
        z-index: 9999999 !important; 
    }}
    .hover-dropdown-content a {{
        color: {text_color};
        padding: 14px 16px;
        text-decoration: none;
        display: block;
        font-size: 0.95rem;
        border-bottom: 1px solid {border_color};
        transition: background-color 0.2s, color 0.2s;
    }}
    .hover-dropdown-content a:last-child {{
        border-bottom: none;
    }}
    .hover-dropdown-content a:hover {{
        background-color: {"rgba(212, 175, 55, 0.15)" if dark else "rgba(15, 32, 75, 0.1)"};
        color: {"#D4AF37" if dark else "#0F204B"} !important;
        text-decoration: none;
    }}
    .hover-dropdown:hover .hover-dropdown-content {{
        display: block;
        animation: fadeInLeft 0.3s ease;
    }}
    /* Yan taraftan açılma animasyonu */
    @keyframes fadeInLeft {{
        from {{ opacity: 0; transform: translateX(-10px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
</style>
""", unsafe_allow_html=True)











# ── Geri Bildirim (RLHF) Yardımcı Fonksiyonları ──
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_feedback(question, answer, rating):
    feedbacks = load_feedback()
    feedbacks.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer[:500],  # Cevabın ilk 500 karakteri
        "rating": rating  # "positive" veya "negative"
    })
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, ensure_ascii=False, indent=2)

def get_feedback_stats():
    feedbacks = load_feedback()
    total = len(feedbacks)
    positive = sum(1 for f in feedbacks if f["rating"] == "positive")
    negative = total - positive
    return total, positive, negative

# ── Sohbet Dışa Aktarma ──
def export_chat_as_txt():
    if not st.session_state.get("messages"):
        return ""
    lines = ["İÜC Akademik Asistan — Sohbet Geçmişi", f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "=" * 50, ""]
    for msg in st.session_state.messages:
        role = "🧑 Siz" if msg["role"] == "user" else "🤖 Asistan"
        lines.append(f"{role}:")
        lines.append(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            sources_clean = [get_display_name(s) for s in msg["sources"] if s]
            if sources_clean:
                lines.append(f"  📚 Kaynaklar: {', '.join(sources_clean)}")
        lines.append("")
    return "\n".join(lines)

@st.cache_resource
def get_chunk_count():
    try:
        with open(os.path.join(VECTORDB_DIR, "chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
        return len(chunks)
    except:
        return 0


def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json().get("message", "")
        return False, f"API Hata Kodu: {response.status_code}"
    except Exception as e:
        return False, "API Sunucusu Ayakta Değil."

def process_query_via_api(query, model_choice, temperature, chat_history):
    start_time = time.time()
    payload = {
        "query": query,
        "model_choice": model_choice,
        "temperature": temperature,
        "chat_history": chat_history
    }
    try:
        # Timeout 120 saniye. Eger model cok yavas yanit verirse veya cokerse 
        # asagidaki except bloguna dusecek.
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
        else:
            result = {
                "answer": f"⚠️ **Sistem Uyarı:** API Sunucusu beklenmeyen bir hata döndürdü (Kod {response.status_code}). Lütfen tekrar deneyin.",
                "sources": [],
                "chunks": [],
                "elapsed": 0.0
            }
    except requests.exceptions.Timeout:
        result = {
            "answer": "⏳ **Zaman Aşımı (Timeout):** Model şu anda aşırı yoğun veya sistem yanıt veremeyecek kadar yavaş çalışıyor. Lütfen sorunuzu basitleştirerek tekrar deneyin veya farklı bir model seçin.",
            "sources": [],
            "chunks": [],
            "elapsed": 120.0
        }
    except requests.exceptions.ConnectionError:
        result = {
            "answer": "🔌 **Bağlantı Hatası:** API sunucusuna ulaşılamıyor. FastAPI sunucusunun (uvicorn) arka planda çalıştığından emin olun.",
            "sources": [],
            "chunks": [],
            "elapsed": 0.0
        }
    except Exception as e:
        result = {
            "answer": f"🛠️ **Teknik Hata:** API sunucusuna bağlanırken teknik bir sorun oluştu. (Detay: {str(e)[:150]})",
            "sources": [],
            "chunks": [],
            "elapsed": 0.0
        }
        
    if "elapsed" not in result or result["elapsed"] == 0.0:
        result["elapsed"] = time.time() - start_time
    return result

def process_query_via_api_stream(query, model_choice, temperature, chat_history, message_placeholder):
    start_time = time.time()
    payload = {
        "query": query,
        "model_choice": model_choice,
        "temperature": temperature,
        "chat_history": chat_history
    }
    
    result = {
        "answer": "",
        "sources": [],
        "chunks": [],
        "elapsed": 0.0,
        "engine": "API"
    }
    
    try:
        response = requests.post(f"{API_URL}/ask_stream", json=payload, stream=True, timeout=120)
        
        if response.status_code != 200:
            result["answer"] = f"⚠️ **Sistem Uyarı:** API Sunucusu hata döndürdü (Kod {response.status_code})."
            message_placeholder.markdown(f'<div class="assistant-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
            return result
            
        full_answer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "meta":
                            result["sources"] = data.get("sources", [])
                            result["chunks"] = data.get("chunks", [])
                            result["engine"] = data.get("engine", "API")
                            # RAG taraması bitti, LLM başladı (Gerçek State 2)
                            message_placeholder.markdown('<div class="thinking-realtime">🧠 İlgili maddeler yapay zeka tarafından sentezleniyor...</div>', unsafe_allow_html=True)
                        elif data.get("type") == "chunk":
                            full_answer += data.get("content", "")
                            # Cursor (▌) efekti ile anlik guncelle
                            message_placeholder.markdown(f'<div class="assistant-bubble">{full_answer} ▌</div>', unsafe_allow_html=True)
                    except json.JSONDecodeError:
                        pass
        
        # Final tam halini bas
        message_placeholder.markdown(f'<div class="assistant-bubble">{full_answer}</div>', unsafe_allow_html=True)
        result["answer"] = full_answer
        
    except requests.exceptions.Timeout:
        result["answer"] = "⏳ **Zaman Aşımı (Timeout):** API sunucusu yanıt vermedi."
        message_placeholder.markdown(f'<div class="assistant-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
    except requests.exceptions.ConnectionError:
        result["answer"] = "🔌 **Bağlantı Hatası:** API sunucusuna ulaşılamıyor."
        message_placeholder.markdown(f'<div class="assistant-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
    except Exception as e:
        result["answer"] = f"🛠️ **Teknik Hata:** {str(e)[:150]}"
        message_placeholder.markdown(f'<div class="assistant-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
        
    result["elapsed"] = time.time() - start_time
    return result

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def trigger_example(q):
    st.session_state.trigger_query = q

# ── Geri Bildirim Callback'leri ──
def handle_feedback(msg_index, rating):
    msgs = st.session_state.messages
    if msg_index < len(msgs):
        msg = msgs[msg_index]
        # Bir önceki mesaj kullanıcının sorusu
        question = ""
        if msg_index > 0:
            question = msgs[msg_index - 1].get("content", "")
        save_feedback(question, msg["content"], rating)
        if "feedback_given" not in st.session_state:
            st.session_state.feedback_given = set()
        st.session_state.feedback_given.add(msg_index)

# API Baglantı Kontrolü
api_alive, api_msg = check_api_health()
if not api_alive:
    st.markdown("""
    <div class="main-header">
        <h1>🎓 İÜC Akademik Asistan</h1>
        <p>İstanbul Üniversitesi-Cerrahpaşa · Yapay Zeka Destekli Bilgi Sistemi</p>
    </div>
    """, unsafe_allow_html=True)
    st.error("⚠️ **İÜC RAG API Sunucusu Bağlantı Hatası!**")
    st.info(f"""
    Uygulamanın çalışabilmesi için önce API sunucusunu başlatmanız gerekmektedir.
    
    **API Sunucusunu Başlatmak İçin:**
    Proje kök dizininde yeni bir terminal açın ve aşağıdaki komutu çalıştırın:
    ```bash
    uvicorn iuc-rag-chatbot.src.api:app --reload --port 8000
    ```
    API başarıyla başladıktan sonra bu sayfayı yenileyin. (Hata Detayı: {api_msg})
    """)
    st.stop()

COMPLAINTS_FILE = os.path.join(BASE_DIR, "data", "istek_sikayet.json")

def save_complaint(konu, email, mesaj):
    os.makedirs(os.path.dirname(COMPLAINTS_FILE), exist_ok=True)
    data = []
    if os.path.exists(COMPLAINTS_FILE):
        try:
            with open(COMPLAINTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
            
    data.append({
        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "konu": konu,
        "email": email,
        "mesaj": mesaj
    })
    
    with open(COMPLAINTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

@st.dialog("İstek, Şikayet ve Öneri Formu")
def feedback_dialog():
    st.markdown("""
    <div style='text-align: center; padding: 15px; margin-bottom: 20px; background: rgba(212, 175, 55, 0.1); border-radius: 10px; border: 1px solid rgba(212, 175, 55, 0.3); box-shadow: 0 4px 10px rgba(0,0,0,0.05);'>
        <h3 style='margin-bottom: 5px; color: #D4AF37; font-weight: 700; font-size: 1.2rem;'>Üniversitemizi Birlikte Geliştirelim</h3>
        <p style='font-size: 0.9rem; margin-bottom: 0; opacity: 0.9;'>Görüşleriniz, kampüs yaşantısını ve dijital sistemlerimizi iyileştirmek için çok değerlidir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        konu = st.selectbox("📌 Bildirim Türü", ["💡 Yeni Bir Öneri", "⚠️ Sistem Şikayeti", "🔧 Teknik Bir Hata", "🎓 Üniversite İşleyişi", "Diğer"])
    with col2:
        email = st.text_input("✉️ E-posta Adresiniz", placeholder="İletişim için (İsteğe bağlı)", help="Size geri dönüş yapabilmemiz için e-posta adresinizi bırakabilirsiniz.")
        
    mesaj = st.text_area("✍️ Mesajınız", placeholder="Karşılaştığınız sorunu veya aklınızdaki harika fikri detaylıca anlatın...", height=150)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    btn_col1, btn_col2, btn_col3 = st.columns([1,2,1])
    with btn_col2:
        if st.button("🚀 Formu Gönder", use_container_width=True, type="primary"):
            if len(mesaj) < 10:
                st.error("Lütfen konuyu biraz daha detaylı açıklayın. (En az 10 karakter)")
            else:
                save_complaint(konu, email, mesaj)
                st.markdown("""
                <div style='text-align: center; padding: 15px; background: rgba(40, 167, 69, 0.1); border-radius: 10px; border: 1px solid rgba(40, 167, 69, 0.3); margin-top: 10px;'>
                    <h4 style='color: #28a745; margin: 0;'>✅ Teşekkürler!</h4>
                    <p style='font-size: 0.9rem; margin-top: 5px; margin-bottom: 0;'>Bildiriminiz ilgili birimlere başarıyla iletilmiştir.</p>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(2.5)
                st.rerun()

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown(f'<img src="data:image/png;base64,{base64_logo}" style="width: 130px; display: block; margin: -15px auto 15px auto;">', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <div style='font-weight: 800; font-size: 1.25rem; letter-spacing: 1.5px; color: #D4AF37;'>İÜC ASİSTAN</div>
        <div style='font-size: 0.75rem; opacity: 0.6; letter-spacing: 0.5px; margin-top: 2px;'>Yapay Zeka Destekli</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Arka plan için sabit ayarlar
    model_choice = "gemma3:4b"
    temperature = 0.1

    # 🌙 Dark Mode Toggle (Pill Tasarımı ile)
    st.markdown("<small style='opacity:0.7;'>⚠️ Soru yanıtlanırken tema değiştirmeyin.</small>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        dark_toggle = st.toggle("🌙 Tema", value=st.session_state.dark_mode, key="dark_toggle")
        if dark_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_toggle
            st.rerun()
    with col_t2:
        if st.button("🗑️ Temizle", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.session_state.total_time = 0.0
            if "feedback_given" in st.session_state:
                del st.session_state.feedback_given
            st.rerun()
    
    # 📢 Güncel Duyurular (Hover Açılır Menü)
    st.markdown("""
    <div class="hover-dropdown">
        <button class="hover-dropdown-btn">
            <span>📢 Güncel Duyurular</span>
            <span style="font-size: 0.8rem; opacity: 0.8;">▶</span>
        </button>
        <div class="hover-dropdown-content">
            <a href="https://www.iuc.edu.tr/tr/duyurular/1/1" target="_blank">🏛️ Üniversite Duyuruları</a>
            <a href="https://bilgisayarmuhendislik.iuc.edu.tr/tr/duyurular/1/1" target="_blank">💻 Bölüm Duyuruları</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 📰 Güncel Haberler (Hover Açılır Menü)
    st.markdown("""
    <div class="hover-dropdown">
        <button class="hover-dropdown-btn">
            <span>📰 Güncel Haberler</span>
            <span style="font-size: 0.8rem; opacity: 0.8;">▶</span>
        </button>
        <div class="hover-dropdown-content">
            <a href="https://www.iuc.edu.tr/tr/haberler/1" target="_blank">🏛️ Üniversite Haberleri</a>
            <a href="https://bilgisayarmuhendislik.iuc.edu.tr/tr/haberler/" target="_blank">💻 Bölüm Haberleri</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 🎓 Akademik Sistemler (Hover Açılır Menü)
    st.markdown("""
    <div class="hover-dropdown">
        <button class="hover-dropdown-btn">
            <span>🎓 Akademik Sistemler</span>
            <span style="font-size: 0.8rem; opacity: 0.8;">▶</span>
        </button>
        <div class="hover-dropdown-content">
            <a href="https://aksis.iuc.edu.tr/Account/LogOn?ReturnUrl=%2f" target="_blank">📊 AKSİS (Kayıt & Not)</a>
            <a href="https://canvas.iuc.edu.tr/login/ldap" target="_blank">📚 CANVAS (Ders & İçerik)</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("📝 Form", use_container_width=True):
            feedback_dialog()
            
    with col_b2:
        chat_txt = export_chat_as_txt()
        if chat_txt:
            st.download_button("📥 Aktar", data=chat_txt, file_name=f"iuc_sohbet_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain", use_container_width=True)
        else:
            st.button("📥 Aktar", disabled=True, use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

is_first_load = len(st.session_state.messages) == 0
hero_text = "Merhaba." if is_first_load else "Size nasıl yardımcı olabilirim?"

st.markdown(f"""
<img src="data:image/png;base64,{base64_logo}" class="bg-watermark">
<div class="hero-section">
    <img src="data:image/png;base64,{base64_logo}" class="hero-logo" alt="IUC Logo">
    <div class="hero-title" {'data-typing="first_load"' if is_first_load else ''}>{hero_text}</div>
    <p class="hero-subtitle">İstanbul Üniversitesi - Cerrahpaşa akademik takvimi, yönetmelikler ve yönergeler hakkında resmi kaynaklara dayalı net cevaplar sunuyorum.<br><br>
    <span style="font-weight: 600; color: #D4AF37; font-size: 0.95rem; letter-spacing: 2px; text-transform: uppercase;">#eniyiolmakiçin</span>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="text-align: center; opacity: 0.7; margin-bottom: 10px; font-size: 0.9rem;">💡 Aşağıdaki konulardan birini seçebilir veya kendi sorunuzu yazabilirsiniz:</div>', unsafe_allow_html=True)
example_questions = [
    "🎓 Çift anadal (ÇAP) şartları nelerdir?",
    "📅 Yatay geçiş başvuruları ne zaman?",
    "📝 Yaz okulunda en fazla kaç kredi alabilirim?",
    "⚖️ Mazeret sınavına kimler girebilir?",
    "🛑 Kayıt dondurma süresi ne kadar?",
    "🌟 Onur öğrencisi olmak için ne gerekir?",
    "📊 Derslere devam zorunluluğu yüzde kaçtır?",
    "❌ Danışman ders kaydını onaylamazsa ne olur?"
]

cols = st.columns(4)
for i, question in enumerate(example_questions):
    with cols[i % 4]:
        st.button(question, on_click=trigger_example, args=(question,), key=f"btn_{i}", use_container_width=True, type="primary")

for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = "🎓" if role == "user" else f"data:image/png;base64,{base64_logo}"
    with st.chat_message(role, avatar=avatar):
        if role == "user":
            st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            if "elapsed" in message:
                engine = message.get("engine", "API")
                st.markdown(f"""<div class="timing-badge">⏱️ {message['elapsed']:.2f}s · {engine} + Hibrit Arama</div>""", unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                sources_clean = [s for s in message["sources"] if s]
                if sources_clean:
                    with st.expander("📚 Kaynaklar"):
                        for source in sources_clean:
                            clean_name = get_display_name(source)
                            st.markdown(f"📄 **{clean_name}**")
            # 👍/👎 Geri Bildirim Butonları
            if idx not in st.session_state.feedback_given:
                st.markdown("<div style='margin-left: 10%; margin-bottom:10px;'>", unsafe_allow_html=True)
                f_col1, f_col2, _ = st.columns([1,1,8])
                with f_col1:
                    st.button("👍", key=f"up_{idx}", on_click=handle_feedback, args=(idx, 1))
                with f_col2:
                    st.button("👎", key=f"down_{idx}", on_click=handle_feedback, args=(idx, -1))
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="feedback-done">✅ Geri bildiriminiz kaydedildi!</div>', unsafe_allow_html=True)

user_query = st.chat_input("Sorunuzu yazın...")

if "trigger_query" in st.session_state:
    user_query = st.session_state.trigger_query
    del st.session_state.trigger_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🎓"):
        st.markdown(f'<div class="user-bubble">{user_query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar=f"data:image/png;base64,{base64_logo}"):
        message_placeholder = st.empty()
        # Gerçek zamanlı arama başlama durumu (State 1)
        message_placeholder.markdown('<div class="thinking-realtime">🔍 Akademik veritabanı taranıyor...</div>', unsafe_allow_html=True)
        
        result = process_query_via_api_stream(user_query, model_choice, temperature, st.session_state.chat_history, message_placeholder)

        engine = result.get("engine", "API")
        st.markdown(f"""<div class="timing-badge">⏱️ {result['elapsed']:.2f}s · {engine} + Hibrit Arama</div>""", unsafe_allow_html=True)

        if result["sources"]:
            sources_clean = [s for s in result["sources"] if s]
            if sources_clean:
                with st.expander("📚 Kaynaklar"):
                    for source in sources_clean:
                        clean_name = get_display_name(source)
                        st.markdown(f"📄 **{clean_name}**")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "elapsed": result["elapsed"],
            "engine": result.get("engine", "API")
        })
        st.session_state.chat_history.append({"user": user_query, "assistant": result["answer"]})
        st.session_state.total_queries += 1
        st.session_state.total_time += result["elapsed"]

        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
            
        # DOM u tam yenilemek (st.rerun) sayfanin titremesine yol aciyordu.
        # Rerun atmamak icin yeni uretilen cevabin butonlarini manuel olarak burada ciziyoruz.
        new_idx = len(st.session_state.messages) - 1
        new_idx = len(st.session_state.messages) - 1
        if new_idx not in st.session_state.feedback_given:
            fb_col1, fb_col2, fb_spacer = st.columns([1, 1, 8])
            with fb_col1:
                if st.button("👍", key=f"fb_pos_{new_idx}", help="Bu cevap faydalıydı"):
                    handle_feedback(new_idx, "positive")
            with fb_col2:
                if st.button("👎", key=f"fb_neg_{new_idx}", help="Bu cevap yetersizdi"):
                    handle_feedback(new_idx, "negative")

js_code = f"""
<script>
    const parentDoc = window.parent.document;
    
    // ====== TYPEWRITER (DAKTİLO) EFEKTİ ======
    const titleElement = parentDoc.querySelector(".hero-title");
    if(titleElement && titleElement.getAttribute('data-typing') === 'first_load') {{
        titleElement.removeAttribute('data-typing'); // Bir kez çalışması için flag'i sil
        
        const titles = ["Merhaba.", "Size nasıl yardımcı olabilirim?", "İÜC Asistan'a Hoş Geldiniz."];
        let titleIndex = 0;
        let charIndex = titles[0].length; // Starts full (Merhaba.)
        let isDeleting = true;
        
        // Start effect after 2 seconds
        setTimeout(() => {{ typeWriter(); }}, 2000);
        
        function typeWriter() {{
            const currentText = titles[titleIndex];
            if (isDeleting) {{
                titleElement.innerText = currentText.substring(0, charIndex - 1);
                charIndex--;
            }} else {{
                titleElement.innerText = currentText.substring(0, charIndex + 1);
                charIndex++;
            }}
            
            titleElement.style.borderRight = "3px solid #D4AF37";
            
            let typeSpeed = isDeleting ? 40 : 80;
            
            if (!isDeleting && charIndex === currentText.length) {{
                typeSpeed = 2500; // Bekleme
                isDeleting = true;
                
                // Animasyonu "Size nasıl yardımcı olabilirim?" yazısında tamamen bitir.
                if (titleIndex === 1) {{
                    titleElement.style.borderRight = "none";
                    return; // Fonksiyondan çık, loop'u bitir
                }}
            }} else if (isDeleting && charIndex === 0) {{
                isDeleting = false;
                titleIndex = (titleIndex + 1) % titles.length;
                typeSpeed = 600; // Silme bittikten sonra bekleme
            }}
            setTimeout(typeWriter, typeSpeed);
        }}
    }}
</script>
"""
components.html(js_code, height=0, width=0)
