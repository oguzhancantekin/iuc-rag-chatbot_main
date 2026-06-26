import streamlit as st
import pickle
import os
import sys
import time
import json
import requests
import base64
from datetime import datetime

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
    /* Filigran */
    [data-testid="stAppViewContainer"]::before {{
        content: ""; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/png;base64,{base64_logo}");
        background-repeat: no-repeat; background-position: center; background-size: 50%;
        opacity: {watermark_opacity}; z-index: 0; pointer-events: none;
    }}
    
    /* Global Themes */
    {"" if not dark else '''
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"], .stApp { background-color: #050814 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: rgba(8, 12, 24, 0.8) !important; border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp div, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #e0e0e0 !important; }
    '''}
    {"" if dark else '''
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"], .stApp { background-color: #f4f6f9 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: rgba(230, 235, 240, 0.9) !important; border-right: 1px solid rgba(0, 0, 0, 0.05); }
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp div, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #111111 !important; }
    '''}

    /* Toggle Switch Rengi (SARI KUTU KESİN ÇÖZÜM) */
    div[data-testid="stToggle"] {{
        background-color: {"rgba(255, 255, 255, 0.05)" if dark else "#D4AF37"} !important;
        padding: 10px 15px !important;
        border-radius: 12px !important;
        border: 2px solid {"rgba(212, 175, 55, 0.5)" if dark else "#000000"} !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }}
    div[data-testid="stToggle"] * {{
        color: {"#e0e0e0" if dark else "#000000"} !important;
        font-weight: 700 !important;
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

    /* ====== BEĞEN/BEĞENMEME BUTONLARI (KIND="SECONDARY") ====== */
    div[data-testid="stButton"] button[kind="secondary"] {{
        background: transparent !important;
        background-color: transparent !important;
        border: 1px solid rgba(150,150,150,0.6) !important;
        border-radius: 20px !important;
        color: {text_color} !important;
        box-shadow: none !important;
        padding: 4px 12px !important;
        transform: none !important;
        min-height: 0px !important;
        height: auto !important;
    }}
    div[data-testid="stButton"] button[kind="secondary"] * {{
        color: {text_color} !important;
    }}
    div[data-testid="stButton"] button[kind="secondary"]:hover {{
        background: rgba(150,150,150,0.1) !important;
        background-color: rgba(150,150,150,0.1) !important;
        border-color: rgba(150,150,150,0.8) !important;
    }}

    /* ====== ANA BUTONLAR (KIND="PRIMARY" ve DOWNLOAD BUTONU) ====== */
    div[data-testid="stButton"] button[kind="primary"], 
    div.stDownloadButton > button {{
        background: linear-gradient(135deg, rgba(15, 32, 75, 0.9) 0%, rgba(20, 40, 80, 0.8) 100%) !important;
        background-color: transparent !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    div[data-testid="stButton"] button[kind="primary"] *, 
    div.stDownloadButton > button * {{ color: #ffffff !important; }}
    
    div[data-testid="stButton"] button[kind="primary"]:hover, 
    div.stDownloadButton > button:hover {{
        background: linear-gradient(135deg, rgba(20, 40, 80, 1) 0%, rgba(30, 50, 100, 0.9) 100%) !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
        color: #D4AF37 !important;
        transform: translateY(-2px) !important;
    }}
    div[data-testid="stButton"] button[kind="primary"]:active, 
    div.stDownloadButton > button:active {{
        transform: scale(0.95) !important;
        box-shadow: inset 0 3px 8px rgba(0,0,0,0.4) !important;
    }}
    
    /* Cards & Headers */
    .main-header {{ background: linear-gradient(135deg, rgba(15, 32, 75, 0.8) 0%, rgba(26, 43, 76, 0.6) 100%); border-bottom: 3px solid #D4AF37; padding: 2rem; border-radius: 16px; margin-bottom: 2rem; color: white !important; }}
    .main-header h1 {{ color: #E5C158 !important; font-size: 2.2rem !important; margin: 0 !important; font-weight: 800 !important; }}
    .main-header p {{ color: rgba(255,255,255,0.85) !important; margin: 0.5rem 0 0 0 !important; font-size: 1.05rem !important; }}
    .stat-card {{ background: {stat_bg}; border: 1px solid {border_color}; border-left: 4px solid #D4AF37; border-radius: 12px; padding: 1.2rem; text-align: center; margin-bottom: 1rem; }}
    .stat-value {{ font-size: 1.8rem; font-weight: 800; color: #D4AF37; }}
    .stat-label {{ font-size: 0.85rem; opacity: 0.9; font-weight: 600; text-transform: uppercase; color: {text_color}; }}
    .welcome-box {{ background: {bg_card}; border-top: 4px solid #D4AF37; border: 1px solid {border_color}; border-radius: 16px; padding: 2rem; margin-bottom: 2rem; }}
    .feature-item {{ display: flex; align-items: center; gap: 0.8rem; margin: 0.8rem 0; font-size: 1.05rem; color: {text_color}; }}
    .user-bubble {{ background: {user_bubble_bg}; border: 1px solid {border_color}; border-left: 4px solid #D4AF37; padding: 15px 20px; border-radius: 15px 15px 0 15px; color: {user_bubble_text} !important; margin-left: 10%; margin-bottom: 15px; }}
    .assistant-bubble {{ background: {assistant_bubble_bg}; border: 1px solid {border_color}; border-left: 4px solid #4A90E2; padding: 15px 20px; border-radius: 15px 15px 15px 0; color: {assistant_bubble_text} !important; line-height: 1.6; margin-right: 10%; margin-bottom: 15px; }}
    [data-testid="stChatMessage"] {{ background-color: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }}
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

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(LOGO_PATH, width=130)
    st.markdown("---")
    st.markdown("## ⚙️ Ayarlar")
    model_choice = st.selectbox(
        "Model Seçimi",
        ["gemma3:4b", "llama3:8b", "phi3:3.8b"],
        index=0
    )
    temperature = st.slider("Sıcaklık (Creativity)", 0.0, 1.0, 0.1, 0.05)

    # 🌙 Dark Mode Toggle (Pill Tasarımı ile)
    st.markdown("<small style='opacity:0.7;'>⚠️ Soru yanıtlanırken tema değiştirmeyin.</small>", unsafe_allow_html=True)
    dark_toggle = st.toggle("🌙 Karanlık Mod", value=st.session_state.dark_mode, key="dark_toggle")
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Geçmişi Sil", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.session_state.total_time = 0.0
            if "feedback_given" in st.session_state:
                del st.session_state.feedback_given
            st.rerun()
    with col_btn2:
        chat_txt = export_chat_as_txt()
        if chat_txt:
            st.download_button(
                "📥 Dışa Aktar",
                data=chat_txt,
                file_name=f"iuc_sohbet_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.button("📥 Dışa Aktar", disabled=True, use_container_width=True)

    st.divider()
    st.markdown("### 📊 Sistem Metrikleri")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="stat-card"><div class="stat-value">FastAPI</div><div class="stat-label">Altyapı</div></div>""", unsafe_allow_html=True)
    with col2:
        chunk_count = get_chunk_count()
        st.markdown(f"""<div class="stat-card"><div class="stat-value">{chunk_count}</div><div class="stat-label">Chunk</div></div>""", unsafe_allow_html=True)

    if "total_queries" in st.session_state and st.session_state.total_queries > 0:
        st.divider()
        st.markdown("### 💬 Bu Oturum")
        avg_time = st.session_state.total_time / st.session_state.total_queries
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""<div class="stat-card"><div class="stat-value">{st.session_state.total_queries}</div><div class="stat-label">Soru</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="stat-card"><div class="stat-value">{avg_time:.1f}s</div><div class="stat-label">Ort. Süre</div></div>""", unsafe_allow_html=True)

    # 📈 RLHF Geri Bildirim İstatistikleri
    fb_total, fb_pos, fb_neg = get_feedback_stats()
    if fb_total > 0:
        st.divider()
        st.markdown("### 📈 Kullanıcı Geri Bildirimleri")
        col5, col6, col7 = st.columns(3)
        with col5:
            st.markdown(f"""<div class="stat-card"><div class="stat-value">{fb_total}</div><div class="stat-label">Toplam</div></div>""", unsafe_allow_html=True)
        with col6:
            st.markdown(f"""<div class="stat-card"><div class="stat-value" style="color:#22c55e;">👍 {fb_pos}</div><div class="stat-label">Beğeni</div></div>""", unsafe_allow_html=True)
        with col7:
            st.markdown(f"""<div class="stat-card"><div class="stat-value" style="color:#ef4444;">👎 {fb_neg}</div><div class="stat-label">Beğenmedi</div></div>""", unsafe_allow_html=True)
        if fb_total > 0:
            satisfaction = (fb_pos / fb_total) * 100
            st.progress(fb_pos / fb_total, text=f"Memnuniyet Oranı: %{satisfaction:.0f}")

st.markdown("""
<div class="main-header">
    <h1>🏛️ İÜC Akademik Asistan</h1>
    <p>İstanbul Üniversitesi-Cerrahpaşa · Yapay Zeka Destekli Bilgi Sistemi</p>
</div>
""", unsafe_allow_html=True)

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

st.markdown("""
<div class="welcome-box">
    <h3 style="margin-top:0">👋 Merhaba! Size nasıl yardımcı olabilirim?</h3>
    <p style="opacity:0.8; font-size:0.9rem;">Yönetmelikler, akademik takvim ve yönergeler hakkında sorularınızı sorabilirsiniz.</p>
    <div class="feature-item">✅ Kaynak gösteriyor — hangi yönetmelikten geldiğini belirtiyor</div>
    <div class="feature-item">✅ API Tabanlı Mimari — FastAPI ile hızlı ve güvenli veri akışı</div>
    <div class="feature-item">✅ Geri Bildirim — Cevapları 👍/👎 ile değerlendirin</div>
</div>
""", unsafe_allow_html=True)

st.markdown("**💡 Örnek sorular — tıklayarak deneyin:**")
example_questions = [
    "Derslere devam zorunluluğu yüzde kaçtır?",
    "Yaz okulunda en fazla kaç kredi alabilirim?",
    "Onur öğrencisi olmak için ne gerekir?",
    "Kayıt dondurma süresi ne kadar?",
    "Çap şartları nelerdir?",
    "Hoca ders kaydını onaylamazsa ne olur?",
    "Mazeret sınavına kimler girebilir?",
    "Yatay geçiş başvuruları ne zaman?"
]

cols = st.columns(4)
for i, question in enumerate(example_questions):
    with cols[i % 4]:
        st.button(question, on_click=trigger_example, args=(question,), key=f"btn_{i}", use_container_width=True, type="primary")

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
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
                
                fb_col1, fb_col2, fb_spacer = st.columns([1, 1, 8])
                with fb_col1:
                    if st.button("👍", key=f"fb_pos_{idx}", help="Bu cevap faydalıydı"):
                        handle_feedback(idx, "positive")
                        st.rerun()
                with fb_col2:
                    if st.button("👎", key=f"fb_neg_{idx}", help="Bu cevap yetersizdi"):
                        handle_feedback(idx, "negative")
                        st.rerun()
                
            else:
                st.markdown('<div class="feedback-done">✅ Geri bildiriminiz kaydedildi!</div>', unsafe_allow_html=True)

user_query = st.chat_input("Sorunuzu yazın...")

if "trigger_query" in st.session_state:
    user_query = st.session_state.trigger_query
    del st.session_state.trigger_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-bubble">{user_query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Belgeler taranıyor..."):
            result = process_query_via_api(user_query, model_choice, temperature, st.session_state.chat_history)

        engine = result.get("engine", "API")
        st.markdown(f'<div class="assistant-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
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
            
        # DOM u tam yenilemek icin (Beğen butonlarının ilk soruda hemen çıkması için)
        st.rerun()
