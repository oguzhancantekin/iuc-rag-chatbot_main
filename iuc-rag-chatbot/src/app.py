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

# Logo yolunu belirle
LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.png")

# Watermark için logoyu base64'e çevir
base64_logo = ""
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as image_file:
        base64_logo = base64.b64encode(image_file.read()).decode("utf-8")

# ── Dark Mode State ──
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# ── Renk Paleti (Tema Bazlı) ──
if dark:
    bg_primary = "#0a0f1a"
    bg_card = "rgba(212, 175, 55, 0.08)"
    text_color = "#e0e0e0"
    border_color = "rgba(212, 175, 55, 0.25)"
    user_bubble_bg = "linear-gradient(135deg, rgba(212, 175, 55, 0.20) 0%, rgba(212, 175, 55, 0.08) 100%)"
    user_bubble_text = "#f0e6c8"
    assistant_bubble_bg = "linear-gradient(135deg, rgba(30, 50, 100, 0.40) 0%, rgba(20, 35, 70, 0.25) 100%)"
    assistant_bubble_text = "#d0d8e8"
    stat_bg = "rgba(212, 175, 55, 0.10)"
    stat_border = "rgba(212, 175, 55, 0.30)"
else:
    bg_primary = "#ffffff"
    bg_card = "rgba(15, 32, 75, 0.07)"
    text_color = "#111"
    border_color = "rgba(15, 32, 75, 0.15)"
    user_bubble_bg = "linear-gradient(135deg, rgba(212, 175, 55, 0.15) 0%, rgba(212, 175, 55, 0.05) 100%)"
    user_bubble_text = "#111"
    assistant_bubble_bg = "linear-gradient(135deg, rgba(15, 32, 75, 0.06) 0%, rgba(15, 32, 75, 0.02) 100%)"
    assistant_bubble_text = "#111"
    stat_bg = "rgba(15, 32, 75, 0.07)"
    stat_border = "rgba(15, 32, 75, 0.15)"

st.markdown(f"""
<style>
    /* Arka plan filigran (Watermark) */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("data:image/png;base64,{base64_logo}");
        background-repeat: no-repeat;
        background-position: center;
        background-size: 40%;
        opacity: {0.06 if dark else 0.12};
        z-index: 0;
        pointer-events: none;
    }}
    /* Dark Mode global overrides */
    {"" if not dark else '''
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"], .stApp { background-color: #0a0f1a !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background-color: #0d1526 !important; }
    
    /* Metin renklerini düzeltme */
    .stApp p, .stApp li, .stApp span, .stApp label, .stApp div, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #e0e0e0 !important; }
    
    /* Girdi alanları ve alt kısım düzeltmeleri */
    .stTextInput input, .stSelectbox select, [data-testid="stChatInput"] textarea { background-color: #1a2540 !important; color: #e0e0e0 !important; border: 1px solid rgba(212, 175, 55, 0.3) !important; }
    [data-testid="stChatInput"] { background-color: #0a0f1a !important; }
    '''}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, #0F204B 0%, #1A2B4C 100%);
        border-bottom: 4px solid #D4AF37;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .main-header h1 {{ color: #D4AF37 !important; font-size: 2rem !important; margin: 0 !important; font-weight: 800 !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }}
    .main-header p {{ color: rgba(255,255,255,0.9) !important; margin: 0.3rem 0 0 0 !important; font-size: 0.95rem !important; letter-spacing: 0.5px; }}
    .stat-card {{
        background: {stat_bg};
        border: 1px solid {stat_border};
        border-left: 4px solid #0F204B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
        transition: transform 0.2s;
        backdrop-filter: blur(5px);
    }}
    .stat-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
    .stat-value {{ font-size: 1.5rem; font-weight: 800; color: #D4AF37; }}
    .stat-label {{ font-size: 0.8rem; opacity: 0.8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
    .welcome-box {{
        border-top: 4px solid #D4AF37;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        background: {bg_card};
        border: 1px solid {border_color};
        backdrop-filter: blur(5px);
    }}
    .feature-item {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.6rem 0; font-size: 0.95rem; }}
    .timing-badge {{
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(212, 175, 55, 0.1);
        color: #D4AF37;
        border: 1px solid rgba(212, 175, 55, 0.3);
        display: inline-block;
    }}
    div.stButton > button {{
        background-color: #0F204B !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    div.stButton > button:hover {{
        background-color: #1A2B4C !important;
        box-shadow: 0 4px 8px rgba(15, 32, 75, 0.2) !important;
        color: #D4AF37 !important;
    }}
    /* Chat bubbles styling */
    .user-bubble {{
        background: {user_bubble_bg};
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-left: 5px solid #D4AF37;
        padding: 15px 20px;
        border-radius: 15px 15px 0 15px;
        color: {user_bubble_text} !important;
        font-weight: 500;
        margin-left: 10%;
        box-shadow: 0 4px 10px rgba(212, 175, 55, 0.08);
    }}
    .assistant-bubble {{
        background: {assistant_bubble_bg};
        border: 1px solid rgba(15, 32, 75, 0.15);
        border-left: 5px solid #0F204B;
        padding: 15px 20px;
        border-radius: 15px 15px 15px 0;
        color: {assistant_bubble_text} !important;
        line-height: 1.6;
        margin-right: 10%;
        box-shadow: 0 4px 10px rgba(15, 32, 75, 0.05);
    }}
    /* Hide the default Streamlit chat background so our bubbles pop */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }}
    /* Feedback butonları */
    .feedback-container {{
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }}
    .feedback-btn {{
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        cursor: pointer;
        border: 1px solid rgba(150,150,150,0.3);
        background: transparent;
        transition: all 0.2s;
    }}
    .feedback-btn:hover {{ transform: scale(1.1); }}
    .feedback-done {{
        font-size: 0.75rem;
        color: #D4AF37;
        font-weight: 600;
        padding: 4px 0;
    }}
</style>
""", unsafe_allow_html=True)

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
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
        else:
            result = {
                "answer": f"API Sunucusu Hata Döndürdü (Kod {response.status_code}): {response.text}",
                "sources": [],
                "chunks": [],
                "elapsed": 0.0
            }
    except Exception as e:
        result = {
            "answer": f"API Sunucusuna bağlanırken teknik bir hata oluştu. (Detay: {str(e)[:150]})",
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

    # 🌙 Dark Mode Toggle
    st.markdown("<small style='opacity:0.7;'>⚠️ Soru yanıtlanırken tema değiştirmeyin.</small>", unsafe_allow_html=True)
    dark_toggle = st.toggle("🌙 Karanlık Mod", value=st.session_state.dark_mode, key="dark_toggle")
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Geçmişi Sil", use_container_width=True):
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

if len(st.session_state.messages) == 0:
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
        "çap şartları",
        "hoca onay vermezse ne olur?",
        "Mazeret sınavına kimler girebilir?",
        "Yatay geçiş başvuruları ne zaman?"
    ]

    cols = st.columns(4)
    for i, question in enumerate(example_questions):
        with cols[i % 4]:
            st.button(question, on_click=trigger_example, args=(question,), key=f"btn_{i}", use_container_width=True)

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            if "elapsed" in message:
                st.markdown(f"""<div class="timing-badge">⏱️ {message['elapsed']:.2f}s · API + Hibrit Arama</div>""", unsafe_allow_html=True)
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

        st.markdown(f'<div class="assistant-bubble">{result["answer"]}</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="timing-badge">⏱️ {result['elapsed']:.2f}s · API + Hibrit Arama</div>""", unsafe_allow_html=True)

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
            "elapsed": result["elapsed"]
        })
        st.session_state.chat_history.append({"user": user_query, "assistant": result["answer"]})
        st.session_state.total_queries += 1
        st.session_state.total_time += result["elapsed"]

        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]
