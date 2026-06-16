import streamlit as st
import pickle
import os
import sys
import time
import requests
import base64

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import VECTORDB_DIR
from shared import get_display_name

# NOT: SOURCE_DISPLAY_NAMES ve get_display_name() burada rag_engine.py ile
# birebir kopya halindeydi (iki dosyada ayni sozluk, ayni fonksiyon).
# Artik shared.py'den import ediliyor; yeni bir kaynak dosyasi eklendiginde
# tek yerde guncelleme yeterli.

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="İÜC Akademik Asistan",
    page_icon="🏛️",
    layout="wide"
)

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
        opacity: 0.12; /* Görünürlük artırıldı */
        z-index: 0;
        pointer-events: none;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0F204B 0%, #1A2B4C 100%);
        border-bottom: 4px solid #D4AF37;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 { color: #D4AF37 !important; font-size: 2rem !important; margin: 0 !important; font-weight: 800 !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); }
    .main-header p { color: rgba(255,255,255,0.9) !important; margin: 0.3rem 0 0 0 !important; font-size: 0.95rem !important; letter-spacing: 0.5px; }
    .stat-card {
        background: rgba(15, 32, 75, 0.07);
        border: 1px solid rgba(15, 32, 75, 0.15);
        border-left: 4px solid #0F204B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
        transition: transform 0.2s;
        backdrop-filter: blur(5px);
    }
    .stat-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .stat-value { font-size: 1.5rem; font-weight: 800; color: #D4AF37; } /* Altın sarısı ile daha belirgin */
    .stat-label { font-size: 0.8rem; opacity: 0.8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .welcome-box {
        border-top: 4px solid #D4AF37;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        background: rgba(15, 32, 75, 0.06);
        border: 1px solid rgba(15, 32, 75, 0.15);
        backdrop-filter: blur(5px);
    }
    .feature-item { display: flex; align-items: center; gap: 0.5rem; margin: 0.6rem 0; font-size: 0.95rem; } /* color:#333 silindi */
    .timing-badge {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(212, 175, 55, 0.1); /* Altın arka plan */
        color: #D4AF37; /* Altın metin */
        border: 1px solid rgba(212, 175, 55, 0.3);
        display: inline-block;
    }
    div.stButton > button {
        background-color: #0F204B !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1A2B4C !important;
        box-shadow: 0 4px 8px rgba(15, 32, 75, 0.2) !important;
        color: #D4AF37 !important;
    }
    /* Chat bubbles styling */
    [data-testid="stChatMessage"] {
        border-radius: 15px !important;
        padding: 15px 20px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    /* Kullanıcı Mesajı (Altın Tonlu) */
    [data-testid="stChatMessage"]:nth-child(odd) { 
        background-color: rgba(212, 175, 55, 0.05) !important; 
        border: 1px solid rgba(212, 175, 55, 0.2) !important; 
        border-left: 4px solid #D4AF37 !important;
    }
    /* Asistan Mesajı (Lacivert Tonlu) */
    [data-testid="stChatMessage"]:nth-child(even) { 
        background-color: rgba(15, 32, 75, 0.03) !important; 
        border: 1px solid rgba(15, 32, 75, 0.1) !important; 
        border-left: 4px solid #0F204B !important;
    }
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
    st.divider()

    if st.button("🗑️ Geçmişi Sil", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.session_state.total_time = 0.0
        st.rerun()

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

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-box">
        <h3 style="margin-top:0">👋 Merhaba! Size nasıl yardımcı olabilirim?</h3>
        <p style="opacity:0.8; font-size:0.9rem;">Yönetmelikler, akademik takvim ve yönergeler hakkında sorularınızı sorabilirsiniz.</p>
        <div class="feature-item">✅ Kaynak gösteriyor — hangi yönetmelikten geldiğini belirtiyor</div>
        <div class="feature-item">✅ API Tabanlı Mimari — FastAPI ile hızlı ve güvenli veri akışı</div>
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "elapsed" in message:
                st.markdown(f"""<div class="timing-badge">⏱️ {message['elapsed']:.2f}s · API + Hibrit Arama</div>""", unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                sources_clean = [s for s in message["sources"] if s]
                if sources_clean:
                    with st.expander("📚 Kaynaklar"):
                        for source in sources_clean:
                            clean_name = get_display_name(source)
                            st.markdown(f"📄 **{clean_name}**")

user_query = st.chat_input("Sorunuzu yazın...")

if "trigger_query" in st.session_state:
    user_query = st.session_state.trigger_query
    del st.session_state.trigger_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Belgeler taranıyor..."):
            result = process_query_via_api(user_query, model_choice, temperature, st.session_state.chat_history)

        st.write_stream(stream_data(result["answer"]))
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
