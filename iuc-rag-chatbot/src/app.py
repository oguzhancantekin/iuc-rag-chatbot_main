import streamlit as st
import pickle
import os
import sys
import time
import requests
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import VECTORDB_DIR

API_URL = "http://localhost:8000"

# Kaynak dosya adlarini kullaniciya gosterilecek okunabilir isimlere cevirir.
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

st.set_page_config(
    page_title="İÜC Akademik Asistan",
    page_icon="🎓",
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

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #C41E3A 0%, #8B0000 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white !important; font-size: 1.8rem !important; margin: 0 !important; font-weight: 700 !important; }
    .main-header p { color: rgba(255,255,255,0.85) !important; margin: 0.3rem 0 0 0 !important; font-size: 0.9rem !important; }
    .stat-card {
        background: rgba(196, 30, 58, 0.1);
        border: 1px solid rgba(196, 30, 58, 0.3);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #C41E3A; }
    .stat-label { font-size: 0.75rem; opacity: 0.7; }
    .welcome-box {
        border: 1px solid rgba(196, 30, 58, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background: rgba(196, 30, 58, 0.03);
    }
    .feature-item { display: flex; align-items: center; gap: 0.5rem; margin: 0.4rem 0; font-size: 0.9rem; }
    .timing-badge {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 0.3rem;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(196, 30, 58, 0.05);
        border: 1px solid rgba(196, 30, 58, 0.1);
        display: inline-block;
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
    <h1>🎓 İÜC Akademik Asistan</h1>
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

        st.markdown(result["answer"])
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