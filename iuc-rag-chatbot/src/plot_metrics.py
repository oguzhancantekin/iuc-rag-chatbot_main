import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Türkçe karakter desteği ve stil ayarları
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_FILE = os.path.join(BASE_DIR, "evaluation_results.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "assets", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_plots():
    if not os.path.exists(EVAL_FILE):
        print(f"HATA: {EVAL_FILE} bulunamadi! Önce evaluation.py calistirilmali.")
        return

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    metrics = data.get("metrics", {})
    if not metrics:
        print("Metrik verisi bulunamadi.")
        return

    # 1. RAG vs RAG'siz Doğruluk Karşılaştırması (Bar Chart)
    labels = ['Konu İsabeti\n(Topic Acc)', 'İçerik İsabeti\n(Content Acc)']
    rag_scores = [metrics.get("rag_topic_accuracy", 0), metrics.get("rag_content_accuracy", 0)]
    ragless_scores = [metrics.get("ragless_topic_accuracy", 0), metrics.get("ragless_content_accuracy", 0)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rag_scores, width, label='RAG Sistemi', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width/2, ragless_scores, width, label='RAG\'siz (Saf LLM)', color='#ff7f0e', edgecolor='black')

    ax.set_ylabel('Doğruluk Oranı (%)', fontsize=12, fontweight='bold')
    ax.set_title('RAG ve RAG\'siz Sistem Doğruluk Karşılaştırması', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Değerleri barlara ekleme
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'%{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rag_vs_ragless_accuracy.png"), dpi=300)
    print(f"Oluşturuldu: {os.path.join(OUTPUT_DIR, 'rag_vs_ragless_accuracy.png')}")
    plt.close()

    # 2. Yanıt Süresi (Latency) Karşılaştırması
    categories = ['RAG Sistemi\n(Erişim + Üretim)', 'RAG\'siz\n(Sadece Üretim)']
    latency = [metrics.get("avg_latency_rag", 0), metrics.get("avg_latency_ragless", 0)]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, latency, color=['#2ca02c', '#d62728'], width=0.5, edgecolor='black')
    plt.ylabel('Ortalama Süre (Saniye)', fontsize=12, fontweight='bold')
    plt.title('Ortalama Yanıt Süresi (Latency) Karşılaştırması', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # max value for y scale
    plt.ylim(0, max(latency) * 1.2)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(latency)*0.02), f'{yval:.2f} sn', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "latency_comparison.png"), dpi=300)
    print(f"Oluşturuldu: {os.path.join(OUTPUT_DIR, 'latency_comparison.png')}")
    plt.close()

    # 3. Bilgi Getirimi (Retrieval) Performansı
    labels = ['Recall@5', 'MRR (Sıralama Başarısı)']
    
    # MRR 0-1 arasındadır, Recall 0-100 arasındadır. Ekranda düzgün dursun diye MRR'yi 100 ile çarpıyoruz.
    scores = [metrics.get("avg_recall_at_5", 0), metrics.get("avg_mrr", 0)*100]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, scores, color=['#9467bd', '#8c564b'], width=0.5, edgecolor='black')
    plt.ylabel('Başarı Oranı (%)', fontsize=12, fontweight='bold')
    plt.title('Hibrit Arama Performansı\n(FAISS + BM25 + CrossEncoder)', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 115)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        if i == 0:
            val_text = f'%{yval:.1f}'
        else:
            val_text = f'{yval/100:.3f}'  # Gerçek MRR değerini (0-1 arası) göster
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, val_text, 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "retrieval_performance.png"), dpi=300)
    print(f"Oluşturuldu: {os.path.join(OUTPUT_DIR, 'retrieval_performance.png')}")
    plt.close()

    print("\nTum grafikler basariyla olusturuldu! Sunumda kullanabilirsiniz.")

if __name__ == "__main__":
    create_plots()
