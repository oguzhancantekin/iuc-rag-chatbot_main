import matplotlib.pyplot as plt
import numpy as np

# Veriler
kategoriler = ['Groq (RAG\'li)', 'Groq (RAG\'siz)', 'Ollama (RAG\'li)', 'Ollama (RAG\'siz)']
dogruluk = [81.2, 48.0, 73.5, 26.0]
gecikme = [2.60, 1.55, 10.80, 6.20]

x = np.arange(len(kategoriler))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Doğruluk (Accuracy) için çubuklar
color = 'tab:blue'
ax1.set_xlabel('Motor ve Konfigürasyon')
ax1.set_ylabel('Doğruluk (%)', color=color)
bars1 = ax1.bar(x - width/2, dogruluk, width, label='İçerik Doğruluğu', color=color, alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 100)

# Değerleri çubukların üzerine yazdır
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'%{yval}', ha='center', va='bottom', fontsize=10, color=color, fontweight='bold')

# İkinci Y ekseni (Gecikme)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Yanıt Süresi (Saniye)', color=color)
bars2 = ax2.bar(x + width/2, gecikme, width, label='Yanıt Süresi', color=color, alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 12)

# Değerleri çubukların üzerine yazdır
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval}s', ha='center', va='bottom', fontsize=10, color=color, fontweight='bold')

plt.title('RAG Sistem Performans Analizi: Doğruluk vs. Yanıt Süresi', fontweight='bold', pad=20)
plt.xticks(x, kategoriler)

# Lejant
fig.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)

plt.savefig('performans_grafik.png', dpi=300, bbox_inches='tight')
print("Grafik 'performans_grafik.png' adıyla başarıyla oluşturuldu.")
