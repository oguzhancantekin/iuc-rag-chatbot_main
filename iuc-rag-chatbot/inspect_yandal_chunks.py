import pickle

with open("vectordb/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Önce mevcut tum source isimlerini gorelim (yandal icerenleri)
print("=== 'yandal' iceren tum source isimleri ===")
sources_seen = set()
for c in chunks:
    src = c["metadata"]["source"]
    if "yandal" in src.lower():
        sources_seen.add(src)

for s in sources_seen:
    print(f"  - {s}")

print()

# Bu source'lara ait chunk'lari incele
yandal_chunks = [c for c in chunks if "yandal" in c["metadata"]["source"].lower()]

print(f"Toplam yandal-ilgili chunk sayisi: {len(yandal_chunks)}\n")

for c in yandal_chunks:
    src = c["metadata"]["source"]
    chunk_id = c["metadata"]["chunk_id"]
    content = c["content"]
    has_agno = "agno" in content.lower()
    has_25 = "2.5" in content or "2,5" in content

    flag = ""
    if has_agno:
        flag += " [AGNO]"
    if has_25:
        flag += " [2.5x BULUNDU!]"

    print(f"--- {chunk_id} (src: {src}){flag} ---")
    if "chunk_2" in chunk_id or "chunk_3" in chunk_id:
        print(content)  # TAM ICERIK
    else:
        print(content[:200])
    print()