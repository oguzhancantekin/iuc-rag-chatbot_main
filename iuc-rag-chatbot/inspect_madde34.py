import pickle

with open("vectordb/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("=== 'MADDE 34' veya 'onur' iceren chunk'lar ===\n")
for c in chunks:
    content = c["content"]
    if "MADDE 34" in content or "Onur" in content or "onur" in content:
        chunk_id = c["metadata"]["chunk_id"]
        has_300 = "3,00" in content or "3.00" in content
        has_349 = "3,49" in content or "3.49" in content
        has_350 = "3,50" in content or "3.50" in content
        has_400 = "4,00" in content or "4.00" in content

        flags = []
        if has_300: flags.append("3.00")
        if has_349: flags.append("3.49")
        if has_350: flags.append("3.50")
        if has_400: flags.append("4.00")

        print(f"--- {chunk_id} [{', '.join(flags)}] ---")
        print(content[:500])
        print()