from PIL import Image
import os

path = 'arayuz.png'
if not os.path.exists(path):
    print("arayuz.png bulunamadi!")
else:
    img = Image.open(path)
    old_size = img.size
    # Shrink proportionally to a max width/height of 1000px
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    img.save('arayuz.png')
    print(f"Basariyla kucultuldu: {old_size} -> {img.size}")
