import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
import sys
import json
from urllib.parse import urljoin

# GÜVENLİK KİLİDİ: config.py'yi her klasörden sorunsuz bulması için
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PDF_DIR, HTML_DIR, RAW_DIR

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TARGET_PAGES = [
    "https://ogrenci.iuc.edu.tr/tr/content/mevzuat/yonetmelikler",
    "https://ogrenci.iuc.edu.tr/tr/content/mevzuat/yonergeler",
    "https://ogrenci.iuc.edu.tr/tr/content/mevzuat/esaslar-kararlar-kilavuzlar",
    "https://ogrenci.iuc.edu.tr/tr/content/mevzuat/kanunlar",
    "https://ogrenci.iuc.edu.tr/tr/content/akademik-takvim/onlisans~2Flisans-akademik-takvim",
    "https://ogrenci.iuc.edu.tr/tr/content/sss/",
    "https://ogrenci.iuc.edu.tr/tr/duyurular/1/1",
    "https://ogrenci.iuc.edu.tr/tr/duyurular/1/2",
    "https://ogrenci.iuc.edu.tr/tr/duyurular/1/3",
    "https://ogrenci.iuc.edu.tr/tr/duyurular/1/4",
    "https://ogrenci.iuc.edu.tr/tr/duyurular/1/5",
    "https://iuc.edu.tr/tr/content/aday-ogrenci/universitemiz-hakkinda",
    "https://iuc.edu.tr/tr/content/iuc-kart/iuc-kart-nedir",
    "https://iuc.edu.tr/tr/",
]

metadata = []

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

def download_pdf(url, filename=None):
    try:
        if not filename:
            filename = url.split("/")[-1].split("?")[-1]
            if not filename.endswith(".pdf"):
                filename += ".pdf"
        filename = filename.replace("%20", "_").replace(" ", "_")
        filepath = os.path.join(PDF_DIR, filename)
        if os.path.exists(filepath):
            print(f"Zaten var: {filename}")
            return
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(r.content)
            print(f"PDF indirildi: {filename}")
            metadata.append({"url": url, "filename": filename, "type": "pdf"})
        time.sleep(1)
    except Exception as e:
        print(f"PDF hatası {url}: {e}")

def save_html(url, content, filename):
    filepath = os.path.join(HTML_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"HTML kaydedildi: {filename}")
    metadata.append({"url": url, "filename": filename, "type": "html"})

def scrape_with_selenium(url):
    driver = get_driver()
    try:
        print(f"\nSayfa açılıyor: {url}")
        driver.get(url)
        time.sleep(3)

        # HTML kaydet
        filename = url.replace("https://", "").replace("/", "_") + ".html"
        save_html(url, driver.page_source, filename)

        # PDF linklerini bul
        links = driver.find_elements("tag name", "a")
        for link in links:
            href = link.get_attribute("href")
            if href and ".pdf" in href.lower():
                print(f"PDF bulundu: {href}")
                download_pdf(href)

    except Exception as e:
        print(f"Selenium hatası {url}: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    print("Scraping başlıyor...\n")
    
    for page in TARGET_PAGES:
        scrape_with_selenium(page)
        time.sleep(2)

    # Metadata kaydet (Gizli hatayı düzelttik: OUTPUT_DIR yerine RAW_DIR kullanıyoruz)
    with open(os.path.join(RAW_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nTamamlandı!")
    print(f"Toplam kayıt: {len(metadata)}")