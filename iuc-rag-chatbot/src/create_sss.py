import json
import os
import sys

# GÜVENLİK KİLİDİ: config.py'yi her dizinden sorunsuz bulması için
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dinamik HTML dizinini config'den çekiyoruz
from config import HTML_DIR

SSS_DATA = [
    {"soru": "Ders kaydını kim onaylar?", "cevap": "Ders kaydını akademik danışman onaylar. Danışman onayı olmadan ders kaydı geçerli sayılmaz."},
    {"soru": "Danışman ders kaydını onaylamazsa ne olur?", "cevap": "Danışman onayı olmayan ders kayıtları geçersiz sayılır ve öğrenci o dönem ders almamış kabul edilir."},
    {"soru": "Kaç dönem üst üste başarısız olursam ilişiğim kesilir?", "cevap": "Azami öğrenim süresi içinde mezun olanamayan öğrencilerin ilişiği kesilir."},
    {"soru": "Bütünleme sınavına kimler girebilir?", "cevap": "Yarıyıl sonu sınavında başarısız olan veya sınava giremeyen öğrenciler bütünleme sınavına girebilir."},
    {"soru": "Tek ders sınavı nedir?", "cevap": "Mezuniyeti için tek dersi kalan öğrencilere verilen özel sınav hakkıdır."},
    {"soru": "Not itirazı nasıl yapılır?", "cevap": "Öğrenciler sınav sonuçlarına itirazlarını sonuçların ilanından itibaren beş iş günü içinde ilgili birime yazılı olarak yapabilir."},
    {"soru": "Çift anadal ile yandal arasındaki fark nedir?", "cevap": "Çift anadal tam bir diploma programıdır ve mezuniyette iki ayrı diploma verilir. Yandal ise sadece sertifika ile sonuçlanan ek bir programdır."},
    {"soru": "Erasmus programına kimler başvurabilir?", "cevap": "En az bir yıl öğrenim görmüş, AGNO 2.20 ve üzeri olan öğrenciler Erasmus programına başvurabilir."},
    {"soru": "Staj zorunlu mu?", "cevap": "Staj zorunluluğu bölüme göre değişmektedir. Mühendislik fakültesi öğrencileri için staj zorunludur."},
    {"soru": "Kayıt yenileme nasıl yapılır?", "cevap": "Kayıt yenileme akademik takvimde belirlenen tarihlerde AKSİS sistemi üzerinden yapılır."},
    {"soru": "Harç ödenmezse ne olur?", "cevap": "Katkı payı veya öğrenim ücretini ödemeyen öğrencilerin kaydı yenilenmez ve o dönem öğrencilik haklarından yararlanamazlar."},
    {"soru": "Mezuniyet için minimum AGNO kaçtır?", "cevap": "Mezuniyet için genel not ortalamasının 4.00 üzerinden en az 2.00 olması gerekmektedir."},
    {"soru": "Ders muafiyeti nasıl alınır?", "cevap": "Daha önce alınan derslerden muafiyet için kayıt döneminde dilekçe ile ilgili birime başvurulması gerekmektedir."},
    {"soru": "Yaz okulu derslerini ortalamam etkiler mi?", "cevap": "Yaz okulunda alınan dersler AGNO hesabına dahil edilir."},
    {"soru": "Öğrenci belgesi nereden alınır?", "cevap": "Öğrenci belgesi AKSİS sistemi üzerinden veya öğrenci işleri biriminden alınabilir."},
    {"soru": "Transkript nasıl alınır?", "cevap": "Transkript AKSİS sistemi üzerinden veya öğrenci işleri biriminden temin edilebilir."},
    {"soru": "Ders bırakma mümkün mü?", "cevap": "Öğrenciler akademik takvimde belirlenen ders ekleme bırakma süresi içinde ders bırakabilirler."},
    {"soru": "Üstten ders alabilir miyim?", "cevap": "Üstten ders alabilmek için bulunduğunuz yarıyılın tüm derslerini almış ve AGNO şartını sağlamış olmanız gerekir."},
    {"soru": "Alttan ders almak zorunda mıyım?", "cevap": "Başarısız olunan dersler öncelikli olarak tekrar alınmalıdır."},
    {"soru": "Diploma ne zaman verilir?", "cevap": "Diploma mezuniyet töreninde veya mezuniyet sonrasında öğrenci işleri biriminden teslim alınabilir."},
    {"soru": "Yaz okulunda en fazla kaç kredi alabilirim?", "cevap": "Yaz okulunda bir dönemde en fazla 10 ulusal kredi (yaklaşık 15-16 AKTS) değerinde ders alınabilir."},
    {"soru": "Onur öğrencisi olmak için not ortalaması kaç olmalı?", "cevap": "Genel not ortalaması 3.00 ile 3.49 arasında olan öğrenciler Onur Öğrencisi sayılır."},
    {"soru": "Yüksek onur öğrencisi kime denir?", "cevap": "Genel not ortalaması 3.50 ve üzeri olan öğrenciler Yüksek Onur Öğrencisi sayılır."},
    {"soru": "Onur öğrencisi olmak için gerekli not ortalaması nedir?", "cevap": "Genel not ortalaması 3.00 ile 3.49 arasında olan öğrenciler Onur Öğrencisi sayılır. 3.50 ve üzeri olanlar ise Yüksek Onur Öğrencisi sayılır."},
    {"soru": "Öğrenci kimlik kartımı kaybettim ne yapmalıyım?", "cevap": "Kimlik kartınızı kaybettiğinizde öğrenci işleri birimine dilekçe ile başvurarak yeni kimlik kartı çıkartabilirsiniz. İÜC Kart ile ilgili işlemler için Kart İşlemleri birimine başvurmanız gerekmektedir."},
    {"soru": "Akademik takvime nereden erişebilirim?", "cevap": "Akademik takvim İstanbul Üniversitesi-Cerrahpaşa'nın resmi web sitesinden ve öğrenci işleri sayfasından erişilebilir. Akademik takvimde kayıt, sınav, tatil tarihleri ve diğer önemli akademik tarihler yer almaktadır."},
]

def create_sss_documents():
    output_dir = HTML_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    content = "<html><body>\n<h1>Sıkça Sorulan Sorular</h1>\n"
    for item in SSS_DATA:
        content += f"<h2>{item['soru']}</h2>\n"
        content += f"<p>{item['cevap']}</p>\n"
    content += "</body></html>"
    
    filepath = os.path.join(output_dir, "sss_manuel.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"SSS dosyası oluşturuldu: {filepath}")
    print(f"Toplam {len(SSS_DATA)} soru-cevap çifti eklendi.")

if __name__ == "__main__":
    create_sss_documents()