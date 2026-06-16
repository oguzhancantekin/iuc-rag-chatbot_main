import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset

# Model ve kayıt yolları
from config import MODELS_DIR
OUTPUT_DIR = os.path.join(MODELS_DIR, "llama3-iuc-finetuned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# İÜC Fine-tuning veri seti
TRAINING_DATA = [
    {"instruction": "Derslere devam zorunluluğu yüzde kaçtır?", 
     "response": "Teorik derslerin en az %70'ine, uygulamalı derslerin en az %80'ine devam etmek zorunludur. (Kaynak: İÜC Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği, MADDE 19)"},
    {"instruction": "Onur öğrencisi olmak için not ortalaması kaç olmalı?",
     "response": "Genel not ortalaması 3.00 ile 3.49 arasında olan öğrenciler Onur Öğrencisi sayılır. (Kaynak: İÜC Yönetmeliği, MADDE 34)"},
    {"instruction": "Yüksek onur öğrencisi kime denir?",
     "response": "Genel not ortalaması 3.50 ve üzeri olan öğrenciler Yüksek Onur Öğrencisi sayılır. (Kaynak: İÜC Yönetmeliği, MADDE 34)"},
    {"instruction": "Yaz okulunda en fazla kaç kredi alabilirim?",
     "response": "Yaz okulunda bir dönemde en fazla 10 ulusal kredi (yaklaşık 15-16 AKTS) değerinde ders alınabilir."},
    {"instruction": "Kayıt dondurma süresi en fazla ne kadardır?",
     "response": "Kayıt dondurma süresi toplamda normal öğrenim süresinin en çok yarısı kadar olabilir. (Kaynak: İÜC Yönetmeliği, MADDE 29)"},
    {"instruction": "Çift anadal başvuru şartları nelerdir?",
     "response": "Başvuru anında anadal not ortalamasının en az 3.00 olması ve sınıfında başarı sıralamasında ilk %20'de bulunması gerekir. (Kaynak: ÇAP Yönergesi, MADDE 5)"},
    {"instruction": "Mazeret sınavına kimler girebilir?",
     "response": "Haklı ve geçerli nedenlerle ara sınava giremeyen ve mazereti Yönetim Kurulu tarafından kabul edilen öğrenciler mazeret sınavına girebilir. (Kaynak: İÜC Yönetmeliği, MADDE 24)"},
    {"instruction": "Mezuniyet için minimum AGNO kaçtır?",
     "response": "Mezuniyet için genel not ortalamasının 4.00 üzerinden en az 2.00 olması gerekmektedir. (Kaynak: İÜC Yönetmeliği)"},
    {"instruction": "Staj defterini ne zaman teslim etmeliyim?",
     "response": "Staj defteri, staj bitimini takip eden akademik dönemin ilk 3 haftası içinde teslim edilmelidir. (Kaynak: Lisans Staj Yönergesi, MADDE 11)"},
    {"instruction": "Yatay geçiş başvuruları ne zaman yapılır?",
     "response": "Yatay geçiş başvuruları akademik takvimde belirtilen tarihlerde yapılır. (Kaynak: Yatay Geçiş Yönergesi, MADDE 6)"},
    {"instruction": "Bütünleme sınavına kimler girebilir?",
     "response": "Yarıyıl sonu sınavında başarısız olan veya devamsızlık nedeniyle sınava giremeyen öğrenciler bütünleme sınavına girebilir."},
    {"instruction": "Tek ders sınavı nedir?",
     "response": "Mezuniyeti için tek dersi kalan öğrencilere verilen özel sınav hakkıdır."},
    {"instruction": "Yandal programına başvuru şartları nelerdir?",
     "response": "En az bir yıl öğrenim görmüş ve AGNO 2.20 ve üzeri olan öğrenciler yandal programına başvurabilir. (Kaynak: Yandal Yönergesi)"},
    {"instruction": "Çift anadal ile yandal arasındaki fark nedir?",
     "response": "Çift anadal tam bir diploma programıdır, mezuniyette iki ayrı diploma verilir. Yandal ise sadece sertifika ile sonuçlanan ek bir programdır."},
    {"instruction": "Kayıt yenileme nasıl yapılır?",
     "response": "Kayıt yenileme işlemi AKSİS sistemi üzerinden akademik takvimde belirlenen tarihlerde yapılır."},
    {"instruction": "Yaz okulu dersleri not ortalamamı etkiler mi?",
     "response": "Evet, yaz okulunda alınan dersler AGNO hesabına dahil edilir."},
    {"instruction": "hoca onay vermezse?",
     "response": "Danışman onayı olmayan ders kayıtları geçersiz sayılır ve öğrenci o dönem ders almamış kabul edilir. (Kaynak: İÜC Yönetmeliği, MADDE 9)"},
    {"instruction": "çap şartları",
     "response": "Çift Anadal Programına başvuru için AGNO 3.00 ve üzeri olması, sınıfında ilk %20'de bulunması gerekir. (Kaynak: ÇAP Yönergesi, MADDE 5)"},
    {"instruction": "devam zorunluluğu",
     "response": "Teorik derslerin en az %70'ine, uygulamalı derslerin en az %80'ine devam zorunludur. (Kaynak: İÜC Yönetmeliği, MADDE 19)"},
    {"instruction": "not itirazı nasıl yapılır?",
     "response": "Not itirazları, sonuçların ilanından itibaren beş iş günü içinde yazılı olarak ilgili birime yapılabilir."},
]

def format_instruction(sample):
    return f"""### Soru:
{sample['instruction']}

### Yanıt:
{sample['response']}"""

def prepare_dataset():
    formatted = [{"text": format_instruction(d)} for d in TRAINING_DATA]
    return Dataset.from_list(formatted)

def train():
    print("Fine-tuning başlıyor...")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "")

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Model yükleniyor... (bu birkaç dakika sürebilir)")
    
    # Yerel GGUF modeli kullanamayız, HuggingFace'den yükleyeceğiz
    # Alternatif: daha küçük bir model kullanalım
    model_id = "unsloth/Llama-3.2-3B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # LoRA konfigürasyonu
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Veri seti
    dataset = prepare_dataset()
    print(f"Eğitim verisi: {len(dataset)} örnek")

    # Eğitim parametreleri
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Eğitim başlıyor...")
    trainer.train()

    # Modeli kaydet
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model kaydedildi: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
