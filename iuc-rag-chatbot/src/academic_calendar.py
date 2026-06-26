import os
import json
from config import DATA_DIR

def load_calendar_json():
    json_path = os.path.join(DATA_DIR, "academic_calendar.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON okuma hatası: {e}")
        return None

def format_calendar_context():
    """Takvim JSON'ını LLM'in anlayabileceği metin formatına çevirir."""
    data = load_calendar_json()
    if not data:
        return "Akademik takvim verisi bulunamadı."
    return json.dumps(data, ensure_ascii=False, indent=2)
