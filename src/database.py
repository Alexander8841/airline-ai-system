import os
import json
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIGHTS_PATH = os.path.join(BASE_DIR, "data", "flights.json")
BOOKINGS_PATH = os.path.join(BASE_DIR, "data", "bookings.json")
POLICIES_PATH = os.path.join(BASE_DIR, "data", "policies.json")
PROFILE_PATH = os.path.join(BASE_DIR, "data", "passenger_profile.json")

DB = {}

def load_json_file(file_path):
    """Безопасная загрузка JSON файла."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл базы данных не найден: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_profile() -> dict:
    """Загрузка профиля пассажира из JSON файла. Возвращает пустой словарь при ошибке."""
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            save_profile({})
            return {}
    return {}

def save_profile(profile: dict) -> None:
    """Сохранение профиля пассажира в JSON файл."""
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

def fresh_db() -> dict:
    """Загрузка исходного состояния базы данных из JSON файлов."""
    flights_list = load_json_file(FLIGHTS_PATH)
    bookings_dict = load_json_file(BOOKINGS_PATH)
    policies_dict = load_json_file(POLICIES_PATH)

    flights_dict = {}
    for flight in flights_list:
        key = flight.get("flight_key")
        if not key:
            date_clean = flight["date"].replace("-", "")[4:]
            key = f"{flight['id']}_{date_clean}"
            flight["flight_key"] = key
            
        flights_dict[key] = {
            "id": flight["id"],
            "from": flight["from"],
            "to": flight["to"],
            "date": flight["date"],
            "time": flight["time"],
            "price": flight["price"],
            "seats": flight["seats"],
            "class": flight.get("class", "economy")
        }

    return {
        "flights": flights_dict,
        "bookings": bookings_dict,
        "policies": policies_dict
    }

def reset_db():
    """Сброс глобальной БД к исходному чистому состоянию."""
    global DB
    DB.clear()
    DB.update(fresh_db())

    if not os.path.exists(PROFILE_PATH):
        save_profile({})

def snapshot_db(db_state: dict) -> dict:
    """Создает глубокую копию состояния БД. Используется для сравнения состояний."""
    return copy.deepcopy(db_state)

reset_db()

