import os
import json
import datetime
from langchain_core.tools import tool
from langgraph.types import interrupt
from src.database import DB, load_profile, save_profile

@tool
def update_passenger_profile(key: str, value: str) -> str:
    """Обновляет информацию в профиле пассажира. 
    Рекомендуемые ключи: name, passport, email, seat_preference (место), meal_preference (еда)."""
    profile = load_profile()
    profile[key] = value
    save_profile(profile)
    return json.dumps({"status": "success", "message": f"Профиль обновлен: {key} = '{value}'"}, ensure_ascii=False)

@tool
def get_booking(booking_id: str) -> str:
    """Получение информации о бронировании по ID. Возвращает детали брони, имя пассажира и информацию о рейсе."""
    b = DB.get("bookings", {}).get(booking_id)
    if not b:
        return json.dumps({"error": f"Бронирование {booking_id} не найдено"}, ensure_ascii=False)
    f = DB.get("flights", {}).get(b["flight_key"], {})
    return json.dumps({"booking_id": booking_id, **b, "flight": f}, ensure_ascii=False)

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Поиск доступных рейсов. origin/destination должны быть 3-х буквенными кодами (например, MOW, PAR, LON). Формат даты: YYYY-MM-DD."""
    results = []
    for f_key, f in DB.get("flights", {}).items():
        if f["from"] == origin and f["to"] == destination and f["date"] == date and f["seats"] > 0:
            flight_copy = dict(f)
            # Имитация бага v1: удаление flight_key, чтобы агент не нашел его для изменения брони
            buggy = os.getenv("BUGGY_TOOLS", "false").lower() == "true"
            if buggy:
                # v1: удаляем flight_key, оставляя только номер рейса
                if "flight_key" in flight_copy:
                    del flight_copy["flight_key"]
            else:
                # v2 (исправлено): гарантируем, что flight_key присутствует в ответе
                flight_copy["flight_key"] = f_key
            results.append(flight_copy)

    # Сортировка по времени вылета
    sorted_results = sorted(results, key=lambda x: x["time"])
    return json.dumps(sorted_results, ensure_ascii=False)

@tool
def change_booking(booking_id: str, new_flight_key: str) -> str:
    """Изменение рейса в бронировании. Требуется booking_id (например, ABC123) и новый flight_key (например, SU2456_0220)."""
    b = DB.get("bookings", {}).get(booking_id)
    if not b:
        return json.dumps({"error": "Бронирование не найдено"}, ensure_ascii=False)
    
    nf = DB.get("flights", {}).get(new_flight_key)
    if not nf:
        return json.dumps({"error": f"Рейс с ключом '{new_flight_key}' не найден."}, ensure_ascii=False)
    
    if nf["seats"] <= 0:
        return json.dumps({"error": "Нет свободных мест на выбранном рейсе"}, ensure_ascii=False)
    
    old_f = DB.get("flights", {}).get(b["flight_key"], {})
    diff = nf["price"] - old_f.get("price", 0)
    
    # Human-in-the-Loop Interrupt
    approval = interrupt({
        "action": "change_booking",
        "booking_id": booking_id,
        "new_flight": nf,
        "price_diff": diff
    })
    
    if approval != "approved":
        return json.dumps({"error": f"Действие отменено оператором: {approval}"}, ensure_ascii=False)
    
    # Применение изменений
    b["flight_key"] = new_flight_key
    b["status"] = "changed"
    nf["seats"] -= 1
    
    # Возврат места на старом рейсе, если он существует
    if old_f:
        old_f["seats"] += 1
        
    return json.dumps({"success": True, "new_flight": nf["id"], "price_diff": diff}, ensure_ascii=False)

@tool
def cancel_booking(booking_id: str) -> str:
    """Отмена активного бронирования."""
    b = DB.get("bookings", {}).get(booking_id)
    if not b:
        return json.dumps({"error": "Бронирование не найдено"}, ensure_ascii=False)
    
    if b["status"] == "cancelled":
        return json.dumps({"error": f"Бронирование {booking_id} уже отменено"}, ensure_ascii=False)

    f = DB.get("flights", {}).get(b["flight_key"])
    
    # Human-in-the-Loop Interrupt
    approval = interrupt({
        "action": "cancel_booking",
        "booking_id": booking_id,
        "refund": f["price"] if f else 0
    })
    
    if approval != "approved":
        return json.dumps({"error": f"Действие отменено оператором: {approval}"}, ensure_ascii=False)
        
    b["status"] = "cancelled"
    
    # Возврат места на рейс
    if f:
        f["seats"] += 1
        
    return json.dumps({"success": True, "refund": f["price"] if f else 0}, ensure_ascii=False)

# Стоп-слова для RAG
STOP_WORDS = {
    "а", "и", "но", "в", "на", "с", "по", "к", "о", "об", "у", "для", 
    "из", "от", "до", "за", "над", "под", "перед", "при", "без", "я", "ты", "он", 
    "она", "оно", "мы", "вы", "они", "меня", "тебя", "его", "ее", "нас", "вас", 
    "их", "мой", "твой", "наш", "ваш", "этот", "тот", "такой", "какой", "что", 
    "кто", "как", "где", "когда", "почему", "зачем", "откуда", "куда", "чтобы", 
    "если", "хотя", "потому", "так", "тоже", "также", "уже", "еще", "да", "нет",
    "не", "ни", "быть", "был", "была", "было", "были", "буду", "будет", "будут",
    "есть", "нет", "можно", "нужно", "надо", "очень", "все", "всегда", "никогда"
}

def load_policies():
    import os
    policies_path = os.path.join("data", "policies.json")
    if os.path.exists(policies_path):
        with open(policies_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

def keyword_score(query: str, chunk: dict) -> int:
    """Подсчет совпадений ключевых слов (без стоп-слов)."""
    words = set(query.lower().split()) - STOP_WORDS
    text = (chunk["title"] + " " + chunk["content"]).lower()
    return sum(1 for w in words if w in text)

@tool
def lookup_policy(query: str) -> str:
    """Поиск правил авиакомпании по ключевым словам. Используем технику HyDE: передаем гипотетический текст правила как запрос."""
    policies = load_policies()
    if not policies:
        return "База правил пуста."
        
    scored = [(chunk, keyword_score(query, chunk)) for chunk in policies]
    
    SCORE_THRESHOLD = 2 # Минимальный порог (можно настроить)
    relevant = sorted(
        [(chunk, score) for chunk, score in scored if score >= SCORE_THRESHOLD],
        key=lambda x: x[1],
        reverse=True,
    )[:2]

    if not relevant:
        return "Не найдено релевантных правил."

    parts = []
    for chunk, score in relevant:
        parts.append(f"### {chunk['title']}\n{chunk['content']}")
    return "\n\n".join(parts)


@tool
def get_flight_details(flight_number: str) -> str:
    """Получение детальной информации о рейсе по номеру рейса/ID (например, SU2454)."""
    flights = []
    for f in DB.get("flights", {}).values():
        if f["id"].upper() == flight_number.upper():
            flights.append(f)
            
    if flights:
        return json.dumps({"status": "success", "flights": flights}, ensure_ascii=False)
        
    return json.dumps({"status": "error", "message": f"Рейс {flight_number} не найден"}, ensure_ascii=False)

@tool
def update_booking(booking_id: str, new_flight_number: str, new_date: str) -> str:
    """Обновление бронирования: изменение рейса по номеру рейса (например, SU2456) и дате (YYYY-MM-DD)."""
    booking = DB.get("bookings", {}).get(booking_id)
    if not booking:
        return json.dumps({"status": "error", "message": f"Бронирование {booking_id} не найдено"}, ensure_ascii=False)

    new_flight = None
    new_flight_key = None
    for f_key, f in DB.get("flights", {}).items():
        if f["id"].upper() == new_flight_number.upper() and f["date"] == new_date:
            new_flight = f
            new_flight_key = f_key
            break

    if not new_flight:
        return json.dumps({"status": "error", "message": "Рейс не найден"}, ensure_ascii=False)

    if booking["class"] != new_flight["class"]:
        return json.dumps({"status": "error", "message": "Несовпадение классов"}, ensure_ascii=False)

    if new_flight["seats"] <= 0:
        return json.dumps({"status": "error", "message": "Нет свободных мест на выбранном рейсе"}, ensure_ascii=False)

    old_flight = DB.get("flights", {}).get(booking["flight_key"], {})
    
    # Human-in-the-Loop Interrupt
    approval = interrupt({
        "action": "update_booking",
        "booking_id": booking_id,
        "new_flight": new_flight,
    })
    
    if approval != "approved":
        return json.dumps({"error": f"Действие отменено оператором: {approval}"}, ensure_ascii=False)

    # Применение изменений
    booking["flight_key"] = new_flight_key
    booking["status"] = "rebooked"
    new_flight["seats"] -= 1
    if old_flight:
        old_flight["seats"] += 1
    
    return json.dumps({
        "status": "success",
        "message": f"Бронирование {booking_id} изменено",
        "updated_booking": booking
    }, ensure_ascii=False)

@tool
def get_flight_by_id(flight_id: str) -> str:
    """Получение информации о рейсе по номеру рейса/ID (например, 'SU2454')."""
    return get_flight_details(flight_id)

@tool
def flight_duration(flight_id: str) -> str:
    """Расчет продолжительности рейса по времени вылета и прилета. flight_id - это номер рейса (например, SU2454)."""
    flight = None
    for f in DB.get("flights", {}).values():
        if f["id"].upper() == flight_id.upper():
            flight = f
            break
            
    if not flight:
        return json.dumps({"status": "error", "message": f"Рейс {flight_id} не найден"}, ensure_ascii=False)
        
    duration = "3h 30m"
    if flight["to"] == "LON":
        duration = "4h 00m"
        
    return json.dumps({
        "flight_id": flight_id,
        "departure_time": flight["time"],
        "flight_duration": duration
    }, ensure_ascii=False)

ALL_TOOLS = [
    update_passenger_profile,
    get_booking, 
    search_flights, 
    change_booking, 
    cancel_booking, 
    lookup_policy, 
    get_flight_details, 
    update_booking, 
    get_flight_by_id, 
    flight_duration
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
