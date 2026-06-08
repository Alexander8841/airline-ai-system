from src.config import extract_text
import re
import json
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph import END

PII_PATTERNS = [
    # Российсий паспорт
    (re.compile(r'\b\d{4}\s\d{6}\b'), "[PASSPORT]"),
    # Email
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b'), "[EMAIL]"),
    # Номер карты
    (re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'), "[CARD]"),
    # Номер телефона
    (re.compile(r'\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'), "[PHONE]"),
]

def mask_pii(text: str) -> str:
    """Заменяет PII-паттерны плейсхолдерами для безопасного логирования."""
    for pattern, placeholder in PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text

def log_message(role: str, content: str) -> None:
    """Логирует сообщение с PII-плейсхолдерами."""
    masked = mask_pii(content)
    print(f"[LOG] {role}: {masked[:120]}")

def get_is_on_topic_classifier(llm):
    """Фабрика для создания классификатора релевантности, привязанного к LLM."""
    def is_on_topic(user_message: str) -> bool:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a relevance classifier for an airline support chatbot. "
                "Respond with exactly 'yes' or 'no'.\n\n"
                "Is the following message related to airline support "
                "(flights, booking, baggage, policies, travel, passenger info)?"
            )),
            HumanMessage(content=user_message),
        ])
        answer = extract_text(response).strip().lower()
        print(f"[GUARD] is_on_topic -> {answer}")
        return answer.startswith("yes")
    return is_on_topic

def make_input_guard(llm):
    """Создает узел input_guard"""
    classifier = get_is_on_topic_classifier(llm)
    
    def input_guard(state: dict) -> dict:
        messages = state["messages"]
        last_msg = messages[-1]
        content = extract_text(last_msg)

        log_message("user", content)

        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        
        # Проверяем на оффтопик только если это первый запрос в диалоге
        if len(user_messages) == 1 and not classifier(content):
            print(f"[GUARD] Off-topic request blocked: '{mask_pii(content)[:60]}'")
            block_msg = AIMessage(
                content="Извините, но я авиа-ассистент. Я могу помочь только с бронированием билетов, правилами багажа и другими вопросами о перелетах."
            )
            return {"messages": [block_msg]}

        return {}
    return input_guard

# Паттерны инъекций для поиска в результатах вызова инструментов
INJECTION_PATTERNS = re.compile(
    r'\[SYSTEM[:\s]|ignore\s+|disregard\s+|'
    r'new\s+instructions?|override\s+|you\s+are\s+now\s+|'
    r'forget\s+|act\s+as\s+if',
    re.IGNORECASE
)

def tool_output_guard(state: dict) -> dict:
    """Узел проверки на prompt injection."""
    messages = state["messages"]
    last_msg = messages[-1]

    # Проверяем только ToolMessage от search_flights
    if not isinstance(last_msg, ToolMessage):
        return {}
    if last_msg.name != "search_flights":
        return {}

    try:
        flights = json.loads(last_msg.content)
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(flights, list):
        return {}

    # Исключаем рейсы, в правилах тарифа которых (fare_rules/description) есть инъекция
    clean_flights = []
    for flight in flights:
        fare_rules = flight.get("fare_rules", flight.get("description", ""))
        if INJECTION_PATTERNS.search(fare_rules):
            print(f"[GUARD] Flight {flight.get('id', 'unknown')} dropped: injection detected")
        else:
            clean_flights.append(flight)

    if len(clean_flights) == len(flights):
        return {} 

    # Заменяем содержимое сообщения инструмента очищенным результатом
    cleaned_msg = ToolMessage(
        content=json.dumps(clean_flights, ensure_ascii=False),
        tool_call_id=last_msg.tool_call_id,
        name=last_msg.name,
        id=last_msg.id,  
    )
    return {"messages": [cleaned_msg]}

def route_after_input_guard(state: dict) -> str:

    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        return END
    return "agent"
