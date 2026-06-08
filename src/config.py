import os
import json
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Обходной путь (Workaround) для прокси-сервера OpenAI
def _unwrap_proxy(response: httpx.Response) -> None:
    """Некоторые прокси оборачивают реальный ответ OpenAI в дополнительное поле 'response'.
    Этот хук прозрачно распаковывает его, чтобы LangChain получал стандартный формат."""
    if response.status_code == 200:
        try:
            response.read()
            data = response.json()
            if isinstance(data, dict) and 'response' in data and isinstance(data['response'], dict):
                inner = data['response']
                response._content = json.dumps(inner).encode('utf-8')
        except Exception:
            pass

HTTP_CLIENT_UNWRAP = httpx.Client(
    event_hooks={'response': [_unwrap_proxy]},
    verify=False,
)

def extract_text(message) -> str:
    """Универсальное извлечение текста из объекта сообщения или сырого content."""
    content = message.content if hasattr(message, "content") else message
    
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        return ''.join(
            block.get('text', '') 
            for block in content 
            if isinstance(block, dict) and block.get('type') == 'text'
        )
        
    return str(content)


def get_llm(temperature: float = 0.0, max_tokens: int = 2048, model='gemini-3.1-flash-lite'):
    """Фабрика для получения правильного экземпляра LLM от LangChain в зависимости от настроек."""
    if LLM_PROVIDER == "yandex-cloud":
        
        return ChatOpenAI(
            api_key=YANDEX_API_KEY,
            base_url="https://llm.api.cloud.yandex.net/v1",
            model=f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest",
            temperature=temperature,
            max_tokens=max_tokens
        )

    else:
        use_unwrap = os.getenv("USE_HTTP_UNWRAP", "false").lower() == "true"
        client_kwargs = {}
        if use_unwrap:
            client_kwargs["http_client"] = HTTP_CLIENT_UNWRAP
            
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            **client_kwargs
        )

# Системный промпт для базового ReAct-агента
REACT_SYSTEM_PROMPT = """You are an airline customer support agent. Follow the ReAct loop:
THOUGHT: Analyze the request, plan the steps.
ACTION: Call the appropriate tool.
OBSERVATION: Analyze tool result.

Available tools:
- search_flights: search flights by route and date.
- get_flight_details: detailed flight information.
- get_booking: booking information.
- lookup_policy: lookup airline policies using HyDE keyword search.
- update_booking: rebook to a different flight.

Today's Date is: February 20, 2026 (2026-02-20).

IMPORTANT RULES:
1. Request confirmation before any booking change or cancellation.
2. Retrieve the current booking first, then suggest changes.
3. Report the price difference when changing a flight.
4. Do not disclose other passengers' data.
5. If information is insufficient — ask the user.
6. Do not fabricate information — only use data from tools.
7. Be polite and helpful.
"""
# Системный промпт для специалиста по поиску рейсов
FLIGHT_AGENT_PROMPT = """You are a flight search expert. Your tasks:
- Find flights matching the user's criteria
- Compare options by price, time, and convenience
- Provide a brief recommendation
Use only the available tools. Do not fabricate data.
Today is February 20, 2026."""
# Системный промпт для специалиста по правилам авиакомпании
POLICY_AGENT_PROMPT = """You are an airline policy expert. Your tasks:
- Provide accurate policy information
- Explain fees and restrictions
- Highlight important conditions (deadlines, class restrictions)
## Tool: lookup_policy
Look up airline policies (baggage, refunds, rebooking, delays, etc.).

## Technique: HyDE Policy Lookup
When the passenger asks a question about policies, use the HyDE technique:

1. Think: what would a relevant policy document say about this topic?
2. Generate a short hypothetical policy excerpt (2-3 sentences, formal tone,
   using policy keywords like "компенсация", "допускается", "штраф", etc.)
3. Pass THAT hypothetical excerpt as the query to lookup_policy.

Example:
  User asks: "Can I bring my cat on the plane?"
  HyDE query: "Перевозка питомцев. Небольшие домашние животные могут перевозиться в салоне в специальной переноске. Требуется предварительное бронирование. Взимается плата."
"""
# Системный промпт для специалиста по бронированию
BOOKING_AGENT_PROMPT = """You are a booking management expert. Your tasks:
- Retrieve current booking data
- Execute rebooking when instructed (via update_booking)
- Check booking status and details
IMPORTANT: before updating a booking, ensure a specific flight and date are provided."""

# Системные промпты для Multi-Agent системы (MAS)
# Промпт для Координатора, который декомпозирует задачу на подзадачи
COORDINATOR_SYSTEM_PROMPT = """You are the coordinator of an airline agent team.
You have 3 specialists:
- flight_agent: flight search (search_flights, get_flight_details)
- policy_agent: company policies (lookup_policy). Uses semantic search.
- booking_agent: bookings (get_booking, update_booking)

Break the user's request into subtasks for the specialists.
Specify priority (1=highest) and execution order.

CRITICAL RULES for subtask descriptions:
- Each subtask description MUST be self-contained — the specialist has NO context beyond its description.
- ALWAYS include explicit city names (e.g. "Moscow to Paris"), dates with year (e.g. "April 16, 2026"), booking IDs, and class.
- NEVER write vague descriptions like "search for flights on the route of the booking" — write "search for flights from Moscow to Paris on April 16, 2026".
- Current year is 2026. Today is February 20, 2026."""
# Промпт для синтеза итогового ответа
COORDINATOR_SYNTHESIS_PROMPT = """You are the coordinator. Combine the specialists' results
into a single coherent response for the customer. Be polite and informative.
Do not repeat internal details, only include information useful to the customer.
Use ONLY facts from the specialist results below. Do NOT invent or assume any data."""
# Промпт для агента-критика (проверка качества ответа)
CRITIC_SYSTEM_PROMPT = """You are a quality assurance critic for an airline's customer responses.

Evaluate the proposed answer using ONLY the specialist data provided below. Check:

1. COMPLETENESS: Does the answer address all parts of the customer's question?
2. CORRECTNESS: Does the answer accurately reflect the specialist data? Do NOT flag missing info that was never requested.
3. SAFETY: Are there risky or misleading recommendations?
4. POLITENESS: Is the tone customer-friendly?
5. SPECIFICITY: Are concrete numbers (prices, fees, flight times) included when available in the data?

IMPORTANT:
- Only flag issues that are DIRECTLY supported by the specialist data or the customer's question.
- Do NOT invent problems or request information the customer did not ask for.
- If the specialist data contains facts and the answer reflects them correctly, that is sufficient.

Score from 0 to 10. Set approved=true only when score >= 7."""
