import os
import sys
import pandas as pd
import streamlit as st

# Add src to Python path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
import uuid
from src.config import get_llm, REACT_SYSTEM_PROMPT
from src.database import DB, reset_db
from src.tools import ALL_TOOLS
from src.graph import build_react_graph, get_multi_agent_system
from src.evaluation import TASKS

# Настройки страницы и стилизация
st.set_page_config(
    page_title="AeroAgent - Flight Booking AI System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stCodeBlock, code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Gradient Title */
    .title-gradient {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #1D4ED8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-styled {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic card styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #60A5FA;
    }
    
    .db-header {
        font-weight: 600;
        color: #3B82F6;
        border-bottom: 2px solid #2563EB;
        padding-bottom: 5px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    /* Collapsible containers (thought logs) styling */
    .thought-container {
        border-left: 3px solid #10B981;
        padding-left: 10px;
        margin-bottom: 10px;
    }
    
    /* Custom CSS transition animations for sidebars/metrics */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# Управление состоянием (State Management) и компиляторы агентов
@st.cache_resource
def load_llm():
    return get_llm(temperature=0.1)

@st.cache_resource
def get_memory():
    return MemorySaver()

@st.cache_resource
def compile_react_agent(_llm, _memory):
    return build_react_graph(_llm, REACT_SYSTEM_PROMPT, ALL_TOOLS, memory=_memory)

@st.cache_resource
def compile_mas_system(_llm, _memory):
    return get_multi_agent_system(_llm, max_revisions=1, memory=_memory)

llm = load_llm()
memory = get_memory()
react_agent = compile_react_agent(llm, memory)
mas_system = compile_mas_system(llm, memory)

# Инициализация сессий Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Боковая панель (Sidebar)
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/airplane-take-off.png", width=120)
    st.markdown("<h2 style='margin-top:0;'>Control Dashboard</h2>", unsafe_allow_html=True)
    
    # 1. Выбор архитектуры
    st.markdown("### 🤖 Architecture Mode")
    arch_mode = st.radio(
        "Выберите бэкенд агента:",
        options=["Мультиагентная система (MAS)", "Одиночный агент (ReAct)"],
        index=0
    )
    
    # 2. Предустановленные сценарии (Presets)
    st.markdown("### 📋 Predefined Scenarios")
    # Фильтрация задач-пресетов (только задачи с осмысленными описаниями)
    preset_options = ["None"] + [f"{t['id']}: {t['query']}" for t in TASKS[:12]]
    selected_preset = st.selectbox("Загрузить пресет задачи:", options=preset_options)
    
    # Обработка события выбора пресета
    if selected_preset != "None" and ("last_preset" not in st.session_state or st.session_state.last_preset != selected_preset):
        task_id = selected_preset.split(":")[0]
        target_task = next(t for t in TASKS if t["id"] == task_id)
        
        task_query = target_task["query"]
        if not isinstance(task_query, str):
            task_query = str(task_query)
        
        # Сброс БД и истории для чистого демо-запуска
        reset_db()
        st.session_state.messages = [{"role": "user", "content": task_query}]
        st.session_state.agent_messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=task_query)
        ]
        st.session_state.last_preset = selected_preset
        st.session_state.thread_id = str(uuid.uuid4())
        # Принудительный перезапуск для отображения загруженного пресета
        st.rerun()

    # 3. Мониторинг базы данных (Live Mock DB Viewer)
    st.markdown("<div class='db-header'>📊 Live Database Monitor</div>", unsafe_allow_html=True)
    
    # Активные бронирования
    st.markdown("**Активные бронирования клиентов:**")
    bookings_data = []
    for bid, b in DB["bookings"].items():
        flight = DB["flights"].get(b["flight_key"], {})
        bookings_data.append({
            "ID": bid,
            "Passenger": b["passenger"],
            "Route": f"{flight.get('from', '')} ➔ {flight.get('to', '')}",
            "Status": b["status"].upper()
        })
    st.dataframe(pd.DataFrame(bookings_data), hide_index=True)
    
    # Места на рейсах
    st.markdown("**Доступность мест на рейсах:**")
    flights_data = []
    for fkey, f in DB["flights"].items():
        flights_data.append({
            "Flight": f["id"],
            "Route": f"{f['from']}➔{f['to']}",
            "Date": f["date"],
            "Price": f"{f['price']}",
            "Seats": f["seats"]
        })
    st.dataframe(pd.DataFrame(flights_data), hide_index=True)

    # Кнопка сброса базы данных
    if st.button("🔄 Reset Database & Chat"):
        reset_db()
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.session_state.last_preset = "None"
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# --- Главное окно чата ---
st.markdown("<div class='title-gradient'>AeroAgent Orchestration System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-styled'>Интерактивная демонстрация ReAct и мультиагентных (MAS) архитектур с проверкой Критиком.</div>", unsafe_allow_html=True)

# Показ инструкций, если история чата пуста
if len(st.session_state.messages) == 0:
    st.info(
        "👋 Добро пожаловать! Выберите **Архитектуру** или загрузите **Сценарий** на боковой панели. "
        "Или напишите свой запрос в поле ввода ниже (например, *'Перенеси бронирование ABC123 на более поздний рейс'*)."
    )

# Отображение предыдущих сообщений чата
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Проверка состояния графа ReAct на предмет ожидающего прерывания (Human-in-the-Loop)
state = react_agent.get_state(thread_config) if arch_mode == "Одиночный агент (ReAct)" else None
is_interrupted = bool(state and state.next)

# Действие пользователя (отправка сообщения)
user_query = st.chat_input("Задайте вопрос авиа-ассистенту...", disabled=is_interrupted)

if is_interrupted:
    with st.chat_message("assistant"):
        interrupt_val = state.tasks[0].interrupts[0].value
        st.warning("⚠️ Требуется подтверждение оператора (Human-in-the-Loop)!")
        st.markdown("**Действие требует авторизации. Детали:**")
        st.json(interrupt_val)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Одобрить", use_container_width=True):
                with st.spinner("Выполнение действия..."):
                    result = react_agent.invoke(Command(resume="approved"), config=thread_config)
                    final_text = result["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    st.session_state.agent_messages = result["messages"]
                    st.rerun()
        with col2:
            if st.button("❌ Отклонить", use_container_width=True):
                with st.spinner("Отмена действия..."):
                    result = react_agent.invoke(Command(resume="rejected"), config=thread_config)
                    final_text = result["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    st.session_state.agent_messages = result["messages"]
                    st.rerun()

elif user_query:
    # Добавление запроса в историю
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Инициализация состояния сообщений агента, если оно пустое
    if not st.session_state.agent_messages:
        st.session_state.agent_messages = [
            SystemMessage(content=REACT_SYSTEM_PROMPT),
            HumanMessage(content=user_query)
        ]
    else:
        st.session_state.agent_messages.append(HumanMessage(content=user_query))

    # Вызов UI ответа агента
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # --- Режим 1: Одиночный агент (Single-Agent ReAct) ---
        if arch_mode == "Одиночный агент (ReAct)":
            with st.spinner("Агент думает..."):
                # Мы передаем только новый запрос в langgraph, так как история хранится в memory
                # Для надежности можно передать полный список или только последние
                result = react_agent.invoke({"messages": st.session_state.agent_messages}, config=thread_config)
                
                # Проверяем, не прервался ли граф прямо сейчас
                new_state = react_agent.get_state(thread_config)
                if new_state.next:
                    st.rerun() # Перезапускаем страницу для отрисовки кнопок подтверждения
                
                # Показ мыслей и вызовов инструментов в раскрывающемся контейнере
                thoughts_expander = st.expander("🔍 Просмотр лога выполнения (ReAct Trajectory)", expanded=True)
                with thoughts_expander:
                    # Разбор шагов
                    for msg in result["messages"]:
                        # Показ мыслей LLM
                        if isinstance(msg, AIMessage):
                            if msg.content:
                                st.markdown(f"**Мысль (Thought):** *{msg.content}*")
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    st.markdown(f"🛠️ **Действие (Вызов инструмента):** `{tc['name']}({tc['args']})`")
                        elif isinstance(msg, ToolMessage):
                            st.markdown(f"👁️ **Наблюдение (Результат инструмента):** `{msg.content[:400]}`")
                
                # Отрисовка финального ответа
                final_text = result["messages"][-1].content
                response_placeholder.markdown(final_text)
                
                # Обновление состояния сессии
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                st.session_state.agent_messages = result["messages"]
                st.rerun()

        # --- Режим 2: Мультиагентная система (MAS with QA Critic) ---
        else:
            with st.spinner("Координатор планирует и делегирует..."):
                # Запуск координатора с проверкой качества
                final_answer, specialist_results, critic_feedback = mas_system.process_query_with_qc(user_query)
                
                # 1. Показ шагов декомпозиции координатором
                mas_expander = st.expander("🛠️ Просмотр логов специалистов и инструментов", expanded=True)
                with mas_expander:
                    st.markdown("**Иерархический лог выполнения:**")
                    for idx, res in enumerate(specialist_results, 1):
                        st.markdown(f"**{idx}. Специалист: `{res.agent_name}`** | Статус: `{res.status}`")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Вывод:* {res.result[:300]}...")
                        if res.tools_used:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Вызваны инструменты:* `{', '.join(res.tools_used)}`")
                        st.divider()

                # 2. Показ фидбека Критика
                st.markdown("### ⚖️ QA Critic Review")
                color = "green" if critic_feedback.approved else "orange"
                st.markdown(
                    f"**Общая оценка:** <span style='color:{color}; font-size:1.4rem; font-weight:600;'>"
                    f"{critic_feedback.score}/10</span> (Одобрено: `{critic_feedback.approved}`)", 
                    unsafe_allow_html=True
                )
                
                st.markdown(f"**Обоснование критики:** *{critic_feedback.reasoning}*")
                if critic_feedback.issues:
                    st.markdown("**Выявленные проблемы:**")
                    for issue in critic_feedback.issues:
                        st.markdown(f"- ❌ {issue}")
                if critic_feedback.suggestions:
                    st.markdown("**Примененные предложения:**")
                    for sug in critic_feedback.suggestions:
                        st.markdown(f"- 💡 {sug}")
                
                st.divider()

                # 3. Вывод итогового ответа
                response_placeholder.markdown(final_answer)
                
                # Обновление состояния сессии
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                # Для MAS мы добавляем финальный ответ в историю, сохраняя простоту
                st.session_state.agent_messages.append(AIMessage(content=final_answer))
                st.rerun()
