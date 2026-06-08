from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.config import (
    REACT_SYSTEM_PROMPT,
    FLIGHT_AGENT_PROMPT,
    POLICY_AGENT_PROMPT,
    BOOKING_AGENT_PROMPT
)
from src.tools import (
    search_flights,
    get_flight_details,
    lookup_policy,
    get_booking,
    update_booking,
    change_booking,
    cancel_booking,
    ALL_TOOLS,
    update_passenger_profile
)
from src.guardrails import make_input_guard, tool_output_guard, route_after_input_guard
from src.agents import SpecializedAgent, EnhancedCoordinator
from src.database import load_profile


def make_agent_node(llm, system_prompt: str, tools_list: list):
    """Создает узел, вызывающий LLM с привязанными инструментами.
    Также динамически внедряет профиль пассажира в системный промпт.
    Если LLM возвращает только вызов инструмента без объяснения, принудительно запрашивается мысль (Thought)."""
    
    if type(llm).__name__ == 'ChatGoogleGenerativeAI':
        llm_with_tools = llm.bind_tools(tools_list)
    else:
        llm_with_tools = llm.bind_tools(tools_list, parallel_tool_calls=False)

    def agent_node(state: MessagesState) -> dict:
        profile = load_profile()
        full_prompt = system_prompt
        if profile:
            profile_text = "\n".join(f"  {k}: {v}" for k, v in profile.items())
            full_prompt += f"\n\n## Текущий профиль пассажира\n{profile_text}\n"
        else:
            full_prompt += "\n\n## Текущий профиль пассажира\n  (пусто — данных пока нет)\n"

        messages = [SystemMessage(content=full_prompt)] + state['messages']       
        response = llm_with_tools.invoke(messages)

        if response.tool_calls and not response.content:
            tool_names = ', '.join(tc['name'] for tc in response.tool_calls)
            
            thought_prompt = (
                f"Вы выбрали инструмент: {tool_names}. "
                "В 1 предложении объясните, почему это правильный следующий шаг. "
                "Ответьте ТОЛЬКО рассуждением (без вызова инструментов)."
            )
            
            thought = llm.invoke(messages + [HumanMessage(content=thought_prompt)])
            response.content = thought.content
            
        return {'messages': [response]}
    
    return agent_node

def route_after_agent(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'tools'
    return END


def build_react_graph(llm, system_prompt: str, tools_list: list, memory=None):
    """Компилирует полный граф состояния (StateGraph) для ReAct агента."""
    tool_node = ToolNode(tools_list)
    agent_node = make_agent_node(llm, system_prompt, tools_list)
    input_guard_node = make_input_guard(llm)
    
    graph = StateGraph(MessagesState)
    graph.add_node('input_guard', input_guard_node)
    graph.add_node('agent', agent_node)
    graph.add_node('tools', tool_node)
    graph.add_node('tool_output_guard', tool_output_guard)

    graph.add_edge(START, 'input_guard')
    graph.add_conditional_edges('input_guard', route_after_input_guard, {'agent': 'agent', END: END})
    graph.add_conditional_edges('agent', route_after_agent, {'tools': 'tools', END: END})
    graph.add_edge("tools", "tool_output_guard")
    graph.add_edge("tool_output_guard", "agent")
    
    return graph.compile(checkpointer=memory)


def get_multi_agent_system(llm, max_revisions: int = 1, memory=None) -> EnhancedCoordinator:
    """Помощник, который компилирует специализированных агентов и регистрирует их в EnhancedCoordinator."""

    flight_tools = [search_flights, get_flight_details, update_passenger_profile]
    flight_graph = build_react_graph(
        llm=llm, 
        system_prompt=FLIGHT_AGENT_PROMPT, 
        tools_list=flight_tools,
        memory=memory
    )
    flight_worker = SpecializedAgent(
        name='FlightAgent', 
        graph=flight_graph, 
        tools_list=flight_tools
    )

    policy_tools = [lookup_policy]
    policy_graph = build_react_graph(
        llm=llm, 
        system_prompt=POLICY_AGENT_PROMPT, 
        tools_list=policy_tools,
        memory=memory
    )
    policy_worker = SpecializedAgent(
        name='PolicyAgent', 
        graph=policy_graph, 
        tools_list=policy_tools
    )

    booking_tools = [get_booking, update_booking, update_passenger_profile]
    booking_graph = build_react_graph(
        llm=llm, 
        system_prompt=BOOKING_AGENT_PROMPT, 
        tools_list=booking_tools,
        memory=memory
    )
    booking_worker = SpecializedAgent(
        name='BookingAgent', 
        graph=booking_graph, 
        tools_list=booking_tools
    )

    specialist_registry = {
        'flight_agent': flight_worker,
        'policy_agent': policy_worker,
        'booking_agent': booking_worker
    }

    return EnhancedCoordinator(
        specialist_agents=specialist_registry,
        planning_llm=llm,
        synthesis_llm=llm,
        max_revisions=max_revisions
    )

