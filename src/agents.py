from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import (
    get_llm,
    COORDINATOR_SYSTEM_PROMPT,
    COORDINATOR_SYNTHESIS_PROMPT,
    CRITIC_SYSTEM_PROMPT
)

# Pydantic схемы

class SubTask(BaseModel):
    """Подзадача для специализированного агента, выделенная координатором."""
    agent_name: Literal['flight_agent', 'policy_agent', 'booking_agent'] = Field(
        description="Целевой специализированный агент для этой задачи"
    )
    description: str = Field(
        description="Самодостаточное описание того, что нужно сделать. Должно явно содержать названия городов, даты, классы и ID."
    )
    priority: int = Field(
        ge=1, le=3, description="1=наивысший приоритет (выполнить первым), 3=низший приоритет"
    )

class CoordinatorPlan(BaseModel):
    reasoning: str = Field(description="Пошаговое обоснование выбранной декомпозиции задачи.")
    subtasks: List[SubTask] = Field(description="Список самодостаточных подзадач в порядке их выполнения.")

class AgentResult(BaseModel):
    agent_name: str = Field(description="Имя специализированного агента")
    status: Literal['success', 'error', 'partial'] = Field(description="Статус результата выполнения задачи")
    result: str = Field(description="Итоговое сообщение или данные, возвращаемые координатору")
    tools_used: List[str] = Field(default_factory=list, description="Список инструментов, вызванных во время работы")

class CriticFeedback(BaseModel):
    approved: bool = Field(description="True, если ответ соответствует стандартам качества (оценка >= 7), иначе False")
    score: float = Field(ge=0.0, le=10.0, description="Общая оценка качества ответа от 0.0 до 10.0")
    issues: List[str] = Field(default_factory=list, description="Список выявленных конкретных ошибок или недочетов")
    suggestions: List[str] = Field(default_factory=list, description="Практические предложения по исправлению")
    reasoning: str = Field(description="Объяснение оценки и критики качества")


# Реализации Агентов

class SpecializedAgent:
    """Специализированный рабочий агент, оборачивающий скомпилированный конвейер LangGraph."""
    
    def __init__(self, name: str, graph, tools_list: list):
        self.name = name
        self.graph = graph
        self.tools = tools_list

    def process(self, task_description: str) -> AgentResult:
        """Обрабатывает назначенную подзадачу, вызывая ее поток ReAct в LangGraph."""
        try:
            # Запуск скомпилированного потока LangGraph
            result = self.graph.invoke({
                'messages': [HumanMessage(content=task_description)]
            })

            # Изучение траектории для отслеживания использования инструментов
            tools_used = []
            for msg in result.get('messages', []):
                # В состоянии сообщений LangGraph содержатся AIMessages с tool_calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tools_used.extend([tc['name'] for tc in msg.tool_calls])

            final_content = result['messages'][-1].content
            return AgentResult(
                agent_name=self.name,
                status='success',
                result=final_content,
                tools_used=tools_used
            )
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                status='error',
                result=f"Ошибка выполнения в специалисте '{self.name}': {str(e)}",
                tools_used=[]
            )


class CoordinatorAgent:
    """Координатор мультиагентной системы, использующий паттерн иерархического планирования/синтеза."""
    
    def __init__(self, specialist_agents: dict, planning_llm=None, synthesis_llm=None):
        self.agents = specialist_agents
        
        # Загрузка LLM, если они не предоставлены
        p_llm = planning_llm if planning_llm else get_llm()
        self.planning_llm = p_llm.with_structured_output(CoordinatorPlan)
        self.synthesis_llm = synthesis_llm if synthesis_llm else get_llm()

    def create_plan(self, user_query: str) -> CoordinatorPlan:
        """Шаг 1: Декомпозиция запроса на структурированные подзадачи."""
        return self.planning_llm.invoke([
            SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ])

    def execute_plan(self, plan: CoordinatorPlan) -> List[AgentResult]:
        """Шаг 2: Делегирование подзадач специалистам в порядке приоритета."""
        results = []
        # Сортировка по возрастанию приоритета (1 = наивысший, выполняется первым)
        sorted_tasks = sorted(plan.subtasks, key=lambda t: t.priority)

        for task in sorted_tasks:
            agent = self.agents.get(task.agent_name)
            if not agent:
                results.append(AgentResult(
                    agent_name=task.agent_name, 
                    status='error',
                    result=f"Специалист '{task.agent_name}' не найден в реестре координатора",
                    tools_used=[]
                ))
                continue

            # Обработка задачи с помощью целевого специалиста
            result = agent.process(task.description)
            results.append(result)
            
        return results

    def synthesize(self, user_query: str, results: List[AgentResult]) -> str:
        """Шаг 3: Объединение результатов специалистов в финальный ответ клиенту."""
        results_text = '\n\n'.join([
            f'[{r.agent_name}] ({r.status}):\n{r.result}' for r in results
        ])

        response = self.synthesis_llm.invoke([
            SystemMessage(content=COORDINATOR_SYNTHESIS_PROMPT),
            HumanMessage(content=f"Запрос клиента: {user_query}\n\nРезультаты специалистов:\n{results_text}"),
        ])
        return response.content

    def process_query(self, user_query: str) -> tuple:
        """Выполнение полного цикла иерархической координации."""
        plan = self.create_plan(user_query)
        results = self.execute_plan(plan)
        final_answer = self.synthesize(user_query, results)
        return final_answer, results


class CriticAgent:
    """Агент-критик (QA), который проверяет и оценивает предложенный ответ с помощью структурированной обратной связи."""
    
    def __init__(self, critic_llm=None):
        llm = critic_llm if critic_llm else get_llm()
        self.critic_llm = llm.with_structured_output(CriticFeedback)

    def review(self, user_query: str, proposed_answer: str, agent_results: List[AgentResult] = None) -> CriticFeedback:
        """Проверяет точность, полноту ответа и соблюдение правил."""
        context = ""
        if agent_results:
            context = "\n\nДанные от специалистов:\n" + "\n".join([
                f"[{r.agent_name}]: {r.result[:800]}" for r in agent_results
            ])

        feedback = self.critic_llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=f"Запрос клиента: {user_query}\n\nПредложенный ответ:\n{proposed_answer}{context}"),
        ])
        return feedback


class EnhancedCoordinator(CoordinatorAgent):
    """Агент-Координатор с циклом проверки критиком для обработки исправлений перед завершением задач."""
    
    def __init__(self, specialist_agents: dict, planning_llm=None, synthesis_llm=None, max_revisions: int = 1):
        super().__init__(specialist_agents, planning_llm, synthesis_llm)
        self.critic = CriticAgent(critic_llm=synthesis_llm)
        self.max_revisions = max_revisions

    def process_query_with_qc(self, user_query: str) -> tuple:
        """Обработка запроса с активной проверкой качества и циклом доработки."""
        from src.guardrails import get_is_on_topic_classifier, mask_pii
        
        # Проверка на оффтоп перед началом планирования
        # Используем synthesis_llm, так как planning_llm обернут в with_structured_output
        classifier = get_is_on_topic_classifier(self.synthesis_llm)
        if not classifier(user_query):
            print(f"[GUARD MAS] Off-topic request blocked: '{mask_pii(user_query)[:60]}'")
            ans = "Извините, но я авиа-ассистент. Я могу помочь только с бронированием билетов, правилами багажа и другими вопросами о перелетах."
            fake_fb = CriticFeedback(
                approved=True, 
                score=10.0, 
                issues=[], 
                suggestions=[], 
                reasoning="Запрос отклонен защитным механизмом (оффтоп)."
            )
            return ans, [], fake_fb

        # 1. Планирование
        plan = self.create_plan(user_query)
        # 2. Выполнение
        results = self.execute_plan(plan)
        # 3. Синтез первоначального черновика
        answer = self.synthesize(user_query, results)
        
        # 4. Проверка критиком
        feedback = self.critic.review(user_query, answer, results)
        
        # 5. Доработка в случае отклонения, если есть доступные попытки
        revisions = 0
        while not feedback.approved and revisions < self.max_revisions:
            results_text = '\n'.join([
                f'[{r.agent_name}]: {r.result[:800]}' for r in results
            ])
            revision_prompt = (
                f"Улучшите ответ клиенту на основе QA фидбека от критика.\n"
                f"Обнаруженные проблемы: {feedback.issues}\n"
                f"Предложения: {feedback.suggestions}\n\n"
                f"Текущий ответ:\n{answer}\n\n"
                f"Данные специалистов (НЕ выдумывайте факты):\n{results_text}"
            )
            
            revised = self.synthesis_llm.invoke([
                SystemMessage(
                    content="Вы - координатор. Улучшите ответ на основе фидбека критика. "
                            "Используйте ТОЛЬКО предоставленные данные специалистов. НЕ выдумывайте и не предполагайте никакие факты."
                ),
                HumanMessage(content=revision_prompt),
            ])
            answer = revised.content
            revisions += 1
            
            # Повторная оценка
            feedback = self.critic.review(user_query, answer, results)
            
        return answer, results, feedback

