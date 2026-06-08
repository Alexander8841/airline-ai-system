import os
import json
import re
import copy
import time
from typing import List, Optional, Any
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from src.config import get_llm, REACT_SYSTEM_PROMPT
# Re-map REACT_SYSTEM_PROMPT as SYSTEM_PROMPT for lecture 4 compatibility
SYSTEM_PROMPT = REACT_SYSTEM_PROMPT

# Static declaration of policies for the policy grader lookup
POLICIES = [
    "Request confirmation before any booking change or cancellation",
    "Retrieve the current booking first, then suggest changes",
    "Report the price difference when changing a flight",
    "Do not disclose other passengers' data",
    "If information is insufficient — ask the user",
]

# --- Dataclasses to Track Trajectories ---

@dataclass
class Step:
    type: str  # 'action' | 'observation' | 'response'
    content: Any

@dataclass
class Trajectory:
    task_id: str
    query: str
    steps: List[Step] = field(default_factory=list)
    final_response: str = ""
    db_before: Optional[dict] = field(default=None, repr=False)
    db_after: Optional[dict] = field(default=None, repr=False)

    def show(self):
        print(f"{'─'*60}\n📋 {self.task_id}: {self.query}")
        for s in self.steps:
            icons = {"action": "🔧", "observation": "👁", "response": "💬"}
            txt = s.content if isinstance(s.content, str) else json.dumps(s.content, ensure_ascii=False)
            print(f"  {icons.get(s.type, '?')} {txt[:200]}")
        print()

    def as_text(self) -> str:
        lines = []
        for s in self.steps:
            txt = s.content if isinstance(s.content, str) else json.dumps(s.content, ensure_ascii=False)
            lines.append(f"[{s.type}] {txt[:500]}")
        return "\n".join(lines)


# --- Task Basket Definitions ---

TASKS = [
    # ── Easy: simple informational queries ──
    {
        "id": "t01", "query": "Show information about booking ABC123",
        "category": "info", "difficulty": "easy", "needs_dialogue": False,
        "expected_state_changes": None, "policies_to_check": [],
        "scenario": None, "user_context": None
    },
    {
        "id": "t02", "query": "Find flights from Moscow to Paris on February 20",
        "category": "search", "difficulty": "easy", "needs_dialogue": False,
        "expected_state_changes": None, "policies_to_check": [],
        "scenario": None, "user_context": None
    },
    {
        "id": "t03", "query": "What is the status of booking XYZ789?",
        "category": "info", "difficulty": "easy", "needs_dialogue": False,
        "expected_state_changes": None, "policies_to_check": [],
        "scenario": None, "user_context": None
    },
    # ── Medium: require multiple steps or dialogue ──
    {
        "id": "t04", "query": "I want to move booking ABC123 to a later flight on the same day",
        "category": "change", "difficulty": "medium", "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.status": "changed"},
        "validation_rule": {"type": "compare_time", "booking_path": "bookings.ABC123", "field": "time", "operator": "gt"},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to move booking ABC123 to a later flight on the same day",
        "user_context": "Booking ABC123, passenger Ivan Petrov, current flight in the morning"
    },
    {
        "id": "t05", "query": "Cancel my booking ABC123",
        "category": "cancel", "difficulty": "medium", "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.status": "cancelled"},
        "policies_to_check": [0],
        "scenario": "You want to cancel booking ABC123",
        "user_context": "Booking ABC123, passenger Ivan Petrov"
    },
    {
        "id": "t06", "query": "Find flights to London and compare with Paris on price on February 20",
        "category": "search", "difficulty": "medium", "needs_dialogue": False,
        "expected_state_changes": None, "policies_to_check": [],
        "scenario": None, "user_context": None
    },
    {
        "id": "t07", "query": "Move booking ABC123 to the afternoon flight at 12:30",
        "category": "change", "difficulty": "medium", "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.flight_key": "SU2456_0220", "bookings.ABC123.status": "changed"},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to move booking ABC123 to the 12:30 flight",
        "user_context": "Booking ABC123, you want flight SU2456 at 12:30"
    },
    # ── Hard: complex conditions and edge cases ──
    {
        "id": "t08", "query": "Move ABC123 to tomorrow, but only if it will be cheaper than the current flight",
        "category": "change", "difficulty": "hard", "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.flight_key": "SU2454_0221", "bookings.ABC123.status": "changed"},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to move ABC123 to tomorrow, but ONLY if it is cheaper. If more expensive — refuse",
        "user_context": "Booking ABC123, current price 25000"
    },
    {
        "id": "t09", "query": "Show booking FAKE000",
        "category": "edge", "difficulty": "hard", "needs_dialogue": False,
        "expected_state_changes": None, "policies_to_check": [],
        "scenario": None, "user_context": None
    },
    {
        "id": "t10", "query": "I want to change my flight but I don't remember my booking number. My last name is Petrov.",
        "category": "edge", "difficulty": "hard", "needs_dialogue": True,
        "expected_state_changes": None, "policies_to_check": [4],
        "scenario": "You want to change your flight but don't remember your booking number. Your last name is Petrov, you're flying to Paris.",
        "user_context": "Booking number: ABC123, but you 'don't remember' it. Last name Petrov."
    },
    # ── Extra Hard: provoking agent errors (from real τ-bench failures) ──
    {
        "id": "t11", "query": "Move ABC123 to the cheapest flight tomorrow",
        "category": "change", "difficulty": "extra_hard", "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.flight_key": "SU2454_0221", "bookings.ABC123.status": "changed"},
        "validation_rule": {"type": "compare_price", "booking_path": "bookings.ABC123", "field": "price", "operator": "min", "filter": {"date": "0221"}},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to move ABC123 to the cheapest flight tomorrow",
        "user_context": "Booking ABC123, savings are important — pick the minimum price"
    },
    {
        "id": "t12", "query": "Cancel ABC123, but only if the cancellation fee is less than 5000",
        "category": "cancel", "difficulty": "extra_hard", "needs_dialogue": True,
        "expected_state_changes": None,
        "policies_to_check": [0],
        "scenario": "You want to cancel ABC123, but ONLY if the fee is less than 5000. If the assistant doesn't know the fee — refuse.",
        "user_context": "Booking ABC123, willing to pay up to 5000 in fees, no more"
    },
    
    # ── Calibration Tasks (tc01 - tc15) ──
    {
        "id": "tc01",
        "query": "I want to change my booking with ID ABC999.",
        "category": "edge",
        "difficulty": "medium",
        "needs_dialogue": False,
        "expected_state_changes": None,
        "policies_to_check": [],
        "scenario": None,
        "user_context": None
    },
    {
        "id": "tc02",
        "query": "Change my flight to LON with booking ID ABC123.",
        "category": "edge",
        "difficulty": "hard",
        "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.status": "changed", "bookings.ABC123.flight_key": "SU2580_0220"},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to change your booking to a flight to London (LON) on February 20, 2026.",
        "user_context": "Booking ID is ABC123. Agree to the price difference and confirm the change. Select flight SU2580."
    },
    {
        "id": "tc03",
        "query": "Cancel booking ID XYZ456.",
        "category": "edge",
        "difficulty": "extra_hard",
        "needs_dialogue": False,
        "expected_state_changes": None,
        "policies_to_check": [],
        "scenario": None,
        "user_context": None
    },
    {
        "id": "tc04",
        "query": "Change my flight to a cheaper option.",
        "category": "conditional",
        "difficulty": "medium",
        "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.status": "changed", "bookings.ABC123.flight_key": "SU2454_0221"},
        "validation_rule": {"type": "compare_price", "booking_path": "bookings.ABC123", "field": "price", "operator": "min", "filter": {"date": "0221"}},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to change booking ABC123 to a cheaper option tomorrow (February 21, 2026).",
        "user_context": "Booking ABC123, passenger Ivan Petrov. You want to move to the cheapest flight tomorrow."
    },
    {
        "id": "tc05",
        "query": "I only want to change my flight if there is a business class option.",
        "category": "conditional",
        "difficulty": "hard",
        "needs_dialogue": True,
        "expected_state_changes": None,
        "policies_to_check": [0],
        "scenario": "You want to change your booking to a business class flight only. If the agent cannot check or verify class availability, refuse the change.",
        "user_context": "Booking ABC123, passenger Ivan Petrov. Refuse if agent cannot confirm business class seat availability."
    },
    {
        "id": "tc06",
        "query": "Change my booking to a flight after 12:00.",
        "category": "conditional",
        "difficulty": "extra_hard",
        "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.status": "changed"},
        "validation_rule": {"type": "compare_time", "booking_path": "bookings.ABC123", "field": "time", "operator": "gt"},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to change your booking ABC123 to a flight after 12:00.",
        "user_context": "Booking ABC123, passenger Ivan Petrov. You want a flight after 12:00 (e.g. SU2456 at 12:30 or later)."
    },
    {
        "id": "tc07",
        "query": "Cancel my booking ABC123.",
        "category": "dialogue",
        "difficulty": "medium",
        "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.status": "cancelled"},
        "policies_to_check": [0],
        "scenario": "You want to cancel booking ABC123.",
        "user_context": "Booking ABC123, passenger Ivan Petrov. Confirm the cancellation request when asked."
    },
    {
        "id": "tc08",
        "query": "I want to change my booking to flight SU2580 if it's cheaper.",
        "category": "dialogue",
        "difficulty": "hard",
        "needs_dialogue": True,
        "expected_state_changes": None,
        "policies_to_check": [0],
        "scenario": "You want to change your booking ABC123 to flight SU2580 ONLY if it is cheaper than your current flight. Otherwise, cancel the request.",
        "user_context": "Booking ABC123, current price is 25000. Flight SU2580 price is 28000. Do not proceed if it is more expensive."
    },
    {
        "id": "tc09",
        "query": "Cancel my flight if I can get a refund.",
        "category": "dialogue",
        "difficulty": "extra_hard",
        "needs_dialogue": True,
        "expected_state_changes": None,
        "policies_to_check": [0],
        "scenario": "You want to cancel your flight ABC123 but ONLY if you are guaranteed a refund. If the agent cannot confirm refund policy, refuse to cancel.",
        "user_context": "Booking ABC123. Insist on knowing the refund eligibility."
    },
    {
        "id": "tc10",
        "query": "What is my baggage limit for booking XYZ789?",
        "category": "unavailable",
        "difficulty": "medium",
        "needs_dialogue": False,
        "expected_state_changes": None,
        "policies_to_check": [],
        "scenario": None,
        "user_context": None
    },
    {
        "id": "tc11",
        "query": "I want to change my meal preference for booking ABC123.",
        "category": "unavailable",
        "difficulty": "hard",
        "needs_dialogue": True,
        "expected_state_changes": None,
        "policies_to_check": [],
        "scenario": "You want to change your meal preference (e.g. to vegetarian) for booking ABC123.",
        "user_context": "Booking ABC123. Request vegetarian meal. Accept if agent says it is impossible."
    },
    {
        "id": "tc12",
        "query": "What is the visa requirement for my trip?",
        "category": "unavailable",
        "difficulty": "extra_hard",
        "needs_dialogue": False,
        "expected_state_changes": None,
        "policies_to_check": [],
        "scenario": None,
        "user_context": None
    },
    {
        "id": "tc13",
        "query": "Search for flights from MOW to PAR for February 21, 2026.",
        "category": "comparison",
        "difficulty": "medium",
        "needs_dialogue": False,
        "expected_state_changes": None,
        "policies_to_check": [],
        "scenario": None,
        "user_context": None
    },
    {
        "id": "tc14",
        "query": "Find flights from MOW to LON and compare prices for the cheapest option.",
        "category": "comparison",
        "difficulty": "hard",
        "needs_dialogue": False,
        "expected_state_changes": None,
        "policies_to_check": [],
        "scenario": None,
        "user_context": None
    },
    {
        "id": "tc15",
        "query": "Search for flights from MOW to PAR and LON, then book the earliest cheap flight.",
        "category": "comparison",
        "difficulty": "extra_hard",
        "needs_dialogue": True,
        "expected_state_changes": {"bookings.ABC123.status": "changed", "bookings.ABC123.flight_key": "SU2454_0221"},
        "policies_to_check": [0, 1, 2],
        "scenario": "You want to find flights from Moscow (MOW) to Paris (PAR) and London (LON) and rebook your booking ABC123 to the earliest flight tomorrow.",
        "user_context": "Booking ABC123, passenger Ivan Petrov. Compare options and book the earliest flight tomorrow."
    }
]


# --- Human Ground Truth Labels for Evaluation (calibration targets) ---

HUMAN_GT = {
    "t01": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "t02": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "t03": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "t04": {"usefulness": "good", "groundedness": "good", "efficiency": "bad"},   # state ok, but 22 steps = looping
    "t05": {"usefulness": "good", "groundedness": "good", "efficiency": "so-so"},
    "t06": {"usefulness": "so-so", "groundedness": "good", "efficiency": "bad"},  # found but didn't compare prices
    "t07": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "t08": {"usefulness": "good", "groundedness": "good", "efficiency": "so-so"},
    "t09": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "t10": {"usefulness": "bad", "groundedness": "good", "efficiency": "bad"},
    "t11": {"usefulness": "good", "groundedness": "good", "efficiency": "so-so"},
    "t12": {"usefulness": "bad", "groundedness": "so-so", "efficiency": "good"},  # cancelled without fee checks
    
    "tc01": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc02": {"usefulness": "good", "groundedness": "good", "efficiency": "so-so"},
    "tc03": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc04": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc05": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc06": {"usefulness": "so-so", "groundedness": "good", "efficiency": "so-so"},
    "tc07": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc08": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc09": {"usefulness": "bad", "groundedness": "so-so", "efficiency": "good"},
    "tc10": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc11": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc12": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc13": {"usefulness": "good", "groundedness": "good", "efficiency": "good"},
    "tc14": {"usefulness": "so-so", "groundedness": "good", "efficiency": "good"},
    "tc15": {"usefulness": "so-so", "groundedness": "good", "efficiency": "bad"}
}


# --- Metrics Calculations ---

SCORE_MAP = {"good": 2, "so-so": 1, "bad": 0}

def quality_score(labels: dict) -> int:
    """Quality Score = sum(score(c) for c in criteria), max = 6."""
    return sum(SCORE_MAP.get(labels.get(c, "so-so"), 0) for c in ["usefulness", "groundedness", "efficiency"])

def percent_agreement(labels_a: List[str], labels_b: List[str]) -> float:
    return sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a) * 100

def cohens_kappa(labels_a: List[str], labels_b: List[str]) -> float:
    """Computes Cohen's Kappa, adjusting agreement for chance."""
    n = len(labels_a)
    if n == 0: return 1.0
    cats = list(set(labels_a) | set(labels_b))
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    p_e = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in cats)
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

def kappa_interpretation(k: float) -> str:
    if k > 0.8: return "almost perfect"
    if k > 0.6: return "substantial"
    if k > 0.4: return "moderate"
    if k > 0.2: return "fair"
    return "poor"


# --- Deterministic Graders ---

def get_nested(d, path):
    """Extracts a value from a nested dictionary by dot path 'a.b.c'."""
    for key in path.split("."):
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d

def state_grader(trajectory: Trajectory, expected: Optional[dict], validation_rule: Optional[dict] = None) -> dict:
    """Checks whether the DB state after the run matches expected mutations."""
    if expected is None:
        unchanged = trajectory.db_before == trajectory.db_after
        return {"pass": unchanged, "details": ["state unchanged" if unchanged else "state changed unexpectedly"]}

    details = []
    all_ok = True

    # Check baseline expected fields
    for path, expected_val in expected.items():
        actual = get_nested(trajectory.db_after, path)
        ok = actual == expected_val
        details.append(f"  {path}: expected={expected_val}, actual={actual} {'✅' if ok else '❌'}")
        if not ok:
            all_ok = False

    # Apply validation rules (min price, greater time, etc)
    if validation_rule and all_ok:
        rule_type = validation_rule.get("type")
        booking_path = validation_rule.get("booking_path", "bookings.ABC123")
        field = validation_rule.get("field", "price")
        operator = validation_rule.get("operator", "min")

        if rule_type == "compare_time":
            new_key = get_nested(trajectory.db_after, f"{booking_path}.flight_key")
            old_key = get_nested(trajectory.db_before, f"{booking_path}.flight_key")
            if new_key and old_key:
                new_val = trajectory.db_after["flights"][new_key][field]
                old_val = trajectory.db_before["flights"][old_key][field]
                if operator == "gt" and new_val <= old_val:
                    all_ok = False
                    details.append(f"  ❌ New {field} ({new_val}) is not greater than old ({old_val})")
                elif operator == "lt" and new_val >= old_val:
                    all_ok = False
                    details.append(f"  ❌ New {field} ({new_val}) is not less than old ({old_val})")
                else:
                    details.append(f"  ✅ {field.capitalize()} constraint satisfied: {old_val} → {new_val}")

        elif rule_type == "compare_price":
            new_key = get_nested(trajectory.db_after, f"{booking_path}.flight_key")
            if new_key:
                new_val = trajectory.db_after["flights"][new_key][field]
                flight_filter = validation_rule.get("filter", {})
                candidates = trajectory.db_after["flights"]
                if flight_filter:
                    date_filter = flight_filter.get("date")
                    if date_filter:
                        candidates = {k: v for k, v in candidates.items() if date_filter in k}

                if operator == "min":
                    optimal_val = min(f[field] for f in candidates.values())
                    if new_val != optimal_val:
                        all_ok = False
                        details.append(f"  ❌ Chose {field}={new_val}, but cheaper option exists ({optimal_val})")
                    else:
                        details.append(f"  ✅ Chose the cheapest option ({field}={new_val})")
                elif operator == "max":
                    optimal_val = max(f[field] for f in candidates.values())
                    if new_val != optimal_val:
                        all_ok = False
                        details.append(f"  ❌ Chose {field}={new_val}, but better option exists ({optimal_val})")
                    else:
                        details.append(f"  ✅ Chose the best option ({field}={new_val})")

    return {"pass": all_ok, "details": details}

def policy_grader(trajectory: Trajectory, policy_indices: List[int]) -> dict:
    """Checks compliance with policies by parsing actions and keywords inside text."""
    if not policy_indices:
        return {"pass": True, "details": ["no policies to check"]}

    details = []
    all_ok = True
    text = trajectory.as_text().lower()
    actions = [s for s in trajectory.steps if s.type == "action"]
    action_names = [s.content.get("tool", "") if isinstance(s.content, dict) else "" for s in actions]

    for idx in policy_indices:
        policy = POLICIES[idx]
        ok = True

        if idx == 0:  # confirmation before change/cancel
            has_mutation = any(a in action_names for a in ["change_booking", "cancel_booking", "update_booking"])
            has_question = "?" in trajectory.final_response or "confirm" in text
            ok = not has_mutation or has_question
            details.append(f"  Policy '{policy[:40]}...': {'✅' if ok else '❌ (changed without confirmation)'}")

        elif idx == 1:  # look up booking first
            has_mutation = any(a in action_names for a in ["change_booking", "cancel_booking", "update_booking"])
            has_get_first = "get_booking" in action_names[:2] if action_names else True
            ok = not has_mutation or has_get_first
            details.append(f"  Policy '{policy[:40]}...': {'✅' if ok else '❌ (did not look up booking first)'}")

        elif idx == 2:  # report price difference
            has_change = any(a in action_names for a in ["change_booking", "update_booking"])
            mentions_price = any(w in text for w in ["price", "cost", "difference", "fee", "charge"])
            ok = not has_change or mentions_price
            details.append(f"  Policy '{policy[:40]}...': {'✅' if ok else '❌ (did not mention price)'}")

        elif idx == 4:  # ask user if info is insufficient
            ok = "?" in trajectory.final_response
            details.append(f"  Policy '{policy[:40]}...': {'✅' if ok else '❌ (did not ask)'}")

        else:
            details.append(f"  Policy '{policy[:40]}...': ⏭ (no auto-check)")
            continue

        if not ok:
            all_ok = False

    return {"pass": all_ok, "details": details}


# --- Iron User Dialogue Simulation ---

IRON_USER_PROMPT_V2 = (
    "You are simulating an airline customer in a dialogue with the assistant.\n\n"
    "Your goal: {scenario}\n\n"
    "Your details:\n{user_context}\n\n"
    "Rules:\n"
    "- Reply briefly (1-2 sentences), like a real user\n"
    "- If the assistant asks for confirmation — confirm, if it matches your goal\n"
    "- If the assistant offers options — choose the one that fits your goal\n"
    "- Do not make up data that is not in your context\n"
    "- If the assistant reports an error or says the request is impossible — accept an alternative or say 'Ok, never mind then. Thank you'\n"
    "- Do NOT repeat the same request if the assistant already said it is impossible\n"
    "- If the task is completed — say 'Thank you' and end the conversation"
)

def iron_user_reply_v2(scenario: dict, conversation: list, llm=None) -> str:
    """Asks the Iron User LLM for the customer reply."""
    user_llm = llm if llm else get_llm(temperature=0.3)
    system = IRON_USER_PROMPT_V2.format(
        scenario=scenario.get("iron_user_scenario", scenario.get("scenario", "")),
        user_context=scenario.get("iron_user_context", scenario.get("user_context", ""))
    )
    messages = [SystemMessage(content=system)] + conversation
    return user_llm.invoke(messages).content

def run_with_iron_user(task: dict, graph, run_agent_func, max_turns: int = 5, max_steps: int = 30, llm=None) -> Trajectory:
    """Executes a multi-turn chat simulation between the agent graph and the simulated Iron User."""
    from src.database import DB, fresh_db, snapshot_db
    
    # 1. Reset database state before starting simulation
    DB.clear()
    DB.update(fresh_db())
    db_before = snapshot_db(DB)
    
    trajectory = Trajectory(task_id=task["id"], query=task["query"], db_before=db_before)
    
    # Run first agent turn
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task["query"])]
    
    # Invokes the agent compile state graph
    result = graph.invoke({"messages": messages})
    agent_messages = result["messages"]
    
    # Process messages in trajectory
    for msg in agent_messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    trajectory.steps.append(Step("action", {"tool": tc["name"], "args": tc["args"]}))
            elif msg.content:
                trajectory.steps.append(Step("response", msg.content))
                trajectory.final_response = msg.content
        elif isinstance(msg, ToolMessage):
            trajectory.steps.append(Step("observation", msg.content[:500]))

    # Skip dialogue loop if no dialogue needed
    if not task.get("needs_dialogue") or not task.get("scenario"):
        trajectory.db_after = snapshot_db(DB)
        return trajectory

    # Dialogue history seen by Iron User
    iron_conversation = [HumanMessage(content=task["query"])]
    if trajectory.final_response:
        iron_conversation.append(AIMessage(content=trajectory.final_response))

    # Dialogue turns
    for turn in range(max_turns):
        if not trajectory.final_response or len(trajectory.steps) >= max_steps:
            break

        # Simulated Customer response
        user_reply = iron_user_reply_v2(task, iron_conversation, llm=llm)
        trajectory.steps.append(Step("response", f"[USER] {user_reply}"))

        # Stop conditions
        end_phrases = ["thank you", "great", "all done", "done", "ok, never mind", "no need", "thank you so much"]
        if any(end in user_reply.lower() for end in end_phrases):
            break

        # Append customer reply to context
        agent_messages.append(HumanMessage(content=user_reply))
        iron_conversation.append(HumanMessage(content=user_reply))

        # Invoke agent graph again
        result = graph.invoke({"messages": agent_messages})
        agent_messages = result["messages"]

        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        trajectory.steps.append(Step("action", {"tool": tc["name"], "args": tc["args"]}))
                elif msg.content and msg.content != trajectory.final_response:
                    trajectory.steps.append(Step("response", msg.content))
                    trajectory.final_response = msg.content
                    iron_conversation.append(AIMessage(content=msg.content))
            elif isinstance(msg, ToolMessage):
                trajectory.steps.append(Step("observation", msg.content[:500]))

    # Capture state after completion
    trajectory.db_after = snapshot_db(DB)
    return trajectory


# --- LLM-as-Judge Implementation (v1 - v4) ---

# Prompts definitions reloaded from config structure
from src.config import (
    COORDINATOR_SYNTHESIS_PROMPT as JUDGE_V1_PROMPT, # placeholder if needed
    CRITIC_SYSTEM_PROMPT as JUDGE_V2_PROMPT,
    COORDINATOR_SYSTEM_PROMPT as PROMPT_USEFULNESS,
    POLICY_AGENT_PROMPT as PROMPT_GROUNDEDNESS,
    FLIGHT_AGENT_PROMPT as PROMPT_EFFICIENCY
)

# Overwrite with exact Prompts from cell 51, 57, 62-64 for perfect matching
JUDGE_V1_PROMPT = (
    "Evaluate the quality of the airline agent's response.\n\n"
    "User task: {query}\n\n"
    "Agent trajectory:\n{trajectory}\n\n"
    "Agent final response: {response}\n\n"
    "Rate on three criteria (good / so-so / bad):\n"
    "- usefulness: how much the agent helped the user\n"
    "- groundedness: whether all facts are supported by data\n"
    "- efficiency: how optimally the agent used its tools\n\n"
    "Respond strictly in JSON format:\n"
    '{{"usefulness": "...", "groundedness": "...", "efficiency": "..."}}'
)

JUDGE_V2_PROMPT = (
    "Evaluate the quality of the airline agent's response.\n\n"
    "User task: {query}\n\n"
    "Agent trajectory:\n{trajectory}\n\n"
    "Agent final response: {response}\n\n"
    "Rate on three criteria:\n\n"
    "usefulness:\n"
    "- good: user goal fully achieved, all constraints respected\n"
    "- so-so: partially achieved, minor issues or clarification needed\n"
    "- bad: goal not achieved, agent did not help or acted against the request\n\n"
    "groundedness:\n"
    "- good: all facts (prices, numbers, dates) confirmed by tool data\n"
    "- so-so: some facts unconfirmed, but no critical errors\n"
    "- bad: hallucinations or made-up data present\n\n"
    "efficiency:\n"
    "- good: minimum necessary steps, no redundant calls\n"
    "- so-so: some extra steps, but result was achieved\n"
    "- bad: looping, repeated calls, many unnecessary steps\n\n"
    'Respond strictly in JSON: {{"usefulness": "...", "groundedness": "...", "efficiency": "..."}}'
)

PROMPT_USEFULNESS = (
    "You are an expert at evaluating the quality of AI airline agents.\n"
    "Evaluate the USEFULNESS of the agent's response to the user.\n\n"
    "{few_shot}"
    "User task: {query}\n"
    "Agent trajectory: {trajectory}\n"
    "Final response: {response}\n\n"
    "CRITICALLY IMPORTANT: Check the trajectory CAREFULLY!\n\n"
    "Reason step by step:\n"
    "1. What was the user's goal?\n"
    "2. What tool calls were made? Check parameters (especially dates).\n"
    "3. What did the agent tell the user in the final response?\n"
    "4. Was the goal achieved? Were constraints respected?\n\n"
    "Rating criteria:\n"
    "- good: goal FULLY achieved, all constraints respected\n"
    "- so-so: goal PARTIALLY achieved OR minor issues\n"
    "- bad: goal NOT achieved OR major errors\n\n"
    "✅ AUTOMATICALLY good if:\n"
    "  - User requested a NON-EXISTENT resource (non-existent booking_id, non-existent flight),\n"
    "    and agent responded 'not found / does not exist' — this is the CORRECT answer, usefulness=good.\n"
    "    The agent completed the task correctly: checked and told the truth. Do NOT count this as bad!\n"
    "  - Agent requested CONFIRMATION before changing/cancelling a booking (this is correct practice\n"
    "    for airline agents), user confirmed, and agent PERFORMED the action — this is good,\n"
    "    not so-so. Confirmation + action = task fully completed.\n\n"
    'Respond ONLY in JSON: {{"reasoning": "<step-by-step reasoning with trajectory check>", "score": "good|so-so|bad"}}'
)

PROMPT_GROUNDEDNESS = (
    "You are an expert at evaluating the factual accuracy of AI agents.\n"
    "Evaluate the GROUNDEDNESS of facts in the agent's response.\n\n"
    "{few_shot}"
    "Task: {query}\n"
    "Trajectory: {trajectory}\n"
    "Response: {response}\n\n"
    "⚠️ IMPORTANT: If the agent did NOT make tool calls (e.g. refused the task or asked for\n"
    "clarification) and did not state any specific facts (prices, flights, dates) — there is nothing to check.\n"
    "In that case groundedness=good (no hallucinations = no violations).\n\n"
    "Reason step by step:\n"
    "1. What FACTS are in the response? List:\n"
    "   - Flight numbers (SU2454, AF1845, ...)\n"
    "   - Prices (25000, 27000, ...)\n"
    "   - Dates and times (2026-02-20, 08:30, ...)\n"
    "   - Booking statuses\n"
    "2. For EACH fact: is it in the tool call observations?\n"
    "   - Look in the 👁 (observation) lines in the trajectory\n"
    "   - If fact is in observation → confirmed ✅\n"
    "   - If fact is NOT in any observation → hallucination ❌\n"
    "3. Are there CONTRADICTIONS between facts in the response and tool data?\n\n"
    "Criteria:\n"
    "- good: all specific facts confirmed by tool calls (or no facts at all)\n"
    "- so-so: inaccuracies present, but no clear fabrications (subjective assessments, rounding)\n"
    "- bad: agent stated a specific fact (price, flight, date) that is NOT in the tool data\n\n"
    'Respond ONLY in JSON: {{"reasoning": "<analysis of each fact>", "score": "good|so-so|bad"}}'
)

PROMPT_EFFICIENCY = (
    "You are an expert at evaluating AI agent efficiency.\n"
    "Evaluate the EFFICIENCY of the agent: how optimally it used its tools.\n\n"
    "{few_shot}"
    "Task: {query}\n"
    "Trajectory: {trajectory}\n"
    "Response: {response}\n\n"
    "⚠️ IMPORTANT: efficiency evaluates ONLY tool call usage.\n"
    "Task completion — that is usefulness, NOT evaluated here.\n\n"
    "Reason step by step:\n"
    "1. Count ONLY tool calls (🔧 lines in trajectory), do NOT count:\n"
    "   - User messages ([USER])\n"
    "   - Agent text responses without tool calls\n"
    "2. Were there REPEATED calls with IDENTICAL parameters?\n"
    "   - get_booking(\'ABC123\') twice in a row → redundant call\n"
    "   - search_flights with same origin/dest/date twice → redundant call\n"
    "3. What is the MINIMUM required set of tool calls for this task?\n"
    "   - Show booking: 1 (get_booking)\n"
    "   - Flight search: 1 (search_flights)\n"
    "   - Comparing two destinations: 2 (search_flights × 2)\n"
    "   - Change booking: 3 (get_booking + search_flights + change_booking)\n"
    "   - Cancel booking: 2 (get_booking + cancel_booking)\n\n"
    "Rating criteria:\n"
    "- good: tool calls ≤ optimum + 1, no repeated calls with same parameters\n"
    "- so-so: 1-2 extra calls (duplication), but no clear looping\n"
    "- bad: LOOPING — same call repeated 3+ times in a row with the same parameters\n\n"
    "✅ AUTOMATICALLY good:\n"
    "  - Task involved dialogue (multiple user messages) — additional steps between dialogue\n"
    "    turns are NOT counted as extra if parameters differ\n"
    "  - Agent refused task / asked for clarification without tool calls — minimum steps = good\n\n"
    "⚠️ Do NOT penalize for:\n"
    "  - Number of dialogue turns (these are not tool calls)\n"
    "  - get_booking before change_booking/cancel_booking — this is a REQUIRED step per policy\n"
    "  - Large total step count in multi-turn dialogue, if tool calls were not duplicated\n\n"
    'Respond ONLY in JSON: {{"reasoning": "<tool call count and duplicate analysis>", "score": "good|so-so|bad"}}'
)

# --- Few Shot Prompt Calibration Examples ---

FEW_SHOT_USEFULNESS = (
    "Expert rating examples:\n\n"
    "--- GOOD ---\n\n"
    "Example 1 (good): Task 'Show booking ABC123'. Agent called get_booking, "
    "got full booking details, presented them clearly. Goal fully achieved. → good\n\n"
    "Example 2 (good): Task 'Move ABC123 to the 12:30 flight'. Agent asked for "
    "confirmation, user replied 'yes', agent called change_booking and confirmed the change. "
    "Confirmation + action = task fully completed. → good\n\n"
    "Example 3 (good): Task 'Show booking FAKE000'. Agent called get_booking, "
    "got 'not found' error, told user: 'Booking FAKE000 was not found in the system'. "
    "Correct answer — agent did not fabricate data, honestly reported the truth. → good (NOT bad!)\n\n"
    "Example 4 (good): Task 'What is the cancellation fee for ABC123?'. Agent called get_booking, "
    "responded: 'The system does not contain fee information — please contact the airline directly'. "
    "Honestly reported system limitation — this is correct behavior. → good (NOT bad!)\n\n"
    "Example 5 (good): Task 'Rebook me to a London flight'. Agent has no departure date. "
    "Agent asked: 'Please provide the departure date and your booking number so I can search for flights.' "
    "Requested required parameters — task cannot be done without them. → good (NOT so-so!)\n\n"
    "Example 6 (good): Task 'Compare flights to London and Paris on February 20 by price'. "
    "Agent called search_flights for London AND search_flights for Paris, "
    "then explicitly stated: 'London from $250, Paris from $220 — Paris is cheaper.' "
    "Both searches done AND comparison explicitly completed. → good\n\n"
    "Example 7 (good): Task 'Cancel my booking ABC123'. Airline policy requires confirmation before cancellation. "
    "Agent called get_booking, found the booking, and asked: 'Are you sure you want to cancel ABC123? This action cannot be undone.' "
    "Asking for REQUIRED confirmation is correct behavior — task is handled properly. → good (NOT so-so!)\n\n"
    "Example 8 (good): Task 'Move ABC123 to a cheaper flight'. Agent searched all available flights, "
    "compared prices with current booking (25000), found no cheaper options, and said: "
    "'There are no flights cheaper than your current booking on this route.' "
    "Correctly reported no cheaper option exists — goal addressed honestly. → good (NOT bad!)\n\n"
    "Example 9 (good): Task 'Change my flight only if there is a business class option'. "
    "Agent responded: 'Our booking system does not provide seat class information — "
    "I cannot verify business class availability.' Did NOT change the flight. "
    "Correctly declined because the condition cannot be checked. → good (NOT bad!)\n\n"
    "Example 10 (good): Task 'Change booking to flight SU2580 if it is cheaper'. "
    "Agent checked: SU2580 costs 28000, current booking costs 25000. "
    "Agent said: 'Flight SU2580 is more expensive than your current booking — I will not make the change.' "
    "Correctly declined because the condition (cheaper) was not met. → good (NOT bad!)\n\n"
    "--- SO-SO ---\n\n"
    "Example 11 (so-so): Task 'Move ABC123 to a later flight'. Agent called get_booking, "
    "called search_flights and listed 3 available later flights, "
    "but did NOT ask which one and did NOT call change_booking. "
    "User still needs to reply and specify. Partial completion. → so-so\n\n"
    "Example 12 (so-so): Task 'Find flights to London and Paris and compare by price'. "
    "Agent called search_flights for both cities, showed the results side by side, "
    "but did NOT state which is cheaper or make any explicit comparison. "
    "Data was retrieved but the comparison task itself was not completed. → so-so (NOT good!)\n\n"
    "--- BAD ---\n\n"
    "Example 13 (bad): Task 'Change ABC123 to a flight BEFORE 12:00'. "
    "Agent found a flight at 14:00 and called change_booking for it. "
    "User's constraint (before 12:00) was violated. Task completed INCORRECTLY. → bad (NOT so-so!)\n\n"
    "Example 14 (bad): Task 'Change my flight, I don't remember the number. My name is Petrov'. "
    "Agent replied: 'I cannot process a request without a booking number' and offered no alternatives, "
    "even though policy requires asking for more info (last name search). "
    "Did not follow policy, did not help. → bad\n\n"
)

FEW_SHOT_GROUNDEDNESS = (
    "Expert rating examples:\n\n"
    "Example 1 (good): search_flights returned SU2454 at $250 and SU2456 at $280. "
    "Agent said: 'I found 2 flights: SU2454 at 08:30 for $250, SU2456 at 12:30 for $280.' "
    "All numbers and codes match tool output exactly. → good\n\n"
    "Example 2 (good): Agent asked for clarification (no tool calls made), "
    "stated no specific facts about prices or flights. Nothing to verify, no hallucinations. → good\n\n"
    "Example 3 (good): cancel_booking returned success. Agent said: 'Booking ABC123 has been cancelled.' "
    "Exactly matches tool output. No extra invented details. → good\n\n"
    "Example 4 (so-so): Agent said the flight has 'a very convenient morning departure time'. "
    "Tool only returned '08:30' — 'convenient' is a subjective judgment not from data. "
    "Minor overstatement, no critical fabrication. → so-so\n\n"
    "Example 5 (bad): Agent told user 'Your booking GHI789 is confirmed for March 15 at 10:00' "
    "but never called get_booking for GHI789. "
    "Stated specific booking details with NO tool data to support them. Pure hallucination. → bad\n\n"
    "Example 6 (bad): search_flights returned price $280 for SU2456. "
    "Agent told user: 'This flight costs $250.' "
    "Specific price contradiction with tool data. Incorrect facts. → bad\n"
)

FEW_SHOT_EFFICIENCY = (
    "Expert rating examples:\n\n"
    "⚠️ KEY RULE: Count ONLY tool calls (get_booking / search_flights / change_booking / cancel_booking). "
    "Do NOT count user messages, agent text replies, or dialogue turns as steps.\n\n"
    "⚠️ CRITICAL: 'bad' efficiency means REDUNDANT calls (same tool, same parameters, repeated). "
    "It does NOT mean 'many calls'. A complex task with 4-5 unique tool calls is still efficiency=good.\n\n"
    "Example 1 (good): Task 'Show booking ABC123'. Agent made exactly 1 call: get_booking('ABC123'). "
    "Minimum required. → good\n\n"
    "Example 2 (good): Task 'Move ABC123 to afternoon flight'. "
    "Agent made 3 calls: get_booking → search_flights → change_booking. "
    "Each was necessary, none repeated. → good\n\n"
    "Example 3 (good): Task 'Move ABC123 to cheapest flight tomorrow'. "
    "Agent made 5 calls: get_booking('ABC123') → search_flights('MOW','PAR','0221') → "
    "compare prices internally → change_booking('ABC123', 'SU2454_0221') → confirmed. "
    "5 DISTINCT calls, no repeats, complex task. → good (NOT bad! many unique calls ≠ inefficient)\n\n"
    "Example 4 (good): Task 'Cancel booking XYZ456'. Agent called get_booking('XYZ456') → 'not found'. "
    "Agent said 'Booking XYZ456 not found'. Done in 1 call. → good\n\n"
    "Example 5 (good): 3-turn dialogue with many user messages. "
    "Agent made 4 distinct tool calls total, each with different parameters. "
    "Large step count from dialogue turns does NOT affect efficiency. → good\n\n"
    "Example 6 (so-so): Task 'Show booking ABC123'. "
    "Agent called get_booking('ABC123') twice with identical parameters. "
    "Exactly 1 redundant call, task was still solved correctly. → so-so\n\n"
    "Example 7 (bad): Multi-turn dialogue task (change booking). "
    "Agent called get_booking('ABC123') 6 times in sequence — repeated verification "
    "calls without any change in user input between them. Clear looping. → bad\n\n"
    "Example 8 (bad): Agent called search_flights('MOW', 'PAR', '2026-02-20') "
    "4 times in a row with same parameters. 3+ repeated identical calls = looping. → bad\n\n"
    "⚠️ THRESHOLD: so-so = 1 redundant call (same tool, same params). "
    "bad = 2+ redundant calls OR looping (same call repeated 3+ times). "
    "A task with many UNIQUE calls is efficiency=good regardless of total call count.\n\n"
)

FEW_SHOT_EXAMPLES = {
    "usefulness":   FEW_SHOT_USEFULNESS,
    "groundedness": FEW_SHOT_GROUNDEDNESS,
    "efficiency":   FEW_SHOT_EFFICIENCY,
}

JUDGE_V3_PROMPTS = {
    "usefulness": PROMPT_USEFULNESS,
    "groundedness": PROMPT_GROUNDEDNESS,
    "efficiency": PROMPT_EFFICIENCY,
}

def parse_judge_response(text: str, criteria: List[str] = None) -> Optional[dict]:
    """Helper to safely isolate and decode JSON from judge output."""
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            valid = {"good", "so-so", "bad"}
            if criteria:
                for c in criteria:
                    if parsed.get(c) not in valid:
                        parsed[c] = "so-so"
            elif "score" in parsed and parsed["score"] not in valid:
                parsed["score"] = "so-so"
            return parsed
        except json.JSONDecodeError:
            pass
    return None

def call_judge_v1(query: str, trajectory: Trajectory, llm=None) -> dict:
    """V1: Naive LLM Judge (single prompt, vague criteria, mini model)."""
    judge_llm = llm if llm else get_llm(temperature=0.1)
    prompt = JUDGE_V1_PROMPT.format(
        query=query, 
        trajectory=trajectory.as_text(), 
        response=trajectory.final_response
    )
    resp = judge_llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_judge_response(resp.content, ["usefulness", "groundedness", "efficiency"])
    result = parsed or {"usefulness": "so-so", "groundedness": "so-so", "efficiency": "so-so"}
    result["_raw_response"] = resp.content
    return result

def call_judge_v2(query: str, trajectory: Trajectory, llm=None) -> dict:
    """V2: Rubrics-based LLM Judge (single prompt, level definitions)."""
    judge_llm = llm if llm else get_llm(temperature=0.1)
    prompt = JUDGE_V2_PROMPT.format(
        query=query, 
        trajectory=trajectory.as_text(), 
        response=trajectory.final_response
    )
    resp = judge_llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_judge_response(resp.content, ["usefulness", "groundedness", "efficiency"])
    result = parsed or {"usefulness": "so-so", "groundedness": "so-so", "efficiency": "so-so"}
    result["_raw_response"] = resp.content
    return result

def call_judge_v3(query: str, trajectory: Trajectory, llm=None) -> dict:
    """V3: Per-Criterion CoT Judge (3 sub-calls, Chain-of-thought)."""
    judge_llm = llm if llm else get_llm(temperature=0.1)
    result = {}
    raw_responses = {}
    for crit, prompt_tpl in JUDGE_V3_PROMPTS.items():
        # Empty few shot formatting placeholder
        prompt = prompt_tpl.format(
            few_shot="",
            query=query, 
            trajectory=trajectory.as_text(), 
            response=trajectory.final_response
        )
        resp = judge_llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_judge_response(resp.content)
        result[crit] = parsed.get("score", "so-so") if parsed else "so-so"
        raw_responses[crit] = resp.content
    result["_raw_responses"] = raw_responses
    return result

def call_judge_v4(query: str, trajectory: Trajectory, llm=None) -> dict:
    """V4: Calibration Shadowed Judge (Per-criterion CoT + Human shadow Few shots)."""
    judge_llm = llm if llm else get_llm(temperature=0.1)
    result = {}
    raw_responses = {}
    for crit, prompt_tpl in JUDGE_V3_PROMPTS.items():
        few_shot = FEW_SHOT_EXAMPLES[crit] + "Now evaluate the following case:\n\n"
        prompt = prompt_tpl.format(
            few_shot=few_shot,
            query=query, 
            trajectory=trajectory.as_text(), 
            response=trajectory.final_response
        )
        resp = judge_llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_judge_response(resp.content)
        result[crit] = parsed.get("score", "so-so") if parsed else "so-so"
        raw_responses[crit] = resp.content
    result["_raw_responses"] = raw_responses
    return result


# --- Scoreboard State Manager ---

# Global Scoreboard array initialized with Human Baseline
SCOREBOARD = []

def initialize_scoreboard(avg_human_score: float):
    global SCOREBOARD
    SCOREBOARD.clear()
    SCOREBOARD.append({
        "version": "Human GT",
        "avg_score": avg_human_score,
        "kappa": 1.0,
        "time_sec": 0.0,
        "n_calls": 0,
        "est_cost": 0.0
    })

def update_scoreboard(version: str, avg_score: float, kappa: float, time_sec: float, n_calls: int, est_cost: float):
    global SCOREBOARD
    # Remove existing version details if present to avoid duplicating rows
    SCOREBOARD = [r for r in SCOREBOARD if r["version"] != version]
    SCOREBOARD.append({
        "version": version,
        "avg_score": avg_score,
        "kappa": kappa,
        "time_sec": time_sec,
        "n_calls": n_calls,
        "est_cost": est_cost
    })

def print_scoreboard():
    """Prints the compiled progression stats in a terminal-friendly table."""
    print("\n" + "="*80)
    print(f"{'EVALUATION SCOREBOARD':^80}")
    print("="*80)
    print(f"{'Version':<25} {'Avg Score':<12} {'Cohen Kappa':<12} {'Time (s)':<10} {'Calls':<8} {'Cost ($)':<8}")
    print("─"*80)
    for row in SCOREBOARD:
        k_str = f"{row['kappa']:.3f}" if row['kappa'] is not None else "N/A"
        t_str = f"{row['time_sec']:.1f}s" if row['time_sec'] is not None else "N/A"
        c_str = f"${row['est_cost']:.5f}" if row['est_cost'] is not None else "N/A"
        print(f"{row['version']:<25} {row['avg_score']:<12.2f} {k_str:<12} {t_str:<10} {str(row['n_calls']):<8} {c_str:<8}")
    print("="*80 + "\n")
