from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    reformulated_query: str
    chunks: List[str]
    answer: str
    critic_verdict: str
    critic_reason: str
    iterations: int
    final_answer: str