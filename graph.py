from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import retrieve, generate, critic, reformulate


graph_builder = StateGraph(AgentState)


graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("critic", critic)
graph_builder.add_node("reformulate", reformulate)


graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", "critic")


def route_critic(state: AgentState):
    if state["critic_verdict"] == "pass":
        return "end"
    elif state.get("iterations", 0) >= 3:
        return "end"
    else:
        return "reformulate"

graph_builder.add_conditional_edges(
    "critic",
    route_critic,
    {
        "end": END,
        "reformulate": "reformulate"
    }
)

graph_builder.add_edge("reformulate", "retrieve")


graph = graph_builder.compile()