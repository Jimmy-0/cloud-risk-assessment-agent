from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.core.state import AgentState
from src.core import nodes

checkpointer = MemorySaver()

REASONING_NODE = [
    "reason",
    "report",
    "summary",
    "insight",
    "assessment",
    "remediation",
    "effort",
    "conclude",
]


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("intent", nodes.classify_user_intent)
    builder.add_node("querydb", nodes.execute_db_query)
    builder.add_node("summary", nodes.generate_summary_report)
    builder.add_node("insight", nodes.generate_insights)
    builder.add_node("conclude", nodes.finalize_conclusion)
    builder.add_node("reason", nodes.provide_explanation)
    builder.add_node("report", nodes.invoke_llm)

    builder.add_edge(START, "intent")
    builder.add_edge("summary", "insight")
    builder.add_edge("insight", "conclude")
    builder.add_edge("querydb", "reason")
    builder.add_edge("conclude", END)
    builder.add_edge("reason", END)

    return builder.compile(checkpointer=checkpointer)


# Pre-build a default graph instance for use by the Chainlit app
graph = build_graph()
