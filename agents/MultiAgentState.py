from typing import Any, Dict, List
from typing import TypedDict,Annotated
from .MessageManager import MessagerManager
from langchain_core.messages import AnyMessage


# 多智能体状态定义
class MultiAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], MessagerManager(max_woking_memory=100, max_history=500)]
    current_agent: str
    task_plan: str
    execution_result: str
    evaluation_result: str
    user_query: str
    agent_history: List[Dict[str, Any]]
    step: str
    completed: bool
    execution_count: int
    planned_tools: List[str]
    executed_tools: List[str]
    planned_tool_calls: List[Dict[str, Any]]
    executed_tool_calls: List[Dict[str, Any]]
    current_tool_call_index: int

if __name__ == "__main__":
    print("no error")