from functools import partial

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .nodes import call_model_node, call_tool_node, should_continue_node
from .models.graph_state import AgentState


@tool
def search(query: str):
    """
    주어진 쿼리에 대한 검색을 수행합니다.
    (이 함수는 실제 구현이 아닌 모의 함수입니다.)
    """
    print(f"Searching for: {query}")
    return "검색 결과: 서울의 날씨는 맑음."

def build_workflow():
    """
    LangGraph 워크플로우를 구축하고 컴파일합니다.
    """
    # LLM 및 도구 목록 초기화
    model = ChatOpenAI(temperature=0)
    tools = [search]

    # 노드 함수에 model과 tools를 바인딩
    call_model_with_tools = partial(call_model_node, model=model, tools=tools)

    # 그래프 정의
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("agent", call_model_with_tools)
    workflow.add_node("tool", call_tool_node)

    # 엣지 추가
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue_node,
        {
            "tool": "tool",
            "end": END,
        },
    )
    workflow.add_edge("tool", "agent")

    # 그래프 컴파일
    return workflow.compile()
