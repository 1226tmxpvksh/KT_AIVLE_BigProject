import operator
from typing import List

from langchain_core.messages import FunctionMessage
from langchain_openai import ChatOpenAI

from src.langgraph_workflow.models.graph_state import AgentState
from src.regression_model.regression import RegressionModel

# 이 파일은 워크플로우의 노드들을 정의합니다.


def call_regression_model_node(state: AgentState, model: RegressionModel):
    """
    RegressionModel을 호출하여 응답을 생성합니다.
    """
    user_input = state["messages"][-1].content
    regression_model = RegressionModel(user_input)
    return_value = regression_model.evaluate()

    return {"regreesion_return": return_value,
            "messages": [return_value]}

def call_model_node(state: AgentState, model: ChatOpenAI, tools: list):
    """
    LLM을 호출하여 응답을 생성하거나 도구를 사용하도록 요청합니다.
    """
    response = model.invoke(state["messages"], tools=tools)
    return {"messages": [response]}

def call_tool_node(state: AgentState):
    """
    마지막 메시지에 있는 도구 호출을 실행합니다.
    """
    last_message = state["messages"][-1]
    # 실제 도구 실행 로직이 여기에 들어갑니다.
    # 여기서는 간단한 예시로 FunctionMessage를 생성합니다.
    tool_call = last_message.tool_calls[0]
    # 이 예시에서는 도구 실행 결과를 하드코딩합니다.
    # 실제로는 tool_call["name"]에 따라 적절한 함수를 호출해야 합니다.
    if tool_call["name"] == "search":
        # 간단한 검색 결과 시뮬레이션
        result = "서울의 현재 날씨는 맑음입니다."
        return {"messages": [FunctionMessage(content=result, tool_call_id=tool_call["id"])]}
    else:
        # 지원되지 않는 도구
        return {"messages": [FunctionMessage(content="알 수 없는 도구입니다.", tool_call_id=tool_call["id"])]}


def should_continue_node(state: AgentState):
    """
    다음에 어떤 노드를 실행할지 결정합니다.
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # LLM이 도구 사용을 요청한 경우
        return "tool"
    else:
        # LLM이 직접 답변한 경우
        return "end"
