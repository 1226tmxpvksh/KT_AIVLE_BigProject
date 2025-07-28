from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

from src.langgraph_workflow.models.graph_state import AgentState
from src.regression_model.regression import RegressionModel
from src.sentiment_analysis.analysis import SentimentAnalyzer
from src.rag_retriever.retriever import RagRetriever


def call_regression_model_node(state: AgentState):
    """RegressionModel을 호출하여 예측을 수행하고 결과를 상태에 추가합니다."""
    print("---회귀 모델 호출---")
    user_input = state["user_input"]
    regression_model = RegressionModel()
    prediction = regression_model.predict(user_input)
    return {"regression_return": str(prediction)}

def call_sentiment_analysis_node(state: AgentState):
    """SentimentAnalyzer를 호출하여 고객 리뷰를 분석하고 결과를 상태에 추가합니다."""
    print("---고객 리뷰 감성 분석 호출---")
    reviews = state["customer_reviews"]
    analyzer = SentimentAnalyzer()
    analysis_result = analyzer.analyze(reviews)
    return {"sentiment_return": analysis_result}

def call_rag_node(state: AgentState):
    """RagRetriever를 호출하여 관련 시장 동향을 검색하고 요약합니다."""
    print("---시장 동향 분석(RAG) 호출---")
    query = state["user_input"]
    retriever = RagRetriever()
    rag_result = retriever.retrieve_and_summarize(query)
    return {"rag_return": rag_result}

def generate_report_node(state: AgentState):
    """
    지금까지 수집된 모든 데이터를 바탕으로 최종 보고서를 생성합니다.
    """
    print("---최종 보고서 생성 시작---")
    
    # 1. 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 데이터 기반 의사결정을 돕는 전문 비즈니스 분석가입니다.
        아래에 제공된 세 가지 핵심 데이터(정량 예측, 고객 반응, 시장 환경)를 바탕으로,
        "상품 현황 분석 및 향후 운영 방향 권고"에 대한 전문적인 보고서를 작성해주세요.
        
        보고서는 반드시 [최종 결론], [종합 평가], [세부 분석 및 판단 근거], [최종 권고 사항]의 목차로 구성되어야 합니다.
        각 데이터가 어떻게 최종 결론에 영향을 미쳤는지 명확하게 서술해주세요.

        ---
        [1. 정량 예측 데이터 (미래 판매량)]
        {regression_data}

        [2. 고객 반응 데이터 (리뷰 기반 감성 분석)]
        {sentiment_data}

        [3. 시장 환경 데이터 (RAG 기반 시장 조사)]
        {rag_data}
        ---
        
        이제 위 데이터를 바탕으로 보고서를 작성해주세요.
        """
    )
    
    # 2. LLM 모델 초기화
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # 3. 체인 생성 및 실행
    chain = prompt | llm
    
    # state의 딕셔너리 데이터를 보기 좋은 문자열로 변환
    sentiment_data_str = json.dumps(state['sentiment_return'], indent=2, ensure_ascii=False)
    
    response = chain.invoke({
        "regression_data": state['regression_return'],
        "sentiment_data": sentiment_data_str,
        "rag_data": state['rag_return']
    })
    
    print("---최종 보고서 생성 완료---")
    return {"final_report": response.content}
