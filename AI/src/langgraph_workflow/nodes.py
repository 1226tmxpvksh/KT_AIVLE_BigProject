from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

from .models.graph_state import AgentState # 주석 해제 및 임포트
from src.regression_model.regression import RegressionModel
from src.sentiment_analysis.analysis import SentimentAnalyzer
from src.rag_retriever.retriever import RagRetriever

# AgentState는 이제 직접 사용되지 않으므로 제거합니다.
# from .models.graph_state import AgentState 

# 노드 함수들은 이제 state 딕셔너리를 직접 받습니다.

def call_regression_model_node(state: AgentState): # 타입을 AgentState로 변경
    """RegressionModel을 호출하고 결과를 JSON 문자열로 상태에 추가합니다."""
    print("---회귀 모델 호출---")
    user_input = state["user_input"]
    # RegressionModel은 한 번만 초기화되어야 효율적이지만, 여기서는 단순성을 위해 매번 생성합니다.
    regression_model = RegressionModel()
    prediction_json = regression_model.predict(user_input)
    return {"regression_return": prediction_json}

def call_sentiment_analysis_node(state: AgentState): # 타입을 AgentState로 변경
    """SentimentAnalyzer를 호출하여 고객 리뷰를 분석하고 결과를 상태에 추가합니다."""
    print("---고객 리뷰 감성 분석 호출---")
    reviews = state["customer_reviews"]
    analyzer = SentimentAnalyzer()
    analysis_result = analyzer.analyze(reviews)
    return {"sentiment_return": analysis_result}

def call_rag_node(state: AgentState): # 타입을 AgentState로 변경
    """RagRetriever를 호출하여 관련 시장 동향을 검색하고 요약합니다."""
    print("---시장 동향 분석(RAG) 호출---")
    query = state["user_input"]
    retriever = RagRetriever()
    rag_result = retriever.retrieve_and_summarize(query)
    return {"rag_return": rag_result}

def generate_report_node(state: AgentState): # 타입을 AgentState로 변경
    """
    지금까지 수집된 모든 데이터를 바탕으로 최종 보고서를 생성합니다.
    """
    print("---최종 보고서 생성 시작---")
    
    # 1. 회귀 분석 결과(JSON) 파싱 및 자연스러운 문장으로 변환
    try:
        reg_data = json.loads(state['regression_return'])
        reg_type = reg_data.get('type', 'error')
        
        if reg_type == 'product_specific':
            reg_text = f"제품 '{reg_data['name']}'의 고유 데이터를 기반으로 예측한 다음 달 예상 매출액은 {reg_data['value']} 입니다."
        elif reg_type == 'sub_category_fallback':
            reg_text = f"해당 제품의 데이터가 부족하여, 상위 카테고리 '{reg_data['name']}'의 데이터를 기반으로 예측한 다음 달 예상 매출액은 {reg_data['value']} 입니다."
        elif reg_type == 'sub_category_specific':
            reg_text = f"카테고리 '{reg_data['name']}' 전체의 데이터를 기반으로 예측한 다음 달 예상 매출액은 {reg_data['value']} 입니다."
        else: # error case
            reg_text = f"예측 실패: {reg_data.get('reason', '알 수 없는 오류')}"
            
    except (json.JSONDecodeError, KeyError):
        reg_text = "회귀 분석 결과 처리 중 오류가 발생했습니다."

    # 2. 감성 분석 결과(dict)를 텍스트로 변환
    sentiment_data_str = json.dumps(state['sentiment_return'], indent=2, ensure_ascii=False)

    # 3. 프롬프트 템플릿 정의
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
    
    # 4. LLM 모델 초기화 및 체인 실행
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    chain = prompt | llm
    
    response = chain.invoke({
        "regression_data": reg_text,
        "sentiment_data": sentiment_data_str,
        "rag_data": state['rag_return']
    })
    
    print("---최종 보고서 생성 완료---")
    return {"final_report": response.content}
