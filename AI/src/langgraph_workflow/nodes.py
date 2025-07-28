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
        sales_prediction = reg_data.get('sales_prediction', 'N/A')
        prediction_name = reg_data.get('name', 'N/A')
        
        reg_text = f"예상 매출액은 {sales_prediction}으로 예측됩니다."
        
        # 컨텍스트 정보가 있는 경우, 평가 문구 추가
        if 'context' in reg_data:
            avg_sales = reg_data['context'].get('avg_sales', 'N/A')
            max_sales = reg_data['context'].get('max_sales', 'N/A')
            reg_text += (
                f" 이 예측치는 해당 카테고리(또는 상품)의 과거 평균 월 매출({avg_sales}) 및 "
                f"최고 월 매출({max_sales}) 데이터와 비교하여 평가되었습니다."
            )

        if reg_data.get('type') == 'sub_category_fallback':
            reason = reg_data.get('reason', '데이터 부족으로 상위 카테고리 모델 사용')
            reg_text = f"제품 '{prediction_name}'에 대한 예측: {reg_text} ({reason})"

    except (json.JSONDecodeError, TypeError):
        reg_text = f"회귀 분석 결과(문자열): {state['regression_return']}"

    # 2. 감성 분석 결과(dict)를 텍스트로 변환
    sentiment_text = json.dumps(state.get("sentiment_analysis_result", {}), indent=2, ensure_ascii=False)
    
    # 3. RAG 결과(str)
    rag_text = state.get("rag_result", "데이터 없음")

    # 4. LLM에 전달할 프롬프트 정의
    prompt_template = f"""
당신은 냉철한 비즈니스 분석가입니다. 주어진 데이터를 바탕으로 특정 상품의 현황을 분석하고, 향후 운영 방향을 결정하여 상세한 보고서를 작성해야 합니다.

아래의 3가지 데이터를 사용하여 종합적으로 판단하고, 최종 보고서를 "최종 결론", "종합 평가", "세부 분석 및 판단 근거", "최종 권고 사항"의 4가지 항목으로 나누어 작성해주세요.

---
[데이터 1: 정량 예측 데이터 (미래 판매량)]
{reg_text}

[데이터 2: 고객 반응 데이터 (리뷰 기반 감성 분석)]
{sentiment_text}

[데이터 3: 시장 환경 데이터 (RAG 기반 시장 조사)]
{rag_text}
---

[보고서 작성 가이드라인]
- **세부 분석**: 각 데이터(정량, 고객 반응, 시장 환경)를 개별적으로 심층 분석하고, 그 의미를 해석해주세요.
- **종합 판단**: 개별 분석들을 종합하여, 이 상품이 현재 시장에서 어떤 상태인지 (성장 가능성, 유지 필요, 개선 시급, 위험 등) 명확하게 평가해주세요.
- **최종 권고**: 종합 판단을 바탕으로, 앞으로 이 상품을 어떻게 운영해야 할지에 대한 구체적인 권고 사항을 제시해주세요. (예: 제품 개선, 마케팅 강화, 가격 조정 등)
- **단종 고려 조건**: 만약 모든 데이터를 종합적으로 분석했을 때, **판매량 예측치가 과거 평균에 비해 현저히 낮고, 고객 리뷰에서 심각한 결함이 지속적으로 언급되며, 시장 트렌드도 비관적이라면** '제품 단종 고려'를 포함한 과감한 권고를 내릴 수 있습니다. 그 외의 경우에는 구체적인 개선안을 제시해주세요.

결과는 반드시 한국어로, 위에 제시된 4가지 항목을 모두 포함한 보고서 형식으로만 출력해주세요.
"""
    
    # 5. LLM 모델 초기화 및 체인 실행
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    # f-string으로 완성된 프롬프트를 사용하므로, ChatPromptTemplate.from_template을 직접 사용하지 않습니다.
    # 대신, 완성된 문자열을 HumanMessage로 감싸서 전달합니다.
    chain = llm
    
    response = chain.invoke(prompt_template)
    
    print("---최종 보고서 생성 완료---")
    return {"final_report": response.content}
