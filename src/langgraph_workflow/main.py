import sys
from pathlib import Path
import os
from langchain_core.messages import HumanMessage

# 프로젝트 루트를 시스템 경로에 추가
# 이렇게 하면 다른 폴더에 있는 모듈을 임포트할 수 있습니다.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.langgraph_workflow.build_workflow import build_workflow

# .env 파일을 수동으로 읽고 환경 변수 설정
try:
    dotenv_path = project_root / '.env'
    with open(dotenv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
except Exception as e:
    # .env 파일이 없거나 읽기 오류가 있어도 일단 진행
    pass


def run_analysis_pipeline(payload: dict):
    """
    백엔드로부터 받은 데이터(payload)를 기반으로 전체 분석 워크플로우를 실행하고,
    최종 보고서를 반환하는 메인 처리 함수입니다.
    """
    print("--- 분석 파이프라인 시작 ---")
    
    # 1. 워크플로우 빌드
    app = build_workflow()

    # 2. payload 데이터를 LangGraph의 AgentState 형식으로 변환
    #    - 회귀 모델 입력은 지금 문자열만 받으므로, 간단히 상품명과 첫 feature를 합칩니다.
    #    - 리뷰 리스트는 하나의 큰 텍스트 덩어리로 합칩니다.
    initial_state = {
        "user_input": f"{payload['product_name']} (월 판매량: {payload['regression_features']['monthly_sales']})",
        "customer_reviews": "\n".join(payload['review_list']),
    }

    # 3. 워크플로우 실행 및 최종 결과 캡처
    # 파이프라인 실행 및 최종 상태 확인
    final_state = app.invoke(initial_state)

    # 최종 상태에서 보고서 추출
    final_report = final_state.get("final_report", "보고서 생성에 실패했습니다.")
    
    print("--- 분석 파이프라인 종료 ---")
    return final_report


def main():
    """
    백엔드 API 서버를 시뮬레이션하는 메인 실행 함수입니다.
    """
    # 1. 백엔드가 API 요청을 받았다고 가정하고, 가짜 요청 데이터를 정의합니다.
    mock_api_request_payload = {
        "product_name": "AI 스피커",
        "regression_features": { "monthly_sales": 1500 },
        "review_list": [
            "이 상품 정말 좋아요! 배송도 빠르고 만족합니다.",
            "그런데 가격이 조금 비싸네요.",
            "배터리도 좀 빨리 닳는 것 같아요."
        ]
    }

    # 2. 메인 처리 함수를 호출하여 분석을 수행하고 최종 보고서를 받습니다.
    final_report = run_analysis_pipeline(mock_api_request_payload)

    # 3. 최종적으로 생성된 보고서를 출력합니다. (실제로는 이 결과를 API 응답으로 반환)
    print("\n\n\n--- 최종 생성된 보고서 ---")
    print(final_report)


if __name__ == "__main__":
    main() 