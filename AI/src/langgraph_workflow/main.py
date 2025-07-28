import sys
from pathlib import Path
import os
from typing import List

# 경로 설정
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / 'AI'))

from src.langgraph_workflow.build_workflow import build_workflow

# .env 파일 로드 (경로 견고성 강화)
try:
    dotenv_path = project_root / 'AI' / '.env'
    if not dotenv_path.exists():
        dotenv_path_alt = project_root / '.env'
        if dotenv_path_alt.exists():
            dotenv_path = dotenv_path_alt

    with open(dotenv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip().strip("'\"")
except Exception:
    pass # .env 파일 없어도 진행

def load_reviews_from_file(file_path: Path) -> List[str]:
    """텍스트 파일에서 리뷰 목록을 읽어옵니다."""
    if not file_path.is_file():
        print(f"!!! 경고: 리뷰 파일을 찾을 수 없습니다: {file_path}")
        return ["리뷰 데이터가 없습니다."]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]
        return reviews
    except Exception as e:
        print(f"!!! 에러: 리뷰 파일 읽기 중 오류 발생: {e}")
        return ["리뷰 데이터를 읽는 데 실패했습니다."]

def run_analysis_pipeline(payload: dict):
    """
    백엔드로부터 받은 데이터(payload)를 기반으로 전체 분석 워크로우를 실행합니다.
    """
    print("--- 분석 파이프라인 시작 ---")
    
    app = build_workflow()

    # user_input을 형식에 맞게 조합
    user_input_str = f"{payload['product_name']}, 예측 기간: {payload['prediction_period']}"
    
    # 파일 경로에서 리뷰 로드
    review_file_path = project_root / payload['review_file_path']
    reviews = load_reviews_from_file(review_file_path)

    initial_state = {
        "user_input": user_input_str,
        "customer_reviews": "\n".join(reviews),
    }

    final_state = app.invoke(initial_state)
    final_report = final_state.get("final_report", "보고서 생성에 실패했습니다.")
    
    print("--- 분석 파이프라인 종료 ---")
    return final_report


def main():
    """
    백엔드 API 서버를 시뮬레이션하는 메인 실행 함수입니다.
    """
    mock_api_request_payload = {
        "product_name": "Bretford CR4500 Series Slim Rectangular Table", 
        "prediction_period": "다음 1개월",
        "review_file_path": "data/table_reviews.txt"
    }

    final_report = run_analysis_pipeline(mock_api_request_payload)

    print("\n\n\n--- 최종 생성된 보고서 ---")
    print(final_report)


if __name__ == "__main__":
    # pandas에서 날짜 파싱 관련 경고가 나올 수 있으나, 실행에 문제는 없습니다.
    main() 