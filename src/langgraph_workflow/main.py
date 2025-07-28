import sys
from pathlib import Path

# 프로젝트 루트를 시스템 경로에 추가
# 이렇게 하면 다른 폴더에 있는 모듈을 임포트할 수 있습니다.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from langchain_core.messages import HumanMessage
import os

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


def main():
    """
    워크플로우를 실행하고 사용자와 상호작용하는 메인 함수
    """
    # 워크플로우 빌드
    app = build_workflow()

    # 초기 상태 설정
    # user_input: 회귀 모델과 RAG에 전달될 초기 데이터
    # messages: HumanMessage는 첫 번째 agent 노드에 전달될 메시지
    initial_state = {
        "user_input": "AI 스피커 신제품 데이터",
        "customer_reviews": "이 상품 정말 좋아요! 배송도 빠르고 만족합니다. 그런데 가격이 조금 비싸네요. 배터리도 좀 빨리 닳는 것 같아요.",
        "messages": [HumanMessage(content="이 상품에 대한 종합 보고서를 작성해줘.")]
    }

    # 워크플로우 실행 및 결과 출력
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"--- {key} ---")
            if "messages" in value:
                for msg in value["messages"]:
                    msg.pretty_print()
            else:
                print(value)
        print("\n---\n")

if __name__ == "__main__":
    main() 