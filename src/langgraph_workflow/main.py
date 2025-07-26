import sys
from pathlib import Path

# 프로젝트 루트를 시스템 경로에 추가
# 이렇게 하면 다른 폴더에 있는 모듈을 임포트할 수 있습니다.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# import dotenv # 더 이상 사용하지 않음
from langchain_core.messages import HumanMessage
import os

from src.langgraph_workflow.build_workflow import build_workflow

# .env 파일을 수동으로 읽고 환경 변수 설정
try:
    dotenv_path = project_root / '.env'
    print(f"Attempting to read .env file from: {dotenv_path}")
    with open(dotenv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
                print(f"Manually set {key}")
except Exception as e:
    print(f"Error reading .env file manually: {e}")


# --- 환경 변수 로드 테스트 ---
api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded OPENAI_API_KEY: {api_key[:5]}..." if api_key else "API key not found")
# ---------------------------

def main():
    """
    워크플로우를 실행하고 사용자와 상호작용하는 메인 함수
    """
    # 워크플로우 빌드
    app = build_workflow()

    # 초기 상태 설정
    # user_input: 회귀 모델에 전달될 초기 데이터
    # messages: HumanMessage는 첫 번째 agent 노드에 전달될 메시지
    initial_state = {
        "user_input": "상품 A 데이터",
        "messages": [HumanMessage(content="이 상품에 대한 보고서를 작성해줘.")]
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