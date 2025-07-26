import sys
from pathlib import Path

# 프로젝트 루트를 시스템 경로에 추가
# 이렇게 하면 다른 폴더에 있는 모듈을 임포트할 수 있습니다.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import dotenv
from langchain_core.messages import HumanMessage

from src.langgraph_workflow.build_workflow import build_workflow

# .env 파일 로드
dotenv.load_dotenv()

def main():
    """
    워크플로우를 실행하고 사용자와 상호작용하는 메인 함수
    """
    # 워크플로우 빌드
    app = build_workflow()

    # 사용자 입력
    inputs = {"messages": [HumanMessage(content="서울 날씨 어때?")]}

    # 워크플로우 실행 및 결과 출력
    for output in app.stream(inputs):
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