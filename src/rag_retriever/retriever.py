import time

class RagRetriever:
    def __init__(self):
        """
        실제로는 여기서 Vector DB에 연결해야 합니다.
        지금은 가짜 DB (파이썬 딕셔너리)를 사용합니다.
        """
        print("RagRetriever 초기화. (가짜 DB 사용)")
        self.fake_db = {
            "AI 스피커": "AI 스피커 시장은 매년 20%씩 성장하고 있으며, 주요 경쟁사는 구글과 아마존입니다. 최근에는 화면이 달린 '스마트 디스플레이' 형태의 제품이 인기를 끌고 있습니다.",
            "스마트 워치": "최근 스마트 워치 시장의 핵심 기능은 혈당 측정 기능이며, 젊은 층에서 인기가 높습니다. 애플 워치가 시장 점유율 1위를 유지하고 있습니다.",
            "무선 이어폰": "무선 이어폰 시장은 '노이즈 캔슬링' 기능이 기본으로 자리 잡았으며, 저가형과 고급형 시장으로 양분화되는 추세입니다."
        }
        # LLM을 사용한 요약 기능을 시뮬레이션하기 위한 가짜 LLM
        # 실제로는 ChatOpenAI 같은 객체가 될 것입니다.
        self.fake_llm = lambda query, context: f"'{query}'에 대한 시장 동향 요약: {context}"

    def retrieve_and_summarize(self, query: str) -> str:
        """
        주어진 쿼리(상품명 등)에 대해 DB에서 관련 문서를 검색하고 요약합니다.
        """
        print(f"RAG 검색 시작: '{query}'")
        time.sleep(1) # 실제 검색 및 요약에 걸리는 시간을 시뮬레이션

        # 1. 검색 (Retrieve)
        # fake_db에서 query와 관련된 내용을 찾습니다. 여기서는 간단하게 키워드 포함 여부로 확인합니다.
        retrieved_context = ""
        for key, value in self.fake_db.items():
            if key in query:
                retrieved_context = value
                break
        
        if not retrieved_context:
            print("관련 시장 동향 데이터를 찾을 수 없습니다.")
            return "관련 시장 동향 데이터를 찾을 수 없습니다."

        # 2. 생성 (Generate / Summarize)
        # 검색된 내용을 바탕으로 LLM이 최종 답변을 생성(요약)합니다.
        summary = self.fake_llm(query, retrieved_context)
        print("RAG 요약 완료.")
        return summary 