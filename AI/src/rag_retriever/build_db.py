import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import gc
import torch
import os

# --- 설정 ---
# 프로젝트 루트 디렉토리의 절대 경로를 찾습니다.
# 이 파일(build_db.py)의 위치를 기준으로 상위 폴더로 이동하여 경로를 구성합니다.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# CSV 파일 경로 (data 폴더를 기준으로)
CSV_FILE_PATH = os.path.join(PROJECT_ROOT, "AI", "data", "news_with_keywords_global.csv")
# ChromaDB 저장 경로 (프로젝트 루트에 chroma_db 폴더 생성)
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")
# 사용할 임베딩 모델
EMBEDDING_MODEL = 'upskyy/bge-m3-korean'
# ChromaDB 컬렉션 이름
COLLECTION_NAME = "news_trend"

def build_database():
    """
    CSV 파일에서 데이터를 읽어 ChromaDB 벡터 데이터베이스를 구축합니다.
    """
    print("--- ChromaDB 데이터베이스 구축 시작 ---")

    # 1. 임베딩 모델 준비
    print(f"임베딩 모델 로딩: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 2. CSV 데이터 불러오기
    print(f"CSV 파일 로딩: {CSV_FILE_PATH}")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"오류: {CSV_FILE_PATH}를 찾을 수 없습니다. 'data' 폴더에 파일이 있는지 확인하세요.")
        return

    df = pd.read_csv(CSV_FILE_PATH)
    print(f"원본 데이터 행 수: {len(df)}")

    # 3. 데이터 정제 (중복 제거)
    df.drop_duplicates(subset=["desc", "content", "keyword", "global_keywords"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"중복 제거 후 데이터 행 수: {len(df)}")

    # 4. 크로마DB 클라이언트 및 컬렉션 준비
    print(f"ChromaDB 준비. 저장 경로: {PERSIST_DIRECTORY}")
    client = chromadb.Client(Settings(persist_directory=PERSIST_DIRECTORY))
    
    # 기존 컬렉션이 있다면 삭제 (새로운 데이터로 완전히 대체)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"기존 컬렉션 '{COLLECTION_NAME}'을 삭제합니다.")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"'{COLLECTION_NAME}' 컬렉션 생성 완료.")

    # 5. 벡터화 및 적재를 위한 데이터 준비
    texts = []
    metadatas = []
    ids = []

    print("DB에 적재할 데이터 포맷팅 중...")
    for idx, row in df.iterrows():
        # title, media, date가 null 값인 경우를 대비하여 안전하게 처리
        title = str(row.get("title", "")).strip()
        media = str(row.get("media", "")).strip()
        date = str(row.get("date", "")).strip()

        for field, doc_type in [
            ("desc", "desc"),
            ("content", "content"),
            ("keyword", "keyword"),
            ("global_keywords", "global_kw")
        ]:
            text = str(row.get(field, "")).strip()
            if text and text.lower() != "nan":
                texts.append(text)
                metadatas.append({
                    "type": doc_type,
                    "title": title,
                    "media": media,
                    "date": date,
                })
                ids.append(f"{doc_type}_{idx}")
    
    if not texts:
        print("오류: DB에 추가할 텍스트 데이터가 없습니다.")
        return

    # 6. 배치 임베딩 및 ChromaDB에 적재
    print(f"{len(texts)}개의 텍스트에 대한 임베딩 생성 및 DB 적재 중...")
    try:
        vectors = embedder.encode(
            texts,
            batch_size=32,  # 배치 크기를 낮춰 메모리 문제 방지
            show_progress_bar=True
        )

        collection.add(
            documents=texts,
            embeddings=[vec.tolist() for vec in vectors],
            metadatas=metadatas,
            ids=ids
        )
        print("ChromaDB 벡터DB 적재 완료!")

    except Exception as e:
        print(f"DB 적재 중 오류 발생: {e}")
        print("메모리 부족 문제일 수 있습니다. 배치 크기를 줄여보세요.")

    finally:
        # 7. 메모리 정리
        del embedder, vectors, texts, metadatas, ids, df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("메모리 정리 완료.")

if __name__ == "__main__":
    build_database() 