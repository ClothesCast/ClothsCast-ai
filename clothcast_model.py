import os
import json
from chroma_db import setup_chroma_db, ensure_vector_db
from config import DATASET_FOLDER, JSON_FILEPATH
from query_engine import query_db
from gpt_styling import generate_styling_recommendation_with_gpt

# ------------------------------------------------------------------------------
# 전체 파이프라인: PDF 벡터화 → JSON 파일로부터 실시간 사용자 입력 → 추천 생성 → (문서 기반 추가 활용)
# ------------------------------------------------------------------------------

def main():
    # (1) 벡터 데이터베이스 설정 및 업데이트 (로컬 PDF 파일 벡터화)
    pdf_vdb = setup_chroma_db()
    ensure_vector_db(pdf_vdb, DATASET_FOLDER)
    
    # (2) JSON 파일에서 실시간 사용자 데이터를 직접 로드
    if not os.path.exists(JSON_FILEPATH):
        print("지정한 JSON 파일이 존재하지 않습니다.")
        return
    with open(JSON_FILEPATH, "r", encoding="utf-8") as f:
        user_data = json.load(f)
    
    # (3) 사용자 입력 기반 쿼리 텍스트 생성 (예: 기온과 스타일 정보)
    temperature = user_data.get("main", {}).get("temp", 20)
    desired_style = user_data.get("style", "casual")
    query_text = f"Temperature: {temperature}, Desired style: {desired_style}"
    
    # (4) 벡터 DB에서 유사 PDF 문서 검색 후 결과 출력
    results = query_db(pdf_vdb, query_text, n_results=2)
    print("\n[검색된 PDF 문서 정보]")
    if results.get("documents") and results["documents"][0]:
        for idx, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][idx] or {}
            print(f"PDF ID: {meta.get('pdf_id', 'N/A')} (페이지 {meta.get('page', 'N/A')}) / Distance: {results['distances'][0][idx]} / File: {meta.get('file_path', 'N/A')}")
    else:
        print("검색된 PDF 문서가 없습니다.")
    
    # (5) LangChain을 활용하여 최종 스타일링 추천 생성
    styling_recommendation = generate_styling_recommendation_with_gpt(user_data)
    print("\n[AI 스타일링 추천]")
    print(styling_recommendation)
