# ------------------------------------------------------------------------------
# 5. 전체 파이프라인: PDF 벡터화 → JSON 파일로부터 실시간 사용자 입력 → 추천 생성 → (문서 기반 추가 활용)
# ------------------------------------------------------------------------------
import os
import json
from chroma_db import setup_chroma_db, ensure_vector_db
from config import DATASET_FOLDER, JSON_FILEPATH
from query_engine import query_db
from gpt_styling import generate_styling_recommendation_with_gpt

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


# app = Flask(__name__)

# # Flask 앱 실행과 main() 실행 분리: 
# if __name__ == "__main__":
# #     # Flask 서버를 실행하려면 주석 해제
#     app.run(host="0.0.0.0", port=8000)
    
# #     # JSON 파일 기반 실행 (테스트용)
# main()

# @app.route('/recommand', methods=['POST'])
# def predict():
#     """
#     클라이언트로부터 POST 요청으로 JSON 데이터를 받습니다.
#     JSON에는 아래와 같은 형식이 포함되어야 합니다.
    
#     주요 키:
#       - ownedClothes: 상, 하, 아우터, 신발 정보 (불리언 값)
#       - style: 사용자 선호 스타일
#       - location: { "latitude": 숫자, "longitude": 숫자 }
#       - main: { "temp": 숫자, ... }
#       - weather: [ { "description": 문자열, ... } ]
      
#     이 정보들은 실시간으로 변경되므로, 매 요청마다 최신 정보를 사용합니다.
#     """
#     user_data = request.json

#     try:
#         recommendation = generate_styling_recommendation_with_gpt(user_data)
#         return jsonify({'result': recommendation})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    

