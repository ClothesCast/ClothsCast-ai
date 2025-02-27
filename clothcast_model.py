import os
import json
from dotenv import load_dotenv
import openai
from chromadb import PersistentClient
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
from langchain.schema import BaseOutputParser

# ------------------------------------------------------------------------------
# 1. 환경 변수 및 API 키 설정
# ------------------------------------------------------------------------------
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# openai.api_key = OPENAI_API_KEY

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# # DATASET_FOLDER는 PDF 파일들이 저장된 폴더입니다.
# DATASET_FOLDER = os.path.join(SCRIPT_DIR, "fashion_dataset_text")
# # JSON 파일 경로
# JSON_FILEPATH = os.path.join(SCRIPT_DIR, "user_data.json")

# ------------------------------------------------------------------------------
# 2. Chroma 벡터 데이터베이스 설정 및 PDF 텍스트 벡터화
# ------------------------------------------------------------------------------
# def setup_chroma_db():
#     """
#     Chroma 벡터 데이터베이스를 persistent client를 사용해 설정합니다.
#     OpenAI 임베딩 함수를 활용하여 PDF 파일 내 텍스트 데이터를 벡터화합니다.
#     """
#     vdb_path = os.path.join(SCRIPT_DIR, "pdf_vdb")
#     chroma_client = PersistentClient(path=vdb_path)
#     embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#     # EmbeddingFunction 인터페이스 변경에 따른 래퍼 클래스 정의
#     class EmbeddingWrapper:
#         def __init__(self, model):
#             self.model = model

#         def __call__(self, input: list) -> list:
#             return self.model.embed_documents(input)

#     embedding_fn = EmbeddingWrapper(embedding_model)

#     pdf_vdb = chroma_client.get_or_create_collection(
#         name="pdf",
#         embedding_function=embedding_fn
#     )
#     return pdf_vdb

# def add_pdfs_to_db(pdf_vdb, dataset_folder, max_docs=101):
#     """
#     dataset_folder 내의 모든 PDF 파일을 재귀적으로 검색하여,
#     각 PDF의 각 페이지를 별도의 문서로 처리해 벡터 DB에 추가합니다.
#     각 문서는 고유 ID를 가지며, 예를 들어 파일명이 "sample.pdf"이고 3페이지라면
#     ID는 "sample_page3" 형식으로 생성됩니다.
#     """
#     ids = []
#     documents = []
#     metadatas = []
#     count = 0

#     for root, dirs, files in os.walk(dataset_folder):
#         for filename in sorted(files):
#             if filename.lower().endswith('.pdf'):
#                 file_path = os.path.join(root, filename)
#                 relative_path = os.path.relpath(file_path, dataset_folder)
#                 base_pdf_id = os.path.splitext(relative_path.replace(os.sep, "_"))[0]
#                 try:
#                     loader = PyPDFLoader(file_path)
#                     docs = loader.load()
#                 except Exception as e:
#                     print(f"PDF 로드 중 에러 발생 ({file_path}):", e)
#                     continue
#                 for i, doc in enumerate(docs):
#                     page_text = doc.page_content
#                     doc_id = f"{base_pdf_id}_page{i+1}"
#                     ids.append(doc_id)
#                     documents.append(page_text)
#                     metadatas.append({"pdf_id": base_pdf_id, "file_path": file_path, "page": i+1})
#                     count += 1
#                     if count >= max_docs:
#                         break
#             if count >= max_docs:
#                 break
#         if count >= max_docs:
#             break

#     if not ids or not documents:
#         print("데이터셋 폴더에서 PDF 파일을 찾지 못했습니다. 폴더 구조와 파일명을 확인해주세요.")
#         return

#     pdf_vdb.add(documents=documents, metadatas=metadatas, ids=ids)
#     print(f"총 {count} 개의 페이지가 벡터 DB에 성공적으로 추가되었습니다.")

# def ensure_vector_db(pdf_vdb, dataset_folder):
#     """
#     벡터 DB에 PDF 데이터가 없다면, 로컬 데이터셋 폴더의 PDF 파일들을 추가합니다.
#     """
#     if pdf_vdb.count() == 0:
#         add_pdfs_to_db(pdf_vdb, dataset_folder)
#     else:
#         print("벡터 DB에 PDF 데이터가 이미 존재합니다. 추가 작업을 생략합니다.")

# ------------------------------------------------------------------------------
# 3. 사용자 JSON 데이터를 이용한 체인 실행 및 쿼리 생성
# ------------------------------------------------------------------------------


# def query_db(pdf_vdb, query_text, n_results=2):
#     """
#     벡터 DB에 대해 주어진 쿼리 텍스트를 사용하여 유사 PDF 문서를 검색합니다.
#     결과에는 metadatas, documents, distances가 포함됩니다.
#     """
#     results = pdf_vdb.query(
#         query_texts=[query_text],
#         n_results=n_results,
#         include=["metadatas", "documents", "distances"]
#     )
#     return results

# ------------------------------------------------------------------------------
# 4. LangChain을 활용한 GPT API 호출 및 최종 스타일링 추천 생성 (동적 변수 전달)
# ------------------------------------------------------------------------------
# def generate_styling_recommendation_with_gpt(user_data: dict) -> str:
#     """
#     사용자 데이터를 기반으로 LangChain을 활용해 GPT에게 프롬프트를 전달하여
#     최종 스타일링 추천 문장을 생성합니다.
#     실시간으로 전달되는 정보를 동적으로 프롬프트에 반영합니다.
#     """
#     location_data = user_data.get("location", {})
#     latitude = location_data.get("latitude", 0)
#     longitude = location_data.get("longitude", 0)
#     location_str = "좌표 기반 위치 정보"  # [수정됨] 실제 위치 변환 로직 추가 가능

#     temperature = user_data.get("main", {}).get("temp", {})
#     weather_info = user_data.get("weather", [{}])[0]
#     weather_desc = weather_info.get("description", {})
#     desired_style = user_data.get("style", {})
    
#     owned_clothes = user_data.get("ownedClothes", {})
#     topwear = owned_clothes.get("topwear", {})
#     tops = ', '.join([item for item, owned in topwear.items() if owned])
#     bottomwear = owned_clothes.get("bottomwear", {})
#     bottoms = ', '.join([item for item, owned in bottomwear.items() if owned])
#     outerwear = owned_clothes.get("outerwear", {})
#     outer = ', '.join([item for item, owned in outerwear.items() if owned])
    
#     shoes_items = []
#     for shoe in ["sneakers", "boots", "sandals", "sportsShoes"]:
#         if owned_clothes.get(shoe, False):
#             shoes_items.append(shoe)
#     shoes = ', '.join(shoes_items)

#     temperature_guide = """
# [온도별 로직]
# - 영하 5도 이하: shortPadding, longPadding, knit, sweatshirt, hoodie, cottonPants
# - 영하 5도 초과 0도 이하 : shortPadding, knit, sweatshirt, hoodie, cottonPants, denimPants, slacks
# - 영상 0도 초과 영상 5도 이하 : shortPadding, coat, knit, sweatshirt, hoodie, denimPants, cottonPants, slacks, longSkirt, slacks
# - 영상 5도 초과 영상 10도 이하 : coat, jacket, sweatshirt, hoodie, shirt, denimPants, slacks, longSkirt
# - 영상 10도 초과 영상 15도 이하 : jacket, cardigan, zipup, sweatshirt, shirt, denimPants, slacks, miniSkirt, longSkirt
# - 영상 15도 초과 영상 20도 이하 : cardigan, zipup, shirt, sweatshirt, denimPants, slacks, longSkirt, miniSkirt
# - 영상 20도 초과 영상 25도 이하 : shortSleeve, shirt, slacks, miniSkirt, denimPants
# - 영상 25도 초과 : sleeveless, shortSleeve, shortPants, miniSkirt, slacks, denimPants
# - 영하 5도 이하, 영하 5도 초과 영상 10도 이하 신발 : boots
# - 영상 25도 초과 신발 : sandals
# - boots, sandals을 제외한 sneakers, sportsShoes는 모든 온도에 반영해줘 : sneakers, sportsShoes
# """

#     prompt_template = PromptTemplate(
#         input_variables=["location_str", "latitude", "longitude", "temperature", "weather_desc", "desired_style", "outer", "tops", "bottoms", "shoes", "temperature_guide"],
#         template="""
# 너는 전문 패션 스타일리스트야. 아래 사용자 정보를 참고해서 최종 스타일링 추천 문장을 만들어줘.
# 실시간 위치 좌표: 위도 {latitude}, 경도 {longitude}. 이 좌표 정보를 바탕으로, 최종 출력에서 사용자의 위치에 해당하는 부분에 나라, 시, 주의 정보를 포함해줘.

# 사용자 위치: {location_str}
# 기온: {temperature}°C
# 날씨: {weather_desc}
# 선호 스타일: {desired_style}

# 사용자 옷장에 있는 의류 목록:
# 아우터: {outer}
# 상의: {tops}
# 하의: {bottoms}
# 신발: {shoes}

# {temperature_guide}

# 주의사항:
# 1. 추천된 의류는 반드시 위 옷장 목록에 있는 항목 중에서 선택해야 해.
# 2. 사용자의 선호 스타일에 맞춰서 어떤 옷을 입으면 좋을지 색감이나 재질 등을 고려하여 옷을 매칭해줘.
# 3. 날씨에 따라서 어떤 옷을 입으면 좋을지 온도별 추천 가이드를 참고하여 추천해줘.
# 4. 추천 순서는 아우터, 상의, 하의, 신발 순서로 작성해줘.
# 5. 날씨 설명에 따라 적절한 이모지를 포함시켜줘.
#    - 예: "Clear"이면 맑음 ☀️, "Clouds"이면 흐림 ☁️, "Rain" 또는 "우천"이면 비 🌧, 그 외에는 빈 문자열로 처리해.
# 6. 모든 출력값은 °C를 제외하고 한국어로 출력해줘.

# 출력 형식:
# "[사용자 위치]는 [기온]도, 날씨는 [날씨]입니다 [해당 날씨 이모지] [선호 스타일] 스타일링으로는 [아우터], [상의], [하의], [신발]을 매치하면 좋을 것 같아요!"

# 예시:
# "대한민국 서울특별시는 0°C, 날씨는 맑음입니다 ☀️ 캐주얼 스타일링으로는 패딩, 니트, 청바지, 어그 부츠를 매치하면 좋을 것 같아요!" 

# 데이터에 따른 변경 예시:
# "미국 로스엔젤레스는 10°C, 날씨는 흐림입니다 ☁️ 페미닌한 스타일링으로는 코트, 셔츠, 롱스커트, 부츠를 매치하면 좋을 것 같아요!" 
# """
#     )

#     llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")
#     chain = LLMChain(llm=llm, prompt=prompt_template)
#     try:
#         recommendation = chain.run({
#             "location_str": location_str,
#             "latitude": latitude,
#             "longitude": longitude,
#             "temperature": temperature,
#             "weather_desc": weather_desc,
#             "desired_style": desired_style,
#             "outer": outer,
#             "tops": tops,
#             "bottoms": bottoms,
#             "shoes": shoes,
#             "temperature_guide": temperature_guide
#         })
#     except Exception as e:
#         recommendation = f"GPT 호출 중 에러 발생: {e}"
    
#     return recommendation

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
    

