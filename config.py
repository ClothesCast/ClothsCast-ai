import os
import openai
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. 환경 변수 및 API 키 설정
# ------------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_FOLDER는 PDF 파일들이 저장된 폴더입니다.
DATASET_FOLDER = os.path.join(SCRIPT_DIR, "fashion_dataset_text")
# JSON 파일 경로
JSON_FILEPATH = os.path.join(SCRIPT_DIR, "user_data.json")

# 클라우드 서버에서 학습 데이터 가져오기
# import os
# import boto3
# from dotenv import load_dotenv

# 환경 변수 로드
# load_dotenv()

# AWS S3 설정
# BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "clothcast")
# OBJECT_KEY = os.getenv("S3_OBJECT_KEY", "패션 코디네이션에 관한 연구.pdf")
# DOWNLOAD_PATH = "/home/ubuntu/ai_server/downloaded.pdf"  # 실제 경로로 수정 필요

# def download_pdf_from_s3():
#     """AWS S3에서 PDF 파일 다운로드"""
#     try:
#         s3 = boto3.client("s3")
#         s3.download_file(BUCKET_NAME, OBJECT_KEY, DOWNLOAD_PATH)
#         print(f"✅ PDF 다운로드 완료: {DOWNLOAD_PATH}")
#         return DOWNLOAD_PATH
#     except Exception as e:
#         print(f"❌ 오류 발생: {e}")
#         return None

# if name == "main":
#     download_pdf_from_s3()