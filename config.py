import os
import openai
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 환경 변수 및 API 키 설정
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
