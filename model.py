import requests
import openai
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# .env 파일 로드
load_dotenv()

# API 키 불러오기
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 예외 처리: API 키가 없을 경우
if not OPENWEATHER_API_KEY or not OPENAI_API_KEY:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# OpenAI 설정
openai.api_key = OPENAI_API_KEY
chat_model = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY) 
   
# 스타일 설명 데이터
STYLE_DATA = [
    {"style": "캐주얼", "text": "캐주얼 스타일은 편안하고 실용적인 의류로 구성됩니다. 대표적인 아이템은 후드티, 청바지, 스니커즈입니다."},
    {"style": "포멀", "text": "포멀 스타일은 비즈니스 또는 격식 있는 자리에서 입기 좋은 의류로 구성됩니다. 대표적인 아이템은 셔츠, 정장 바지, 구두입니다."},
    {"style": "스트릿", "text": "스트릿 스타일은 자유롭고 개성 있는 의류로 구성됩니다. 대표적인 아이템은 오버사이즈 티셔츠, 조거팬츠, 스니커즈입니다."},
    {"style": "모던", "text": "모던 스타일은 세련되고 깔끔한 디자인이 특징입니다. 대표적인 아이템은 슬랙스, 블레이저, 로퍼입니다."}
]

# ✅ 10도 단위 온도별 추천 의류 목록 (상의, 아우터, 하의, 신발)
CLOTHING_RULES = {
    "-10": {
        "top": ["내복", "기모 폴라티"],
        "outer": ["두꺼운 니트", "패딩"],
        "bottom": ["기모 바지", "두꺼운 슬랙스"],
        "shoes": ["부츠", "어그 부츠"]
    },
    "0": {
        "top": ["폴라티", "니트"],
        "outer": ["두꺼운 패딩", "목도리"],
        "bottom": ["울 슬랙스", "기모 청바지"],
        "shoes": ["부츠", "운동화"]
    },
    "10": {
        "top": ["맨투맨", "가디건"],
        "outer": ["후드티", "코트"],
        "bottom": ["슬랙스", "청바지"],
        "shoes": ["운동화", "로퍼"]
    },
    "20": {
        "top": ["긴팔 셔츠", "가벼운 니트"],
        "outer": ["얇은 재킷"],
        "bottom": ["면바지", "치노 팬츠"],
        "shoes": ["스니커즈", "로퍼"]
    },
    "30": {
        "top": ["반팔 티셔츠", "린넨 셔츠"],
        "outer": ["없음"],
        "bottom": ["반바지", "얇은 면바지"],
        "shoes": ["샌들", "가벼운 스니커즈"]
    },
    "40": {
        "top": ["민소매", "얇은 반팔"],
        "outer": ["없음"],
        "bottom": ["반바지", "린넨 팬츠"],
        "shoes": ["샌들", "슬리퍼"]
    }
}

# 날씨 이모지 설정
WEATHER_EMOJIS = {
    "clear": "☀️",
    "clouds": "☁️",
    "rain": "🌧️",
    "snow": "❄️",
    "thunderstorm": "⛈️",
    "drizzle": "🌦️",
    "mist": "🌫️"
}

# 온도 구간 찾기 (-10도 단위로 나누기)
def get_temperature_category(temp):
    if temp < -5:
        return "-10"
    elif -5 <= temp < 5:
        return "0"
    elif 5 <= temp < 15:
        return "10"
    elif 15 <= temp < 25:
        return "20"
    elif 25 <= temp < 35:
        return "30"
    else:
        return "40"

# 날씨 정보 가져오기
def get_weather(location="Seoul"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        weather_main = data["weather"][0]["main"].lower()
        description = data["weather"][0]["description"]
        weather_icon = WEATHER_EMOJIS.get(weather_main, "🌍")  # 기본값: 지구 이모지
        return temp, description, weather_icon
    else:
        print(f"API 요청 실패: {response.status_code}")
        return None, None, None

# ChatGPT를 활용한 코디 추천
def generate_outfit(style, clothing_items, temp, weather, weather_icon):
    top_items = ", ".join(clothing_items["top"])
    outer_items = ", ".join(clothing_items["outer"]) if clothing_items["outer"] != ["없음"] else "없음"
    bottom_items = ", ".join(clothing_items["bottom"])
    shoe_items = ", ".join(clothing_items["shoes"])

    prompt = f"""
    오늘의 날씨는 {weather}이며 기온은 {temp}도입니다. 날씨 이모지는 {weather_icon}입니다.
    사용자는 "{style}" 스타일을 선호합니다.
    "{STYLE_DATA}"
    위 스타일의 설명과 아래 온도별 의류 목록을 참고하여 스타일에 맞게 추천 코디를 구성해주세요.
    상의, 아우터, 하의, 신발 아이템은 하나씩만 추천해주세요.

    상의: {top_items}
    아우터: {outer_items}
    하의: {bottom_items}
    신발: {shoe_items}

    [출력 형식 예시]
    "오늘은 {temp}도의 {weather} 날씨입니다 {weather_icon} {style} 스타일에 어울리는 코디는 아우터, 상의, 하의, 신발 입니다."

    반드시 위 출력 형식처럼 답변해 주세요.
    """

    response = chat_model.invoke(prompt)
    return response.content

# 최종 실행 함수
def recommend_outfit():
    location = input("현재 위치를 입력하세요 (예: Seoul): ").strip()
    style = input("선호하는 스타일을 입력하세요 (캐주얼, 포멀, 스트릿, 모던): ").strip()

    temp, weather, weather_icon = get_weather(location)
    if temp is None:
        return "날씨 정보를 가져올 수 없습니다. 위치를 확인해주세요."

    category = get_temperature_category(temp)
    clothing_items = CLOTHING_RULES[category]

    return generate_outfit(style, clothing_items, temp, weather, weather_icon)

# 실행
if __name__ == "__main__":
    recommendation = recommend_outfit()
    print(recommendation)
