from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import clothcast_model
import json  # Pretty JSON 출력을 위한 모듈
from datetime import datetime

app = FastAPI()

# clothcast_model.py 내의 추천 로직을 감싸는 AppModel 클래스 정의
class AppModel:
    def __init__(self):
        pass

    def predict(self, user_data: dict) -> str:
        return clothcast_model.generate_styling_recommendation_with_gpt(user_data)

# AppModel 인스턴스 생성
model = AppModel()

# 요청 데이터 모델 정의
class Location(BaseModel):
    latitude: float
    longitude: float

class OwnedClothes(BaseModel):
    topwear: Dict[str, bool]
    bottomwear: Dict[str, bool]
    outerwear: Dict[str, bool]
    shoes: Dict[str, bool]

class WeatherDetail(BaseModel):
    description: str

class Weather(BaseModel):
    list: List[Dict[str, Any]]  # JSON 구조를 반영하여 리스트 형태로 받기

    def get_condition(self) -> str:
        """첫 번째 날씨 정보의 'weather'에서 'description' 값을 추출"""
        if self.list and "weather" in self.list[0]:
            return self.list[0]["weather"][0]["description"]
        return "Unknown"

    def get_noon_temp(self) -> float:
        """첫 번째로 등장하는 오후 12시(12:00:00)의 temp 값을 반환"""
        if not self.list:
            print("⚠️ No weather data available! Returning default 0.0")
            return 0.0  # 데이터가 없으면 기본값 반환

        for entry in self.list:
            if "dt_txt" in entry:
                dt_obj = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")

                # 첫 번째로 등장하는 12:00 데이터를 찾으면 반환
                if dt_obj.hour == 12:
                    noon_temp = entry["main"].get("temp", 0.0)
                    print(f"✅ Found Noon Temp at {dt_obj}: {noon_temp}°C")
                    return noon_temp  # 첫 번째 해당 데이터를 찾으면 즉시 반환

        print("⚠️ No Noon Temp Found at 12:00! Returning default 0.0")
        return 0.0  # 12시 데이터가 없으면 기본값 반환

class RecommendRequest(BaseModel):
    style: str
    location: Location
    ownedClothes: OwnedClothes
    weather: Weather

@app.post("/recommend")
def recommend(request: RecommendRequest):
    # 요청 받은 데이터 로그에 Pretty JSON 형식으로 출력
    request_json = request.dict()
    print("Received JSON:")
    print(json.dumps(request_json, indent=4, ensure_ascii=False))

    # clothscast_model.py의 함수가 기대하는 데이터 구조로 변환
    transformed_data = {
        "style": request.style,
        "location": request.location.dict(),
        "ownedClothes": request.ownedClothes.dict(),
        "weather": [{"description": request.weather.get_condition()}],  # 첫 번째 날씨 정보 추출
        "temp": {"temp": request.weather.get_noon_temp()}  # ✅ 첫 번째 오후 12시 temp 값만 추출
    }

    print("Transformed Data:")
    print(json.dumps(transformed_data, indent=4, ensure_ascii=False))

    # 추천 결과 생성
    recommendation = model.predict(transformed_data)

    response_data = {
        "message": "Success",
        "recommendation": recommendation
    }
    return response_data
