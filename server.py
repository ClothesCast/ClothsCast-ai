
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import clothcast_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# clothscast_model.py 내의 추천 로직을 감싸는 AppModel 클래스 정의 (없다면 아래와 같이 추가)
class AppModel:
    def __init__(self):
        # 필요한 경우 초기화 작업 (예, Chroma DB 초기화 등) 수행
        pass

    def predict(self, user_data: dict) -> str:
        # clothscast_model.py에 정의된 generate_styling_recommendation_with_gpt 함수를 호출
        return clothcast_model.generate_styling_recommendation_with_gpt(user_data)
    
# ****************************

# AppModel 인스턴스 생성
model = AppModel()

# 요청 데이터 모델 정의
class Location(BaseModel):
    latitude: float
    longitude: float

class OwnedClothes(BaseModel):
    topwear: dict
    bottomwear: dict
    outerwear: dict
    shoes: dict

class Weather(BaseModel):
    condition: str
    

class RecommendRequest(BaseModel):
    style: str
    location: Location
    ownedClothes: OwnedClothes
    weather: Weather

@app.post("/recommend")
def recommend(request: RecommendRequest):
    # 요청 받은 데이터 출력 (테스트용)
    print(request.dict())
    
    # clothscast_model.py의 함수가 기대하는 데이터 구조로 변환
    transformed_data = {
        "style": request.style,
        "location": request.location.dict(),
        "ownedClothes": request.ownedClothes.dict(),
        "weather": [{"description": request.weather.condition}],
        "main": {"temp": request.main}
    }
    
    # 추천 결과 생성
    recommendation = model.predict(transformed_data)
    
    response_data = {
        "message": "Success",
        "recommendation": recommendation
    }
    return response_data


