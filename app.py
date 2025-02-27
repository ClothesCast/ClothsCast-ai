from flask import Flask, request, jsonify
from gpt_styling import generate_styling_recommendation_with_gpt
from clothcast_model import main

app = Flask(__name__)

# Flask 앱 실행과 main() 실행 분리: 
if __name__ == "__main__":
#     # Flask 서버를 실행하려면 주석 해제
    app.run(host="0.0.0.0", port=8000)
    
#     # JSON 파일 기반 실행 (테스트용)
main()

@app.route('/recommand', methods=['POST'])
def predict():
    """
    클라이언트로부터 POST 요청으로 JSON 데이터를 받습니다.
    JSON에는 아래와 같은 형식이 포함되어야 합니다.
    
    주요 키:
      - ownedClothes: 상, 하, 아우터, 신발 정보 (불리언 값)
      - style: 사용자 선호 스타일
      - location: { "latitude": 숫자, "longitude": 숫자 }
      - weather.main: { "temp": 숫자, ... }
      - weather: [ { "description": 문자열, ... } ]
      
    이 정보들은 실시간으로 변경되므로, 매 요청마다 최신 정보를 사용합니다.
    """
    user_data = request.json

    print("Received JSON data:", user_data)

    try:
        recommendation = generate_styling_recommendation_with_gpt(user_data)
        return jsonify({'result': recommendation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# import json

# # JSON 파일 열기 (파일 경로를 원하는 경로로 수정)
# with open('user_data', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # 데이터 보기 쉽게 출력 (indent를 사용하여 계층 구조를 들여쓰기)
# print(json.dumps(data, indent=4, ensure_ascii=False))
