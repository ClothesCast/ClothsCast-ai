from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY

import json
# ------------------------------------------------------------------------------
# 4. LangChain을 활용한 GPT API 호출 및 최종 스타일링 추천 생성 (동적 변수 전달)
# ------------------------------------------------------------------------------
def generate_styling_recommendation_with_gpt(user_data: dict) -> str:
    """
    사용자 데이터를 기반으로 LangChain을 활용해 GPT에게 프롬프트를 전달하여
    최종 스타일링 추천 문장을 생성합니다.
    실시간으로 전달되는 정보를 동적으로 프롬프트에 반영합니다.
    """
    # print('*'* 10)
    # print(user_data)
    location_data = user_data.get("location", {})
    latitude = location_data.get("latitude", 0)
    longitude = location_data.get("longitude", 0)
    location_str = "좌표 기반 위치 정보"  # [수정됨] 실제 위치 변환 로직 추가 가능

    # temperature = user_data.get("main", {}).get("temp", {})
    # print('t', temperature)
    # temperature = None
    # for item in user_data:
    #     if item.get("dt_txt") == "2025-02-27 09:00:00":
    #         temperature = item.get("main", {}).get("temp", {})
    #         break
    # print(temperature)


    # temperature = None
    # for item in user_data:
    #     # item이 문자열이면 JSON 파싱
    #     if isinstance(item, str):
    #         try:
    #             item = json.loads(item)
    #         except json.JSONDecodeError:
    #             continue  # JSON 파싱에 실패하면 건너뜁니다.
    #     if item.get("dt_txt") == "2025-02-27 09:00:00":
    #         temperature = item.get("main", {}).get("temp")
    #         break
    temperature = user_data.get("temp", {})
    print(temperature)
    weather_info = user_data.get("weather", [{}])[0]
    weather_desc = weather_info.get("description", {})
    desired_style = user_data.get("style", {})
    
    owned_clothes = user_data.get("ownedClothes", {})
    topwear = owned_clothes.get("topwear", {})
    tops = ', '.join([item for item, owned in topwear.items() if owned])
    bottomwear = owned_clothes.get("bottomwear", {})
    bottoms = ', '.join([item for item, owned in bottomwear.items() if owned])
    outerwear = owned_clothes.get("outerwear", {})
    outer = ', '.join([item for item, owned in outerwear.items() if owned])
    
    shoes_items = []
    for shoe in ["sneakers", "boots", "sandals", "sportsShoes"]:
        if owned_clothes.get(shoe, False):
            shoes_items.append(shoe)
    shoes = ', '.join(shoes_items)

    temperature_guide = """
[온도별 로직]
- 영하 5도 이하: shortPadding, longPadding, knit, sweatshirt, hoodie, cottonPants
- 영하 5도 초과 0도 이하 : shortPadding, knit, sweatshirt, hoodie, cottonPants, denimPants, slacks
- 영상 0도 초과 영상 5도 이하 : shortPadding, coat, knit, sweatshirt, hoodie, denimPants, cottonPants, slacks, longSkirt, slacks
- 영상 5도 초과 영상 10도 이하 : coat, jacket, sweatshirt, hoodie, shirt, denimPants, slacks, longSkirt
- 영상 10도 초과 영상 15도 이하 : jacket, cardigan, zipup, sweatshirt, shirt, denimPants, slacks, miniSkirt, longSkirt
- 영상 15도 초과 영상 20도 이하 : cardigan, zipup, shirt, sweatshirt, denimPants, slacks, longSkirt, miniSkirt
- 영상 20도 초과 영상 25도 이하 : shortSleeve, shirt, slacks, miniSkirt, denimPants
- 영상 25도 초과 : sleeveless, shortSleeve, shortPants, miniSkirt, slacks, denimPants
- 영하 5도 이하, 영하 5도 초과 영상 10도 이하 신발 : boots
- 영상 25도 초과 신발 : sandals
- boots, sandals을 제외한 sneakers, sportsShoes는 모든 온도에 반영해줘 : sneakers, sportsShoes
"""

    prompt_template = PromptTemplate(
        input_variables=["location_str", "latitude", "longitude", "temperature", "weather_desc", "desired_style", "outer", "tops", "bottoms", "shoes", "temperature_guide"],
        template="""
너는 전문 패션 스타일리스트야. 아래 사용자 정보를 참고해서 최종 스타일링 추천 문장을 만들어줘.
실시간 위치 좌표: 위도 {latitude}, 경도 {longitude}. 이 좌표 정보를 바탕으로, 최종 출력에서 사용자의 위치에 해당하는 부분에 나라, 시, 주의 정보를 포함해줘.

사용자 위치: {location_str}
기온: {temperature}°C
날씨: {weather_desc}
선호 스타일: {desired_style}

사용자 옷장에 있는 의류 목록:
아우터: {outer}
상의: {tops}
하의: {bottoms}
신발: {shoes}

{temperature_guide}

주의사항:
1. 추천된 의류는 반드시 위 옷장 목록에 있는 항목 중에서 선택해야 해.
2. 사용자의 선호 스타일에 맞춰서 어떤 옷을 입으면 좋을지 색감이나 재질 등을 고려하여 옷을 매칭해줘.
3. 날씨에 따라서 어떤 옷을 입으면 좋을지 온도별 추천 가이드를 참고하여 추천해줘.
4. 추천 순서는 아우터, 상의, 하의, 신발 순서로 작성해줘.
5. 날씨 설명에 따라 적절한 이모지를 포함시켜줘.
   예: "Clear"이면 맑음 ☀️, "Clouds"이면 흐림 ☁️, "Rain" 또는 "우천"이면 비 🌧, 그 외에는 빈 문자열로 처리해.
6. 모든 출력값은 °C를 제외하고 한국어로 꼭 출력해줘.
7. 출력값은 문장마다 줄 바꿈을 해줘.

출력 형식:
"[사용자 위치]는 [기온]도, 날씨는 [날씨]입니다 [해당 날씨 이모지] 
[선호 스타일] 스타일링에 대한 설명
[아우터] 선택한 이유 설명
[상의] 선택한 이유 설명
[하의] 선택한 이유 설명
[신발] 선택한 이유 설명
나온 아우터, 상의, 하의, 신발의 조합을 설명해줘."

예시:
"대한민국 서울특별시의 온도는 2°C, 날씨는 맑음입니다 ☀️

캐주얼한 스타일링을 추천해드릴게요!

아우터: 자켓을 추천해요. 추운 날씨지만 맑은 날이라 가벼운 보온이 가능한 자켓을 선택했습니다.
상의: 밝은 색상의 티셔츠를 추천해요. 무거운 느낌을 줄 수 있는 겨울 스타일에 밝은 색상을 더해 생동감을 주었습니다.
하의: 청바지를 추천해요. 캐주얼하면서도 다양한 상의와 잘 어울리는 실용적인 선택입니다.
신발: 운동화를 추천해요. 편안함과 활동성을 고려해 운동화를 매치했습니다.

이 스타일링은 맑은 겨울 날씨에 적절한 보온을 유지하면서도, 밝은 색상의 티셔츠로 생동감을 주는 균형 잡힌 캐주얼룩입니다.
옷장 안에서 자켓, 티셔츠, 청바지, 운동화를 꺼내 캐쥬얼한 스타일링을 코디 해보세요!"
"""
    )

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    try:
        recommendation = chain.run({
            "location_str": location_str,
            "latitude": latitude,
            "longitude": longitude,
            "temperature": temperature,
            "weather_desc": weather_desc,
            "desired_style": desired_style,
            "outer": outer,
            "tops": tops,
            "bottoms": bottoms,
            "shoes": shoes,
            "temperature_guide": temperature_guide
        })
    except Exception as e:
        recommendation = f"GPT 호출 중 에러 발생: {e}"
    
    return recommendation


