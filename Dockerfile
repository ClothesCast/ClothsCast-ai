#  Python 3.12 slim 이미지를 사용 (가벼운 환경)
FROM python:3.12

#  작업 디렉토리 설정
WORKDIR /app

#  필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#  Python 패키지 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  애플리케이션 코드 복사
COPY . .

#  컨테이너에서 8000번 포트 오픈
EXPOSE 8000

#  Flask 애플리케이션 실행
CMD ["python", "app.py"]