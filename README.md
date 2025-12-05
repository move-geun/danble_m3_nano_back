# 백엔드 로컬 실행 가이드

## 사전 준비

1. **Python 3.11 이상** 설치 확인
2. **Gemini API 키** 준비

## 실행 방법

### 1. 의존성 설치

```bash
cd backend
pip install -r requirements.txt
```

또는 가상환경 사용 (권장):

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:

```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

또는 직접 편집:

```bash
# .env 파일
GOOGLE_API_KEY=your_api_key_here
TEMPLATE_PATH=case.png  # 선택사항, 기본값: case.png
```

### 3. 템플릿 파일 복사 (선택사항)

템플릿 파일이 필요합니다:

```bash
# tmp_input/case.png를 backend/로 복사
cp ../tmp_input/case.png ./case.png
```

또는 환경 변수로 경로 지정:

```bash
# .env 파일에 추가
TEMPLATE_PATH=/absolute/path/to/case.png
```

### 4. 서버 실행

```bash
# 방법 1: uvicorn 직접 실행
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 방법 2: Python으로 실행
python api.py
```

### 5. 확인

서버가 실행되면 다음 URL로 접속:

- API 문서: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health
- API 정보: http://localhost:8000/

## 주요 엔드포인트

- `POST /api/generate` - 이미지 생성
- `GET /api/download/{filename}` - 이미지 다운로드
- `GET /health` - 헬스체크
- `GET /docs` - Swagger API 문서

## 문제 해결

### 포트가 이미 사용 중인 경우

```bash
# 다른 포트 사용
uvicorn api:app --reload --port 8001
```

### API 키 오류

`.env` 파일에 `GOOGLE_API_KEY`가 올바르게 설정되었는지 확인하세요.

### 템플릿 파일 오류

`case.png` 파일이 `backend/` 디렉토리에 있거나 `TEMPLATE_PATH` 환경 변수로 올바른 경로를 지정했는지 확인하세요.

