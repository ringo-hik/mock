# LLM 기반 API 자동화 프로젝트

본 프로젝트는 LLM을 활용하여 API 명세서(OpenAPI)를 바탕으로 자동으로 API를 호출하고, 사용자와의 간단한 웹 기반 챗봇 인터페이스를 통해 상호작용하는 것을 목표로 합니다.

## 주요 기능
- API 명세서(OpenAPI YAML/JSON) 파싱 및 동적 도구 생성
- LLM(OpenAI GPT 모델)을 통한 자연어 이해 및 적절한 API 호출 결정
- 실제 API 호출 실행 및 결과 반환
- 간단한 위젯형 챗봇 UI 제공 (FastAPI 및 HTML/JavaScript)
- 쉬운 설정 및 사용법 (주요 설정은 환경 변수 및 코드 내 상단 변수)

## 요구사항
- Python 3.8 이상
- OpenAI API Key

## 설치 방법
1.  **저장소 클론:**
    ```bash
    # git clone <저장소_URL> # (저장소가 있다면)
    # cd <프로젝트_디렉토리>/llm_api_automation
    # 또는 파일들을 llm_api_automation 디렉토리에 위치시킵니다.
    ```

2.  **가상 환경 생성 및 활성화 (권장):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **필수 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

## 설정 방법
1.  **OpenAI API 키 설정:**
    LLM과의 연동을 위해 OpenAI API 키가 필요합니다. 환경 변수 `OPENAI_API_KEY`에 발급받은 키를 설정해주세요.
    ```bash
    export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Linux/macOS
    # set OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Windows Command Prompt
    # $env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # Windows PowerShell
    ```
    이 환경 변수는 `llm_api_automation/llm_handler.py`에서 사용됩니다.

2.  **API 명세서 파일 경로 설정 (선택 사항):**
    기본적으로 프로젝트는 `llm_api_automation/sample_openapi.yaml` 파일을 API 명세서로 사용합니다. (이 파일은 `api_parser.py` 실행 시 자동으로 생성될 수 있는 샘플입니다.)
    만약 다른 OpenAPI 명세서 파일을 사용하고 싶다면, `llm_api_automation/main.py` 파일 상단에 있는 `DEFAULT_API_SPEC_PATH` 변수의 값을 원하는 파일 경로로 수정해주세요.
    ```python
    # llm_api_automation/main.py 파일 상단
    # DEFAULT_API_SPEC_PATH = os.path.join(os.path.dirname(__file__), 'sample_openapi.yaml')
    DEFAULT_API_SPEC_PATH = "경로/내_API_명세서.yaml" # 또는 .json
    ```
    **중요:** API 명세서는 OpenAPI 3.0.x 버전을 기준으로 작성되어야 합니다.

## 실행 방법
1.  **FastAPI 서버 실행:**
    `llm_api_automation` 디렉토리 내에서 다음 명령어를 실행합니다. (가상 환경이 활성화된 상태여야 합니다.)
    ```bash
    python main.py
    ```
    서버가 정상적으로 실행되면 다음과 같은 메시지가 나타납니다:
    ```
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    ```

2.  **챗봇 사용:**
    웹 브라우저를 열고 `http://localhost:8000` 주소로 접속합니다.
    간단한 챗봇 UI가 나타나며, 메시지 입력창에 자연어로 명령을 입력하여 API 기능을 테스트할 수 있습니다.
    예를 들어, `sample_openapi.yaml` (기본 명세서)을 사용 중이라면 다음과 같은 명령을 시도해볼 수 있습니다:
    - "1번 사용자 정보 알려줘"
    - "가격이 1000원인 '새로운 아이템'이라는 이름의 아이템을 만들어줘"

## 프로젝트 구조
- \`main.py\`: FastAPI 애플리케이션, 웹 UI, 채팅 엔드포인트, API 호출 로직 포함.
- \`api_parser.py\`: OpenAPI 명세서를 파싱하여 LLM이 사용할 도구 목록을 생성.
- \`llm_handler.py\`: OpenAI LLM과의 통신 및 도구 사용 결정 로직 처리.
- \`requirements.txt\`: 필요한 Python 라이브러리 목록.
- \`sample_openapi.yaml\`: 예제 OpenAPI 명세서 파일.
- \`README.md\`: 본 파일 (프로젝트 설명 및 사용법 안내).

## 주의사항
- 본 프로젝트는 데모 및 학습 목적으로 개발되었습니다. 실제 운영 환경에 사용하기 위해서는 보안, 오류 처리, 확장성 등을 추가적으로 고려해야 합니다.
- LLM API 사용에는 비용이 발생할 수 있습니다. OpenAI의 과금 정책을 확인하세요.
- API 호출은 실제 서버로 요청을 보냅니다. 테스트 시 주의하세요.

EOL
