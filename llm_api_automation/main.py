from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles # 현재 미사용
from pydantic import BaseModel, Field # Field 추가
from typing import Dict, Any, Optional, List # List 추가

import json
import os
import httpx # 외부 API 호출을 위한 비동기 HTTP 클라이언트

# --- 프로젝트 내 모듈 임포트 ---
# API 명세서 파싱 관련 모듈
from api_parser import load_api_spec, extract_tools_from_spec
# LLM 연동 관련 모듈
from llm_handler import get_llm_response_with_tools, get_final_response_after_tool_call, client as openai_client

# --- 기본 설정 ---
# API 명세서 파일의 기본 경로 설정.
# 사용자가 다른 명세서를 사용하려면 이 값을 변경하거나, 요청 시 경로를 지정해야 합니다.
# __file__은 현재 파일(main.py)의 경로를 나타냅니다.
# os.path.dirname은 해당 파일이 위치한 디렉토리 경로를 반환합니다.
# os.path.join은 OS에 맞는 경로 구분자를 사용하여 경로를 결합합니다.
DEFAULT_API_SPEC_PATH = os.path.join(os.path.dirname(__file__), 'sample_openapi.yaml')

# --- FastAPI 애플리케이션 초기화 ---
app = FastAPI(
    title="LLM API 자동화 에이전트",
    description="OpenAPI 명세서를 기반으로 LLM이 API를 호출하고 응답하는 챗봇 애플리케이션입니다.",
    version="0.1.0"
)

# --- 데이터 모델 (Pydantic BaseModel 사용) ---
# Pydantic 모델은 요청/응답 데이터의 유효성 검사 및 문서화에 사용됩니다.

class ChatMessage(BaseModel):
    """채팅 요청 시 사용될 데이터 모델"""
    message: str = Field(..., description="사용자가 입력한 메시지")
    api_spec_path: Optional[str] = Field(DEFAULT_API_SPEC_PATH, description="사용할 API 명세서 파일 경로 (지정하지 않으면 기본값 사용)")
    # session_id: Optional[str] = None # 추후 대화 히스토리 관리를 위해 확장 가능

class ToolCallInfo(BaseModel):
    """LLM의 도구 호출 시도 정보를 담는 모델"""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str

class ApiResponseInfo(BaseModel):
    """실제 API 호출 결과를 담는 모델"""
    tool_name: str
    api_path: str
    method: str
    request_data: Optional[Dict] = None # 실제 요청에 사용된 URL, 파라미터, 본문 등
    response_data: Any # API로부터 받은 응답 (성공 시)
    error: Optional[str] = None # API 호출 실패 시 오류 메시지 또는 상태 코드

class ChatResponse(BaseModel):
    """채팅 응답 데이터 모델"""
    final_response: Optional[str] = Field(None, description="LLM이 생성한 최종 사용자 응답 메시지")
    tool_calls_info: Optional[List[ToolCallInfo]] = Field(None, description="LLM의 도구 호출 시도 정보 목록")
    tool_responses_info: Optional[List[ApiResponseInfo]] = Field(None, description="실제 API 호출 결과 목록")
    error_message: Optional[str] = Field(None, description="처리 중 발생한 오류 메시지")


# --- 전역 변수 및 캐시 ---
# 로드된 API 명세서와 추출된 도구 정보를 캐싱하여 반복적인 파일 로드/파싱을 줄입니다.
# 간단한 인메모리 캐시이며, 실제 운영 환경에서는 Redis 등 외부 캐시 사용을 고려할 수 있습니다.
loaded_specs_cache: Dict[str, Dict[str, Any]] = {}

# --- 유틸리티 함수 ---
async def get_or_load_specs_and_tools(spec_path: str) -> Dict[str, Any]:
    """
    지정된 경로의 API 명세서를 로드/파싱하고 도구를 추출합니다. (캐시 우선 확인)
    결과에는 'spec'(원본 명세서), 'tools'(추출된 도구), 'server_url'(API 서버 기본 URL)이 포함됩니다.
    """
    if spec_path in loaded_specs_cache:
        # print(f"캐시에서 API 명세서 사용: {spec_path}")
        return loaded_specs_cache[spec_path]

    print(f"API 명세서 파일 로드 시도: {spec_path}")
    try:
        api_spec = load_api_spec(spec_path)
        tools = extract_tools_from_spec(api_spec)

        # API 명세서에서 서버 URL 추출 (첫 번째 서버 URL 사용)
        server_url = ""
        if isinstance(api_spec.get('servers'), list) and api_spec['servers']:
            server_url = api_spec['servers'][0].get('url', '')

        # 캐시에 저장
        cache_data = {"spec": api_spec, "tools": tools, "server_url": server_url}
        loaded_specs_cache[spec_path] = cache_data

        print(f"'{spec_path}'에서 {len(tools)}개의 도구 추출 완료. 서버 URL: '{server_url if server_url else '없음'}'")
        return cache_data
    except FileNotFoundError:
        print(f"오류: API 명세서 파일을 찾을 수 없습니다 - {spec_path}")
        raise HTTPException(status_code=404, detail=f"지정된 API 명세서 파일을 찾을 수 없습니다: {spec_path}")
    except ValueError as e: # load_api_spec 또는 extract_tools_from_spec 에서 발생 가능
        print(f"오류: API 명세서 처리 중 값 오류 - {spec_path}: {e}")
        raise HTTPException(status_code=400, detail=f"API 명세서 처리 중 오류 발생 (잘못된 형식 또는 내용): {str(e)}")
    except Exception as e:
        print(f"오류: API 명세서 처리 중 예상치 못한 오류 - {spec_path}: {e}")
        raise HTTPException(status_code=500, detail=f"API 명세서 처리 중 서버 내부 오류: {str(e)}")

# --- FastAPI 엔드포인트 정의 ---

@app.get("/", response_class=HTMLResponse, summary="챗봇 UI 제공", tags=["UI"])
async def get_chat_ui_page(request: Request):
    """
    사용자와 상호작용할 수 있는 기본 HTML 웹 페이지를 반환합니다.
    간단한 채팅 인터페이스를 제공합니다.
    """
    # (HTML 내용은 이전과 동일하게 유지 - 주석 처리 생략)
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM API 자동화 챗봇</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f4f4f9; display: flex; justify-content: center;}
            #chatContainer { width: 100%; max-width: 800px; }
            #chatbox { background: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }
            #messages { list-style-type: none; padding: 0; margin: 0 0 20px 0; height: 50vh; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 4px; }
            #messages li { margin-bottom: 10px; padding: 8px 12px; border-radius: 4px; word-wrap: break-word; line-height: 1.4; }
            .user { background-color: #e1f5fe; text-align: right; margin-left: auto; max-width: 70%; clear: both; float: right; }
            .assistant { background-color: #f0f0f0; text-align: left; margin-right: auto; max-width: 70%; clear: both; float: left; }
            .tool-call { background-color: #fff9c4; border-left: 3px solid #fdd835; font-size: 0.9em; color: #545454; max-width: 90%; clear: both; float: left; }
            .tool-response { background-color: #e6ee9c; border-left: 3px solid #c0ca33; font-size: 0.9em; color: #4f592a; max-width: 90%; clear: both; float: left; }
            .error { background-color: #ffcdd2; border-left: 3px solid #f44336; color: #b71c1c; max-width: 90%; clear: both; float: left; }
            #inputArea { display: flex; margin-top: 10px; }
            input[type="text"]#userInput { flex-grow: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px 0 0 4px; }
            button#sendButton { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 0 4px 4px 0; cursor: pointer; }
            button#sendButton:hover { background-color: #0056b3; }
            #apiSpecInfo { font-size: 0.8em; color: #555; margin-top: 5px; text-align: right; }
        </style>
    </head>
    <body>
        <div id="chatContainer">
            <div id="chatbox">
                <h2>LLM API 자동화 에이전트</h2>
                <ul id="messages"></ul>
                <div id="inputArea">
                    <input type="text" id="userInput" placeholder="메시지를 입력하세요..." autocomplete="off">
                    <button id="sendButton" onclick="sendMessage()">전송</button>
                </div>
                <div id="apiSpecInfo">API Spec: <span id="apiSpecPathDisplay"></span></div>
            </div>
        </div>
        <script>
            const messagesList = document.getElementById('messages');
            const userInput = document.getElementById('userInput');
            const sendButton = document.getElementById('sendButton');
            const apiSpecPathDisplay = document.getElementById('apiSpecPathDisplay');
            let currentApiSpecPath = '';

            async function loadDefaultApiSpecPath() {
                try {
                    const response = await fetch('/default_api_spec_path');
                    if (response.ok) {
                        const data = await response.json();
                        currentApiSpecPath = data.path;
                        apiSpecPathDisplay.textContent = data.path.split('/').pop(); // 파일명만 표시
                    } else {
                        apiSpecPathDisplay.textContent = "기본 경로 로드 실패";
                    }
                } catch (e) { apiSpecPathDisplay.textContent = "경로 로드 중 오류"; console.error(e); }
            }

            window.onload = loadDefaultApiSpecPath;

            async function sendMessage() {
                const messageText = userInput.value.trim();
                if (!messageText) return;

                appendMessage(messageText, 'user');
                userInput.value = '';
                userInput.disabled = true;
                sendButton.disabled = true;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: messageText, api_spec_path: currentApiSpecPath })
                    });

                    const data = await response.json(); // 응답은 ChatResponse 모델 형식

                    if (!response.ok) {
                        appendMessage(`오류: ${data.error_message || data.detail || response.statusText}`, 'error');
                        return;
                    }

                    if (data.tool_calls_info && data.tool_calls_info.length > 0) {
                        data.tool_calls_info.forEach(call => {
                             appendMessage(`[도구 호출] ${call.tool_name}(${JSON.stringify(call.tool_args)})`, 'tool-call');
                        });
                    }
                    if (data.tool_responses_info && data.tool_responses_info.length > 0) {
                         data.tool_responses_info.forEach(resp => {
                             let content = `[도구 응답] ${resp.tool_name}: `;
                             if(resp.error) content += `오류(${resp.error}) - ${JSON.stringify(resp.response_data)}`;
                             else content += JSON.stringify(resp.response_data);
                             appendMessage(content, resp.error ? 'error' : 'tool-response');
                        });
                    }
                    if (data.final_response) {
                        appendMessage(data.final_response, 'assistant');
                    } else if (data.error_message) {
                         appendMessage(data.error_message, 'error');
                    } else if (!data.tool_calls_info && !data.tool_responses_info) {
                        appendMessage("응답을 받지 못했습니다.", "error");
                    }

                } catch (error) {
                    console.error('메시지 전송/처리 중 오류:', error);
                    appendMessage('클라이언트 오류: 메시지 처리 중 문제가 발생했습니다.', 'error');
                } finally {
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    userInput.focus();
                }
            }

            function appendMessage(text, type) {
                const listItem = document.createElement('li');
                listItem.className = type;
                // XSS 방지를 위해 textContent 사용 (HTML 직접 삽입 시 주의)
                listItem.textContent = text;
                messagesList.appendChild(listItem);
                messagesList.scrollTop = messagesList.scrollHeight;
            }

            userInput.addEventListener('keypress', function(event) {
                if (event.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/default_api_spec_path", summary="기본 API 명세서 경로 제공", tags=["Config"])
async def get_default_spec_path():
    """현재 서버에 설정된 기본 API 명세서 파일의 경로를 반환합니다."""
    return {"path": DEFAULT_API_SPEC_PATH}

@app.post("/chat", response_model=ChatResponse, summary="챗봇 메시지 처리", tags=["Chat"])
async def handle_chat_message(chat_message: ChatMessage):
    """
    사용자 메시지를 받아 LLM과 상호작용하고, 필요시 API를 호출하여 응답을 생성합니다.
    """
    if not openai_client: # llm_handler에서 OpenAI 클라이언트 초기화 실패 시
        raise HTTPException(status_code=500, detail="OpenAI 클라이언트가 초기화되지 않았습니다. 서버 로그 및 API 키 설정을 확인하세요.")

    user_message = chat_message.message
    api_spec_path = chat_message.api_spec_path if chat_message.api_spec_path else DEFAULT_API_SPEC_PATH

    # API 명세서 로드 및 도구 추출 (캐시 활용)
    try:
        spec_data = await get_or_load_specs_and_tools(api_spec_path)
    except HTTPException as e: # get_or_load_specs_and_tools에서 발생한 HTTP 예외 재전달
        return ChatResponse(error_message=e.detail) # JSON 응답으로 오류 전달

    tools = spec_data.get("tools", [])
    api_spec = spec_data.get("spec", {}) # 전체 명세서 (API 호출 시 상세 정보 참조용)
    base_server_url = spec_data.get("server_url", "") # API 호출 시 사용할 기본 서버 URL

    # 1. LLM에게 사용자 메시지와 도구 목록을 전달하여 1차 응답 (도구 사용 결정) 받기
    # 시스템 프롬프트 예시 (필요에 따라 llm_handler 또는 여기서 동적으로 구성 가능)
    system_prompt = "당신은 사용자의 요청을 이해하고, 등록된 API 도구를 사용하여 정보를 찾거나 작업을 수행하는 AI 에이전트입니다. API 호출 결과에 기반하여 사용자에게 친절하게 답변해주세요."
    llm_initial_response_msg = get_llm_response_with_tools(user_message, tools, system_prompt=system_prompt)

    if not llm_initial_response_msg:
        return ChatResponse(error_message="LLM으로부터 초기 응답을 받지 못했습니다. 서버 로그를 확인하세요.")

    # 반환할 정보 초기화
    tool_calls_attempted_info: List[ToolCallInfo] = []
    api_responses_received_info: List[ApiResponseInfo] = []
    final_llm_output: Optional[str] = None

    # 2. LLM이 도구 사용을 결정했는지 확인
    if llm_initial_response_msg.tool_calls:
        print(f"LLM이 {len(llm_initial_response_msg.tool_calls)}개의 도구 호출을 요청했습니다.")

        # LLM에게 최종 응답을 요청할 때 전달할, 각 도구 호출의 실제 실행 결과 목록
        tool_call_execution_results_for_llm: List[Dict[str, Any]] = []

        async with httpx.AsyncClient() as http_client: # 여러 API 호출을 위해 세션 사용
            for tool_call in llm_initial_response_msg.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print(f"오류: LLM이 제공한 도구 인수(arguments)가 JSON 형식이 아닙니다: {tool_call.function.arguments}")
                    # LLM에게 오류 상황 전달
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": "LLM provided invalid arguments format."})
                    })
                    # 사용자에게 보여줄 정보에도 기록
                    api_responses_received_info.append(ApiResponseInfo(
                        tool_name=tool_name, api_path="N/A", method="N/A", error="잘못된 인수 형식"
                    ))
                    continue # 다음 도구 호출로 넘어감

                tool_calls_attempted_info.append(ToolCallInfo(tool_name=tool_name, tool_args=tool_args, tool_call_id=tool_call_id))
                print(f"  처리할 도구: '{tool_name}', 인수: {tool_args}")

                # API 명세(api_spec)에서 현재 도구(operationId)에 해당하는 API 상세 정보 찾기
                current_api_details = None
                for path_str, path_item_spec in api_spec.get('paths', {}).items():
                    for http_method_str, operation_spec in path_item_spec.items():
                        if operation_spec.get('operationId') == tool_name:
                            current_api_details = {
                                "path_template": path_str, # 예: /users/{userId}
                                "method": http_method_str.upper(), # 예: GET, POST
                                "operation_spec": operation_spec, # 해당 operation의 전체 명세
                                "base_url": base_server_url # 명세서에 정의된 서버 URL
                            }
                            break
                    if current_api_details: break

                # API 응답 정보 객체 초기화
                current_api_call_response = ApiResponseInfo(tool_name=tool_name, api_path="N/A", method="N/A")

                if not current_api_details:
                    error_detail = f"API 명세에서 '{tool_name}'에 해당하는 operationId를 찾을 수 없습니다."
                    print(f"오류: {error_detail}")
                    current_api_call_response.error = error_detail
                    current_api_call_response.response_data = {"error": error_detail}
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": error_detail}) # LLM에게 오류 전달
                    })
                    api_responses_received_info.append(current_api_call_response)
                    continue

                # API 호출 준비
                http_method = current_api_details["method"]
                api_path_template = current_api_details["path_template"]
                operation_spec = current_api_details["operation_spec"]

                current_api_call_response.api_path = api_path_template # 템플릿 경로로 우선 설정
                current_api_call_response.method = http_method

                # Path 파라미터 치환하여 실제 호출 경로(resolved_path) 만들기
                resolved_path = api_path_template
                for arg_name, arg_value in tool_args.items():
                    if f"{{{arg_name}}}" in resolved_path:
                        resolved_path = resolved_path.replace(f"{{{arg_name}}}", str(arg_value))

                # 최종 호출 URL 구성
                full_url = f"{current_api_details['base_url'].rstrip('/')}{resolved_path}"
                current_api_call_response.api_path = full_url # 실제 호출 URL로 업데이트

                # Query 파라미터, Header 파라미터, Request Body 구성
                query_params: Dict[str, Any] = {}
                headers: Dict[str, str] = {} # TODO: 헤더 파라미터 처리 필요
                json_body_payload: Optional[Dict[str, Any]] = None

                # OpenAPI 명세의 파라미터 정의를 보고 tool_args에서 적절히 분배
                for param_spec in operation_spec.get('parameters', []):
                    param_name = param_spec.get('name')
                    if param_name in tool_args:
                        if param_spec.get('in') == 'query':
                            query_params[param_name] = tool_args[param_name]
                        elif param_spec.get('in') == 'header':
                            headers[param_name] = str(tool_args[param_name])
                        # 'path' 파라미터는 이미 resolved_path에 반영됨

                # RequestBody 처리 (JSON 형식만 우선 가정)
                if operation_spec.get('requestBody'):
                    # api_parser.py에서 요청 본문의 속성들을 최상위 파라미터로 올렸다고 가정.
                    # tool_args에서 경로/쿼리/헤더 파라미터가 아닌 것들을 본문으로 간주.
                    # 좀 더 정확하려면, requestBody 스키마의 속성들을 명시적으로 찾아야 함.
                    body_candidate_args = {
                        k: v for k, v in tool_args.items()
                        if f"{{{k}}}" not in api_path_template and k not in query_params and k not in headers
                    }
                    if body_candidate_args:
                        json_body_payload = body_candidate_args

                current_api_call_response.request_data = {
                    "url": full_url, "method": http_method,
                    "params": query_params or None, "json_body": json_body_payload or None, "headers": headers or None
                }
                print(f"  API 호출 실행: {http_method} {full_url}")
                print(f"    Query: {query_params}, Body: {json_body_payload}, Headers: {headers}")

                try:
                    api_http_response = await http_client.request(
                        method=http_method, url=full_url,
                        params=query_params if query_params else None, # GET 등은 params 사용
                        json=json_body_payload if json_body_payload and http_method in ["POST", "PUT", "PATCH"] else None, # POST 등은 json 사용
                        headers=headers if headers else None,
                        timeout=15.0 # 타임아웃 15초
                    )
                    # HTTP 오류 상태 코드(4xx, 5xx) 발생 시 예외 발생
                    api_http_response.raise_for_status()

                    # 응답 내용 처리 (JSON 가정)
                    try:
                        response_content_json = api_http_response.json()
                    except json.JSONDecodeError: # JSON이 아닌 응답 (예: 일반 텍스트)
                        response_content_json = {"raw_response": api_http_response.text}

                    current_api_call_response.response_data = response_content_json
                    print(f"  API 호출 성공 ({tool_name}): 상태코드={api_http_response.status_code}, 응답길이={len(api_http_response.content)}")
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps(response_content_json, ensure_ascii=False) # LLM에게는 JSON 문자열로
                    })
                except httpx.HTTPStatusError as e: # 4xx, 5xx 오류
                    error_detail = f"API 호출 실패 ({tool_name}): 상태코드={e.response.status_code}"
                    try: # 오류 응답도 JSON일 수 있음
                        error_response_body = e.response.json()
                    except json.JSONDecodeError:
                        error_response_body = e.response.text
                    print(f"오류: {error_detail} - {error_response_body}")
                    current_api_call_response.error = str(e.response.status_code)
                    current_api_call_response.response_data = {"error": error_detail, "body": error_response_body}
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": error_detail, "response_body": error_response_body}, ensure_ascii=False)
                    })
                except httpx.RequestError as e: # 연결 오류, 타임아웃 등
                    error_detail = f"API 호출 중 네트워크/요청 오류 ({tool_name}): {type(e).__name__}"
                    print(f"오류: {error_detail} - {str(e)}")
                    current_api_call_response.error = "네트워크/요청 오류"
                    current_api_call_response.response_data = {"error": error_detail, "details": str(e)}
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": error_detail, "exception_details": str(e)})
                    })
                except Exception as e: # 그 외 예외
                    error_detail = f"API 호출 중 예상치 못한 내부 오류 ({tool_name}): {type(e).__name__}"
                    print(f"오류: {error_detail} - {str(e)}")
                    current_api_call_response.error = "내부 처리 오류"
                    current_api_call_response.response_data = {"error": error_detail, "details": str(e)}
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": error_detail, "exception_details": str(e)})
                    })
                api_responses_received_info.append(current_api_call_response)

        # 3. 모든 도구 호출 결과를 취합하여 LLM에게 최종 사용자 응답 생성 요청
        if tool_call_execution_results_for_llm: # 실제 실행된 도구 결과가 하나라도 있다면
            final_llm_output = get_final_response_after_tool_call(
                user_message,
                llm_initial_response_msg, # LLM의 이전 응답 메시지 객체
                tool_call_execution_results_for_llm,
                system_prompt=system_prompt
            )
        elif tool_calls_attempted_info: # 도구 호출은 시도했으나, 모두 실패하여 LLM에게 전달할 결과가 없는 경우
            final_llm_output = "요청하신 작업을 위해 도구를 호출하려고 했지만, 실행 과정에서 문제가 발생하여 결과를 얻지 못했습니다. 다시 시도해주시거나 관리자에게 문의해주세요."
        # (tool_calls_attempted_info도 비었다면, 애초에 LLM이 tool_calls를 반환하지 않은 경우로 아래에서 처리됨)

    else:
        # LLM이 도구 사용 없이 직접 답변한 경우
        print("LLM이 직접 답변했습니다.")
        final_llm_output = llm_initial_response_msg.content

    return ChatResponse(
        final_response=final_llm_output,
        tool_calls_info=tool_calls_attempted_info if tool_calls_attempted_info else None,
        tool_responses_info=api_responses_received_info if api_responses_received_info else None
    )

# --- 서버 실행 로직 (개발용) ---
if __name__ == "__main__":
    # 서버 시작 시 기본 API 명세 파일이 존재하는지 확인하고, 없으면 샘플 생성 시도
    # (api_parser.py의 if __name__ == '__main__' 에서도 샘플 생성 로직이 있음)
    if not os.path.exists(DEFAULT_API_SPEC_PATH):
        print(f"경고: 기본 API 명세 파일 '{DEFAULT_API_SPEC_PATH}'을 찾을 수 없습니다.")
        # api_parser.py를 실행하여 샘플 명세서 생성 시도
        try:
            print("api_parser.py를 실행하여 샘플 명세서를 생성합니다...")
            import subprocess
            # python 실행파일 경로를 명시적으로 지정할 수도 있음 (예: sys.executable)
            process_result = subprocess.run(["python", "api_parser.py"], capture_output=True, text=True, check=False, cwd=os.path.dirname(__file__))
            if process_result.returncode == 0:
                print("api_parser.py 실행 성공, 샘플 명세서가 생성되었을 수 있습니다.")
                print(process_result.stdout)
            else:
                print("api_parser.py 실행 실패:")
                print(process_result.stdout)
                print(process_result.stderr)

            if not os.path.exists(DEFAULT_API_SPEC_PATH): # 그래도 없으면
                 print(f"'{DEFAULT_API_SPEC_PATH}' 파일이 여전히 없습니다. 수동으로 생성하거나 경로를 확인해주세요.")
        except Exception as e:
            print(f"api_parser.py 실행 중 오류: {e}")
    else:
        print(f"기본 API 명세 파일 확인: '{DEFAULT_API_SPEC_PATH}'")

    # Uvicorn ASGI 서버 실행
    # host="0.0.0.0"은 모든 네트워크 인터페이스에서 접속 허용
    # port=8000은 사용할 포트 번호
    # reload=True는 개발 중 코드 변경 시 자동 재시작 (운영 환경에서는 False)
    import uvicorn
    print("FastAPI 서버를 시작합니다 (http://localhost:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
