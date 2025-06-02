from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse # FileResponse 추가
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import aiofiles # FileResponse를 위해 필요할 수 있음 (FastAPI가 내부적으로 사용)

import json
import os
import httpx

# 프로젝트 내 모듈 임포트
from api_parser import load_api_spec, extract_tools_from_spec
from llm_handler import get_llm_response_with_tools, get_final_response_after_tool_call, client as openai_client

# --- 기본 설정 ---
DEFAULT_API_SPEC_PATH = os.path.join(os.path.dirname(__file__), 'sample_openapi.yaml')
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates") # templates 디렉토리 경로

# --- FastAPI 애플리케이션 초기화 ---
app = FastAPI(
    title="LLM API 자동화 에이전트",
    description="OpenAPI 명세서를 기반으로 LLM이 API를 호출하고 응답하는 챗봇 애플리케이션입니다.",
    version="0.1.3" # 버전 업데이트 (HTML 파일 분리)
)

# --- 데이터 모델 (Pydantic BaseModel 사용) ---
class ChatMessage(BaseModel):
    message: str = Field(..., description="사용자가 입력한 메시지")
    api_spec_path: Optional[str] = Field(DEFAULT_API_SPEC_PATH, description="사용할 API 명세서 파일 경로 (지정하지 않으면 기본값 사용)")

class ToolCallInfo(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str

class ApiResponseInfo(BaseModel):
    tool_name: str
    api_path: str
    method: str
    request_data: Optional[Dict] = None
    response_data: Any
    error: Optional[str] = None

class ChatResponse(BaseModel):
    final_response: Optional[str] = Field(None, description="LLM이 생성한 최종 사용자 응답 메시지")
    tool_calls_info: Optional[List[ToolCallInfo]] = Field(None, description="LLM의 도구 호출 시도 정보 목록")
    tool_responses_info: Optional[List[ApiResponseInfo]] = Field(None, description="실제 API 호출 결과 목록")
    error_message: Optional[str] = Field(None, description="처리 중 발생한 오류 메시지")

# --- 전역 변수 및 캐시 ---
loaded_specs_cache: Dict[str, Dict[str, Any]] = {}

# --- 유틸리티 함수 ---
async def get_or_load_specs_and_tools(spec_path: str) -> Dict[str, Any]:
    if spec_path in loaded_specs_cache:
        return loaded_specs_cache[spec_path]
    print(f"API 명세서 파일 로드 시도: {spec_path}")
    try:
        api_spec = load_api_spec(spec_path)
        tools = extract_tools_from_spec(api_spec)
        server_url = ""
        if isinstance(api_spec.get('servers'), list) and api_spec['servers']:
            server_url = api_spec['servers'][0].get('url', '')
        cache_data = {"spec": api_spec, "tools": tools, "server_url": server_url}
        loaded_specs_cache[spec_path] = cache_data
        print(f"'{spec_path}'에서 {len(tools)}개의 도구 추출 완료. 서버 URL: '{server_url if server_url else '없음'}'")
        return cache_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"지정된 API 명세서 파일을 찾을 수 없습니다: {spec_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"API 명세서 처리 중 오류 발생 (잘못된 형식 또는 내용): {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API 명세서 처리 중 서버 내부 오류: {str(e)}")

# --- FastAPI 엔드포인트 정의 ---
@app.get("/", response_class=FileResponse, summary="챗봇 UI 제공", tags=["UI"])
async def get_chat_ui_page():
    chat_ui_html_path = os.path.join(TEMPLATES_DIR, "chat_ui.html")
    if not os.path.exists(chat_ui_html_path):
        # 이 오류는 서버 측 오류이므로 사용자에게 직접 노출하기보다는 로깅하고 일반 오류 메시지를 반환하는 것이 좋음
        print(f"오류: chat_ui.html 파일을 찾을 수 없습니다. 경로: {chat_ui_html_path}")
        raise HTTPException(status_code=500, detail="챗봇 UI 파일을 로드할 수 없습니다.")
    return FileResponse(chat_ui_html_path)

@app.get("/default_api_spec_path", summary="기본 API 명세서 경로 제공", tags=["Config"])
async def get_default_spec_path():
    return {"path": DEFAULT_API_SPEC_PATH}

@app.post("/chat", response_model=ChatResponse, summary="챗봇 메시지 처리", tags=["Chat"])
async def handle_chat_message(chat_message: ChatMessage):
    if not openai_client:
        return ChatResponse(error_message="OpenAI 클라이언트가 초기화되지 않았습니다. 서버 로그 및 API 키 설정을 확인하세요.")

    user_message = chat_message.message
    api_spec_path = chat_message.api_spec_path if chat_message.api_spec_path else DEFAULT_API_SPEC_PATH

    try:
        spec_data = await get_or_load_specs_and_tools(api_spec_path)
    except HTTPException as e:
        return ChatResponse(error_message=e.detail)

    tools = spec_data.get("tools", [])
    api_spec = spec_data.get("spec", {})
    base_server_url = spec_data.get("server_url", "")

    system_prompt = "당신은 사용자의 요청을 이해하고, 등록된 API 도구를 사용하여 정보를 찾거나 작업을 수행하는 AI 에이전트입니다. API 호출 결과에 기반하여 사용자에게 친절하게 답변해주세요."
    llm_initial_response_msg = get_llm_response_with_tools(user_message, tools, system_prompt=system_prompt)

    if not llm_initial_response_msg:
        return ChatResponse(error_message="LLM으로부터 초기 응답을 받지 못했습니다. 서버 로그를 확인하세요.")

    tool_calls_attempted_info: List[ToolCallInfo] = []
    api_responses_received_info: List[ApiResponseInfo] = []
    final_llm_output: Optional[str] = None

    if llm_initial_response_msg.tool_calls:
        print(f"LLM이 {len(llm_initial_response_msg.tool_calls)}개의 도구 호출을 요청했습니다.")
        tool_call_execution_results_for_llm: List[Dict[str, Any]] = []

        async with httpx.AsyncClient() as http_client:
            for tool_call in llm_initial_response_msg.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    print(f"오류: LLM이 제공한 도구 인수(arguments)가 JSON 형식이 아닙니다: {tool_call.function.arguments}")
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": "LLM provided invalid arguments format."})
                    })
                    api_responses_received_info.append(ApiResponseInfo(
                        tool_name=tool_name, api_path="N/A", method="N/A", error="잘못된 인수 형식"
                    ))
                    continue

                tool_calls_attempted_info.append(ToolCallInfo(tool_name=tool_name, tool_args=tool_args, tool_call_id=tool_call_id))
                print(f"  처리할 도구: '{tool_name}', 인수: {tool_args}")

                current_api_details = None
                for path_str, path_item_spec in api_spec.get('paths', {}).items():
                    for http_method_str, operation_spec in path_item_spec.items():
                        if operation_spec.get('operationId') == tool_name:
                            current_api_details = {
                                "path_template": path_str,
                                "method": http_method_str.upper(),
                                "operation_spec": operation_spec,
                                "base_url": base_server_url
                            }
                            break
                    if current_api_details: break

                current_api_call_response = ApiResponseInfo(tool_name=tool_name, api_path="N/A", method="N/A")

                if not current_api_details:
                    error_detail = f"API 명세에서 '{tool_name}'에 해당하는 operationId를 찾을 수 없습니다."
                    print(f"오류: {error_detail}")
                    current_api_call_response.error = error_detail
                    current_api_call_response.response_data = {"error": error_detail}
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": error_detail})
                    })
                    api_responses_received_info.append(current_api_call_response)
                    continue

                http_method = current_api_details["method"]
                api_path_template = current_api_details["path_template"]
                operation_spec = current_api_details["operation_spec"]

                current_api_call_response.api_path = api_path_template
                current_api_call_response.method = http_method

                resolved_path = api_path_template
                for arg_name, arg_value in tool_args.items():
                    if f"{{{arg_name}}}" in resolved_path:
                        resolved_path = resolved_path.replace(f"{{{arg_name}}}", str(arg_value))

                full_url = f"{current_api_details['base_url'].rstrip('/')}{resolved_path}"
                current_api_call_response.api_path = full_url

                query_params: Dict[str, Any] = {}
                headers: Dict[str, str] = {}
                json_body_payload: Optional[Dict[str, Any]] = None

                for param_spec in operation_spec.get('parameters', []):
                    param_name = param_spec.get('name')
                    if param_name in tool_args:
                        if param_spec.get('in') == 'query':
                            query_params[param_name] = tool_args[param_name]
                        elif param_spec.get('in') == 'header':
                            headers[param_name] = str(tool_args[param_name])

                if operation_spec.get('requestBody'):
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
                        params=query_params if query_params else None,
                        json=json_body_payload if json_body_payload and http_method in ["POST", "PUT", "PATCH"] else None,
                        headers=headers if headers else None,
                        timeout=15.0
                    )
                    api_http_response.raise_for_status()
                    try:
                        response_content_json = api_http_response.json()
                    except json.JSONDecodeError:
                        response_content_json = {"raw_response": api_http_response.text}

                    current_api_call_response.response_data = response_content_json
                    print(f"  API 호출 성공 ({tool_name}): 상태코드={api_http_response.status_code}, 응답길이={len(api_http_response.content)}")
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps(response_content_json, ensure_ascii=False)
                    })
                except httpx.HTTPStatusError as e:
                    error_detail = f"API 호출 실패 ({tool_name}): 상태코드={e.response.status_code}"
                    try:
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
                except httpx.RequestError as e:
                    error_detail = f"API 호출 중 네트워크/요청 오류 ({tool_name}): {type(e).__name__}"
                    print(f"오류: {error_detail} - {str(e)}")
                    current_api_call_response.error = "네트워크/요청 오류"
                    current_api_call_response.response_data = {"error": error_detail, "details": str(e)}
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": error_detail, "exception_details": str(e)})
                    })
                except Exception as e:
                    error_detail = f"API 호출 중 예상치 못한 내부 오류 ({tool_name}): {type(e).__name__}"
                    print(f"오류: {error_detail} - {str(e)}")
                    current_api_call_response.error = "내부 처리 오류"
                    current_api_call_response.response_data = {"error": error_detail, "details": str(e)}
                    tool_call_execution_results_for_llm.append({
                        "tool_call_id": tool_call_id, "name": tool_name,
                        "content": json.dumps({"error": error_detail, "exception_details": str(e)})
                    })
                api_responses_received_info.append(current_api_call_response)

        if tool_call_execution_results_for_llm:
            final_llm_output = get_final_response_after_tool_call(
                user_message,
                llm_initial_response_msg,
                tool_call_execution_results_for_llm,
                system_prompt=system_prompt
            )
        elif tool_calls_attempted_info:
            final_llm_output = "요청하신 작업을 위해 도구를 호출하려고 했지만, 실행 과정에서 문제가 발생하여 결과를 얻지 못했습니다. 다시 시도해주시거나 관리자에게 문의해주세요."

    else: # LLM이 도구 사용 없이 직접 답변한 경우
        print("LLM이 직접 답변했습니다.")
        final_llm_output = llm_initial_response_msg.content

    return ChatResponse(
        final_response=final_llm_output,
        tool_calls_info=tool_calls_attempted_info if tool_calls_attempted_info else None,
        tool_responses_info=api_responses_received_info if api_responses_received_info else None
    )

# --- 서버 실행 로직 (개발용) ---
if __name__ == "__main__":
    if not os.path.exists(DEFAULT_API_SPEC_PATH):
        print(f"경고: 기본 API 명세 파일 '{DEFAULT_API_SPEC_PATH}'을 찾을 수 없습니다.")
        try:
            print("api_parser.py를 실행하여 샘플 명세서를 생성합니다...")
            import subprocess
            process_result = subprocess.run(["python", "api_parser.py"], capture_output=True, text=True, check=False, cwd=os.path.dirname(__file__))
            if process_result.returncode == 0:
                print("api_parser.py 실행 성공, 샘플 명세서가 생성되었을 수 있습니다.")
            else:
                print(f"api_parser.py 실행 실패:\n{process_result.stderr}") # Corrected to print stderr

            if not os.path.exists(DEFAULT_API_SPEC_PATH):
                 print(f"'{DEFAULT_API_SPEC_PATH}' 파일이 여전히 없습니다. 수동으로 생성하거나 경로를 확인해주세요.")
        except Exception as e:
            print(f"api_parser.py 실행 중 오류: {e}")
    else:
        print(f"기본 API 명세 파일 확인: '{DEFAULT_API_SPEC_PATH}'")

    import uvicorn
    print("FastAPI 서버를 시작합니다 (http://localhost:8000)")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
