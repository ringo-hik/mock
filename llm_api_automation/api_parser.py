import yaml
import json
from typing import Dict, List, Any

# OpenAPI Specification (OAS)의 버전을 명시적으로 다루지는 않지만,
# 주로 OAS 3.0.x의 구조를 가정하고 작성되었습니다.

def load_api_spec(filepath: str) -> Dict[str, Any]:
    """
    지정된 경로의 API 명세서 파일(YAML 또는 JSON)을 로드하여 파싱합니다.

    Args:
        filepath (str): API 명세서 파일의 경로.

    Returns:
        Dict[str, Any]: 파싱된 API 명세서 (Python 딕셔너리 형태).

    Raises:
        FileNotFoundError: 파일을 찾을 수 없는 경우.
        ValueError: 지원하지 않는 파일 형식이거나 파싱 오류 시.
        Exception: 그 외 예외 발생 시.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if filepath.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            elif filepath.endswith('.json'):
                return json.load(f)
            else:
                # 지원하지 않는 파일 확장자일 경우 오류 발생
                raise ValueError("지원하지 않는 파일 형식입니다. YAML 또는 JSON 파일을 사용해주세요.")
    except FileNotFoundError:
        # 원본 예외를 그대로 전달하여 호출 측에서 명확히 알 수 있도록 함
        raise
    except yaml.YAMLError as e:
        # YAML 파싱 중 구체적인 오류를 포함하여 예외 발생
        raise ValueError(f"YAML 파싱 오류: {e}")
    except json.JSONDecodeError as e:
        # JSON 파싱 중 구체적인 오류를 포함하여 예외 발생
        raise ValueError(f"JSON 파싱 오류: {e}")
    except Exception as e:
        # 기타 예외 처리
        raise Exception(f"API 명세서 로드/파싱 중 예상치 못한 오류 발생: {e}")

def extract_tools_from_spec(api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    파싱된 API 명세서에서 LLM이 도구(tool)로 사용할 수 있는 함수 정의 목록을 추출합니다.
    OpenAI Function Calling과 호환되는 형식으로 도구를 정의합니다.

    Args:
        api_spec (Dict[str, Any]): 파싱된 API 명세서.

    Returns:
        List[Dict[str, Any]]: 추출된 도구(함수) 정의 목록.
    """
    tools = []
    if not isinstance(api_spec, dict):
        # API 명세서가 딕셔너리 형태가 아니면 빈 목록 반환 또는 오류 처리
        print("경고: API 명세서가 올바른 형식이 아닙니다 (딕셔너리 타입이 아님).")
        return tools

    # API 명세서의 'paths' 필드가 없거나 비어있으면 도구 추출 불가
    if not api_spec.get('paths'):
        print("경고: API 명세서에 'paths' 정보가 없습니다. 도구를 추출할 수 없습니다.")
        return tools

    # 각 경로(path)와 HTTP 메소드(method)를 순회하며 도구 정보 추출
    for path, path_item in api_spec.get('paths', {}).items():
        if not isinstance(path_item, dict): continue # 경로 아이템 형식이 올바르지 않으면 건너뜀

        for method, operation in path_item.items():
            # HTTP 메소드 (소문자 기준)만 처리
            if method.lower() not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options', 'trace']:
                continue

            if not isinstance(operation, dict): continue # 오퍼레이션 형식이 올바르지 않으면 건너뜀

            # 도구 이름 결정: operationId가 있으면 사용, 없으면 생성
            operation_id = operation.get('operationId')
            if not operation_id:
                # operationId가 없는 경우, HTTP 메소드와 경로를 조합하여 생성
                # 예: "get_/users/{userId}" -> "get_users_userId"
                sanitized_path = path.replace('/', '_').replace('{', '').replace('}', '').replace('-', '_')
                operation_id = f"{method.lower()}{sanitized_path}"

            # 도구 설명 결정: summary > description > 기본 설명 순으로 사용
            summary = operation.get('summary', '')
            description = operation.get('description', '')
            tool_description = summary if summary else description
            if not tool_description: # 설명이 전혀 없는 경우 기본값 생성
                tool_description = f"{method.upper()} 요청을 {path} 경로로 전송합니다."

            # 도구 파라미터 스키마 초기화 (OpenAI Function Calling 형식)
            parameters_schema = {"type": "object", "properties": {}, "required": []}

            # API 파라미터 (path, query, header, cookie) 처리
            for param_info in operation.get('parameters', []):
                if not isinstance(param_info, dict): continue # 파라미터 정보 형식이 올바르지 않으면 건너뜀

                param_name = param_info.get('name')
                if not param_name: continue # 파라미터 이름이 없으면 건너뜀

                # 파라미터 스키마 추출 (OpenAPI 스키마를 JSON 스키마 형태로 최대한 유지)
                # 실제 LLM이 이 스키마를 얼마나 잘 이해하는지에 따라 조정 필요
                param_schema = param_info.get('schema', {"type": "string"}) # 스키마 없으면 string으로 가정

                parameters_schema['properties'][param_name] = {
                    "type": param_schema.get('type', "string"), # 타입 없으면 string으로 가정
                    "description": param_info.get('description', '') # 설명
                    # TODO: enum, format, items (for array type) 등 추가 스키마 정보 처리
                }
                if param_info.get('required', False):
                    parameters_schema['required'].append(param_name)

            # 요청 본문(RequestBody) 처리
            request_body_spec = operation.get('requestBody')
            if isinstance(request_body_spec, dict) and isinstance(request_body_spec.get('content'), dict):
                # 주로 JSON 컨텐츠 타입을 가정하고 처리 (가장 일반적인 케이스)
                # TODO: 다양한 content type 지원 (예: application/x-www-form-urlencoded, multipart/form-data)
                json_content = request_body_spec.get('content', {}).get('application/json')
                if isinstance(json_content, dict) and isinstance(json_content.get('schema'), dict):
                    body_schema = json_content.get('schema', {})

                    # OpenAI Function Calling은 요청 본문의 최상위 속성들을 파라미터로 인식하는 경향이 있음.
                    # 따라서, 요청 본문 스키마가 객체(object)이고 속성(properties)을 가지면,
                    # 해당 속성들을 도구의 파라미터로 직접 추가합니다.
                    if body_schema.get('type') == 'object' and isinstance(body_schema.get('properties'), dict):
                        for prop_name, prop_schema in body_schema.get('properties', {}).items():
                            if not isinstance(prop_schema, dict): continue # 속성 스키마 형식이 올바르지 않으면 건너뜀
                            parameters_schema['properties'][prop_name] = {
                                "type": prop_schema.get('type', "string"),
                                "description": prop_schema.get('description', ''),
                                # TODO: 중첩 스키마, enum, format 등 상세 정보 처리
                            }
                            # 요청 본문 스키마 내의 'required' 필드도 반영
                            if prop_name in body_schema.get('required', []):
                                if prop_name not in parameters_schema['required']: # 중복 추가 방지
                                     parameters_schema['required'].append(prop_name)
                    else:
                        # 요청 본문 스키마가 객체가 아니거나 속성이 없는 경우 (예: 배열, 기본 타입),
                        # 'request_body'라는 이름의 단일 파라미터로 전체 스키마를 전달.
                        # 이 방식은 LLM의 이해도에 따라 동작이 다를 수 있음.
                        parameters_schema['properties']['request_body'] = body_schema
                        if request_body_spec.get('required', False):
                            if 'request_body' not in parameters_schema['required']:
                                parameters_schema['required'].append('request_body')

            # 최종 도구 정의 (OpenAI Function Calling 형식)
            tools.append({
                "type": "function",
                "function": {
                    "name": operation_id,
                    "description": tool_description,
                    "parameters": parameters_schema
                }
            })

    return tools

# --- 개발 및 테스트용 코드 ---
if __name__ == '__main__':
    # 테스트용 샘플 OpenAPI YAML 파일 경로 및 내용
    sample_spec_filename = 'sample_openapi.yaml'
    sample_openapi_content = {
        "openapi": "3.0.0",
        "info": {"title": "샘플 API", "version": "1.0.0"},
        "servers": [{"url": "http://localhost:8000/api/v1"}],
        "paths": {
            "/users/{userId}": {
                "get": {
                    "summary": "특정 사용자 정보 조회",
                    "description": "주어진 ID를 가진 사용자의 상세 정보를 반환합니다.",
                    "operationId": "getUserById",
                    "parameters": [{
                        "name": "userId", "in": "path", "required": True,
                        "description": "조회할 사용자의 고유 ID", "schema": {"type": "integer", "format": "int64"}
                    }],
                    "responses": {"200": {"description": "성공적인 응답"}}
                }
            },
            "/items": {
                "post": {
                    "summary": "새로운 아이템 생성",
                    "operationId": "createItem",
                    "requestBody": {
                        "description": "생성할 아이템의 정보", "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "아이템의 이름"},
                                        "price": {"type": "number", "format": "float", "description": "아이템의 가격"},
                                        "tags": {"type": "array", "items": {"type": "string"}, "description": "아이템 관련 태그 목록"}
                                    },
                                    "required": ["name", "price"]
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "아이템 성공적으로 생성됨"}}
                }
            },
            "/search": { # operationId가 없는 경우 테스트
                "get": {
                    "summary": "항목 검색",
                    "parameters": [{
                         "name": "query", "in": "query", "required": True,
                         "schema": {"type": "string"}
                    }],
                    "responses": {"200": {"description": "검색 결과"}}
                }
            }
        }
    }
    # 테스트 파일 생성
    try:
        with open(sample_spec_filename, 'w', encoding='utf-8') as f:
            yaml.dump(sample_openapi_content, f, allow_unicode=True, sort_keys=False)
        print(f"테스트용 샘플 OpenAPI 명세서 '{sample_spec_filename}' 생성/덮어쓰기 완료.")
    except Exception as e:
        print(f"테스트 파일 생성 중 오류: {e}")

    # 생성된 테스트 파일로 파서 실행
    try:
        print(f"'{sample_spec_filename}' 파일 로드 및 파싱 테스트 시작...")
        loaded_spec = load_api_spec(sample_spec_filename)
        print("API 명세서 로드 성공.")

        extracted_tools = extract_tools_from_spec(loaded_spec)
        print("\n--- 추출된 도구 목록 ---")
        print(json.dumps(extracted_tools, indent=2, ensure_ascii=False))

        if extracted_tools:
            print(f"\n총 {len(extracted_tools)}개의 도구를 추출했습니다.")
            for tool in extracted_tools:
                print(f"  - 도구명: {tool['function']['name']}, 설명: {tool['function']['description'][:30]}...") # 설명 일부만 표시
        else:
            print("추출된 도구가 없습니다.")

    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")
