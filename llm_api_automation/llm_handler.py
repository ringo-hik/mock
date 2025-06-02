import os
import json
from openai import OpenAI, OpenAIError # OpenAIError 명시적 임포트
from typing import List, Dict, Any, Optional, Union # Union 추가

# --- OpenAI 클라이언트 초기화 ---
# API 키는 환경변수 'OPENAI_API_KEY'에서 가져옵니다.
# 이 키가 없으면 OpenAI API 호출 시 오류가 발생합니다.
try:
    # 환경 변수에서 API 키를 읽어 클라이언트 초기화
    # api_key=None 이면 OpenAI 라이브러리가 자동으로 OPENAI_API_KEY 환경 변수를 찾음
    client = OpenAI()
    # 키 존재 여부 테스트 (선택적) - 실제 호출 전까지는 오류 안 날 수 있음
    if not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. OpenAI API 기능이 제한될 수 있습니다.")
except OpenAIError as e: # openai 라이브러리 관련 모든 예외 처리
    client = None # 클라이언트 초기화 실패 시 None으로 설정
    print(f"OpenAI 클라이언트 초기화 실패: {e}. OPENAI_API_KEY 환경 변수를 확인하세요.")
except Exception as e: # 그 외 예외
    client = None
    print(f"OpenAI 클라이언트 초기화 중 예상치 못한 오류: {e}")


def get_llm_response_with_tools(
    user_message: str,
    tools: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo", # 기본 모델, 필요시 "gpt-4o" 등으로 변경 가능
    system_prompt: Optional[str] = None # 시스템 프롬프트 추가 (선택 사항)
) -> Optional[Any]: # OpenAI 라이브러리 버전업에 따라 Message 객체 반환 (Dict 대신)
    """
    사용자 메시지와 사용 가능한 도구 목록을 OpenAI LLM에게 전달하고,
    LLM의 응답 (도구 사용 결정 포함)을 반환합니다.

    Args:
        user_message (str): 사용자가 입력한 메시지.
        tools (List[Dict[str, Any]]): LLM이 사용할 수 있는 도구 목록 (OpenAI 형식).
        model (str): 사용할 OpenAI 모델 이름.
        system_prompt (Optional[str]): LLM에게 전달할 시스템 레벨 지시사항.

    Returns:
        Optional[openai.types.chat.ChatCompletionMessage]: LLM의 응답 메시지 객체.
                                                        오류 발생 시 None 반환.
    """
    if not client:
        print("오류: OpenAI 클라이언트가 초기화되지 않았습니다. API 키 설정을 확인하세요.")
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    try:
        print(f"LLM 요청: 모델='{model}', 메시지='{user_message[:50]}...', 도구 {len(tools)}개")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None, # 도구가 없으면 None 전달
            tool_choice="auto",  # LLM이 메시지를 보고 도구 호출 여부 및 어떤 도구를 호출할지 결정
        )
        # API 응답에서 첫 번째 선택지의 메시지를 반환
        response_message = response.choices[0].message
        # print(f"LLM 응답 수신: {response_message}") # 디버깅용 상세 로그
        return response_message
    except OpenAIError as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        # 특정 오류 코드에 따른 처리 추가 가능 (예: 인증 오류, 속도 제한 등)
        # if isinstance(e, openai.APIStatusError) and e.status_code == 401:
        #     print("오류: OpenAI API 키가 유효하지 않습니다.")
        return None
    except Exception as e: # 예상치 못한 다른 예외
        print(f"LLM API 호출 중 예상치 못한 오류: {e}")
        return None

def get_final_response_after_tool_call(
    user_message: str,
    original_llm_response_message: Any, # get_llm_response_with_tools의 반환값 (Message 객체)
    tool_call_results: List[Dict[str, Any]], # 각 tool_call에 대한 실행 결과
    model: str = "gpt-3.5-turbo",
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """
    도구 호출 결과를 바탕으로 LLM에게 최종 사용자 응답 생성을 요청합니다.

    Args:
        user_message (str): 최초 사용자 메시지.
        original_llm_response_message (ChatCompletionMessage): 이전 LLM 응답 (tool_calls 포함).
        tool_call_results (List[Dict[str, Any]]): 각 도구 호출의 결과.
            각 항목은 {"tool_call_id": str, "name": str, "content": str (JSON)} 형식이어야 함.
        model (str): 사용할 OpenAI 모델 이름.
        system_prompt (Optional[str]): LLM에게 전달할 시스템 레벨 지시사항.

    Returns:
        Optional[str]: LLM이 생성한 최종 사용자 응답 문자열. 오류 시 None.
    """
    if not client:
        print("오류: OpenAI 클라이언트가 초기화되지 않았습니다.")
        return None

    messages: List[Dict[str, Any]] = [] # 타입 명시
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    # LLM의 이전 응답 메시지 객체를 messages 목록에 추가
    # OpenAI 라이브러리 v1.x 이상에서는 .dict()로 변환하거나 필요한 속성만 추출
    if original_llm_response_message:
        # Message 객체를 딕셔너리로 변환하여 추가 (role, content, tool_calls 등 포함)
        # 필요한 속성만 선택적으로 추가할 수도 있음
        msg_dict = {
            "role": original_llm_response_message.role,
            "content": original_llm_response_message.content if original_llm_response_message.content else "", # None일 경우 빈 문자열
        }
        if original_llm_response_message.tool_calls:
            msg_dict["tool_calls"] = [
                {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in original_llm_response_message.tool_calls
            ]
        messages.append(msg_dict)

    # 각 도구 호출 결과 메시지를 messages 목록에 추가
    for tool_result in tool_call_results:
        messages.append(
            {
                "tool_call_id": tool_result["tool_call_id"],
                "role": "tool",
                "name": tool_result["name"],
                "content": tool_result["content"], # 도구 실행 결과 (JSON 문자열 형태)
            }
        )

    try:
        print(f"LLM 최종 응답 요청: 모델='{model}', 메시지 개수={len(messages)}")
        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        final_message_content = final_response.choices[0].message.content
        # print(f"LLM 최종 응답 수신: {final_message_content}") # 디버깅용 상세 로그
        return final_message_content
    except OpenAIError as e:
        print(f"OpenAI API 최종 응답 요청 중 오류 발생: {e}")
        return None
    except Exception as e:
        print(f"LLM API 최종 응답 요청 중 예상치 못한 오류: {e}")
        return None

# --- 개발 및 테스트용 코드 ---
if __name__ == '__main__':
    if not os.environ.get("OPENAI_API_KEY"):
        print("테스트 중단: OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        print("테스트를 진행하려면 'export OPENAI_API_KEY="YOUR_API_KEY"' 와 같이 설정해주세요.")
    elif not client:
        print("테스트 중단: OpenAI 클라이언트가 초기화되지 않았습니다.")
    else:
        print("LLM 핸들러 테스트 시작 (OPENAI_API_KEY 및 클라이언트 감지됨)")

        # 테스트용 샘플 도구 정의 (api_parser.py의 결과와 유사한 형태)
        sample_tools_for_test = [
            {
                "type": "function",
                "function": {
                    "name": "getWeather",
                    "description": "특정 도시의 현재 날씨 정보를 가져옵니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "날씨를 조회할 도시 이름 (예: Seoul, London)"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        # 시나리오 1: LLM이 도구를 사용해야 하는 경우
        user_query_for_tool = "오늘 서울 날씨 어때?"
        print(f"\n--- 시나리오 1: 도구 사용 질의 ('{user_query_for_tool}') ---")

        # 시스템 프롬프트 예시
        system_prompt_example = "당신은 사용자 요청을 분석하여 API 호출을 돕는 AI 어시스턴트입니다."

        llm_response_message_obj = get_llm_response_with_tools(
            user_query_for_tool,
            sample_tools_for_test,
            system_prompt=system_prompt_example
        )

        if llm_response_message_obj:
            print(f"LLM 초기 응답 (메시지 객체): {llm_response_message_obj}")

            if llm_response_message_obj.tool_calls:
                print("LLM이 도구 호출을 요청했습니다.")
                # 실제로는 여러 도구 호출이 있을 수 있으므로 반복 처리 필요
                tool_call = llm_response_message_obj.tool_calls[0] # 첫 번째 도구 호출 사용
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                tool_call_id = tool_call.id

                try:
                    tool_args_dict = json.loads(tool_args_str)
                    print(f"  호출할 도구: {tool_name}, 파라미터: {tool_args_dict}, 호출 ID: {tool_call_id}")

                    # 가상 도구 실행: 실제로는 API 클라이언트가 이 부분을 담당
                    # 예시: 'getWeather' 도구 호출 시뮬레이션
                    if tool_name == "getWeather":
                        city_name = tool_args_dict.get("city", "UnknownCity")
                        # 가상 API 응답
                        simulated_tool_output = json.dumps({
                            "city": city_name, "temperature": "25°C", "condition": "맑음"
                        })
                        print(f"  가상 도구 실행 결과 ({tool_name}): {simulated_tool_output}")
                    else:
                        simulated_tool_output = json.dumps({"status": "error", "message": f"알 수 없는 도구: {tool_name}"})

                    # 도구 실행 결과를 LLM에 전달하기 위한 형식으로 준비
                    tool_call_result_for_llm = {
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": simulated_tool_output # JSON 문자열
                    }

                    # LLM에게 최종 사용자 응답 생성 요청
                    final_user_response = get_final_response_after_tool_call(
                        user_query_for_tool,
                        llm_response_message_obj, # Message 객체 전달
                        [tool_call_result_for_llm], # 결과 목록 전달
                        system_prompt=system_prompt_example
                    )

                    if final_user_response:
                        print(f"\n최종 사용자 응답:\n{final_user_response}")
                    else:
                        print("최종 사용자 응답 생성에 실패했습니다.")

                except json.JSONDecodeError:
                    print(f"오류: LLM이 제공한 도구 인수(arguments)가 올바른 JSON 형식이 아닙니다: {tool_args_str}")
                except Exception as e:
                    print(f"도구 처리 중 예외 발생: {e}")
            else:
                # LLM이 도구를 사용하지 않고 직접 답변한 경우
                print("LLM이 도구 호출을 요청하지 않았습니다. 직접 답변:")
                print(llm_response_message_obj.content)
        else:
            print("LLM으로부터 초기 응답을 받지 못했습니다 (시나리오 1).")

        # 시나리오 2: LLM이 직접 답변할 수 있는 경우 (제공된 도구와 무관)
        user_query_direct_ans = "간단한 자기소개 해줘."
        print(f"\n--- 시나리오 2: 직접 답변 질의 ('{user_query_direct_ans}') ---")
        llm_direct_response_msg_obj = get_llm_response_with_tools(
            user_query_direct_ans,
            sample_tools_for_test, # 도구는 제공하지만, LLM이 사용하지 않을 것으로 예상
            system_prompt=system_prompt_example
        )
        if llm_direct_response_msg_obj:
            if llm_direct_response_msg_obj.tool_calls:
                print("LLM이 도구 호출을 요청했습니다 (예상치 않음).")
            else:
                print("LLM이 직접 답변 (예상대로):")
                print(llm_direct_response_msg_obj.content)
        else:
            print("LLM으로부터 응답을 받지 못했습니다 (시나리오 2).")
