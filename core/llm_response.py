import json


def get_llm_response(response: str):
    return json.loads(response)
