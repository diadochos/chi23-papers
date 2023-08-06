import requests
from urllib.parse import urlparse, urlunparse
import openai


def chat_completion_request(messages, functions=None, result=[], model="gpt-3.5-turbo-0613"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        result.append(response)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")


def insert_path_to_url(url: str, insert_path: str, position: int) -> str:
    parts = urlparse(url)
    new_path = '/'.join(parts.path.split('/')[:position] + [insert_path] + parts.path.split('/')[position:])
    return urlunparse(parts._replace(path=new_path))


def replace_domain(url: str, domain: str) -> str:
    parts = urlparse(url)
    return urlunparse(parts._replace(netloc=domain))
