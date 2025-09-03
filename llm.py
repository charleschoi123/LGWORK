import os, time, json, requests
from typing import List, Optional


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
TIMEOUT = 60
MAX_TOKENS = int(os.getenv("MAX_TOKENS_PER_CALL", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))


class LLMError(Exception):
pass


def _headers():
return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# 兼容 OpenAI 格式的 /chat/completions 接口
# DeepSeek / Qwen / GLM 多数均支持该格式或提供兼容端点


def chat(messages: List[dict], system_prompt: Optional[str] = None) -> str:
if not API_KEY:
raise LLMError("Missing LLM_API_KEY")


payload = {
"model": MODEL_NAME,
"temperature": TEMPERATURE,
"max_tokens": MAX_TOKENS,
"messages": []
}
if system_prompt:
payload["messages"].append({"role": "system", "content": system_prompt})
payload["messages"].extend(messages)


url = BASE_URL.rstrip('/') + "/v1/chat/completions"


# 简单重试：网络波动/429 限流
for attempt in range(3):
try:
resp = requests.post(url, headers=_headers(), json=payload, timeout=TIMEOUT)
if resp.status_code == 429:
time.sleep(1 + attempt)
continue
if resp.status_code >= 400:
raise LLMError(f"[{resp.status_code}] {resp.text}")
data = resp.json()
# OpenAI 兼容返回结构
content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
return content.strip()
except requests.RequestException as e:
if attempt == 2:
raise LLMError(str(e))
time.sleep(1 + attempt)
raise LLMError("Unknown LLM error")
