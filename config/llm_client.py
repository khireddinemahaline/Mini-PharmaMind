from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API")
if not api_key:
    raise ValueError("DEEPSEEK_API environment variable is not set.")

model_client = OpenAIChatCompletionClient(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    model_info={
        "family": "deepseek",
        "function_calling": True,
        "vision": False,
        "json_output": False,
        "structured_output": False,
    },
    reasoning_effort="none",
    extra_body={
        "thinking": {"type": "enabled"}
    },
    parallel_tool_calls=False,
    max_retries=10,
    timeout=200,
    max_tokens=300000
)
