from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from pathlib import Path
import os


# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[1] / ".env"  # Go up 2 levels
load_dotenv(env_path)

# Get the API key from the environment
api_key = os.getenv("DEEPSEEK_API")

if not api_key:
    raise ValueError(
        "DEEPSEEK_API environment variable is not set. Please set it in your .env file."
    )


model_client = OpenAIChatCompletionClient(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    model_info={
        "id": "deepseek-chat",
        "family": "deepseek",
        "function_calling": True,
        "vision": False,
        "json_output": False,
        "structured_output": False,
    },
    max_retries=10,
    time_out=200,
)
