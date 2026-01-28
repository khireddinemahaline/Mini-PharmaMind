from autogen_agentchat.agents import AssistantAgent
from config.llm_client import model_client
from config.sytem_prompts import CRITIQUE_SYSTEM_PROMPT


async def setup_critique_agent():
    
    return AssistantAgent(
        name="Critique",
        description="Handles ALL greetings (hi, hello, hey, good morning), random text (asasdasd, test), off-topic queries (jokes, casual chat), and provides PharmaMind platform documentation. Must be selected FIRST for any non-pharmaceutical or vague queries. DO NOT select ExpertHuman for greetings.",
        model_client=model_client,
        system_message=CRITIQUE_SYSTEM_PROMPT,
        model_client_stream=True,
    )
