from autogen_agentchat.agents import AssistantAgent
from config.llm_client import model_client
from config.sytem_prompts import CRITIQUE_SYSTEM_PROMPT


def setup_critique_agent():
    return AssistantAgent(
        name="Critique",
        description=(
            "anlyze the user's input and refine it and improves prompt quality for clarity, completeness, and scientific accuracy. or handoff to ExpertHuman for clearity and completeness informations. "
            "Handles greetings, off-topic requests, and questions about the platform and guide the user to use the platform effectively. "
            "reviews specialist outputs for completeness and scientific consistency, and recommends Human-in-the-Loop by inviting ExpertHuman to review the results of outputs when necessary. "
            
        ),
        model_client=model_client,
        system_message=CRITIQUE_SYSTEM_PROMPT,
        model_client_stream=True,
    )
