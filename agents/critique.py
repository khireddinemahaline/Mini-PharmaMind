from autogen_agentchat.agents import AssistantAgent
from config.llm_client import model_client
from config.sytem_prompts import CRITIQUE_SYSTEM_PROMPT


def setup_critique_agent():
    return AssistantAgent(
        name="Critique",
        description=(
            "Quality assurance and workflow validation agent for PharmaMind. "
            "Handles greetings, off-topic requests, and questions about the platform. "
            "Validates and refines user requests, identifies missing or ambiguous "
            "information, improves prompt quality, reviews specialist outputs for "
            "completeness and scientific consistency, and recommends Human-in-the-Loop "
            "review by ExpertHuman when scientific judgement or validation is required. "
            "Does not perform biomedical analysis, literature searches, or report generation."
        ),
        model_client=model_client,
        system_message=CRITIQUE_SYSTEM_PROMPT,
        model_client_stream=True,
    )
