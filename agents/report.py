from autogen_agentchat.agents import AssistantAgent

from tools.tool_call import save_to_pdf
from config.llm_client import model_client
from config.sytem_prompts import SYSTEM_PROMPTS_REPORT


def report_agent():
    """
    Create and return the ReportAgent.

    Returns:
        AssistantAgent: Configured report agent for PDF generation
    """
    return AssistantAgent(
        name="ReportAgent",
        description="An automated reporting agent that compiles biomedical analysis results into structured, well-formatted PDF reports. It focuses on transforming research findings into clear, professional outputs using integrated document generation tools.",
        model_client=model_client,
        system_message=SYSTEM_PROMPTS_REPORT,
        tools=[save_to_pdf],
        model_client_stream=True,
    )
