from autogen_agentchat.agents import AssistantAgent
from config.llm_client import model_client
from config.sytem_prompts import PLANNING_SYSTEM_PROMPT


def setup_planning_agent():
    return AssistantAgent(
        name="Planning",
        description=(
    "Analyzes the user's objective, decomposes complex tasks into executable subtasks, "
  "Its primary responsibility is to create a structured plan to achieve the user's gaol"
  "and assigns each step to the most appropriate specialist agent."
  "planning agent could change the plan based on the feedback from the specialist agents and the user."
),
        system_message=PLANNING_SYSTEM_PROMPT,
        model_client =  model_client,
        model_client_stream=True,
    )
