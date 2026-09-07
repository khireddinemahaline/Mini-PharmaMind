"""
Target Search Agent Module

This module provides the TargetSearch agent, specialized in discovering and
analyzing disease-associated therapeutic targets using the Open Targets MCP
server. The agent identifies biologically relevant genes, proteins, pathways,
disease-target associations, biomarkers, and known drug-target interactions
to support drug discovery workflows.

Functions:
    target_search_agent: Factory function to create a configured TargetSearch agent

Example:
    from agents.target_search import target_search_agent

    agent = await target_search_agent()
    # Use the agent in your workflow
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import (
    McpWorkbench,
    StreamableHttpServerParams,
)

from config.llm_client import model_client
from config.sytem_prompts import SYSTEM_PROMPTS_TARGET_SEARCH


async def target_search_agent() -> AssistantAgent:
    """
    Create and configure the TargetSearch agent for therapeutic target discovery.

    The TargetSearch agent specializes in:
    - Searching for diseases and associated therapeutic targets
    - Identifying disease-associated genes, proteins, receptors, enzymes,
      and other biological targets
    - Retrieving biological functions, pathways, and expression profiles
    - Exploring disease-target associations and supporting experimental evidence
    - Identifying biomarkers and known drug-target interactions
    - Providing structured biomedical knowledge to support downstream
      scientific analysis and decision-making

    The agent leverages the OpenTargets MCP server to access integrated
    biomedical databases and retrieve comprehensive disease and target
    information.

    Returns:
        AssistantAgent: Configured TargetSearch agent ready for use in
        multi-agent drug discovery workflows.

    Example:
        >>> agent = await target_search_agent()
        >>> # Agent is now ready to process therapeutic target discovery tasks
    """

    # Official Open Targets Remote MCP Server
    opentarget_workbench = McpWorkbench(
        server_params=StreamableHttpServerParams(
            url="https://mcp.platform.opentargets.org/mcp"
        )
    )

    return AssistantAgent(
        name="TargetSearch",
        description=(
            "A specialized biomedical research agent for therapeutic target "
            "discovery and disease analysis. The agent leverages the "
            "OpenTargets MCP workbench to identify diseases and their associated "
            "genes, proteins, receptors, enzymes, biomarkers, pathways, and "
            "other biologically relevant targets. It retrieves disease "
            "descriptions, target functions, expression profiles, "
            "disease-target associations, known drug-target interactions, and "
            "supporting experimental evidence. The agent provides structured "
            "and scientifically grounded information to support target "
            "identification, hypothesis generation, and downstream drug "
            "discovery workflows."
        ),
        model_client=model_client,
        system_message=SYSTEM_PROMPTS_TARGET_SEARCH,
        workbench=opentarget_workbench,
        model_client_stream=True,
        max_tool_iterations=3,
    )
