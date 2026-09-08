"""
Drug Search Agent Module

This module provides the DrugSearch agent, specialized in discovering and
analyzing drug candidates using the ChEMBL MCP server.

The agent searches for approved drugs, investigational compounds, small
molecules, and relevant bioactivity information to support drug discovery
workflows.

Functions:
    drug_search_agent: Factory function to create a configured DrugSearch agent

Example:
    from agents.drug_search import drug_search_agent

    agent = await drug_search_agent()
    # Use the agent in your workflow
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import (
    McpWorkbench,
    StreamableHttpServerParams,
)

from config.llm_client import model_client
from config.sytem_prompts import SYSTEM_PROMPTS_DRUG_SEARCH


async def setup_drug_search_agent() -> AssistantAgent:
    """
    Create and configure the DrugSearch agent for drug discovery.

    The DrugSearch agent specializes in:
    - Searching for drug candidates and small molecules
    - Finding approved and investigational compounds
    - Retrieving compound information from ChEMBL
    - Searching and analyzing bioactivity data
    - Identifying compounds with relevant IC50, Ki, EC50, and related values
    - Linking compounds to biological targets
    - Exploring drug mechanisms and indications
    - Supporting downstream drug discovery and target analysis workflows

    The agent uses the official ChEMBL MCP server through a remote
    Streamable HTTP connection.

    Returns:
        AssistantAgent: Configured DrugSearch agent ready for use in
        multi-agent drug discovery workflows.

    Example:
        >>> agent = await drug_search_agent()
        >>> # Agent is now ready to process drug discovery tasks
    """

    # Official ChEMBL Remote MCP Server
    chembl_workbench = McpWorkbench(
        server_params=StreamableHttpServerParams(
            url="https://chembl.caseyjhand.com/mcp",
        )
    )

    return AssistantAgent(
        name="DrugSearch",
        description=(
            "A specialized biomedical research agent focused on drug discovery "
            "and compound analysis. The agent leverages the ChEMBL MCP workbench "
            "to discover approved drugs, investigational compounds, and small "
            "molecules relevant to biomedical research questions. It can retrieve "
            "compound information, bioactivity measurements, target associations, "
            "drug mechanisms, indications, and related pharmacological evidence. "
            "The agent provides structured and scientifically grounded results "
            "to support compound identification, candidate prioritization, "
            "mechanism analysis, and downstream drug discovery workflows."
        ),
        model_client=model_client,
        system_message=SYSTEM_PROMPTS_DRUG_SEARCH,
        workbench=chembl_workbench,
        model_client_stream=True,
        max_tool_iterations=3,
    )
