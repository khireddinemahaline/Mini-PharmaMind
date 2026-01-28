"""
Drug Search Agent Module

This module provides the DrugSearch agent, specialized in drug discovery and 
small-molecule compound analysis. The agent leverages chemical databases to 
identify potential drug candidates, analyze their properties, and evaluate 
their activity against biological targets.

Functions:
    setup_drug_search_agent: Async factory function to create a DrugSearch agent

Example:
    from agents.drug_search import setup_drug_search_agent
    
    agent = await setup_drug_search_agent()
    # Use the agent in your drug discovery workflow
"""

from autogen_agentchat.agents import AssistantAgent
from config.llm_client import model_client
from config.sytem_prompts import SYSTEM_PROMPTS_DRUG_SEARCH
from tools.tool_call import (
    compound_search,
    compound_info,
    compound_structure,
    activity_search,
    assay_info,
    batch_compound_lookup_tool
)


async def setup_drug_search_agent():
    """
    Create and configure the DrugSearch agent for drug discovery workflows.
    
    The DrugSearch agent specializes in:
    - Identifying small-molecule drug candidates
    - Analyzing chemical compound properties
    - Evaluating drug-target interactions
    - Retrieving compound activity data from chemical databases
    
    The agent is equipped with tools to:
    - Search for chemical compounds by various criteria
    - Retrieve detailed compound information (ADMET, physicochemical properties)
    - Analyze molecular structures
    - Query bioactivity and assay data
    - Perform batch compound lookups
    
    Returns:
        AssistantAgent: Configured drug search agent with streaming enabled
                       and limited tool iterations for efficient processing
                       
    Example:
        >>> agent = await setup_drug_search_agent()
        >>> # Agent is now ready to process drug discovery queries
        
    Note:
        This is an async function and must be awaited when called.
        The agent has a maximum of 3 tool iterations to prevent excessive API calls.
    """
    
    return AssistantAgent(
        name="DrugSearch",
        description="A specialized biomedical research agent focused on drug discovery. It identifies potential small-molecule candidates for specific protein targets or diseases, retrieves relevant compound data, and analyzes their properties using integrated drug discovery tools.",
        model_client=model_client,
        system_message=SYSTEM_PROMPTS_DRUG_SEARCH,
        tools=[compound_search, compound_info, compound_structure, activity_search, assay_info, batch_compound_lookup_tool],
        model_client_stream=True,
        max_tool_iterations=3,
    )
