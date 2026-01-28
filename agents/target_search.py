"""
Target Search Agent Module

This module provides the TargetSearch agent, specialized in protein target 
discovery and analysis for drug development. The agent uses integrated 
biomedical databases to identify disease-associated proteins and retrieve 
relevant scientific literature.

Functions:
    target_search_agent: Factory function to create a configured TargetSearch agent

Example:
    from agents.target_search import target_search_agent
    
    agent = target_search_agent()
    # Use the agent in your workflow
"""

from autogen_agentchat.agents import AssistantAgent

from tools.tool_call import (
    search_for_articles,
    search_for_disease,
    extract_targets_summary,
)
from config.llm_client import model_client
from config.sytem_prompts import SYSTEM_PROMPTS_TARGET_SEARCH


def target_search_agent():
    """
    Create and configure the TargetSearch agent for protein target discovery.
    
    The TargetSearch agent specializes in:
    - Identifying disease-associated protein targets
    - Retrieving scientific literature and research articles
    - Analyzing target-disease relationships
    - Providing insights into biological roles of targets
    
    The agent is equipped with tools to:
    - Search for scientific articles via PubMed
    - Query disease information from biomedical databases
    - Extract and summarize target protein information

    Returns:
        AssistantAgent: Configured target search agent ready for use in 
                       multi-agent workflows
                       
    Example:
        >>> agent = target_search_agent()
        >>> # Agent is now ready to process target search queries
    """
    return AssistantAgent(
        name="TargetSearch",
        description="An expert biomedical research agent specialized in protein target discovery and analysis. It focuses on identifying disease-associated proteins, retrieving relevant literature, and providing insights into their biological roles using integrated biomedical data tools.",
        model_client=model_client,
        system_message=SYSTEM_PROMPTS_TARGET_SEARCH,
        tools=[search_for_articles, search_for_disease, extract_targets_summary],
        model_client_stream=True,
    )
