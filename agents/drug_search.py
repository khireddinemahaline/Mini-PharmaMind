"""
Drug Search Agent Module
This module provides the DrugSearch agent, specialized in drug discovery and 
small-molecule compound analysis. The agent leverages the ChEMBL MCP endpoint
to identify potential drug candidates, analyze their properties, and evaluate 
their activity against biological targets.

Functions:
    setup_drug_search_agent: Async factory function to create a DrugSearch agent

Example:
    from agents.drug_search import setup_drug_search_agent
    agent = await setup_drug_search_agent()
    # Use the agent in your drug discovery workflow
"""


from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import StreamableHttpServerParams, mcp_server_tools, StdioServerParams
from config.llm_client import model_client
from config.sytem_prompts import SYSTEM_PROMPTS_DRUG_SEARCH
from pathlib import Path

#linicaltrials_MCP_URL = "https://clinicaltrials.caseyjhand.com/mcp"



    
async def setup_drug_search_agent() -> AssistantAgent:
    """
    Create and configure the DrugSearch agent for drug discovery workflows.

    The DrugSearch agent specializes in:
    - Identifying small-molecule drug candidates
    - Analyzing chemical compound properties
    - Evaluating drug-target interactions
    - Retrieving compound activity data from ChEMBL (v34, EMBL-EBI)

    Tools are dynamically fetched from the ChEMBL MCP endpoint, covering:
    - Compound search by name or structure
    - Bioactivity data (IC50, EC50, Ki)
    - ADMET and physicochemical properties
    - Assay and target information
    - Batch compound lookups

    Returns:
        AssistantAgent: Configured drug search agent with streaming enabled
                        and limited tool iterations for efficient processing.

    Raises:
        Exception: If the ChEMBL MCP server is unreachable or returns no tools.

    Example:
        >>> agent = await setup_drug_search_agent()
        >>> # Agent is now ready to process drug discovery queries

    Note:
        This is an async function and must be awaited when called.
        The agent has a maximum of 3 tool iterations to prevent excessive API calls.
    """
    project_root = Path(__file__).resolve().parent.parent
    chembl_server = StdioServerParams(
        command="node",
        args=[
            str(project_root / "mcp-servers" / "ChEMBL-MCP-Server" / "build" / "index.js"),
        ],
    )
    chembl_tools = await mcp_server_tools(chembl_server)

   ## pubchem_server = StdioServerParams(
    ##command="node",
    ##args=["/home/ubuntu/Documents/Agentic/pharmaMind/MCP-serveres/PubChem-MCP-Server-main/build/index.js"],
    ##env={},
##)

    #pubchem_tools = await mcp_server_tools(pubchem_server)



  #  clinicaltrials_server = StreamableHttpServerParams(
   #     url=clinicaltrials_MCP_URL,
    #    headers={"Accept": "application/json, text/event-stream"},
    #)
    #clinicaltrials_tools = await mcp_server_tools(clinicaltrials_server)

    

    return AssistantAgent(
        name="DrugSearch",
        description=(
            "A specialized biomedical research agent focused on discovering and analyzing "
    "drug candidates. It searches for approved drugs, investigational compounds, "
    "and small molecules relevant to the user's request by leveraging mcp tools"
     "identifies promising drug candidates, and provides structured results to "
    "support downstream scientific analysis and decision-making."
        ),
        model_client=model_client,
        system_message=SYSTEM_PROMPTS_DRUG_SEARCH,
        tools=chembl_tools,
        model_client_stream=True,
        max_tool_iterations=3,
    )