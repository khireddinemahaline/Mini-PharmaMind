#!/usr/bin/env python3
"""
Agentic Pharma System - Chainlit Interface

This module provides a conversational AI interface for pharmaceutical research,
enabling multi-agent collaboration for drug discovery workflows. The system
integrates target search, drug search, and report generation agents with
human expertise through an interactive chat interface.

Features:
    - Multi-agent orchestration for drug discovery workflows
    - Authentication and session management
    - Streaming responses with real-time updates
    - State persistence and recovery
    - PDF report generation and download
    - Arize Phoenix integration for multi-agent observability and tracking

Observability:
    We use Arize Phoenix with OpenInference instrumentation instead of MLflow
    because it provides native support for AutoGen AgentChat multi-agent systems,
    offering superior tracking and visualization of agent interactions, tool calls,
    and conversation flows in distributed agent architectures. The OpenTelemetry-based
    tracing automatically captures all agent communications and LLM calls.

Author: Agentic Pharma Team
Last Modified: February 2026
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import cast, List, Optional
import json
from datetime import datetime
from dotenv import load_dotenv
from opentelemetry import trace

# Load environment variables from .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from opentelemetry.sdk.trace.sampling import Sampler, Decision, SamplingResult
from arize.otel import (
    TracerProvider, 
    BatchSpanProcessor, 
    GRPCSpanExporter, 
    Transport
)


from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor
from opentelemetry.sdk.resources import Resource

# 1. Define the Sampler to kill the "autogen create" noise
class AutoGenNoiseSampler(Sampler):
    def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None):
        # Drop the specific noisy span you identified
        if name.startswith("autogen"):
            return SamplingResult(Decision.DROP)
        
        # Keep everything else (LLM calls, etc.)
        return SamplingResult(Decision.RECORD_AND_SAMPLE)

    def get_description(self):
        return "AutoGenNoiseSampler"

# 2. Manually setup the Provider using Arize-aware classes
# This mimics what register() does but adds the Sampler
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")

if not ARIZE_SPACE_ID or not ARIZE_API_KEY:
    raise ValueError("ARIZE_SPACE_ID and ARIZE_API_KEY must be set in the .env file")

exporter = GRPCSpanExporter(
    space_id=ARIZE_SPACE_ID,
    api_key=ARIZE_API_KEY,
)

# Initialize provider with your custom sampler
provider = TracerProvider(
    space_id=ARIZE_SPACE_ID,
    api_key=ARIZE_API_KEY,
    sampler=AutoGenNoiseSampler(),
    resource=Resource(attributes={"model_id": "pharma-mind", "arize.project.name": "pharma-mind"})
)

# Add the batch processor to the provider
provider.add_span_processor(BatchSpanProcessor(exporter))

# 3. Set as global and instrument
trace.set_tracer_provider(provider)
AutogenAgentChatInstrumentor().instrument(tracer_provider=provider)



import chainlit as cl
from autogen_core import CancellationToken
from autogen_agentchat.agents import UserProxyAgent
from chainlit.types import ThreadDict
from pathlib import Path


# Add the project root to Python path for proper module resolution
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Imports
from agents.target_search import target_search_agent
from agents.drug_search import setup_drug_search_agent
from agents.report import report_agent
from agents.critique import setup_critique_agent
from agents.planning import setup_planning_agent
from config.llm_client import model_client
from autogen_agentchat.teams import SelectorGroupChat
from config.sytem_prompts import SELECT_PROMPT
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    SourceMatchTermination,
)
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_agentchat.conditions import ExternalTermination
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.messages import (
    ThoughtEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)



# ==================== Configuration ====================

# Directory for storing agent team state between sessions
STATE_DIR = "session_state"
os.makedirs(STATE_DIR, exist_ok=True)


async def save_team_state_to_disk(
    team: SelectorGroupChat, username: str, thread_id: str
) -> Optional[str]:
    """
    Persist the current state of the agent team to disk with session-specific filename.

    Args:
        team: The SelectorGroupChat team instance to save
        username: The authenticated user's identifier
        thread_id: The unique session/thread identifier

    Returns:
        Optional[str]: Absolute path to saved file on success, None on failure

    Raises:
        No exceptions raised - all errors are caught and logged

    Note:
        Filename format: team_state_{username}_{thread_id}.json
    """
    try:
        filename = f"team_state_{username}_{thread_id}.json"
        filepath = os.path.join(STATE_DIR, filename)

        state = await team.save_state()

        payload = json.dumps(state, indent=2, ensure_ascii=False)
        await asyncio.to_thread(Path(filepath).write_text, payload, encoding="utf-8")
        
        print(f"✅ Team state saved to: {filepath}")
        return filepath
    except (IOError, OSError) as e:
        error_msg = f"File I/O error saving team state: {str(e)}"
        print(f"❌ {error_msg}")
        return None
    except Exception as e:
        error_msg = f"Unexpected error saving team state: {str(e)}"
        print(f"❌ {error_msg}")
        return None


async def load_team_state_from_disk(
    team: SelectorGroupChat, username: str, thread_id: str
) -> bool:
    """
    Restore the agent team state from a session-specific saved JSON file.

    Args:
        team: The SelectorGroupChat team instance to restore state into
        username: The authenticated user's identifier
        thread_id: The unique session/thread identifier

    Returns:
        bool: True if state was successfully loaded, False otherwise

    Raises:
        No exceptions raised - all errors are caught and logged

    Note:
        Looks for filename: team_state_{username}_{thread_id}.json
    """
    try:
        filename = f"team_state_{username}_{thread_id}.json"
        filepath = os.path.join(STATE_DIR, filename)

        if not os.path.exists(filepath):
            print(f"❌ State file does not exist: {filepath}")
            return False

        data = await asyncio.to_thread(Path(filepath).read_text, encoding="utf-8")
        await team.load_state(json.loads(data))
        
        print(f"✅ Team state loaded from: {filepath}")
        return True
    except (IOError, OSError) as e:
        error_msg = f"File I/O error loading team state: {str(e)}"
        print(f"❌ {error_msg}")
        return False
    except (json.JSONDecodeError, ValueError) as e:
        error_msg = f"Invalid JSON in state file: {str(e)}"
        print(f"❌ {error_msg}")
        return False
    except Exception as e:
        error_msg = f"Unexpected error loading team state: {str(e)}"
        print(f"❌ {error_msg}")
        return False


def remove_team_state_from_disk(username: str, thread_id: str) -> bool:
    """
    Delete a session-specific saved team state file from disk to free up storage.

    Args:
        username: The authenticated user's identifier
        thread_id: The unique session/thread identifier

    Returns:
        bool: True if file was successfully deleted, False otherwise

    Raises:
        No exceptions raised - all errors are caught and logged

    Note:
        Deletes file: team_state_{username}_{thread_id}.json
    """
    try:
        # Generate session-specific filename
        filename = f"team_state_{username}_{thread_id}.json"
        filepath = os.path.join(STATE_DIR, filename)

        if not os.path.exists(filepath):
            print(f"⚠️ State file does not exist (already cleaned up): {filepath}")
            return True  # Consider this success since the end goal is achieved
        os.remove(filepath)
        print(f"✅ Team state removed from: {filepath}")
        return True
    except (IOError, OSError, PermissionError) as e:
        print(f"❌ File system error removing team state: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error removing team state: {str(e)}")
        return False

async def user_input_func(
    prompt: str, cancellation_token: CancellationToken | None = None
) -> str:
    """
    Capture user input from the Chainlit UI for the UserProxyAgent.

    Args:
        prompt: The question or request to display to the user
        cancellation_token: Optional token for cancelling the input request

    Returns:
        str: The user's input response, or a default message if timeout/error occurs

    Raises:
        No exceptions raised - all errors return default messages
    """
    try:
        response = await cl.AskUserMessage(
            content=prompt, timeout=300, raise_on_timeout=True
        ).send()
        if response:
            return response["output"]  # type: ignore
        else:
            return "User did not provide any input."
    except TimeoutError:
        print("⚠️ User input request timed out after 300 seconds")
        return "User did not provide any input within the time limit."
    except Exception as e:
        print(f"❌ Error getting user input: {str(e)}")
        return "An error occurred while requesting user input."

async def initialize_agents():
    try:
        termination_word = TextMentionTermination("TERMINATE")
        model_context = BufferedChatCompletionContext(buffer_size=40)  # 10 was too tight — see note
        max_iterations = MaxMessageTermination(30)   # safety net, not primary control
        termination_ext = ExternalTermination()
        termination = termination_word | max_iterations | termination_ext
        # SourceMatchTermination removed entirely — see below

        target_agent = await target_search_agent()
        drug_agent = await setup_drug_search_agent()
        report = report_agent()
        critique_agent = setup_critique_agent()
        planning_agent = setup_planning_agent()
        expert_human = UserProxyAgent(
        name="ExpertHuman",
        description=(
            "A Human-in-the-Loop biomedical expert who reviews and validates "
            "AI-generated findings during the drug discovery workflow. The "
            "expert provides scientific judgement, approves or revises target "
            "and drug rankings, resolves conflicting evidence, answers "
            "clarification requests, and records the final human decision "
            "before the workflow proceeds."
        ),
        input_func=user_input_func,
    )
        team = SelectorGroupChat(
            [planning_agent, target_agent, drug_agent, report, critique_agent, expert_human],
            model_client=model_client,
            termination_condition=termination,
            allow_repeated_speaker=True,
            selector_prompt=SELECT_PROMPT,
            model_context=model_context,
            max_turns=30,
        )
        return team, termination_ext
    except Exception as e:
        print(f"❌ Error initializing agents: {str(e)}")
        raise
    
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("researcher", "easydiscovery##1"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    """
    Configure chat profiles and starter prompts for the pharmaceutical research interface.

    Args:
        current_user: The authenticated user (required by decorator)

    Returns:
        List of chat profiles with starter prompts for common research workflows

    Raises:
        No exceptions raised - returns default profile on any error
    """
    try:
        return [
            cl.ChatProfile(
                name="Drug Discovery Researcher",
                markdown_description="A researcher focused on identifying novel drug targets and compounds.",
                icon="/public/logo.png",
                starters=[
                    cl.Starter(
                        label="Find drug targets for Alzheimer's disease",
                        message="Search for therapeutic targets associated with Alzheimer's disease and identify potential drug candidates that could modulate these targets.",
                        icon="/public/adn.png",
                    ),
                    cl.Starter(
                        label="Analyze aspirin compound",
                        message="Search for aspirin drug information including its molecular structure, mechanism of action, and known targets.",
                        icon="/public/drug.png",
                    ),
                    cl.Starter(
                        label="Cancer drug discovery",
                        message="Identify potential drug compounds for treating breast cancer, including their efficacy data and clinical trial status.",
                        icon="/public/cancer.png",
                    ),
                    cl.Starter(
                        label="Compare anti-inflammatory drugs",
                        message="Compare the mechanisms and efficacy of ibuprofen and naproxen as anti-inflammatory agents.",
                        icon="/public/disease.png",
                    ),
                ],
            )
        ]
    except Exception as e:
        print(f"❌ Error configuring chat profiles: {str(e)}")
        # Return minimal default profile on error
        return [
            cl.ChatProfile(
                name="Drug Discovery Researcher",
                markdown_description="Pharmaceutical research assistant",
                icon="/public/logo.png",
                starters=[],
            )
        ]

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Handle chat session resumption - reload saved state if available.

    Args:
        thread: Dictionary containing thread metadata including ID and user info

    Note:
        Automatically loads session-specific state file if it exists
    """
    try:
        # Get user info from Chainlit context
        user = cl.user_session.get("user")
        if not user:
            print("⚠️ No user found in session during resume")
            return

        username = user.identifier
        thread_id = thread.get("id")

        # Initialize agents first
        team, termination_ext = await initialize_agents()

        # Set session variables
        cl.user_session.set("team", team)
        cl.user_session.set("termination_ext", termination_ext)
        cl.user_session.set("username", username)
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("is_processing", False)
        cl.user_session.set("message_count", 0)
        cl.user_session.set(
            "has_sent_message", True
        )  # Set to True since this is an existing thread

        # Try to load saved state for this session
        state_loaded = await load_team_state_from_disk(team, username, thread_id)

        if state_loaded:
            print(f"✅ Resumed existing thread '{thread_id}' for user '{username}'")
        else:
            print(f"ℹ️ No saved state found for thread '{thread_id}' - starting fresh")

    except Exception as e:
        print(f"❌ Error resuming chat session: {str(e)}")
        await cl.Message(
            content=f"⚠️ **Session Resume Error**\n\n{str(e)}\n\nStarting a fresh session.",
            author="System",
        ).send()

@cl.on_chat_start
async def start_chat() -> None:
    """
    Initialize a new chat session when a user connects to the system.

    Session Variables:
        - team: The initialized SelectorGroupChat instance
        - termination_ext: External termination controller
        - is_processing: Flag to prevent concurrent task execution
        - username: Authenticated user's identifier
        - role: User's role (admin, user, guest)
        - thread_id: Unique session identifier
        - message_count: Counter for messages in this session
        - has_sent_message: Flag to track if user has sent any message

    Raises:
        Exception: If agent initialization fails, displays error to user and re-raises

    Note:
        Session state is NOT saved until user sends their first message.
        This prevents empty threads from being registered in the database.
    """
    # Get authenticated user info safely
    try:
        user = cl.user_session.get("user")
        username = user.identifier
        role = user.metadata.get("role", "guest")
    except Exception as e:
        print(f"⚠️ Error getting user info: {str(e)}")
        username = "unknown"
        role = "guest"

    # Get or create thread ID for this session
    thread_id = cl.context.session.thread_id

    try:
        team, termination_ext = await initialize_agents()

        # Store session state for workflow management
        cl.user_session.set("team", team)
        cl.user_session.set("termination_ext", termination_ext)
        cl.user_session.set("is_processing", False)
        cl.user_session.set("username", username)
        cl.user_session.set("role", role)
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("message_count", 0)
        cl.user_session.set(
            "has_sent_message", False
        )  # Track if user has sent any message

        print(f"🔵 New session initialized for '{username}' on thread '{thread_id}'")
        print(f"⏳ Waiting for first message before saving state...")

    except Exception as e:
        print(f"❌ Critical error in start_chat: {str(e)}")
        await cl.Message(
            content=f"❌ **System initialization failed:** {str(e)}\n\nPlease refresh the page and try again.",
            author="System",
        ).send()
        raise  # Re-raise to ensure proper error handling


@cl.action_callback("clear_session_state")
async def on_clear_session_state(action: cl.Action):
    """Handle the clear session state action button"""
    try:
        username = cl.user_session.get("username")
        thread_id = cl.user_session.get("thread_id")

        # Remove the state file
        success = remove_team_state_from_disk(username, thread_id)

        if success:
            # Reinitialize the team to clear memory
            team, termination_ext = await initialize_agents()
            cl.user_session.set("team", team)
            cl.user_session.set("termination_ext", termination_ext)
            cl.user_session.set("message_count", 0)

            await cl.Message(
                content="✅ **Session History Cleared**\n\n"
                "Your conversation history for this session has been deleted. "
                "Starting fresh!",
                author="System",
            ).send()
        else:
            await cl.Message(
                content="⚠️ **Clear Failed**\n\n"
                "Could not clear session history. It may already be empty.",
                author="System",
            ).send()

    except Exception as e:
        print(f"❌ Error clearing session state: {str(e)}")
        await cl.Message(content=f"❌ **Error**: {str(e)}", author="System").send()


@cl.on_message
async def handle_message(message: cl.Message) -> None:
    """
    Process incoming user messages and orchestrate multi-agent responses.

    Args:
        message: The incoming user message to process

    Raises:
        No exceptions raised - all errors are caught and displayed to user
    """
    if cl.user_session.get("is_processing", False):
        await cl.Message(
            content="⚠️ **System is currently processing another request. Please wait or click stop to cancel.**",
            author="System",
        ).send()
        return

    cl.user_session.set("is_processing", True)

    # Track message count for session management
    message_count = cl.user_session.get("message_count", 0)
    cl.user_session.set("message_count", message_count + 1)

    # Mark that user has sent a message (important for state saving)
    has_sent_message = cl.user_session.get("has_sent_message", False)
    if not has_sent_message:
        cl.user_session.set("has_sent_message", True)
        username = cl.user_session.get("username", "Guest")
        thread_id = cl.user_session.get("thread_id", "unknown")
        print(f"✅ First message received from '{username}' on thread '{thread_id}'")
        print(f"📝 State will now be saved for this session")

    try:
        team = cast(SelectorGroupChat, cl.user_session.get("team"))
        if not team:
            await cl.Message(
                content="❌ Agent team not initialized. Please refresh the page.",
                author="System",
            ).send()
            cl.user_session.set("is_processing", False)
            return

        await cl.Message(
            content="🚀 **Starting Multi-Agent Analysis...**", author="System"
        ).send()

        cancellation_token = CancellationToken()
        cl.user_session.set("cancellation_token", cancellation_token)

        current_streaming_msg: cl.Message | None = None
        
        # Track metrics
        agent_message_count = {}
        tool_call_count = 0
        total_tokens_streamed = 0
        start_time = datetime.now()
        start_time_ns = start_time.timestamp()

        from pathlib import Path

        pdf_dirs = [Path("generated_reports"), Path("resumes_uploaded")]
        known_pdf_state: dict[Path, int] = {}
        for pdf_dir in pdf_dirs:
            if pdf_dir.exists():
                for pdf_path in pdf_dir.glob("*.pdf"):
                    known_pdf_state[pdf_path] = pdf_path.stat().st_mtime_ns

        event_sequence = []
        
        async for msg in team.run_stream(
            task=TextMessage(content=message.content, source="ExpertHuman"),
            cancellation_token=cancellation_token,
        ):
            # Check if task was cancelled
            if cancellation_token.is_cancelled():
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                break
            agent_name = getattr(msg, "source", None)
            msg_type = type(msg).__name__
            
            # Track agent activity
            if agent_name:
                agent_message_count[agent_name] = agent_message_count.get(agent_name, 0) + 1
                event_sequence.append({
                    "agent": agent_name,
                    "event_type": msg_type,
                    "timestamp": datetime.now().isoformat()
                })

            # Thought spinner
            if isinstance(msg, ThoughtEvent):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None
                spinner_msg = cl.Message(content="⏳ *thinking...*", author=agent_name)
                await spinner_msg.send()
                await spinner_msg.remove()

            # Streaming chunks
            elif isinstance(msg, ModelClientStreamingChunkEvent):
                if (
                    current_streaming_msg is None
                    or getattr(current_streaming_msg, "author", None) != agent_name
                ):
                    if current_streaming_msg is not None:
                        await current_streaming_msg.send()
                    current_streaming_msg = cl.Message(content="", author=agent_name)
                await current_streaming_msg.stream_token(msg.content)
                total_tokens_streamed += len(msg.content)

            # Tool call request
            elif isinstance(msg, ToolCallRequestEvent):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None
                for tool_call in msg.content:
                    tool_call_count += 1
                    args_preview = str(tool_call.arguments)
                    if len(args_preview) > 300:
                        args_preview = args_preview[:300] + "... (truncated)"
                    await cl.Message(
                        content=f"`{agent_name}` 🛠️ Calling tool `{tool_call.name}` with args:\n```json\n{args_preview}\n```",
                        author=agent_name,
                    ).send()

            # Tool call summary
            elif isinstance(msg, ToolCallSummaryMessage):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None
                await cl.Message(
                    content=f"`{agent_name}` 🔄 reading tool's result...",
                    author=agent_name,
                ).send()

            # Task finished
            elif isinstance(msg, TaskResult):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                if cancellation_token.is_cancelled():
                    await cl.Message(
                        content="🛑 **Task cancelled by user**\n\nYou can now start a new query.",
                        author="System",
                    ).send()
                else:
                    final_message = "✅ **Task completed successfully**"
                    if msg.stop_reason:
                        final_message += f" ({msg.stop_reason})"
                    await cl.Message(content=final_message, author="System").send()

                # Check if PDF was generated during this task and provide download
                pdf_files = []
                for pdf_dir in pdf_dirs:
                    if pdf_dir.exists():
                        for pdf_path in pdf_dir.glob("*.pdf"):
                            file_mtime_ns = pdf_path.stat().st_mtime_ns
                            if (
                                pdf_path not in known_pdf_state
                                or file_mtime_ns > known_pdf_state[pdf_path]
                                or pdf_path.stat().st_mtime >= start_time_ns
                            ):
                                pdf_files.append(pdf_path)

                if pdf_files:
                    latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime_ns)
                    pdf_bytes = latest_pdf.read_bytes()

                    # Create downloadable file element
                    elements = [
                        cl.Pdf(
                            name=latest_pdf.name,
                            content=pdf_bytes,
                            display="inline",
                            mime="application/pdf",
                        ),
                        cl.File(
                            name=latest_pdf.name,
                            content=pdf_bytes,
                            mime="application/pdf",
                            display="inline",
                        ),
                    ]

                    await cl.Message(
                        content=f"📄 **Report Generated Successfully!**\n\nYour comprehensive analysis report is ready for download:",
                        elements=elements,
                        author="System",
                    ).send()

            # Skip all other messages
            else:
                continue

        # Finalize any remaining stream
        if current_streaming_msg is not None:
            await current_streaming_msg.send()

        # DO NOT reset team - we want to keep conversation history for session resumption
        # await team.reset()

    except CancellationToken.CancelledError:
        print("⚠️ Task was cancelled during execution")
        await cl.Message(
            content="🛑 **Task was cancelled**\n\nYou can start a new query.",
            author="System",
        ).send()
    except Exception as e:
        print(f"❌ Error in handle_message: {str(e)}")
        await cl.Message(
            content=f"❌ **Error occurred during processing**\n\n{str(e)}\n\nPlease try again or contact support if the issue persists.",
            author="System",
        ).send()
    finally:
            # Save state ONLY if user has sent at least one message
            # This prevents empty threads from being registered
            try:
                has_sent_message = cl.user_session.get("has_sent_message", False)

                if has_sent_message:
                    team = cl.user_session.get("team")
                    username = cl.user_session.get("username")
                    thread_id = cl.user_session.get("thread_id")

                    if team:
                        await save_team_state_to_disk(team, username, thread_id)
                        print(f"💾 Auto-saved state for {username} on thread {thread_id}")
                else:
                    print(f"⏭️ Skipping state save - no messages sent yet")
            except Exception as e:
                print(f"⚠️ Error auto-saving team state: {str(e)}")

            # Release processing lock
            cl.user_session.set("is_processing", False)


@cl.on_stop
async def on_stop():
    """Handle stop button click - cancel current task and save state (only if user has sent messages)"""
    try:
        token: CancellationToken = cl.user_session.get("cancellation_token")
        if token:
            token.cancel()

        # Only save state if user has actually sent a message
        has_sent_message = cl.user_session.get("has_sent_message", False)

        if has_sent_message:
            team = cl.user_session.get("team")
            username = cl.user_session.get("username")
            thread_id = cl.user_session.get("thread_id")

            if team:
                await save_team_state_to_disk(team, username, thread_id)
                print(f"💾 State saved after stop for {username}")
        else:
            print(f"⏭️ Skipping state save on stop - no messages sent")

        cl.user_session.set("is_processing", False)
        await cl.Message(
            content="🛑 **Task cancelled**\n\nYou can start a new query or return later.",
            author="System",
        ).send()
        print("🛑 User clicked stop!")
    except Exception as e:
        print(f"⚠️ Error in on_stop: {str(e)}")


@cl.on_chat_end
async def on_chat_end():
    """
    Handle when chat session ends (user refreshes/closes tab).

    Saves state ONLY if user has sent at least one message.
    This prevents empty threads from being saved.
    """
    try:
        token: CancellationToken = cl.user_session.get("cancellation_token")
        if token:
            token.cancel()

        # Only save state if user has actually sent a message
        has_sent_message = cl.user_session.get("has_sent_message", False)

        if has_sent_message:
            team = cl.user_session.get("team")
            username = cl.user_session.get("username")
            thread_id = cl.user_session.get("thread_id")

            if team:
                await save_team_state_to_disk(team, username, thread_id)
                print(
                    f"💾 Final state saved for user '{username}' on thread '{thread_id}'"
                )
        else:
            username = cl.user_session.get("username", "unknown")
            thread_id = cl.user_session.get("thread_id", "unknown")
            print(
                f"⏭️ Session closed without messages - no state saved for '{username}' on thread '{thread_id}'"
            )

    except Exception as e:
        print(f"⚠️ Error saving state on chat end: {str(e)}")


@cl.on_settings_update
async def setup_agent_settings(settings):
    """
    Handle user configuration updates from the settings sidebar.

    Args:
        settings: Dictionary containing updated setting key-value pairs

    Raises:
        No exceptions raised - errors are logged but don't disrupt user experience
    """
    try:
        # Future: Apply settings to agent configurations here
        # For example: update model temperature, tool selection, etc.
        await cl.Message(
            content=f"⚙️ **Settings Updated**\n\nYour preferences have been saved.",
            author="System",
        ).send()
    except Exception as e:
        print(f"❌ Error updating settings: {str(e)}")
        await cl.Message(
            content=f"⚠️ **Settings update failed**\n\nPlease try again.",
            author="System",
        ).send()

