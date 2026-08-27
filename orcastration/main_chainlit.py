#!/usr/bin/env python3
"""
Agentic Pharma System - Chainlit Interface

Multi-agent pharmaceutical research interface using AutoGen AgentChat,
DeepSeek, Chainlit, and Arize/OpenInference instrumentation.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import chainlit as cl
from autogen_core import CancellationToken
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    ExternalTermination,
    MaxMessageTermination,
    SourceMatchTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    TextMessage,
    ThoughtEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import SelectorGroupChat
from chainlit.types import ThreadDict

# Centralized instrumentation must be imported after project_root is available.
from orcastration.instrumentation import tracer_provider  # noqa: F401

from agents.target_search import target_search_agent
from agents.drug_search import setup_drug_search_agent
from agents.report import report_agent
from agents.critique import setup_critique_agent
from agents.planning import setup_planning_agent
from config.llm_client import model_client
from config.sytem_prompts import SELECT_PROMPT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_DIR = PROJECT_ROOT / "session_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

async def save_team_state_to_disk(
    team: SelectorGroupChat,
    username: str,
    thread_id: str,
) -> Optional[str]:
    """Persist the current SelectorGroupChat state to disk."""
    try:
        filename = f"team_state_{username}_{thread_id}.json"
        filepath = STATE_DIR / filename

        state = await team.save_state()
        payload = json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        )

        await asyncio.to_thread(
            filepath.write_text,
            payload,
            encoding="utf-8",
        )

        print(f"✅ Team state saved to: {filepath}")
        return str(filepath)

    except (IOError, OSError) as exc:
        print(f"❌ File I/O error saving team state: {exc}")
        return None

    except Exception as exc:
        print(f"❌ Unexpected error saving team state: {exc}")
        return None


async def load_team_state_from_disk(
    team: SelectorGroupChat,
    username: str,
    thread_id: str,
) -> bool:
    """Restore SelectorGroupChat state from disk."""
    try:
        filename = f"team_state_{username}_{thread_id}.json"
        filepath = STATE_DIR / filename

        if not filepath.exists():
            print(f"ℹ️ No state file found: {filepath}")
            return False

        data = await asyncio.to_thread(
            filepath.read_text,
            encoding="utf-8",
        )

        await team.load_state(json.loads(data))

        print(f"✅ Team state loaded from: {filepath}")
        return True

    except (IOError, OSError) as exc:
        print(f"❌ File I/O error loading team state: {exc}")
        return False

    except (json.JSONDecodeError, ValueError) as exc:
        print(f"❌ Invalid JSON in state file: {exc}")
        return False

    except Exception as exc:
        print(f"❌ Unexpected error loading team state: {exc}")
        return False


def remove_team_state_from_disk(
    username: str,
    thread_id: str,
) -> bool:
    """Delete a saved SelectorGroupChat state file."""
    try:
        filename = f"team_state_{username}_{thread_id}.json"
        filepath = STATE_DIR / filename

        if not filepath.exists():
            print(f"⚠️ State file does not exist: {filepath}")
            return True

        filepath.unlink()
        print(f"✅ Team state removed: {filepath}")
        return True

    except (IOError, OSError, PermissionError) as exc:
        print(f"❌ File system error removing state: {exc}")
        return False

    except Exception as exc:
        print(f"❌ Unexpected error removing state: {exc}")
        return False


# ---------------------------------------------------------------------------
# Human-in-the-loop input
# ---------------------------------------------------------------------------

async def user_input_func(
    prompt: str,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """
    Capture human input through Chainlit.

    Note:
        AutoGen's CancellationToken is accepted by the callback signature,
        but asyncio.CancelledError is the exception used for task cancellation.
    """
    try:
        response = await cl.AskUserMessage(
            content=prompt,
            timeout=300,
            raise_on_timeout=True,
        ).send()

        if response:
            return response["output"]  # type: ignore[index]

        return "User did not provide any input."

    except asyncio.CancelledError:
        print("🛑 Human input request was cancelled.")
        raise

    except TimeoutError:
        print("⚠️ User input request timed out after 300 seconds.")
        return "User did not provide any input within the time limit."

    except Exception as exc:
        print(f"❌ Error getting user input: {exc}")
        return "An error occurred while requesting user input."


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

async def initialize_agents():
    """Create and configure the complete pharmaceutical research team."""
    try:
        # Stop when:
        #   1. An agent explicitly mentions TERMINATE,
        #   2. ExternalTermination is triggered,
        #   3. ReportAgent becomes the source of a terminating message.
        termination_word = TextMentionTermination("TERMINATE")
        termination_ext = ExternalTermination()
        source_match_termination = SourceMatchTermination("ReportAgent")

        termination = (
            termination_word
            | termination_ext
            | source_match_termination
        )

        # Unbounded context avoids the artificial small context restriction
        # that was previously configured.
        from autogen_core.model_context import UnboundedChatCompletionContext

        model_context = UnboundedChatCompletionContext()

        print("🔧 Initializing agents...")

        target_agent = await target_search_agent()
        drug_agent = await setup_drug_search_agent()
        report = report_agent()
        critique_agent = setup_critique_agent()
        planning_agent = setup_planning_agent()

        expert_human = UserProxyAgent(
            name="ExpertHuman",
            description=(
                "A Human-in-the-Loop biomedical expert who reviews and "
                "validates AI-generated findings during the drug discovery "
                "workflow. The expert provides scientific judgement, "
                "approves or revises target and drug rankings, resolves "
                "conflicting evidence, answers clarification requests, "
                "and records the final human decision before the workflow "
                "proceeds."
            ),
            input_func=user_input_func,
        )

        team = SelectorGroupChat(
            [
                planning_agent,
                target_agent,
                drug_agent,
                report,
                critique_agent,
                expert_human,
            ],
            model_client=model_client,
            termination_condition=termination,
            allow_repeated_speaker=True,
            selector_prompt=SELECT_PROMPT,
            model_context=model_context,
        )

        print("✅ Agent team initialized successfully.")

        return team, termination_ext

    except Exception as exc:
        print(f"❌ Error initializing agents: {exc}")
        import traceback
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    Authenticate the Chainlit user.

    Replace this with database-backed authentication in production.
    """
    if (username, password) == ("researcher", "easydiscovery##1"):
        return cl.User(
            identifier="admin",
            metadata={
                "role": "admin",
                "provider": "credentials",
            },
        )

    return None


# ---------------------------------------------------------------------------
# Chat profile
# ---------------------------------------------------------------------------

@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    """Configure the pharmaceutical research chat profile."""
    try:
        return [
            cl.ChatProfile(
                name="Drug Discovery Researcher",
                markdown_description=(
                    "A researcher focused on identifying novel drug "
                    "targets and compounds."
                ),
                icon="/public/logo.png",
                starters=[
                    cl.Starter(
                        label="Find drug targets for Alzheimer's disease",
                        message=(
                            "Search for therapeutic targets associated with "
                            "Alzheimer's disease and identify potential drug "
                            "candidates that could modulate these targets."
                        ),
                        icon="/public/adn.png",
                    ),
                    cl.Starter(
                        label="Analyze aspirin compound",
                        message=(
                            "Search for aspirin drug information including "
                            "its molecular structure, mechanism of action, "
                            "and known targets."
                        ),
                        icon="/public/drug.png",
                    ),
                    cl.Starter(
                        label="Cancer drug discovery",
                        message=(
                            "Identify potential drug compounds for treating "
                            "breast cancer, including efficacy data and "
                            "clinical trial status."
                        ),
                        icon="/public/cancer.png",
                    ),
                    cl.Starter(
                        label="Compare anti-inflammatory drugs",
                        message=(
                            "Compare the mechanisms and efficacy of "
                            "ibuprofen and naproxen as anti-inflammatory drugs."
                        ),
                        icon="/public/disease.png",
                    ),
                ],
            )
        ]

    except Exception as exc:
        print(f"❌ Error configuring chat profiles: {exc}")

        return [
            cl.ChatProfile(
                name="Drug Discovery Researcher",
                markdown_description="Pharmaceutical research assistant",
                icon="/public/logo.png",
                starters=[],
            )
        ]


# ---------------------------------------------------------------------------
# Resume existing Chainlit thread
# ---------------------------------------------------------------------------

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Restore a previously saved agent-team state."""
    try:
        user = cl.user_session.get("user")

        if not user:
            print("⚠️ No user found during chat resume.")
            return

        username = user.identifier
        thread_id = thread.get("id")

        if not thread_id:
            print("⚠️ No thread ID available during chat resume.")
            return

        team, termination_ext = await initialize_agents()

        cl.user_session.set("team", team)
        cl.user_session.set("termination_ext", termination_ext)
        cl.user_session.set("username", username)
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("is_processing", False)
        cl.user_session.set("message_count", 0)
        cl.user_session.set("has_sent_message", True)

        state_loaded = await load_team_state_from_disk(
            team,
            username,
            thread_id,
        )

        if state_loaded:
            print(
                f"✅ Resumed existing thread "
                f"'{thread_id}' for user '{username}'."
            )
        else:
            print(
                f"ℹ️ No saved state for thread '{thread_id}'. "
                "Starting fresh."
            )

    except Exception as exc:
        print(f"❌ Error resuming chat session: {exc}")

        try:
            await cl.Message(
                content=(
                    "⚠️ **Session Resume Error**\n\n"
                    f"{exc}\n\n"
                    "Starting a fresh session."
                ),
                author="System",
            ).send()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# New Chainlit session
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def start_chat() -> None:
    """Initialize a new agent team for the current Chainlit session."""
    try:
        user = cl.user_session.get("user")

        if user:
            username = user.identifier
            role = user.metadata.get("role", "guest")
        else:
            username = "unknown"
            role = "guest"

    except Exception as exc:
        print(f"⚠️ Error getting user info: {exc}")
        username = "unknown"
        role = "guest"

    thread_id = cl.context.session.thread_id

    try:
        team, termination_ext = await initialize_agents()

        cl.user_session.set("team", team)
        cl.user_session.set("termination_ext", termination_ext)
        cl.user_session.set("is_processing", False)
        cl.user_session.set("username", username)
        cl.user_session.set("role", role)
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("message_count", 0)
        cl.user_session.set("has_sent_message", False)

        print(
            f"🔵 New session initialized for '{username}' "
            f"on thread '{thread_id}'."
        )
        print("⏳ Waiting for first message before saving state...")

    except Exception as exc:
        print(f"❌ Critical error in start_chat: {exc}")

        try:
            await cl.Message(
                content=(
                    "❌ **System initialization failed:**\n\n"
                    f"{exc}\n\n"
                    "Please refresh the page and try again."
                ),
                author="System",
            ).send()
        except Exception:
            pass

        raise


# ---------------------------------------------------------------------------
# Clear session state
# ---------------------------------------------------------------------------

@cl.action_callback("clear_session_state")
async def on_clear_session_state(action: cl.Action):
    """Delete persisted state and reinitialize the agent team."""
    try:
        username = cl.user_session.get("username")
        thread_id = cl.user_session.get("thread_id")

        if not username or not thread_id:
            raise RuntimeError("Missing username or thread ID.")

        success = remove_team_state_from_disk(
            username,
            thread_id,
        )

        if success:
            team, termination_ext = await initialize_agents()

            cl.user_session.set("team", team)
            cl.user_session.set(
                "termination_ext",
                termination_ext,
            )
            cl.user_session.set(
                "message_count",
                0,
            )
            cl.user_session.set(
                "has_sent_message",
                False,
            )

            await cl.Message(
                content=(
                    "✅ **Session History Cleared**\n\n"
                    "Your conversation history for this session "
                    "has been deleted. Starting fresh!"
                ),
                author="System",
            ).send()

        else:
            await cl.Message(
                content=(
                    "⚠️ **Clear Failed**\n\n"
                    "Could not clear session history."
                ),
                author="System",
            ).send()

    except Exception as exc:
        print(f"❌ Error clearing session state: {exc}")

        await cl.Message(
            content=f"❌ **Error:** {exc}",
            author="System",
        ).send()


# ---------------------------------------------------------------------------
# Main message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def handle_message(message: cl.Message) -> None:
    """
    Process one user request through the SelectorGroupChat.

    IMPORTANT:
        asyncio.CancelledError is intentionally handled separately.
        Do NOT use CancellationToken.CancelledError because that attribute
        does not exist in AutoGen's CancellationToken class.
    """

    # Prevent overlapping workflows.
    if cl.user_session.get("is_processing", False):
        await cl.Message(
            content=(
                "⚠️ **System is currently processing another request.**\n\n"
                "Please wait or click Stop to cancel the current workflow."
            ),
            author="System",
        ).send()
        return

    cl.user_session.set("is_processing", True)

    message_count = cl.user_session.get("message_count", 0)
    cl.user_session.set(
        "message_count",
        message_count + 1,
    )

    if not cl.user_session.get("has_sent_message", False):
        cl.user_session.set("has_sent_message", True)

        username = cl.user_session.get("username", "Guest")
        thread_id = cl.user_session.get("thread_id", "unknown")

        print(
            f"✅ First message received from '{username}' "
            f"on thread '{thread_id}'."
        )

    try:
        team = cast(
            SelectorGroupChat,
            cl.user_session.get("team"),
        )

        termination_ext = cl.user_session.get(
            "termination_ext"
        )

        username = cl.user_session.get(
            "username",
            "Guest",
        )

        thread_id = cl.user_session.get(
            "thread_id",
            "unknown",
        )

        if team is None:
            raise RuntimeError(
                "Agent team is not initialized."
            )

        # Reset external termination before a new run.
        if termination_ext is not None:
            try:
                termination_ext.reset()
            except Exception as exc:
                print(
                    f"⚠️ Could not reset ExternalTermination: {exc}"
                )

        # The response is streamed into Chainlit.
        response_msg = cl.Message(
            content="",
            author="AgentTeam",
        )

        await response_msg.send()

        # ---------------------------------------------------------------
        # AutoGen streaming
        # ---------------------------------------------------------------

        async for event in team.run_stream(
            task=message.content,
        ):
            if isinstance(
                event,
                ModelClientStreamingChunkEvent,
            ):
                if event.content:
                    await response_msg.stream_token(
                        str(event.content)
                    )

            elif isinstance(event, ThoughtEvent):
                if event.content:
                    print(
                        f"💭 {event.source}: {event.content}"
                    )

            elif isinstance(event, ToolCallRequestEvent):
                print(
                    f"🔧 Tool call request: {event}"
                )

            elif isinstance(event, ToolCallSummaryMessage):
                print(
                    f"🔧 Tool call summary: {event}"
                )

            elif isinstance(event, TextMessage):
                # TextMessage can be emitted after streaming. We do not
                # append it again here to avoid duplicated final output.
                print(
                    f"📝 TextMessage from {event.source}"
                )

            elif isinstance(event, TaskResult):
                print(
                    f"✅ Task completed: "
                    f"{getattr(event, 'stop_reason', None)}"
                )

        await response_msg.update()

        # ---------------------------------------------------------------
        # Persist state after successful completion.
        # ---------------------------------------------------------------

        if username and thread_id:
            state_path = await save_team_state_to_disk(
                team=team,
                username=username,
                thread_id=thread_id,
            )

            if state_path:
                print(
                    f"💾 Auto-saved state for '{username}' "
                    f"on thread '{thread_id}'."
                )

    # -------------------------------------------------------------------
    # Correct cancellation handling
    # -------------------------------------------------------------------

    except asyncio.CancelledError:
        print(
            "🛑 Workflow cancelled by Chainlit/AutoGen."
        )

        # Do not re-raise here. The request was intentionally cancelled.
        return

    # -------------------------------------------------------------------
    # All other errors
    # -------------------------------------------------------------------

    except Exception as exc:
        print(
            f"❌ Error processing user message: "
            f"{type(exc).__name__}: {exc}"
        )

        import traceback
        traceback.print_exc()

        try:
            await cl.Message(
                content=(
                    "❌ **Workflow Error**\n\n"
                    f"`{type(exc).__name__}: {exc}`\n\n"
                    "The agent workflow could not be completed."
                ),
                author="System",
            ).send()
        except Exception as ui_exc:
            print(
                f"❌ Could not send error message: {ui_exc}"
            )

    # -------------------------------------------------------------------
    # ALWAYS release processing lock
    # -------------------------------------------------------------------

    finally:
        cl.user_session.set(
            "is_processing",
            False,
        )

        print(
            "🔓 Processing lock released."
        )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        "🚀 Agentic Pharma Chainlit application."
    )
    print(
        "Run with: chainlit run orcastration/main_chainlit.py"
    )
