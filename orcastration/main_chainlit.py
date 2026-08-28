from pathlib import Path

content = r'''#!/usr/bin/env python3
"""
Agentic Pharma System - Chainlit Interface

Multi-agent pharmaceutical research interface using AutoGen AgentChat,
DeepSeek, Chainlit, and Arize/OpenInference instrumentation.

Workflow:
    User
      -> Planning
      -> Target Search
      -> Drug Search
      -> Critique
      -> Report
      -> PDF generation
      -> ReportAgent emits TERMINATE
      -> TaskResult
      -> Chainlit displays the generated PDF
      -> State is persisted

Important:
    - TERMINATE is the normal workflow completion signal.
    - SourceMatchTermination is intentionally NOT used.
    - ReportAgent should emit TERMINATE only after the final report/PDF
      has been successfully generated.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / project root
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
from autogen_core.model_context import UnboundedChatCompletionContext

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    ExternalTermination,
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

# Centralized observability instrumentation.
from orcastration.instrumentation import tracer_provider  # noqa: F401

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from agents.critique import setup_critique_agent
from agents.drug_search import setup_drug_search_agent
from agents.planning import setup_planning_agent
from agents.report import report_agent
from agents.target_search import target_search_agent
from config.llm_client import model_client
from config.sytem_prompts import SELECT_PROMPT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_DIR = PROJECT_ROOT / "session_state"
REPORT_DIRS = (
    PROJECT_ROOT / "generated_reports",
    PROJECT_ROOT / "resumes_uploaded",
)

STATE_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "generated_reports").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


async def save_team_state_to_disk(
    team: SelectorGroupChat,
    username: str,
    thread_id: str,
) -> Optional[str]:
    """Persist SelectorGroupChat state to a session-specific JSON file."""
    try:
        filepath = STATE_DIR / f"team_state_{username}_{thread_id}.json"

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

        print(f"✅ Team state saved: {filepath}")
        return str(filepath.resolve())

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
        filepath = STATE_DIR / f"team_state_{username}_{thread_id}.json"

        if not filepath.exists():
            print(f"ℹ️ No saved state: {filepath}")
            return False

        data = await asyncio.to_thread(
            filepath.read_text,
            encoding="utf-8",
        )

        await team.load_state(json.loads(data))

        print(f"✅ Team state loaded: {filepath}")
        return True

    except (IOError, OSError) as exc:
        print(f"❌ File I/O error loading state: {exc}")
        return False

    except (json.JSONDecodeError, ValueError) as exc:
        print(f"❌ Invalid state JSON: {exc}")
        return False

    except Exception as exc:
        print(f"❌ Unexpected error loading state: {exc}")
        return False


def remove_team_state_from_disk(
    username: str,
    thread_id: str,
) -> bool:
    """Delete persisted state for the current thread."""
    try:
        filepath = STATE_DIR / f"team_state_{username}_{thread_id}.json"

        if not filepath.exists():
            print(f"ℹ️ State already absent: {filepath}")
            return True

        filepath.unlink()
        print(f"✅ Team state removed: {filepath}")
        return True

    except (IOError, OSError, PermissionError) as exc:
        print(f"❌ Error removing team state: {exc}")
        return False

    except Exception as exc:
        print(f"❌ Unexpected error removing team state: {exc}")
        return False


# ---------------------------------------------------------------------------
# PDF detection
# ---------------------------------------------------------------------------


def snapshot_pdf_state() -> dict[Path, int]:
    """Return modification times of PDFs that already existed before a task."""
    state: dict[Path, int] = {}

    for directory in REPORT_DIRS:
        if not directory.exists():
            continue

        for pdf_path in directory.glob("*.pdf"):
            try:
                state[pdf_path] = pdf_path.stat().st_mtime_ns
            except OSError:
                continue

    return state


def find_new_or_updated_pdfs(
    known_state: dict[Path, int],
    task_start_ns: int,
) -> list[Path]:
    """Find PDFs created or modified during the current task."""
    candidates: list[Path] = []

    for directory in REPORT_DIRS:
        if not directory.exists():
            continue

        for pdf_path in directory.glob("*.pdf"):
            try:
                mtime_ns = pdf_path.stat().st_mtime_ns
            except OSError:
                continue

            previous_mtime = known_state.get(pdf_path)

            if (
                previous_mtime is None
                or mtime_ns > previous_mtime
                or mtime_ns >= task_start_ns
            ):
                candidates.append(pdf_path)

    return candidates


def get_latest_pdf(
    known_state: dict[Path, int],
    task_start_ns: int,
) -> Optional[Path]:
    """Return the most recently created/updated PDF for this task."""
    candidates = find_new_or_updated_pdfs(
        known_state=known_state,
        task_start_ns=task_start_ns,
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
    )


async def send_pdf_to_chainlit(
    pdf_path: Path,
) -> None:
    """Display the generated PDF and a downloadable file in Chainlit."""
    try:
        pdf_bytes = await asyncio.to_thread(pdf_path.read_bytes)

        elements = [
            cl.Pdf(
                name=pdf_path.name,
                content=pdf_bytes,
                display="inline",
                mime="application/pdf",
            ),
            cl.File(
                name=pdf_path.name,
                content=pdf_bytes,
                display="inline",
                mime="application/pdf",
            ),
        ]

        await cl.Message(
            content=(
                "📄 **Report Generated Successfully**\n\n"
                f"Your final pharmaceutical research report is ready: "
                f"`{pdf_path.name}`"
            ),
            elements=elements,
            author="System",
        ).send()

        print(f"📄 PDF displayed in Chainlit: {pdf_path}")

    except Exception as exc:
        print(f"❌ Could not display PDF: {exc}")

        await cl.Message(
            content=(
                "⚠️ The report was generated, but Chainlit could not "
                f"display the PDF automatically.\n\n`{pdf_path}`"
            ),
            author="System",
        ).send()


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------


async def user_input_func(
    prompt: str,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Capture human input through Chainlit."""
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
        print("🛑 Human input request cancelled.")
        raise

    except TimeoutError:
        print("⚠️ Human input timed out.")
        return "User did not provide any input within the time limit."

    except Exception as exc:
        print(f"❌ Error getting human input: {exc}")
        return "An error occurred while requesting user input."


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------


async def initialize_agents():
    """
    Create the pharmaceutical research team.

    Normal completion:
        ReportAgent generates the final report/PDF and emits TERMINATE.

    External cancellation:
        ExternalTermination can be triggered by the application.
    """
    try:
        # IMPORTANT:
        # Do not add SourceMatchTermination("ReportAgent").
        # A ReportAgent message alone must not terminate the workflow.
        termination_word = TextMentionTermination("TERMINATE")
        termination_ext = ExternalTermination()

        termination = termination_word | termination_ext

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
        raise


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate Chainlit user."""
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
# Resume existing thread
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
            print("⚠️ No thread ID available during resume.")
            return

        team, termination_ext = await initialize_agents()

        cl.user_session.set("team", team)
        cl.user_session.set("termination_ext", termination_ext)
        cl.user_session.set("username", username)
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("is_processing", False)
        cl.user_session.set("message_count", 0)
        cl.user_session.set("has_sent_message", True)
        cl.user_session.set("cancellation_token", None)

        state_loaded = await load_team_state_from_disk(
            team,
            username,
            thread_id,
        )

        if state_loaded:
            print(
                f"✅ Resumed thread '{thread_id}' "
                f"for user '{username}'."
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
# New session
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

        thread_id = cl.context.session.thread_id

        team, termination_ext = await initialize_agents()

        cl.user_session.set("team", team)
        cl.user_session.set("termination_ext", termination_ext)
        cl.user_session.set("is_processing", False)
        cl.user_session.set("username", username)
        cl.user_session.set("role", role)
        cl.user_session.set("thread_id", thread_id)
        cl.user_session.set("message_count", 0)
        cl.user_session.set("has_sent_message", False)
        cl.user_session.set("cancellation_token", None)

        print(
            f"🔵 New session initialized for '{username}' "
            f"on thread '{thread_id}'."
        )

    except Exception as exc:
        print(f"❌ Critical error in start_chat: {exc}")

        try:
            await cl.Message(
                content=(
                    "❌ **System initialization failed**\n\n"
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
    """Delete persisted state and initialize a fresh team."""
    try:
        username = cl.user_session.get("username")
        thread_id = cl.user_session.get("thread_id")

        if not username or not thread_id:
            raise RuntimeError("Missing username or thread ID.")

        success = remove_team_state_from_disk(
            username,
            thread_id,
        )

        if not success:
            await cl.Message(
                content=(
                    "⚠️ **Clear Failed**\n\n"
                    "Could not clear session history."
                ),
                author="System",
            ).send()
            return

        team, termination_ext = await initialize_agents()

        cl.user_session.set("team", team)
        cl.user_session.set("termination_ext", termination_ext)
        cl.user_session.set("message_count", 0)
        cl.user_session.set("has_sent_message", False)
        cl.user_session.set("cancellation_token", None)

        await cl.Message(
            content=(
                "✅ **Session History Cleared**\n\n"
                "The persisted agent state has been deleted. "
                "A fresh agent team is now active."
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
    Run one pharmaceutical research task.

    Chainlit displays:
        - agent thinking events
        - streaming model output
        - tool calls
        - tool result notifications
        - final TaskResult
        - generated PDF

    The team terminates normally when ReportAgent emits TERMINATE.
    """
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
    cl.user_session.set("message_count", message_count + 1)
    cl.user_session.set("has_sent_message", True)

    team = cast(
        Optional[SelectorGroupChat],
        cl.user_session.get("team"),
    )

    if team is None:
        cl.user_session.set("is_processing", False)

        await cl.Message(
            content=(
                "❌ **Agent team is not initialized.**\n\n"
                "Please refresh the page and try again."
            ),
            author="System",
        ).send()
        return

    termination_ext = cl.user_session.get("termination_ext")
    username = cl.user_session.get("username", "Guest")
    thread_id = cl.user_session.get("thread_id", "unknown")

    cancellation_token = CancellationToken()
    cl.user_session.set("cancellation_token", cancellation_token)

    task_start = datetime.now()
    task_start_ns = task_start.timestamp() * 1_000_000_000
    known_pdf_state = snapshot_pdf_state()

    current_streaming_msg: Optional[cl.Message] = None
    agent_message_count: dict[str, int] = {}
    tool_call_count = 0
    total_streamed_chars = 0

    try:
        # Reset only the external cancellation condition.
        if termination_ext is not None:
            try:
                termination_ext.reset()
            except Exception as exc:
                print(f"⚠️ Could not reset ExternalTermination: {exc}")

        await cl.Message(
            content="🚀 **Starting Multi-Agent Analysis...**",
            author="System",
        ).send()

        # -------------------------------------------------------------------
        # AutoGen stream
        # -------------------------------------------------------------------

        async for event in team.run_stream(
            task=TextMessage(
                content=message.content,
                source="ExpertHuman",
            ),
            cancellation_token=cancellation_token,
        ):
            if cancellation_token.is_cancelled():
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                break

            agent_name = getattr(event, "source", None)
            agent_name = str(agent_name) if agent_name else "Agent"

            event_type = type(event).__name__

            agent_message_count[agent_name] = (
                agent_message_count.get(agent_name, 0) + 1
            )

            # ---------------------------------------------------------------
            # Agent thought
            # ---------------------------------------------------------------

            if isinstance(event, ThoughtEvent):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                spinner = cl.Message(
                    content="⏳ *thinking...*",
                    author=agent_name,
                )
                await spinner.send()
                await spinner.remove()

                print(f"💭 {agent_name}: {event.content}")

            # ---------------------------------------------------------------
            # Streaming LLM output
            # ---------------------------------------------------------------

            elif isinstance(
                event,
                ModelClientStreamingChunkEvent,
            ):
                if not event.content:
                    continue

                if (
                    current_streaming_msg is None
                    or getattr(
                        current_streaming_msg,
                        "author",
                        None,
                    )
                    != agent_name
                ):
                    if current_streaming_msg is not None:
                        await current_streaming_msg.send()

                    current_streaming_msg = cl.Message(
                        content="",
                        author=agent_name,
                    )

                await current_streaming_msg.stream_token(
                    str(event.content)
                )

                total_streamed_chars += len(str(event.content))

            # ---------------------------------------------------------------
            # Tool call
            # ---------------------------------------------------------------

            elif isinstance(event, ToolCallRequestEvent):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                for tool_call in event.content:
                    tool_call_count += 1

                    args_preview = str(tool_call.arguments)

                    if len(args_preview) > 1000:
                        args_preview = (
                            args_preview[:1000]
                            + "... (truncated)"
                        )

                    await cl.Message(
                        content=(
                            f"`{agent_name}` 🛠️ **Calling tool** "
                            f"`{tool_call.name}`\n\n"
                            "```json\n"
                            f"{args_preview}\n"
                            "```"
                        ),
                        author=agent_name,
                    ).send()

                    print(
                        f"🔧 {agent_name} -> "
                        f"{tool_call.name}({args_preview})"
                    )

            # ---------------------------------------------------------------
            # Tool result
            # ---------------------------------------------------------------

            elif isinstance(
                event,
                ToolCallSummaryMessage,
            ):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                await cl.Message(
                    content=(
                        f"`{agent_name}` 🔄 **Tool result received**"
                    ),
                    author=agent_name,
                ).send()

                print(
                    f"🔧 Tool result from {agent_name}: "
                    f"{event.content}"
                )

            # ---------------------------------------------------------------
            # Normal text message
            # ---------------------------------------------------------------

            elif isinstance(event, TextMessage):
                # Streaming chunks have already been rendered.
                # Do not duplicate the same final text in Chainlit.
                print(
                    f"📝 {agent_name}: "
                    f"{str(event.content)[:500]}"
                )

                # If this is a non-streamed message with content and there
                # is no active stream, show it in the UI.
                if (
                    event.content
                    and current_streaming_msg is None
                    and agent_name != "ExpertHuman"
                ):
                    content = str(event.content)

                    # Avoid displaying a duplicate TERMINATE-only message.
                    if content.strip() != "TERMINATE":
                        await cl.Message(
                            content=content,
                            author=agent_name,
                        ).send()

            # ---------------------------------------------------------------
            # Task finished
            # ---------------------------------------------------------------

            elif isinstance(event, TaskResult):
                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                stop_reason = getattr(
                    event,
                    "stop_reason",
                    None,
                )

                duration = (
                    datetime.now() - task_start
                ).total_seconds()

                print(
                    "✅ TaskResult received | "
                    f"stop_reason={stop_reason} | "
                    f"duration={duration:.2f}s | "
                    f"tool_calls={tool_call_count}"
                )

                if cancellation_token.is_cancelled():
                    await cl.Message(
                        content=(
                            "🛑 **Task cancelled by user.**\n\n"
                            "You can now start a new query."
                        ),
                        author="System",
                    ).send()
                    continue

                # -----------------------------------------------------------
                # Normal completion
                # -----------------------------------------------------------

                final_message = "✅ **Task completed successfully**"

                if stop_reason:
                    final_message += f" ({stop_reason})"

                await cl.Message(
                    content=final_message,
                    author="System",
                ).send()

                # -----------------------------------------------------------
                # Find and display generated PDF
                # -----------------------------------------------------------

                latest_pdf = get_latest_pdf(
                    known_state=known_pdf_state,
                    task_start_ns=int(task_start_ns),
                )

                if latest_pdf is not None:
                    await send_pdf_to_chainlit(latest_pdf)
                else:
                    await cl.Message(
                        content=(
                            "ℹ️ The workflow completed, but no new PDF "
                            "was detected in `generated_reports/`."
                        ),
                        author="System",
                    ).send()

            # ---------------------------------------------------------------
            # Unknown event type
            # ---------------------------------------------------------------

            else:
                print(
                    f"ℹ️ Unhandled AutoGen event: {event_type}"
                )

        # Finalize any remaining streamed message.
        if current_streaming_msg is not None:
            await current_streaming_msg.send()
            current_streaming_msg = None

        print(
            f"📊 Task metrics: agents={agent_message_count}, "
            f"tool_calls={tool_call_count}, "
            f"streamed_chars={total_streamed_chars}"
        )

    except asyncio.CancelledError:
        print("🛑 Workflow cancelled by Chainlit/AutoGen.")

        if current_streaming_msg is not None:
            try:
                await current_streaming_msg.send()
            except Exception:
                pass

        await cl.Message(
            content=(
                "🛑 **Task cancelled.**\n\n"
                "You can start a new query."
            ),
            author="System",
        ).send()

    except Exception as exc:
        print(
            f"❌ Error processing workflow: "
            f"{type(exc).__name__}: {exc}"
        )

        import traceback

        traceback.print_exc()

        await cl.Message(
            content=(
                "❌ **Workflow Error**\n\n"
                f"`{type(exc).__name__}: {exc}`"
            ),
            author="System",
        ).send()

    finally:
        # ---------------------------------------------------------------
        # Persist state after the run.
        # ---------------------------------------------------------------

        try:
            if team is not None and username and thread_id:
                state_path = await save_team_state_to_disk(
                    team=team,
                    username=username,
                    thread_id=thread_id,
                )

                if state_path:
                    print(
                        f"💾 State saved for '{username}' "
                        f"on thread '{thread_id}'."
                    )

        except Exception as exc:
            print(f"⚠️ Error auto-saving team state: {exc}")

        cl.user_session.set("is_processing", False)
        cl.user_session.set("cancellation_token", None)

        print("🔓 Processing lock released.")


# ---------------------------------------------------------------------------
# Stop current task
# ---------------------------------------------------------------------------


@cl.on_stop
async def on_stop():
    """Cancel the current workflow through CancellationToken."""
    try:
        token = cl.user_session.get("cancellation_token")

        if token is not None:
            token.cancel()
            print("🛑 CancellationToken cancelled.")

        cl.user_session.set("is_processing", False)

        await cl.Message(
            content=(
                "🛑 **Stop requested.**\n\n"
                "The current workflow is being cancelled. "
                "You can start a new query."
            ),
            author="System",
        ).send()

    except Exception as exc:
        print(f"⚠️ Error in on_stop: {exc}")


# ---------------------------------------------------------------------------
# Chat end
# ---------------------------------------------------------------------------


@cl.on_chat_end
async def on_chat_end():
    """Save state when the Chainlit chat session ends."""
    try:
        token = cl.user_session.get("cancellation_token")

        if token is not None:
            token.cancel()

        team = cl.user_session.get("team")
        username = cl.user_session.get("username")
        thread_id = cl.user_session.get("thread_id")
        has_sent_message = cl.user_session.get(
            "has_sent_message",
            False,
        )

        if (
            has_sent_message
            and team is not None
            and username
            and thread_id
        ):
            await save_team_state_to_disk(
                team,
                username,
                thread_id,
            )

            print(
                f"💾 Final state saved for '{username}' "
                f"on thread '{thread_id}'."
            )
        else:
            print(
                "⏭️ Chat ended without a saved workflow state."
            )

    except Exception as exc:
        print(f"⚠️ Error saving state on chat end: {exc}")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@cl.on_settings_update
async def setup_agent_settings(settings):
    """Handle Chainlit settings updates."""
    try:
        await cl.Message(
            content=(
                "⚙️ **Settings Updated**\n\n"
                "Your preferences have been received."
            ),
            author="System",
        ).send()

    except Exception as exc:
        print(f"❌ Error updating settings: {exc}")

        await cl.Message(
            content=(
                "⚠️ **Settings update failed**\n\n"
                "Please try again."
            ),
            author="System",
        ).send()


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Agentic Pharma Chainlit application.")
    print(
        "Run with: "
        "chainlit run orcastration/main_chainlit.py"
    )
'''

path = Path("/mnt/data/main_chainlit.py")
path.write_text(content, encoding="utf-8")
print(f"Created: {path}")
print(f"Lines: {len(content.splitlines())}")
