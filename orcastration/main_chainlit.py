"""
Agentic Pharma System - Chainlit Interface

This module provides a conversational AI interface for pharmaceutical research,
enabling multi-agent collaboration for drug discovery workflows.

Key behavior:
    - Multi-agent orchestration with AutoGen SelectorGroupChat
    - Streaming agent output
    - Visible tool-call events in Chainlit
    - Human-in-the-loop support
    - Persistent team state
    - PDF detection and download after TaskResult
    - TERMINATE is the normal workflow completion signal
    - ExternalTermination is reserved for manual cancellation
    - Correct asyncio cancellation handling
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv

# ============================================================================
# Environment / project root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Third-party imports
# ============================================================================

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

# Centralized Arize/OpenInference instrumentation.
from orcastration.instrumentation import tracer_provider  # noqa: F401

# ============================================================================
# Project imports
# ============================================================================

from agents.target_search import target_search_agent
from agents.drug_search import setup_drug_search_agent
from agents.report import report_agent
from agents.critique import setup_critique_agent
from agents.planning import setup_planning_agent
from config.llm_client import model_client
from config.sytem_prompts import SELECT_PROMPT


# ============================================================================
# Configuration
# ============================================================================

STATE_DIR = PROJECT_ROOT / "session_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

PDF_DIRS = (
    PROJECT_ROOT / "generated_reports",
    PROJECT_ROOT / "resumes_uploaded",
)

(PROJECT_ROOT / "generated_reports").mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# TEAM STATE: SAVE
# ============================================================================

async def save_team_state_to_disk(
    team: SelectorGroupChat,
    username: str,
    thread_id: str,
) -> Optional[str]:
    """
    Persist the current SelectorGroupChat state to disk.
    """

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
        return str(filepath.resolve())

    except (IOError, OSError) as exc:
        print(f"❌ File I/O error saving team state: {exc}")
        return None

    except Exception as exc:
        print(f"❌ Unexpected error saving team state: {exc}")
        return None


# ============================================================================
# TEAM STATE: LOAD
# ============================================================================

async def load_team_state_from_disk(
    team: SelectorGroupChat,
    username: str,
    thread_id: str,
) -> bool:
    """
    Restore a previously saved SelectorGroupChat state.
    """

    try:
        filename = f"team_state_{username}_{thread_id}.json"
        filepath = STATE_DIR / filename

        if not filepath.exists():
            print(f"ℹ️ State file does not exist: {filepath}")
            return False

        data = await asyncio.to_thread(
            filepath.read_text,
            encoding="utf-8",
        )

        await team.load_state(
            json.loads(data)
        )

        print(f"✅ Team state loaded from: {filepath}")
        return True

    except (IOError, OSError) as exc:
        print(f"❌ File I/O error loading state: {exc}")
        return False

    except (json.JSONDecodeError, ValueError) as exc:
        print(f"❌ Invalid JSON in state file: {exc}")
        return False

    except Exception as exc:
        print(f"❌ Unexpected error loading state: {exc}")
        return False


# ============================================================================
# TEAM STATE: DELETE
# ============================================================================

def remove_team_state_from_disk(
    username: str,
    thread_id: str,
) -> bool:
    """
    Delete the persisted state for a session.
    """

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


# ============================================================================
# PDF TRACKING
# ============================================================================

def snapshot_pdf_state() -> dict[Path, int]:
    """
    Capture modification times of PDFs that existed before the task started.
    """

    state: dict[Path, int] = {}

    for directory in PDF_DIRS:
        if not directory.exists():
            continue

        for pdf_path in directory.glob("*.pdf"):
            try:
                state[pdf_path] = pdf_path.stat().st_mtime_ns
            except OSError:
                continue

    return state


def find_task_pdfs(
    before_state: dict[Path, int],
    task_start_ns: int,
) -> list[Path]:
    """
    Return PDFs created or modified during the current task.
    """

    candidates: list[Path] = []

    for directory in PDF_DIRS:
        if not directory.exists():
            continue

        for pdf_path in directory.glob("*.pdf"):
            try:
                mtime_ns = pdf_path.stat().st_mtime_ns
            except OSError:
                continue

            old_mtime = before_state.get(pdf_path)

            if (
                old_mtime is None
                or mtime_ns > old_mtime
                or mtime_ns >= task_start_ns
            ):
                candidates.append(pdf_path)

    return candidates


def find_latest_task_pdf(
    before_state: dict[Path, int],
    task_start_ns: int,
) -> Optional[Path]:
    """
    Find the newest PDF generated/updated during this task.
    """

    candidates = find_task_pdfs(
        before_state,
        task_start_ns,
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime_ns,
    )


async def show_pdf(
    pdf_path: Path,
) -> bool:
    """
    Display a PDF in Chainlit and provide a download entry.
    """

    try:
        if not pdf_path.exists():
            print(f"⚠️ PDF not found: {pdf_path}")
            return False

        if pdf_path.suffix.lower() != ".pdf":
            print(f"⚠️ Not a PDF file: {pdf_path}")
            return False

        pdf_bytes = await asyncio.to_thread(
            pdf_path.read_bytes
        )

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
                f"Final PDF: `{pdf_path.name}`"
            ),
            elements=elements,
            author="System",
        ).send()

        print(
            f"📄 PDF displayed in Chainlit: {pdf_path}"
        )

        return True

    except Exception as exc:
        print(f"❌ Error displaying PDF: {exc}")

        await cl.Message(
            content=(
                "⚠️ The report was completed, but the PDF could not "
                f"be displayed automatically.\n\n`{pdf_path}`"
            ),
            author="System",
        ).send()

        return False


# ============================================================================
# HUMAN INPUT
# ============================================================================

async def user_input_func(
    prompt: str,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """
    Capture human input through Chainlit.

    CancellationToken is accepted because AutoGen supplies it to the callback.
    asyncio.CancelledError is the actual Python task-cancellation exception.
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
        print(
            "⚠️ User input request timed out after 300 seconds."
        )
        return (
            "User did not provide any input within the time limit."
        )

    except Exception as exc:
        print(
            f"❌ Error getting user input: {exc}"
        )
        return (
            "An error occurred while requesting user input."
        )


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

async def initialize_agents():
    """
    Initialize the complete agent team.

    Normal termination:
        ReportAgent emits TERMINATE after successful PDF generation.

    Manual stop:
        ExternalTermination is available for explicit cancellation.
    """

    try:
        # --------------------------------------------------------------------
        # TERMINATION
        #
        # IMPORTANT:
        # SourceMatchTermination("ReportAgent") is intentionally removed.
        # A ReportAgent message alone must NOT stop the task.
        #
        # The normal completion signal is TERMINATE.
        # --------------------------------------------------------------------

        termination_word = TextMentionTermination(
            "TERMINATE"
        )

        termination_ext = ExternalTermination()

        termination = (
            termination_word
            | termination_ext
        )

        # --------------------------------------------------------------------
        # Context
        # --------------------------------------------------------------------

        model_context = UnboundedChatCompletionContext()

        # --------------------------------------------------------------------
        # Agents
        # --------------------------------------------------------------------

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

        # --------------------------------------------------------------------
        # SelectorGroupChat
        # --------------------------------------------------------------------

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

        print(
            "✅ Agent team initialized successfully."
        )

        return team, termination_ext

    except Exception as exc:
        print(
            f"❌ Error initializing agents: {exc}"
        )
        raise


# ============================================================================
# AUTHENTICATION
# ============================================================================

@cl.password_auth_callback
def auth_callback(
    username: str,
    password: str,
):
    """
    Authenticate the Chainlit user.

    Replace hard-coded credentials with database authentication
    in production.
    """

    if (
        username,
        password,
    ) == (
        "researcher",
        "easydiscovery##1",
    ):
        return cl.User(
            identifier="admin",
            metadata={
                "role": "admin",
                "provider": "credentials",
            },
        )

    return None


# ============================================================================
# CHAT PROFILE
# ============================================================================

@cl.set_chat_profiles
async def chat_profile(
    current_user: cl.User,
):
    """
    Configure Chainlit chat profile and starters.
    """

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
                        label=(
                            "Find drug targets for Alzheimer's disease"
                        ),
                        message=(
                            "Search for therapeutic targets associated "
                            "with Alzheimer's disease and identify "
                            "potential drug candidates that could "
                            "modulate these targets."
                        ),
                        icon="/public/adn.png",
                    ),
                    cl.Starter(
                        label="Analyze aspirin compound",
                        message=(
                            "Search for aspirin drug information "
                            "including its molecular structure, "
                            "mechanism of action, and known targets."
                        ),
                        icon="/public/drug.png",
                    ),
                    cl.Starter(
                        label="Cancer drug discovery",
                        message=(
                            "Identify potential drug compounds for "
                            "treating breast cancer, including efficacy "
                            "data and clinical trial status."
                        ),
                        icon="/public/cancer.png",
                    ),
                    cl.Starter(
                        label="Compare anti-inflammatory drugs",
                        message=(
                            "Compare the mechanisms and efficacy of "
                            "ibuprofen and naproxen as anti-inflammatory "
                            "drugs."
                        ),
                        icon="/public/disease.png",
                    ),
                ],
            )
        ]

    except Exception as exc:
        print(
            f"❌ Error configuring chat profiles: {exc}"
        )

        return [
            cl.ChatProfile(
                name="Drug Discovery Researcher",
                markdown_description=(
                    "Pharmaceutical research assistant"
                ),
                icon="/public/logo.png",
                starters=[],
            )
        ]


# ============================================================================
# CHAT RESUME
# ============================================================================

@cl.on_chat_resume
async def on_chat_resume(
    thread: ThreadDict,
):
    """
    Restore a previous conversation and team state.
    """

    try:
        user = cl.user_session.get("user")

        if not user:
            print(
                "⚠️ No user found during chat resume."
            )
            return

        username = user.identifier

        thread_id = thread.get("id")

        if not thread_id:
            print(
                "⚠️ No thread ID available during chat resume."
            )
            return

        team, termination_ext = (
            await initialize_agents()
        )

        cl.user_session.set(
            "team",
            team,
        )

        cl.user_session.set(
            "termination_ext",
            termination_ext,
        )

        cl.user_session.set(
            "username",
            username,
        )

        cl.user_session.set(
            "thread_id",
            thread_id,
        )

        cl.user_session.set(
            "is_processing",
            False,
        )

        cl.user_session.set(
            "message_count",
            0,
        )

        cl.user_session.set(
            "has_sent_message",
            True,
        )

        cl.user_session.set(
            "cancellation_token",
            None,
        )

        loaded = await load_team_state_from_disk(
            team,
            username,
            thread_id,
        )

        if loaded:
            print(
                f"✅ Resumed existing thread "
                f"'{thread_id}' for user "
                f"'{username}'."
            )
        else:
            print(
                f"ℹ️ No saved state for thread "
                f"'{thread_id}'. Starting fresh."
            )

    except Exception as exc:
        print(
            f"❌ Error resuming chat session: {exc}"
        )

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


# ============================================================================
# NEW CHAT
# ============================================================================

@cl.on_chat_start
async def start_chat() -> None:
    """
    Initialize a new session.
    """

    try:
        user = cl.user_session.get("user")

        if user:
            username = user.identifier
            role = user.metadata.get(
                "role",
                "guest",
            )
        else:
            username = "unknown"
            role = "guest"

    except Exception as exc:
        print(
            f"⚠️ Error getting user information: {exc}"
        )
        username = "unknown"
        role = "guest"

    thread_id = (
        cl.context.session.thread_id
    )

    try:
        team, termination_ext = (
            await initialize_agents()
        )

        cl.user_session.set(
            "team",
            team,
        )

        cl.user_session.set(
            "termination_ext",
            termination_ext,
        )

        cl.user_session.set(
            "is_processing",
            False,
        )

        cl.user_session.set(
            "username",
            username,
        )

        cl.user_session.set(
            "role",
            role,
        )

        cl.user_session.set(
            "thread_id",
            thread_id,
        )

        cl.user_session.set(
            "message_count",
            0,
        )

        cl.user_session.set(
            "has_sent_message",
            False,
        )

        cl.user_session.set(
            "cancellation_token",
            None,
        )

        print(
            f"🔵 New session initialized for "
            f"'{username}' on thread "
            f"'{thread_id}'."
        )

        print(
            "⏳ Waiting for first message..."
        )

    except Exception as exc:
        print(
            f"❌ Critical error in start_chat: {exc}"
        )

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


# ============================================================================
# CLEAR SESSION STATE
# ============================================================================

@cl.action_callback(
    "clear_session_state"
)
async def on_clear_session_state(
    action: cl.Action,
):
    """
    Delete saved state and reinitialize a clean team.
    """

    try:
        username = cl.user_session.get(
            "username"
        )

        thread_id = cl.user_session.get(
            "thread_id"
        )

        if not username or not thread_id:
            raise RuntimeError(
                "Missing username or thread ID."
            )

        success = remove_team_state_from_disk(
            username,
            thread_id,
        )

        if not success:
            await cl.Message(
                content=(
                    "⚠️ **Could not clear session state.**"
                ),
                author="System",
            ).send()
            return

        team, termination_ext = (
            await initialize_agents()
        )

        cl.user_session.set(
            "team",
            team,
        )

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

        cl.user_session.set(
            "cancellation_token",
            None,
        )

        await cl.Message(
            content=(
                "✅ **Session History Cleared**\n\n"
                "The saved team state was deleted. "
                "A new agent team is active."
            ),
            author="System",
        ).send()

    except Exception as exc:
        print(
            f"❌ Error clearing session state: {exc}"
        )

        await cl.Message(
            content=(
                f"❌ **Error:** {exc}"
            ),
            author="System",
        ).send()


# ============================================================================
# MAIN MESSAGE HANDLER
# ============================================================================

@cl.on_message
async def handle_message(
    message: cl.Message,
) -> None:
    """
    Execute one multi-agent research workflow.

    Normal flow:
        agents -> ReportAgent -> save_to_pdf -> TERMINATE
        -> TaskResult -> Chainlit displays PDF
    """

    # ------------------------------------------------------------------------
    # Concurrency guard
    # ------------------------------------------------------------------------

    if cl.user_session.get(
        "is_processing",
        False,
    ):
        await cl.Message(
            content=(
                "⚠️ **Another request is already being processed.**\n\n"
                "Please wait or stop the current workflow."
            ),
            author="System",
        ).send()
        return

    cl.user_session.set(
        "is_processing",
        True,
    )

    # ------------------------------------------------------------------------
    # Session metrics
    # ------------------------------------------------------------------------

    message_count = cl.user_session.get(
        "message_count",
        0,
    )

    cl.user_session.set(
        "message_count",
        message_count + 1,
    )

    cl.user_session.set(
        "has_sent_message",
        True,
    )

    username = cl.user_session.get(
        "username",
        "Guest",
    )

    thread_id = cl.user_session.get(
        "thread_id",
        "unknown",
    )

    # ------------------------------------------------------------------------
    # Team
    # ------------------------------------------------------------------------

    team = cast(
        Optional[SelectorGroupChat],
        cl.user_session.get("team"),
    )

    if team is None:
        cl.user_session.set(
            "is_processing",
            False,
        )

        await cl.Message(
            content=(
                "❌ **Agent team is not initialized.**\n\n"
                "Please refresh the page."
            ),
            author="System",
        ).send()

        return

    # ------------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------------

    cancellation_token = CancellationToken()

    cl.user_session.set(
        "cancellation_token",
        cancellation_token,
    )

    termination_ext = cl.user_session.get(
        "termination_ext"
    )

    # ------------------------------------------------------------------------
    # PDF tracking
    # ------------------------------------------------------------------------

    task_start = datetime.now()

    task_start_ns = int(
        task_start.timestamp() * 1_000_000_000
    )

    known_pdf_state = snapshot_pdf_state()

    # ------------------------------------------------------------------------
    # Streaming state
    # ------------------------------------------------------------------------

    current_streaming_msg: Optional[
        cl.Message
    ] = None

    agent_message_count: dict[str, int] = {}

    tool_call_count = 0

    total_streamed_chars = 0

    task_completed_normally = False

    try:
        # --------------------------------------------------------------------
        # Reset external termination
        # --------------------------------------------------------------------

        if termination_ext is not None:
            try:
                termination_ext.reset()
            except Exception as exc:
                print(
                    f"⚠️ ExternalTermination reset failed: {exc}"
                )

        await cl.Message(
            content=(
                "🚀 **Starting Multi-Agent Analysis...**"
            ),
            author="System",
        ).send()

        # --------------------------------------------------------------------
        # AutoGen streaming
        # --------------------------------------------------------------------

        async for msg in team.run_stream(
            task=TextMessage(
                content=message.content,
                source="ExpertHuman",
            ),
            cancellation_token=cancellation_token,
        ):

            # ---------------------------------------------------------------
            # Explicit cancellation check
            # ---------------------------------------------------------------

            if cancellation_token.is_cancelled():

                print(
                    "🛑 CancellationToken is cancelled."
                )

                if current_streaming_msg is not None:
                    try:
                        await current_streaming_msg.send()
                    except Exception:
                        pass

                    current_streaming_msg = None

                break

            # ---------------------------------------------------------------
            # Common metadata
            # ---------------------------------------------------------------

            agent_name = getattr(
                msg,
                "source",
                None,
            )

            agent_name = (
                str(agent_name)
                if agent_name
                else "UnknownAgent"
            )

            msg_type = type(msg).__name__

            agent_message_count[
                agent_name
            ] = (
                agent_message_count.get(
                    agent_name,
                    0,
                ) + 1
            )

            # ---------------------------------------------------------------
            # Thought event
            # ---------------------------------------------------------------

            if isinstance(
                msg,
                ThoughtEvent,
            ):

                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                thought_msg = cl.Message(
                    content=(
                        "⏳ *thinking...*"
                    ),
                    author=agent_name,
                )

                await thought_msg.send()

                # Remove immediately to avoid clutter.
                try:
                    await thought_msg.remove()
                except Exception:
                    pass

                print(
                    f"💭 {agent_name}: "
                    f"{getattr(msg, 'content', '')}"
                )

            # ---------------------------------------------------------------
            # Streaming chunk
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                ModelClientStreamingChunkEvent,
            ):

                content = getattr(
                    msg,
                    "content",
                    "",
                )

                if not content:
                    continue

                if (
                    current_streaming_msg is None
                    or getattr(
                        current_streaming_msg,
                        "author",
                        None,
                    ) != agent_name
                ):

                    if current_streaming_msg is not None:
                        await current_streaming_msg.send()

                    current_streaming_msg = cl.Message(
                        content="",
                        author=agent_name,
                    )

                await current_streaming_msg.stream_token(
                    str(content)
                )

                total_streamed_chars += len(
                    str(content)
                )

            # ---------------------------------------------------------------
            # Tool call request
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                ToolCallRequestEvent,
            ):

                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                for tool_call in msg.content:

                    tool_call_count += 1

                    args_preview = str(
                        tool_call.arguments
                    )

                    if len(args_preview) > 500:
                        args_preview = (
                            args_preview[:500]
                            + "... (truncated)"
                        )

                    await cl.Message(
                        content=(
                            f"`{agent_name}` 🛠️ "
                            f"**Calling tool** "
                            f"`{tool_call.name}`\n\n"
                            "```json\n"
                            f"{args_preview}\n"
                            "```"
                        ),
                        author=agent_name,
                    ).send()

                    print(
                        f"🔧 {agent_name} -> "
                        f"{tool_call.name}"
                    )

            # ---------------------------------------------------------------
            # Tool result
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                ToolCallSummaryMessage,
            ):

                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                await cl.Message(
                    content=(
                        f"`{agent_name}` 🔄 "
                        "**Tool result received**"
                    ),
                    author=agent_name,
                ).send()

                print(
                    f"🔧 Tool summary from "
                    f"{agent_name}: "
                    f"{getattr(msg, 'content', '')}"
                )

            # ---------------------------------------------------------------
            # Normal agent text
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                TextMessage,
            ):

                content = str(
                    getattr(msg, "content", "")
                ).strip()

                print(
                    f"📝 {agent_name}: "
                    f"{content[:1000]}"
                )

                # Do not duplicate text that was already delivered through
                # ModelClientStreamingChunkEvent.
                if (
                    content
                    and current_streaming_msg is None
                    and content != "TERMINATE"
                ):

                    await cl.Message(
                        content=content,
                        author=agent_name,
                    ).send()

            # ---------------------------------------------------------------
            # Task result
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                TaskResult,
            ):

                task_completed_normally = True

                if current_streaming_msg is not None:
                    await current_streaming_msg.send()
                    current_streaming_msg = None

                stop_reason = getattr(
                    msg,
                    "stop_reason",
                    None,
                )

                duration = (
                    datetime.now() - task_start
                ).total_seconds()

                print(
                    "🏁 TaskResult received | "
                    f"stop_reason={stop_reason} | "
                    f"duration={duration:.2f}s | "
                    f"tools={tool_call_count}"
                )

                await cl.Message(
                    content=(
                        "✅ **Task completed successfully**"
                        + (
                            f" — {stop_reason}"
                            if stop_reason
                            else ""
                        )
                    ),
                    author="System",
                ).send()

                # -----------------------------------------------------------
                # PDF generated during THIS task
                # -----------------------------------------------------------

                latest_pdf = find_latest_task_pdf(
                    before_state=known_pdf_state,
                    task_start_ns=task_start_ns,
                )

                if latest_pdf is not None:

                    await show_pdf(
                        latest_pdf
                    )

                else:

                    print(
                        "⚠️ Task completed but no new PDF "
                        "was detected."
                    )

                    await cl.Message(
                        content=(
                            "ℹ️ **Task completed, but no new PDF "
                            "was detected.**\n\n"
                            "Check ReportAgent/save_to_pdf and "
                            "the generated_reports directory."
                        ),
                        author="System",
                    ).send()

            # ---------------------------------------------------------------
            # Other events
            # ---------------------------------------------------------------

            else:

                print(
                    f"ℹ️ Unhandled AutoGen event: "
                    f"{msg_type}"
                )

        # --------------------------------------------------------------------
        # Finalize active stream
        # --------------------------------------------------------------------

        if current_streaming_msg is not None:
            try:
                await current_streaming_msg.send()
            except Exception:
                pass

            current_streaming_msg = None

        # --------------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------------

        print(
            "📊 Workflow metrics | "
            f"agents={agent_message_count} | "
            f"tool_calls={tool_call_count} | "
            f"streamed_chars={total_streamed_chars}"
        )

        # --------------------------------------------------------------------
        # Save state after run
        # --------------------------------------------------------------------

        if task_completed_normally:
            await save_team_state_to_disk(
                team,
                username,
                thread_id,
            )

    except asyncio.CancelledError:

        # IMPORTANT:
        # The correct exception is asyncio.CancelledError.
        # CancellationToken itself is not an exception namespace.

        print(
            "🛑 Workflow cancelled by asyncio/Chainlit."
        )

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

        # Do not convert cancellation to a normal application error.

    except Exception as exc:

        print(
            f"❌ Error in handle_message: "
            f"{type(exc).__name__}: {exc}"
        )

        import traceback

        traceback.print_exc()

        await cl.Message(
            content=(
                "❌ **Error occurred during processing**\n\n"
                f"`{type(exc).__name__}: {exc}`"
            ),
            author="System",
        ).send()

    finally:

        # --------------------------------------------------------------------
        # State save fallback
        # --------------------------------------------------------------------

        try:
            has_sent_message = cl.user_session.get(
                "has_sent_message",
                False,
            )

            if (
                has_sent_message
                and team is not None
            ):
                await save_team_state_to_disk(
                    team,
                    username,
                    thread_id,
                )

        except Exception as exc:

            print(
                f"⚠️ Error auto-saving team state: {exc}"
            )

        # --------------------------------------------------------------------
        # Unlock
        # --------------------------------------------------------------------

        cl.user_session.set(
            "is_processing",
            False,
        )

        cl.user_session.set(
            "cancellation_token",
            None,
        )

        print(
            "🔓 Processing lock released."
        )


# ============================================================================
# STOP BUTTON
# ============================================================================

@cl.on_stop
async def on_stop():
    """
    Cancel the currently running workflow.
    """

    try:
        token = cl.user_session.get(
            "cancellation_token"
        )

        if token is not None:
            token.cancel()
            print(
                "🛑 CancellationToken.cancel() called."
            )

        # Also trigger the AutoGen ExternalTermination condition
        # when available. This provides a second cancellation path.
        termination_ext = cl.user_session.get(
            "termination_ext"
        )

        if termination_ext is not None:
            try:
                await termination_ext.set()
            except Exception:
                try:
                    termination_ext.set()
                except Exception:
                    pass

        await cl.Message(
            content=(
                "🛑 **Stop requested.**\n\n"
                "The current workflow is being cancelled."
            ),
            author="System",
        ).send()

    except Exception as exc:

        print(
            f"⚠️ Error in on_stop: {exc}"
        )


# ============================================================================
# CHAT END
# ============================================================================

@cl.on_chat_end
async def on_chat_end():
    """
    Cancel active work and persist the session state.
    """

    try:

        token = cl.user_session.get(
            "cancellation_token"
        )

        if token is not None:
            token.cancel()

        team = cl.user_session.get(
            "team"
        )

        username = cl.user_session.get(
            "username"
        )

        thread_id = cl.user_session.get(
            "thread_id"
        )

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
                f"💾 Final state saved for "
                f"'{username}' / '{thread_id}'."
            )

        else:

            print(
                "⏭️ Chat closed without workflow state."
            )

    except Exception as exc:

        print(
            f"⚠️ Error saving state on chat end: {exc}"
        )


# ============================================================================
# SETTINGS
# ============================================================================

@cl.on_settings_update
async def setup_agent_settings(settings):
    """
    Handle settings updates.
    """

    try:

        await cl.Message(
            content=(
                "⚙️ **Settings Updated**\n\n"
                "Your preferences have been received."
            ),
            author="System",
        ).send()

    except Exception as exc:

        print(
            f"❌ Error updating settings: {exc}"
        )

        await cl.Message(
            content=(
                "⚠️ **Settings update failed.**"
            ),
            author="System",
        ).send()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print(
        "🚀 Agentic Pharma System - Chainlit"
    )
    print(
        "Run with:"
    )
    print(
        "chainlit run "
        "orcastration/main_chainlit.py "
        "-w --host 0.0.0.0 --port 8000"
    )



