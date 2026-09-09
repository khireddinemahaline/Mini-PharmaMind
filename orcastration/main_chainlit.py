"""
Agentic Pharma System - Chainlit Interface

Multi-agent pharmaceutical research interface using:
- Chainlit
- AutoGen SelectorGroupChat
- Streaming output
- Human-in-the-loop
- Persistent team state
- PDF detection
- Workflow cancellation
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv


# ============================================================================
# ENVIRONMENT / PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# LOGGING
# ============================================================================

from autogen_agentchat import EVENT_LOGGER_NAME, TRACE_LOGGER_NAME


logging.basicConfig(level=logging.WARNING)

# Application logger
logger = logging.getLogger(__name__)

# AutoGen trace logger
trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
trace_logger.addHandler(logging.StreamHandler())
trace_logger.setLevel(logging.DEBUG)

# AutoGen event logger
event_logger = logging.getLogger(EVENT_LOGGER_NAME)
event_logger.addHandler(logging.StreamHandler())
event_logger.setLevel(logging.DEBUG)


# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

import chainlit as cl

from autogen_core import CancellationToken
from autogen_core.model_context import UnboundedChatCompletionContext

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import TaskResult

from autogen_agentchat.conditions import (
    ExternalTermination,
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


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from agents.target_search import target_search_agent
from agents.drug_search import setup_drug_search_agent
from agents.report import report_agent

from config.llm_client import model_client
from config.sytem_prompts import SELECT_PROMPT


# ============================================================================
# CONFIGURATION
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
# SAFE CANCELLATION HELPERS
# ============================================================================

def safe_cancel_token(token) -> bool:
    """
    Safely cancel a CancellationToken.

    Returns True if cancellation was requested successfully.
    """

    if token is None:
        return False

    # Detect unexpected coroutine/object corruption.
    if inspect.iscoroutine(token):
        logger.error(
            "Expected CancellationToken but found coroutine: %r",
            token,
        )

        try:
            token.close()
        except Exception:
            pass

        return False

    cancel_method = getattr(token, "cancel", None)

    if cancel_method is None or not callable(cancel_method):
        logger.error(
            "Invalid cancellation token object: %r",
            token,
        )
        return False

    try:
        result = cancel_method()

        # Defensive handling if a future implementation makes it async.
        if inspect.isawaitable(result):
            logger.warning(
                "CancellationToken.cancel() returned an awaitable."
            )

        return True

    except Exception:
        logger.exception(
            "Failed to cancel CancellationToken."
        )
        return False


def safe_set_external_termination(
    termination_ext,
) -> bool:
    """
    Safely trigger AutoGen ExternalTermination.
    """

    if termination_ext is None:
        return False

    set_method = getattr(
        termination_ext,
        "set",
        None,
    )

    if set_method is None or not callable(set_method):
        logger.error(
            "Invalid ExternalTermination object: %r",
            termination_ext,
        )
        return False

    try:
        result = set_method()

        # Some versions could theoretically return an awaitable.
        if inspect.isawaitable(result):
            logger.warning(
                "ExternalTermination.set() returned awaitable."
            )

        return True

    except Exception:
        logger.exception(
            "Failed to trigger ExternalTermination."
        )
        return False


def safe_reset_external_termination(
    termination_ext,
) -> bool:
    """
    Safely reset AutoGen ExternalTermination.
    """

    if termination_ext is None:
        return False

    reset_method = getattr(
        termination_ext,
        "reset",
        None,
    )

    if reset_method is None or not callable(reset_method):
        logger.warning(
            "ExternalTermination has no reset method."
        )
        return False

    try:
        result = reset_method()

        if inspect.isawaitable(result):
            logger.warning(
                "ExternalTermination.reset() returned awaitable."
            )

        return True

    except Exception:
        logger.exception(
            "Failed to reset ExternalTermination."
        )
        return False


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

        filename = (
            f"team_state_{username}_{thread_id}.json"
        )

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

        print(
            f"✅ Team state saved to: {filepath}"
        )

        return str(filepath.resolve())

    except asyncio.CancelledError:
        print(
            "🛑 State saving was cancelled."
        )
        raise

    except (IOError, OSError) as exc:

        logger.exception(
            "File I/O error saving team state."
        )

        print(
            f"❌ File I/O error saving team state: {exc}"
        )

        return None

    except Exception as exc:

        logger.exception(
            "Unexpected error saving team state."
        )

        print(
            f"❌ Unexpected error saving team state: {exc}"
        )

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

        filename = (
            f"team_state_{username}_{thread_id}.json"
        )

        filepath = STATE_DIR / filename

        if not filepath.exists():

            print(
                f"ℹ️ State file does not exist: {filepath}"
            )

            return False

        data = await asyncio.to_thread(
            filepath.read_text,
            encoding="utf-8",
        )

        state = json.loads(data)

        await team.load_state(state)

        print(
            f"✅ Team state loaded from: {filepath}"
        )

        return True

    except asyncio.CancelledError:
        print(
            "🛑 State loading was cancelled."
        )
        raise

    except (IOError, OSError) as exc:

        logger.exception(
            "File I/O error loading team state."
        )

        print(
            f"❌ File I/O error loading state: {exc}"
        )

        return False

    except (
        json.JSONDecodeError,
        ValueError,
    ) as exc:

        logger.exception(
            "Invalid JSON in team state."
        )

        print(
            f"❌ Invalid JSON in state file: {exc}"
        )

        return False

    except Exception as exc:

        logger.exception(
            "Unexpected error loading team state."
        )

        print(
            f"❌ Unexpected error loading state: {exc}"
        )

        return False


# ============================================================================
# TEAM STATE: DELETE
# ============================================================================

def remove_team_state_from_disk(
    username: str,
    thread_id: str,
) -> bool:
    """
    Delete persisted state for a session.
    """

    try:

        filename = (
            f"team_state_{username}_{thread_id}.json"
        )

        filepath = STATE_DIR / filename

        if not filepath.exists():

            print(
                f"⚠️ State file does not exist: {filepath}"
            )

            return True

        filepath.unlink()

        print(
            f"✅ Team state removed: {filepath}"
        )

        return True

    except (
        IOError,
        OSError,
        PermissionError,
    ) as exc:

        logger.exception(
            "File system error removing team state."
        )

        print(
            f"❌ File system error removing state: {exc}"
        )

        return False

    except Exception as exc:

        logger.exception(
            "Unexpected error removing team state."
        )

        print(
            f"❌ Unexpected error removing state: {exc}"
        )

        return False


# ============================================================================
# PDF TRACKING
# ============================================================================

def snapshot_pdf_state() -> dict[Path, int]:
    """
    Capture modification times of existing PDFs.
    """

    state: dict[Path, int] = {}

    for directory in PDF_DIRS:

        if not directory.exists():
            continue

        for pdf_path in directory.glob("*.pdf"):

            try:

                state[pdf_path] = (
                    pdf_path.stat().st_mtime_ns
                )

            except OSError:
                continue

    return state


def find_task_pdfs(
    before_state: dict[Path, int],
    task_start_ns: int,
) -> list[Path]:
    """
    Find PDFs created or modified during this task.
    """

    candidates: list[Path] = []

    for directory in PDF_DIRS:

        if not directory.exists():
            continue

        for pdf_path in directory.glob("*.pdf"):

            try:

                mtime_ns = (
                    pdf_path.stat().st_mtime_ns
                )

            except OSError:
                continue

            old_mtime = before_state.get(
                pdf_path
            )

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
    Return newest PDF generated during this task.
    """

    candidates = find_task_pdfs(
        before_state,
        task_start_ns,
    )

    if not candidates:
        return None

    try:

        return max(
            candidates,
            key=lambda path: (
                path.stat().st_mtime_ns
            ),
        )

    except OSError:

        logger.exception(
            "Failed while selecting latest PDF."
        )

        return None


# ============================================================================
# SHOW PDF
# ============================================================================

async def show_pdf(
    pdf_path: Path,
) -> bool:
    """
    Display PDF in Chainlit.
    """

    try:

        if not pdf_path.exists():

            print(
                f"⚠️ PDF not found: {pdf_path}"
            )

            return False

        if pdf_path.suffix.lower() != ".pdf":

            print(
                f"⚠️ Not a PDF file: {pdf_path}"
            )

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
            f"📄 PDF displayed: {pdf_path}"
        )

        return True

    except asyncio.CancelledError:
        print(
            "🛑 PDF display was cancelled."
        )
        raise

    except Exception as exc:

        logger.exception(
            "Error displaying PDF."
        )

        print(
            f"❌ Error displaying PDF: {exc}"
        )

        try:

            await cl.Message(
                content=(
                    "⚠️ The report was completed, "
                    "but the PDF could not be displayed.\n\n"
                    f"`{pdf_path}`"
                ),
                author="System",
            ).send()

        except Exception:
            logger.exception(
                "Failed to notify user about PDF error."
            )

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
    """

    try:

        response = await cl.AskUserMessage(
            content=prompt,
            timeout=300,
            raise_on_timeout=True,
        ).send()

        if response:

            output = response.get("output")

            if output:
                return str(output)

        return "User did not provide any input."

    except asyncio.CancelledError:

        print(
            "🛑 Human input request was cancelled."
        )

        raise

    except asyncio.TimeoutError:

        print(
            "⚠️ User input timed out."
        )

        return (
            "User did not provide any input "
            "within the time limit."
        )

    except TimeoutError:

        print(
            "⚠️ User input timed out."
        )

        return (
            "User did not provide any input "
            "within the time limit."
        )

    except Exception as exc:

        logger.exception(
            "Error getting human input."
        )

        print(
            f"❌ Error getting user input: {exc}"
        )

        return (
            "An error occurred while requesting "
            "user input."
        )


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

async def initialize_agents():

    """
    Initialize the AutoGen team.
    """

    try:

        # --------------------------------------------------------------------
        # TERMINATION
        # --------------------------------------------------------------------

        termination_word = (
            TextMentionTermination("TERMINATE")
        )

        source_match_termination = (
            SourceMatchTermination("ReportAgent")
        )

        termination_ext = (
            ExternalTermination()
        )

        termination = (
            (
                termination_word
                & source_match_termination
            )
            | termination_ext
        )

        # --------------------------------------------------------------------
        # CONTEXT
        # --------------------------------------------------------------------

        model_context = (
            UnboundedChatCompletionContext()
        )

        # --------------------------------------------------------------------
        # AGENTS
        # --------------------------------------------------------------------

        target_agent = (
            await target_search_agent()
        )

        drug_agent = (
            await setup_drug_search_agent()
        )

        report = report_agent()

        expert_human = UserProxyAgent(

            name="ExpertHuman",

            description=(
                "A Human-in-the-Loop biomedical expert "
                "who reviews and validates AI-generated "
                "findings during the drug discovery workflow."
            ),

            input_func=user_input_func,
        )

        # --------------------------------------------------------------------
        # TEAM
        # --------------------------------------------------------------------

        team = SelectorGroupChat(

            [
                target_agent,
                drug_agent,
                report,
                expert_human,
            ],

            model_client=model_client,

            termination_condition=termination,

            allow_repeated_speaker=False,

            selector_prompt=SELECT_PROMPT,

            model_context=model_context,

            max_selector_attempts=3,
        )

        print(
            "✅ Agent team initialized successfully."
        )

        return (
            team,
            termination_ext,
        )

    except asyncio.CancelledError:
        print(
            "🛑 Agent initialization cancelled."
        )
        raise

    except Exception as exc:

        logger.exception(
            "Error initializing agents."
        )

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

    # Credentials intentionally preserved exactly as requested.

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

    return [

        cl.ChatProfile(

            name="Drug Discovery Researcher",

            markdown_description=(
                "A researcher focused on identifying "
                "novel drug targets and compounds."
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
                        "potential drug candidates."
                    ),
                    icon="/public/adn.png",
                ),

                cl.Starter(
                    label="Analyze aspirin compound",
                    message=(
                        "Search for aspirin drug information "
                        "including molecular structure, mechanism "
                        "of action, and known targets."
                    ),
                    icon="/public/drug.png",
                ),

                cl.Starter(
                    label="Cancer drug discovery",
                    message=(
                        "Identify potential drug compounds "
                        "for treating breast cancer."
                    ),
                    icon="/public/cancer.png",
                ),

                cl.Starter(
                    label=(
                        "Compare anti-inflammatory drugs"
                    ),
                    message=(
                        "Compare ibuprofen and naproxen "
                        "as anti-inflammatory drugs."
                    ),
                    icon="/public/disease.png",
                ),
            ],
        )
    ]


# ============================================================================
# CHAT RESUME
# ============================================================================

@cl.on_chat_resume
async def on_chat_resume(
    thread: ThreadDict,
):

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
                "⚠️ No thread ID available."
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
                f"✅ Resumed thread '{thread_id}'."
            )

        else:

            print(
                f"ℹ️ No saved state for '{thread_id}'."
            )

    except asyncio.CancelledError:
        raise

    except Exception as exc:

        logger.exception(
            "Error resuming chat."
        )

        print(
            f"❌ Error resuming chat: {exc}"
        )


# ============================================================================
# NEW CHAT
# ============================================================================

@cl.on_chat_start
async def start_chat() -> None:

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

        thread_id = (
            cl.context.session.thread_id
        )

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
            f"'{username}' on thread '{thread_id}'."
        )

        print(
            "⏳ Waiting for first message..."
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:

        logger.exception(
            "Critical error in start_chat."
        )

        print(
            f"❌ Critical error in start_chat: {exc}"
        )

        raise


# ============================================================================
# MAIN MESSAGE HANDLER
# ============================================================================

@cl.on_message
async def handle_message(
    message: cl.Message,
) -> None:

    if cl.user_session.get(
        "is_processing",
        False,
    ):

        await cl.Message(
            content=(
                "⚠️ **Another request is already "
                "being processed.**"
            ),
            author="System",
        ).send()

        return

    cl.user_session.set(
        "is_processing",
        True,
    )

    current_streaming_msg = None

    cancellation_token = None

    username = "Guest"
    thread_id = "unknown"

    task_completed_normally = False

    try:

        # --------------------------------------------------------------------
        # SESSION DATA
        # --------------------------------------------------------------------

        message_count = (
            cl.user_session.get(
                "message_count",
                0,
            )
        )

        cl.user_session.set(
            "message_count",
            message_count + 1,
        )

        cl.user_session.set(
            "has_sent_message",
            True,
        )

        username = (
            cl.user_session.get(
                "username",
                "Guest",
            )
        )

        thread_id = (
            cl.user_session.get(
                "thread_id",
                "unknown",
            )
        )

        # --------------------------------------------------------------------
        # TEAM
        # --------------------------------------------------------------------

        team = cast(
            Optional[SelectorGroupChat],
            cl.user_session.get("team"),
        )

        if team is None:

            await cl.Message(
                content=(
                    "❌ **Agent team is not initialized.**"
                ),
                author="System",
            ).send()

            return

        # --------------------------------------------------------------------
        # CANCELLATION
        # --------------------------------------------------------------------

        cancellation_token = (
            CancellationToken()
        )

        cl.user_session.set(
            "cancellation_token",
            cancellation_token,
        )

        termination_ext = (
            cl.user_session.get(
                "termination_ext"
            )
        )

        safe_reset_external_termination(
            termination_ext
        )

        # --------------------------------------------------------------------
        # PDF TRACKING
        # --------------------------------------------------------------------

        task_start = datetime.now()

        task_start_ns = int(
            task_start.timestamp()
            * 1_000_000_000
        )

        known_pdf_state = (
            snapshot_pdf_state()
        )

        # --------------------------------------------------------------------
        # STREAMING STATE
        # --------------------------------------------------------------------

        agent_message_count: dict[str, int] = {}

        tool_call_count = 0

        total_streamed_chars = 0

        await cl.Message(
            content=(
                "🚀 **Starting Multi-Agent Analysis...**"
            ),
            author="System",
        ).send()

        print(
            "▶️ Starting team.run_stream()"
        )

        # --------------------------------------------------------------------
        # RUN STREAM
        # --------------------------------------------------------------------

        async for msg in team.run_stream(

            task=TextMessage(
                content=message.content,
                source="ExpertHuman",
            ),

            cancellation_token=cancellation_token,
        ):

            # ---------------------------------------------------------------
            # CANCELLATION CHECK
            # ---------------------------------------------------------------

            if cancellation_token.is_cancelled():

                print(
                    "🛑 CancellationToken is cancelled."
                )

                break

            # ---------------------------------------------------------------
            # METADATA
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

            agent_message_count[agent_name] = (
                agent_message_count.get(
                    agent_name,
                    0,
                )
                + 1
            )

            # ---------------------------------------------------------------
            # THOUGHT EVENT
            # ---------------------------------------------------------------

            if isinstance(
                msg,
                ThoughtEvent,
            ):

                if current_streaming_msg is not None:

                    try:
                        await current_streaming_msg.send()
                    except Exception:
                        logger.exception(
                            "Failed sending streaming message."
                        )

                    current_streaming_msg = None

                print(
                    f"💭 {agent_name}: "
                    f"{getattr(msg, 'content', '')}"
                )

            # ---------------------------------------------------------------
            # STREAMING CHUNK
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                ModelClientStreamingChunkEvent,
            ):

                content = str(
                    getattr(
                        msg,
                        "content",
                        "",
                    )
                )

                if not content:
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

                    current_streaming_msg = (
                        cl.Message(
                            content="",
                            author=agent_name,
                        )
                    )

                await current_streaming_msg.stream_token(
                    content
                )

                total_streamed_chars += len(
                    content
                )

            # ---------------------------------------------------------------
            # TOOL CALL
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

            # ---------------------------------------------------------------
            # TOOL RESULT
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

            # ---------------------------------------------------------------
            # TEXT MESSAGE
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                TextMessage,
            ):

                content = str(
                    getattr(
                        msg,
                        "content",
                        "",
                    )
                ).strip()

                print(
                    f"📝 {agent_name}: "
                    f"{content[:1000]}"
                )

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
            # TASK RESULT
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
                    datetime.now()
                    - task_start
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

                latest_pdf = (
                    find_latest_task_pdf(
                        before_state=known_pdf_state,
                        task_start_ns=task_start_ns,
                    )
                )

                if latest_pdf is not None:

                    await show_pdf(
                        latest_pdf
                    )

                else:

                    await cl.Message(
                        content=(
                            "ℹ️ **Task completed, but no new "
                            "PDF was detected.**"
                        ),
                        author="System",
                    ).send()

            # ---------------------------------------------------------------
            # OTHER EVENTS
            # ---------------------------------------------------------------

            else:

                print(
                    f"ℹ️ Unhandled AutoGen event: "
                    f"{msg_type}"
                )

        # --------------------------------------------------------------------
        # STREAM FINISHED
        # --------------------------------------------------------------------

        print(
            "⏹️ team.run_stream() exited"
        )

        if current_streaming_msg is not None:

            try:
                await current_streaming_msg.send()
            except Exception:
                logger.exception(
                    "Failed to flush final stream."
                )

            current_streaming_msg = None

        print(
            "📊 Workflow metrics | "
            f"agents={agent_message_count} | "
            f"tool_calls={tool_call_count} | "
            f"streamed_chars={total_streamed_chars}"
        )

        if task_completed_normally:

            await save_team_state_to_disk(
                team,
                username,
                thread_id,
            )

    # ------------------------------------------------------------------------
    # ASYNCIO CANCELLATION
    # ------------------------------------------------------------------------

    except asyncio.CancelledError:

        logger.warning(
            "Workflow cancelled by asyncio/Chainlit.",
            exc_info=True,
        )

        print(
            "🛑 Workflow received asyncio.CancelledError."
        )

        if current_streaming_msg is not None:

            try:
                await current_streaming_msg.send()
            except Exception:
                logger.exception(
                    "Failed to flush stream after cancellation."
                )

        try:

            await cl.Message(
                content=(
                    "🛑 **Task cancelled.**\n\n"
                    "You can start a new query."
                ),
                author="System",
            ).send()

        except Exception:
            logger.exception(
                "Failed sending cancellation notice."
            )

        # Important: do not raise a new NameError here.

    # ------------------------------------------------------------------------
    # GENERIC ERROR
    # ------------------------------------------------------------------------

    except Exception as exc:

        logger.exception(
            "Unhandled error in handle_message | "
            "user=%s thread=%s",
            username,
            thread_id,
        )

        print(
            f"❌ Unhandled error: "
            f"{type(exc).__name__}: {exc}"
        )

        try:

            await cl.Message(
                content=(
                    "❌ **Error occurred during processing**\n\n"
                    f"`{type(exc).__name__}: {exc}`"
                ),
                author="System",
            ).send()

        except Exception:
            logger.exception(
                "Failed to notify user about error."
            )

    # ------------------------------------------------------------------------
    # FINALLY
    # ------------------------------------------------------------------------

    finally:

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

    try:

        print(
            "🛑 on_stop() FIRED"
        )

        token = cl.user_session.get(
            "cancellation_token"
        )

        if token is not None:

            print(
                f"🛑 Cancellation object type: "
                f"{type(token)}"
            )

            safe_cancel_token(token)

        termination_ext = (
            cl.user_session.get(
                "termination_ext"
            )
        )

        safe_set_external_termination(
            termination_ext
        )

        await cl.Message(
            content=(
                "🛑 **Stop requested.**\n\n"
                "The current workflow is being cancelled."
            ),
            author="System",
        ).send()

    except asyncio.CancelledError:
        raise

    except Exception:

        logger.exception(
            "Error in on_stop."
        )


# ============================================================================
# CHAT END
# ============================================================================

@cl.on_chat_end
async def on_chat_end():

    try:

        print(
            "🔴 on_chat_end() FIRED"
        )

        token = cl.user_session.get(
            "cancellation_token"
        )

        if token is not None:

            print(
                f"🔴 Chat-end cancellation object type: "
                f"{type(token)}"
            )

            safe_cancel_token(token)

        team = cl.user_session.get(
            "team"
        )

        username = cl.user_session.get(
            "username"
        )

        thread_id = cl.user_session.get(
            "thread_id"
        )

        has_sent_message = (
            cl.user_session.get(
                "has_sent_message",
                False,
            )
        )

        # Do not save state if a workflow is actively running.
        # Saving AutoGen state while the team is mutating can cause problems.
        is_processing = cl.user_session.get(
            "is_processing",
            False,
        )

        if (
            has_sent_message
            and not is_processing
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

        elif is_processing:

            print(
                "⏭️ Chat ended while workflow was processing. "
                "Skipping concurrent state save."
            )

        else:

            print(
                "⏭️ Chat closed without workflow state."
            )

    except asyncio.CancelledError:

        print(
            "🛑 on_chat_end cancelled."
        )

        raise

    except Exception as exc:

        logger.exception(
            "Error during on_chat_end."
        )

        print(
            f"⚠️ Error saving state on chat end: {exc}"
        )


# ============================================================================
# SETTINGS
# ============================================================================

@cl.on_settings_update
async def setup_agent_settings(settings):

    try:

        await cl.Message(
            content=(
                "⚙️ **Settings Updated**\n\n"
                "Your preferences have been received."
            ),
            author="System",
        ).send()

    except asyncio.CancelledError:
        raise

    except Exception:

        logger.exception(
            "Error updating settings."
        )


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
