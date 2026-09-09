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
    - ReportAgent is responsible for normal TERMINATE completion
    - ExternalTermination is reserved for manual cancellation
    - Correct asyncio cancellation handling
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, cast

import chainlit as cl
from dotenv import load_dotenv

# ============================================================================
# Environment / project root
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

try:
    from autogen_agentchat import (
        EVENT_LOGGER_NAME,
        TRACE_LOGGER_NAME,
    )

    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    trace_logger.setLevel(logging.DEBUG)

    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    event_logger.setLevel(logging.DEBUG)

except Exception:
    trace_logger = logging.getLogger("autogen.trace")
    event_logger = logging.getLogger("autogen.event")


# ============================================================================
# Third-party imports
# ============================================================================

from autogen_core import CancellationToken
from autogen_core.model_context import UnboundedChatCompletionContext

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import (
    ExternalTermination,
    TerminationCondition,
    TerminatedException,
)
from autogen_agentchat.messages import (
    BaseChatMessage,
    BaseAgentEvent,
    ModelClientStreamingChunkEvent,
    TextMessage,
    ThoughtEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_agentchat.teams import SelectorGroupChat

from chainlit.types import ThreadDict


# ============================================================================
# Project imports
# ============================================================================

from agents.target_search import target_search_agent
from agents.drug_search import setup_drug_search_agent
from agents.report import report_agent

from config.llm_client import model_client
from config.sytem_prompts import SELECT_PROMPT


# ============================================================================
# Configuration
# ============================================================================

STATE_DIR = PROJECT_ROOT / "session_state"

STATE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GENERATED_REPORTS_DIR = (
    PROJECT_ROOT / "generated_reports"
)

GENERATED_REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PDF_DIRS = (
    GENERATED_REPORTS_DIR,
    PROJECT_ROOT / "resumes_uploaded",
)


# ============================================================================
# CUSTOM TERMINATION CONDITION
# ============================================================================

class ReportAgentTermination(TerminationCondition):
    """
    Terminates ONLY when ReportAgent explicitly sends:

        TERMINATE

    This avoids the potential latching problem caused by combining:

        TextMentionTermination("TERMINATE")
        &
        SourceMatchTermination("ReportAgent")

    across different messages.
    """

    def __init__(self) -> None:
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(
        self,
        messages: list[
            BaseAgentEvent | BaseChatMessage
        ],
    ) -> Optional[BaseChatMessage]:

        if self._terminated:
            raise TerminatedException(
                "Termination condition has already been reached."
            )

        for message in messages:

            source = getattr(
                message,
                "source",
                None,
            )

            content = getattr(
                message,
                "content",
                None,
            )

            if (
                source == "ReportAgent"
                and isinstance(content, str)
                and content.strip() == "TERMINATE"
            ):

                self._terminated = True

                logger.info(
                    "ReportAgent emitted TERMINATE."
                )

                return TextMessage(
                    content="TERMINATE",
                    source="ReportAgent",
                )

        return None

    async def reset(self) -> None:
        self._terminated = False


# ============================================================================
# HELPER: SAFE TOKEN CANCELLATION
# ============================================================================

def safe_cancel_token(
    token: Any,
    context: str,
) -> bool:
    """
    Safely cancel an AutoGen CancellationToken.

    Returns True if cancellation was successfully requested.
    """

    if token is None:
        logger.debug(
            "%s: no cancellation token.",
            context,
        )
        return False

    if inspect.iscoroutine(token):
        logger.error(
            "%s: cancellation_token is unexpectedly "
            "a coroutine: %r",
            context,
            token,
        )

        return False

    cancel_method = getattr(
        token,
        "cancel",
        None,
    )

    if not callable(cancel_method):

        logger.error(
            "%s: invalid cancellation_token type: %s",
            context,
            type(token).__name__,
        )

        return False

    try:

        result = cancel_method()

        # Defensive handling if an unexpected implementation
        # returns an awaitable.
        if inspect.isawaitable(result):
            logger.warning(
                "%s: token.cancel() returned an awaitable "
                "and cannot be awaited from this sync helper.",
                context,
            )

        logger.info(
            "%s: CancellationToken cancelled.",
            context,
        )

        return True

    except Exception:

        logger.exception(
            "%s: failed to cancel token.",
            context,
        )

        return False


# ============================================================================
# HELPER: SAFE EXTERNAL TERMINATION
# ============================================================================

async def safe_set_external_termination(
    termination_ext: Any,
    context: str,
) -> bool:
    """
    Safely trigger ExternalTermination.

    Supports both synchronous and awaitable implementations.
    """

    if termination_ext is None:
        return False

    set_method = getattr(
        termination_ext,
        "set",
        None,
    )

    if not callable(set_method):

        logger.error(
            "%s: invalid ExternalTermination object: %r",
            context,
            termination_ext,
        )

        return False

    try:

        result = set_method()

        if inspect.isawaitable(result):
            await result

        logger.info(
            "%s: ExternalTermination triggered.",
            context,
        )

        return True

    except Exception:

        logger.exception(
            "%s: failed to trigger ExternalTermination.",
            context,
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

        logger.info(
            "Team state saved: %s",
            filepath,
        )

        return str(
            filepath.resolve()
        )

    except asyncio.CancelledError:

        logger.warning(
            "State save was cancelled."
        )

        raise

    except (
        IOError,
        OSError,
        TypeError,
        ValueError,
    ):

        logger.exception(
            "Failed to save team state."
        )

        return None

    except Exception:

        logger.exception(
            "Unexpected error saving team state."
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

            logger.info(
                "State file does not exist: %s",
                filepath,
            )

            return False

        data = await asyncio.to_thread(
            filepath.read_text,
            encoding="utf-8",
        )

        state = json.loads(data)

        await team.load_state(
            state
        )

        logger.info(
            "Team state loaded: %s",
            filepath,
        )

        return True

    except asyncio.CancelledError:
        raise

    except (
        IOError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        logger.exception(
            "Failed to load team state."
        )

        return False

    except Exception:

        logger.exception(
            "Unexpected error loading team state."
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

            logger.info(
                "State file already absent: %s",
                filepath,
            )

            return True

        filepath.unlink()

        logger.info(
            "Team state removed: %s",
            filepath,
        )

        return True

    except (
        IOError,
        OSError,
        PermissionError,
    ):

        logger.exception(
            "Failed to remove team state."
        )

        return False

    except Exception:

        logger.exception(
            "Unexpected error removing team state."
        )

        return False


# ============================================================================
# PDF TRACKING
# ============================================================================

def snapshot_pdf_state() -> dict[Path, int]:
    """
    Capture modification times of PDFs before the task starts.
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
    Return PDFs created or modified during the current task.
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

            is_new = (
                old_mtime is None
            )

            is_modified = (
                old_mtime is not None
                and mtime_ns > old_mtime
            )

            created_during_task = (
                mtime_ns >= task_start_ns
            )

            if (
                is_new
                or is_modified
                or created_during_task
            ):

                candidates.append(
                    pdf_path
                )

    return candidates


def find_latest_task_pdf(
    before_state: dict[Path, int],
    task_start_ns: int,
) -> Optional[Path]:
    """
    Find newest PDF generated or modified during this task.
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
            "Error selecting latest PDF."
        )

        return None


# ============================================================================
# PDF DISPLAY
# ============================================================================

async def show_pdf(
    pdf_path: Path,
) -> bool:
    """
    Display PDF in Chainlit and provide download access.
    """

    try:

        if not pdf_path.exists():

            logger.warning(
                "PDF not found: %s",
                pdf_path,
            )

            return False

        if (
            pdf_path.suffix.lower()
            != ".pdf"
        ):

            logger.warning(
                "Not a PDF: %s",
                pdf_path,
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

        logger.info(
            "PDF displayed: %s",
            pdf_path,
        )

        return True

    except asyncio.CancelledError:
        raise

    except Exception:

        logger.exception(
            "Error displaying PDF."
        )

        try:

            await cl.Message(
                content=(
                    "⚠️ The report was completed, "
                    "but the PDF could not be displayed "
                    "automatically.\n\n"
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
    cancellation_token: (
        CancellationToken | None
    ) = None,
) -> str:
    """
    Capture human input through Chainlit.
    """

    try:

        if (
            cancellation_token is not None
            and cancellation_token.is_cancelled()
        ):

            raise asyncio.CancelledError

        response = await cl.AskUserMessage(
            content=prompt,
            timeout=300,
            raise_on_timeout=True,
        ).send()

        if response:

            output = response.get(
                "output"
            )

            if output:
                return str(output)

        return (
            "User did not provide any input."
        )

    except asyncio.CancelledError:

        logger.info(
            "Human input request was cancelled."
        )

        raise

    except asyncio.TimeoutError:

        logger.warning(
            "User input timed out after 300 seconds."
        )

        return (
            "User did not provide any input "
            "within the time limit."
        )

    except Exception:

        logger.exception(
            "Error getting human input."
        )

        return (
            "An error occurred while "
            "requesting user input."
        )


# ============================================================================
# AGENT INITIALIZATION
# ============================================================================

async def initialize_agents():
    """
    Initialize the complete agent team.

    Normal termination:
        ReportAgent emits exactly TERMINATE.

    Manual stop:
        ExternalTermination is used.
    """

    try:

        # --------------------------------------------------------------------
        # TERMINATION
        # --------------------------------------------------------------------

        report_termination = (
            ReportAgentTermination()
        )

        termination_ext = (
            ExternalTermination()
        )

        termination = (
            report_termination
            | termination_ext
        )

        # --------------------------------------------------------------------
        # Context
        # --------------------------------------------------------------------

        model_context = (
            UnboundedChatCompletionContext()
        )

        # --------------------------------------------------------------------
        # Agents
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
                "findings during the drug discovery workflow. "
                "The expert provides scientific judgement, "
                "approves or revises target and drug rankings, "
                "resolves conflicting evidence, answers "
                "clarification requests, and records the final "
                "human decision before the workflow proceeds."
            ),

            input_func=user_input_func,
        )

        # --------------------------------------------------------------------
        # SelectorGroupChat
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

        logger.info(
            "Agent team initialized successfully."
        )

        return (
            team,
            termination_ext,
        )

    except asyncio.CancelledError:
        raise

    except Exception:

        logger.exception(
            "Error initializing agents."
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
    """

    # Credentials intentionally preserved
    # exactly as requested.

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

    try:

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
                            "Find drug targets for "
                            "Alzheimer's disease"
                        ),

                        message=(
                            "Search for therapeutic targets "
                            "associated with Alzheimer's disease "
                            "and identify potential drug candidates "
                            "that could modulate these targets."
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
                            "Identify potential drug compounds "
                            "for treating breast cancer, including "
                            "efficacy data and clinical trial status."
                        ),

                        icon="/public/cancer.png",
                    ),

                    cl.Starter(

                        label=(
                            "Compare anti-inflammatory drugs"
                        ),

                        message=(
                            "Compare the mechanisms and efficacy "
                            "of ibuprofen and naproxen as "
                            "anti-inflammatory drugs."
                        ),

                        icon="/public/disease.png",
                    ),
                ],
            )
        ]

    except Exception:

        logger.exception(
            "Error configuring chat profiles."
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

    try:

        user = cl.user_session.get(
            "user"
        )

        if not user:

            logger.warning(
                "No user found during chat resume."
            )

            return

        username = user.identifier

        thread_id = thread.get("id")

        if not thread_id:

            logger.warning(
                "No thread ID during chat resume."
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

            logger.info(
                "Resumed thread '%s' for '%s'.",
                thread_id,
                username,
            )

        else:

            logger.info(
                "No saved state for thread '%s'.",
                thread_id,
            )

    except asyncio.CancelledError:
        raise

    except Exception:

        logger.exception(
            "Error resuming chat session."
        )

        try:

            await cl.Message(

                content=(
                    "⚠️ **Session Resume Error**\n\n"
                    "Starting a fresh session."
                ),

                author="System",
            ).send()

        except Exception:
            logger.exception(
                "Failed to send resume error."
            )


# ============================================================================
# NEW CHAT
# ============================================================================

@cl.on_chat_start
async def start_chat() -> None:

    try:

        user = cl.user_session.get(
            "user"
        )

        if user:

            username = user.identifier

            role = user.metadata.get(
                "role",
                "guest",
            )

        else:

            username = "unknown"
            role = "guest"

    except Exception:

        logger.exception(
            "Error getting user information."
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

        logger.info(
            "New session initialized for '%s' "
            "on thread '%s'.",
            username,
            thread_id,
        )

        logger.info(
            "Waiting for first message..."
        )

    except asyncio.CancelledError:
        raise

    except Exception:

        logger.exception(
            "Critical error in start_chat."
        )

        try:

            await cl.Message(

                content=(
                    "❌ **System initialization failed.**\n\n"
                    "Please refresh the page and try again."
                ),

                author="System",
            ).send()

        except Exception:
            logger.exception(
                "Failed to send initialization error."
            )

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

    try:

        if cl.user_session.get(
            "is_processing",
            False,
        ):

            await cl.Message(
                content=(
                    "⚠️ Cannot clear the session while "
                    "a workflow is running."
                ),
                author="System",
            ).send()

            return

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

    except asyncio.CancelledError:
        raise

    except Exception:

        logger.exception(
            "Error clearing session state."
        )

        await cl.Message(

            content=(
                "❌ **Error clearing session state.**"
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

        agents
        ->
        ReportAgent
        ->
        PDF generation
        ->
        TERMINATE
        ->
        TaskResult
        ->
        PDF displayed
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

    current_streaming_msg: Optional[
        cl.Message
    ] = None

    cancellation_token: Optional[
        CancellationToken
    ] = None

    task_completed_normally = False

    username = "Guest"
    thread_id = "unknown"

    try:

        # --------------------------------------------------------------------
        # Session metrics
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

        username = cl.user_session.get(
            "username",
            "Guest",
        )

        thread_id = cl.user_session.get(
            "thread_id",
            "unknown",
        )

        # --------------------------------------------------------------------
        # Team
        # --------------------------------------------------------------------

        team = cast(

            Optional[SelectorGroupChat],

            cl.user_session.get(
                "team"
            ),
        )

        if team is None:

            raise RuntimeError(
                "Agent team is not initialized."
            )

        # --------------------------------------------------------------------
        # Cancellation
        # --------------------------------------------------------------------

        cancellation_token = (
            CancellationToken()
        )

        cl.user_session.set(
            "cancellation_token",
            cancellation_token,
        )

        logger.info(
            "CancellationToken created."
        )

        termination_ext = (
            cl.user_session.get(
                "termination_ext"
            )
        )

        # --------------------------------------------------------------------
        # PDF tracking
        # --------------------------------------------------------------------

        task_start_ns = time.time_ns()

        known_pdf_state = (
            snapshot_pdf_state()
        )

        # --------------------------------------------------------------------
        # Streaming metrics
        # --------------------------------------------------------------------

        agent_message_count: dict[
            str,
            int,
        ] = {}

        tool_call_count = 0

        total_streamed_chars = 0

        # --------------------------------------------------------------------
        # Reset termination
        # --------------------------------------------------------------------

        if termination_ext is not None:

            reset_method = getattr(
                termination_ext,
                "reset",
                None,
            )

            if callable(reset_method):

                try:

                    result = reset_method()

                    if inspect.isawaitable(
                        result
                    ):
                        await result

                except Exception:

                    logger.exception(
                        "ExternalTermination reset failed."
                    )

        await cl.Message(

            content=(
                "🚀 **Starting Multi-Agent Analysis...**"
            ),

            author="System",
        ).send()

        logger.info(
            "Starting team.run_stream()."
        )

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
            # Explicit cancellation
            # ---------------------------------------------------------------

            if cancellation_token.is_cancelled():

                logger.info(
                    "CancellationToken is cancelled."
                )

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

            msg_type = type(
                msg
            ).__name__

            agent_message_count[
                agent_name
            ] = (
                agent_message_count.get(
                    agent_name,
                    0,
                )
                + 1
            )

            # ---------------------------------------------------------------
            # Thought event
            # ---------------------------------------------------------------

            if isinstance(
                msg,
                ThoughtEvent,
            ):

                if (
                    current_streaming_msg
                    is not None
                ):

                    await current_streaming_msg.send()

                    current_streaming_msg = None

                thought_msg = cl.Message(

                    content="⏳ *thinking...*",

                    author=agent_name,
                )

                await thought_msg.send()

                try:

                    await thought_msg.remove()

                except Exception:
                    pass

                logger.debug(
                    "Thought from %s: %s",
                    agent_name,
                    getattr(
                        msg,
                        "content",
                        "",
                    ),
                )

            # ---------------------------------------------------------------
            # Streaming chunks
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
                    )
                    != agent_name
                ):

                    if (
                        current_streaming_msg
                        is not None
                    ):

                        await current_streaming_msg.send()

                    current_streaming_msg = (
                        cl.Message(
                            content="",
                            author=agent_name,
                        )
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

                if (
                    current_streaming_msg
                    is not None
                ):

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

                    logger.info(
                        "%s -> tool: %s",
                        agent_name,
                        tool_call.name,
                    )

            # ---------------------------------------------------------------
            # Tool result
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                ToolCallSummaryMessage,
            ):

                if (
                    current_streaming_msg
                    is not None
                ):

                    await current_streaming_msg.send()

                    current_streaming_msg = None

                await cl.Message(

                    content=(
                        f"`{agent_name}` 🔄 "
                        "**Tool result received**"
                    ),

                    author=agent_name,
                ).send()

                logger.debug(
                    "Tool summary from %s: %s",
                    agent_name,
                    getattr(
                        msg,
                        "content",
                        "",
                    ),
                )

            # ---------------------------------------------------------------
            # Normal agent text
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

                logger.info(
                    "Message from %s: %s",
                    agent_name,
                    content[:1000],
                )

                # Do not duplicate streamed text.
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
            # TaskResult
            # ---------------------------------------------------------------

            elif isinstance(
                msg,
                TaskResult,
            ):

                task_completed_normally = True

                if (
                    current_streaming_msg
                    is not None
                ):

                    await current_streaming_msg.send()

                    current_streaming_msg = None

                stop_reason = getattr(
                    msg,
                    "stop_reason",
                    None,
                )

                logger.info(
                    "TaskResult received | "
                    "stop_reason=%s | "
                    "tools=%s",
                    stop_reason,
                    tool_call_count,
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
                # PDF generated during this task
                # -----------------------------------------------------------

                latest_pdf = (
                    find_latest_task_pdf(

                        before_state=(
                            known_pdf_state
                        ),

                        task_start_ns=(
                            task_start_ns
                        ),
                    )
                )

                if latest_pdf is not None:

                    await show_pdf(
                        latest_pdf
                    )

                else:

                    logger.warning(
                        "Task completed but no new PDF "
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
            # Other AutoGen events
            # ---------------------------------------------------------------

            else:

                logger.debug(
                    "Unhandled AutoGen event: %s",
                    msg_type,
                )

        logger.info(
            "team.run_stream() exited."
        )

        # --------------------------------------------------------------------
        # Save state
        # --------------------------------------------------------------------

        if task_completed_normally:

            await save_team_state_to_disk(

                team,
                username,
                thread_id,
            )

    except asyncio.CancelledError:

        logger.warning(
            "Workflow cancelled by asyncio/Chainlit."
        )

        if current_streaming_msg is not None:

            try:

                await current_streaming_msg.send()

            except Exception:

                logger.exception(
                    "Failed to flush streaming message."
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
                "Failed to send cancellation notice."
            )

        # Do not raise a new exception here.

    except Exception as exc:

        logger.exception(
            "Unhandled error in handle_message | "
            "user=%s thread=%s message_count=%s",
            username,
            thread_id,
            cl.user_session.get(
                "message_count",
                "-",
            ),
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

    finally:

        # --------------------------------------------------------------------
        # Finalize stream
        # --------------------------------------------------------------------

        if current_streaming_msg is not None:

            try:

                await current_streaming_msg.send()

            except Exception:
                pass

        # --------------------------------------------------------------------
        # Clear cancellation token only if it belongs
        # to this workflow.
        # --------------------------------------------------------------------

        current_token = (
            cl.user_session.get(
                "cancellation_token"
            )
        )

        if (
            cancellation_token is not None
            and current_token is cancellation_token
        ):

            cl.user_session.set(
                "cancellation_token",
                None,
            )

        cl.user_session.set(
            "is_processing",
            False,
        )

        logger.info(
            "Processing lock released."
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

        logger.info(
            "on_stop() fired."
        )

        token = cl.user_session.get(
            "cancellation_token"
        )

        safe_cancel_token(
            token,
            "on_stop",
        )

        # --------------------------------------------------------------------
        # Trigger ExternalTermination
        # --------------------------------------------------------------------

        termination_ext = (
            cl.user_session.get(
                "termination_ext"
            )
        )

        await safe_set_external_termination(
            termination_ext,
            "on_stop",
        )

        await cl.Message(

            content=(
                "🛑 **Stop requested.**\n\n"
                "The current workflow is being cancelled."
            ),

            author="System",
        ).send()

    except asyncio.CancelledError:

        logger.info(
            "on_stop() was cancelled."
        )

        raise

    except Exception:

        logger.exception(
            "Error in on_stop()."
        )


# ============================================================================
# CHAT END
# ============================================================================

@cl.on_chat_end
async def on_chat_end():

    """
    Handle chat shutdown safely.

    Important:
        We cancel active work, but we DO NOT attempt to save
        SelectorGroupChat state while run_stream() may still be active.

    Concurrent save_state() during a cancelled workflow can create
    unstable behavior depending on the AutoGen runtime version.
    """

    try:

        logger.info(
            "on_chat_end() fired."
        )

        token = cl.user_session.get(
            "cancellation_token"
        )

        safe_cancel_token(
            token,
            "on_chat_end",
        )

        termination_ext = (
            cl.user_session.get(
                "termination_ext"
            )
        )

        await safe_set_external_termination(
            termination_ext,
            "on_chat_end",
        )

        is_processing = (
            cl.user_session.get(
                "is_processing",
                False,
            )
        )

        # --------------------------------------------------------------------
        # Do not save while workflow is active.
        # --------------------------------------------------------------------

        if is_processing:

            logger.info(
                "Chat ended while workflow was active. "
                "Cancellation requested; state save skipped."
            )

            return

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

            logger.info(
                "Final state saved for '%s' / '%s'.",
                username,
                thread_id,
            )

        else:

            logger.info(
                "Chat closed without workflow state."
            )

    except asyncio.CancelledError:

        logger.info(
            "on_chat_end() was cancelled."
        )

        raise

    except Exception:

        logger.exception(
            "Error handling chat end."
        )


# ============================================================================
# SETTINGS
# ============================================================================

@cl.on_settings_update
async def setup_agent_settings(
    settings,
):

    """
    Handle settings updates.
    """

    try:

        logger.info(
            "Settings updated: %s",
            settings,
        )

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

        try:

            await cl.Message(

                content=(
                    "⚠️ **Settings update failed.**"
                ),

                author="System",
            ).send()

        except Exception:

            logger.exception(
                "Failed to notify user about settings error."
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
        "--host 0.0.0.0 --port 8000"
    )
