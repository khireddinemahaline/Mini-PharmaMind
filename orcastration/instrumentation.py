#!/usr/bin/env python3
"""Arize / OpenInference instrumentation bootstrap

Provides a single place to register Arize tracing and instrument AutoGen AgentChat.
This module follows the project's preferred template and reads credentials from env.
"""
import os

from arize.otel import register
from openinference.instrumentation.autogen_agentchat import (
    AutogenAgentChatInstrumentor,
)


tracer_provider = register(
    space_id=os.environ["ARIZE_SPACE_ID"],
    api_key=os.environ["ARIZE_API_KEY"],
    project_name=os.environ.get("ARIZE_PROJECT_NAME", "pharma-mind"),
)

AutogenAgentChatInstrumentor().instrument(tracer_provider=tracer_provider)
print("Arize AX tracing initialized for AutoGen AgentChat.")
