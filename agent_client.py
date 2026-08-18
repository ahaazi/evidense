"""Thin wrapper around the Azure AI Foundry agent node.

Keeps every Azure detail out of the UI layer. Also provides a demo mode so the
interface can be run and reviewed without Azure credentials.

An answer arrives in two parts. The agent node returns the prose, which carries
citation markers like ``【6:0†source】`` but nothing behind them: the langchain
node keeps only the final message. The evidence itself — which document each
marker came from, and the passage that was retrieved — lives in the tool calls
of the same response, so :func:`ask` fetches the response by id and hands the
raw output items back with the reply for :mod:`citations` to resolve.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEMO_DIR = Path(__file__).parent / "assets" / "demo"
DEFAULT_DEMO = "hypertension-pregnancy"


class AgentError(RuntimeError):
    """Raised when the agent cannot be reached or returns nothing usable."""


@dataclass
class AgentReply:
    text: str
    model: str | None = None
    tokens: int | None = None
    agent_name: str | None = None
    conversation_id: str | None = None
    response_id: str | None = None
    output_items: list = field(default_factory=list)
    """Raw output items of the response — the tool calls the citations come from."""

    citation_error: str | None = None
    """Set when the prose arrived but its evidence could not be fetched."""


@dataclass
class StreamResult:
    """Out-of-band result for :func:`ask_stream`.

    A generator can't both yield incremental text and return a value, so
    ``ask_stream`` writes its outcome here instead — read ``reply`` or
    ``error`` only after the generator is fully drained.
    """

    reply: AgentReply | None = None
    error: AgentError | None = None


def demo_mode() -> bool:
    if os.environ.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return not (os.environ.get("AZURE_AI_PROJECT_ENDPOINT") and os.environ.get("AGENT_NAME"))


def agent_label() -> str:
    return os.environ.get("AGENT_NAME", "demo agent")


def build_agent_node():
    """Create the agent node. Raises AgentError with an actionable message."""
    from azure.core.exceptions import ClientAuthenticationError
    from azure.identity import DefaultAzureCredential

    from langchain_azure_ai.agents import AgentServiceFactory

    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    name = os.environ.get("AGENT_NAME")
    version = os.environ.get("AGENT_VERSION", "latest")

    missing = [
        key
        for key, value in (("AZURE_AI_PROJECT_ENDPOINT", endpoint), ("AGENT_NAME", name))
        if not value
    ]
    if missing:
        raise AgentError(
            "Add " + " and ".join(missing) + " to your .env file, then reload."
        )

    credential = DefaultAzureCredential()
    try:
        credential.get_token("https://management.azure.com/.default")
    except ClientAuthenticationError as exc:
        raise AgentError(
            "Azure sign-in failed. Run `az login`, or set AZURE_CLIENT_ID, "
            f"AZURE_TENANT_ID, and AZURE_CLIENT_SECRET in .env. ({exc})"
        ) from exc

    factory = AgentServiceFactory(project_endpoint=endpoint, credential=credential)
    try:
        return factory.get_agent_node(name=name, version=version)
    except ValueError as exc:
        raise AgentError(
            f"No agent named '{name}' (version {version}) in this project. "
            f"Check the name and version in the Foundry portal. ({exc})"
        ) from exc


@lru_cache(maxsize=1)
def _responses_api():
    """The Responses client for the same project, used to read back citations."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    client = AIProjectClient(
        endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    return client.get_openai_client().responses


def fetch_output_items(response_id: str) -> list:
    """Read back one response's output items, tool calls included.

    The agent node reports only the assistant's text, so the retrieval that
    produced the citations has to be fetched separately. This is a metadata
    read against a response that already exists — no model work is repeated.
    """
    response = _responses_api().retrieve(response_id)
    items = getattr(response, "output", None) or []
    return [item.model_dump() if hasattr(item, "model_dump") else item for item in items]


def fetch_response_details(response_id: str) -> tuple[list, str | None, int | None]:
    """Like :func:`fetch_output_items`, but also returns model and token usage.

    The streaming path's underlying ``_stream()`` implementation doesn't
    carry model/usage the way a plain ``invoke()`` does, so :func:`ask_stream`
    reads it back from the same response used for citations, in one call.
    """
    response = _responses_api().retrieve(response_id)
    items = getattr(response, "output", None) or []
    items = [item.model_dump() if hasattr(item, "model_dump") else item for item in items]
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "total_tokens", None) if usage else None
    return items, getattr(response, "model", None), tokens


def ask(agent_node, question: str, conversation_id: str | None = None) -> AgentReply:
    """Send one question, optionally continuing an existing conversation."""
    from langchain_core.messages import HumanMessage

    payload: dict = {"messages": [HumanMessage(content=question)]}
    if conversation_id:
        payload["azure_ai_agents_conversation_id"] = conversation_id

    try:
        response = agent_node.invoke(payload)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        if conversation_id:
            # The thread may have expired; retry once as a fresh conversation.
            try:
                response = agent_node.invoke({"messages": payload["messages"]})
            except Exception as retry_exc:  # noqa: BLE001
                raise AgentError(str(retry_exc)) from retry_exc
        else:
            raise AgentError(str(exc)) from exc

    reply = _to_reply(response)

    if reply.response_id:
        try:
            reply.output_items = fetch_output_items(reply.response_id)
        except Exception as exc:  # noqa: BLE001 - an answer without sources still reads
            reply.citation_error = str(exc)

    return reply


# ------------------------------------------------------------------------ streaming
@lru_cache(maxsize=8)
def _stream_graph(agent_node):
    """Wrap the agent node in a one-node graph so tokens can be streamed.

    The node's own ``invoke`` only routes to the underlying chat model's
    ``_stream()`` when a LangGraph-recognized streaming handler is attached
    to the run — that handler is wired up internally by a compiled graph's
    ``stream(..., stream_mode="messages")``, not by calling the node
    directly. Wrapping it here is the supported way to get real token
    streaming out of it.
    """
    from langgraph.graph import END, StateGraph

    from langchain_azure_ai.agents import AgentServiceAgentState

    builder = StateGraph(AgentServiceAgentState)
    builder.add_node("agent", agent_node)
    builder.set_entry_point("agent")
    builder.add_edge("agent", END)
    return builder.compile()


def _stream_once(graph, question: str, conversation_id: str | None) -> Iterator[str]:
    """Run the graph once, yielding text deltas; returns ``(full_text, update)``."""
    from langchain_core.messages import AIMessageChunk, HumanMessage

    payload: dict = {"messages": [HumanMessage(content=question)]}
    if conversation_id:
        payload["azure_ai_agents_conversation_id"] = conversation_id

    text = ""
    update: dict = {}
    for mode, data in graph.stream(payload, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk, _metadata = data
            delta = chunk.content if isinstance(chunk, AIMessageChunk) else None
            if isinstance(delta, str) and delta:
                text += delta
                yield delta
        elif mode == "updates":
            agent_update = data.get("agent")
            if agent_update:
                update = agent_update

    return text, update


def ask_stream(
    agent_node, question: str, conversation_id: str | None, result: StreamResult
) -> Iterator[str]:
    """Send one question, yielding the answer text as it streams in.

    Mirrors :func:`ask`, but yields incremental text deltas instead of
    returning once at the end. The final :class:`AgentReply` (or an
    :class:`AgentError` on failure) is written to ``result`` — read it only
    once this generator has been fully drained.
    """
    graph = _stream_graph(agent_node)

    try:
        text, update = yield from _stream_once(graph, question, conversation_id)
    except Exception as exc:  # noqa: BLE001
        if not conversation_id:
            result.error = AgentError(str(exc))
            return
        # The thread may have expired; retry once as a fresh conversation.
        try:
            text, update = yield from _stream_once(graph, question, None)
        except Exception as retry_exc:  # noqa: BLE001
            result.error = AgentError(str(retry_exc))
            return

    if not text:
        result.error = AgentError("The agent returned an empty answer. Try rephrasing the question.")
        return

    reply = AgentReply(
        text=text,
        conversation_id=update.get("azure_ai_agents_conversation_id") or conversation_id,
        response_id=update.get("azure_ai_agents_previous_response_id"),
    )
    if reply.response_id:
        try:
            reply.output_items, reply.model, reply.tokens = fetch_response_details(
                reply.response_id
            )
        except Exception as exc:  # noqa: BLE001 - an answer without sources still reads
            reply.citation_error = str(exc)

    result.reply = reply


# ----------------------------------------------------------------------- demo mode
@lru_cache(maxsize=1)
def _demo_fixtures() -> dict[str, dict]:
    """Stored real responses, so sample mode exercises the same code path."""
    fixtures: dict[str, dict] = {}
    for path in sorted(DEMO_DIR.glob("*.json")):
        try:
            fixtures[path.stem] = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return fixtures


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3}


def ask_demo(question: str) -> AgentReply:
    """Answer from the closest stored response rather than calling Azure."""
    fixtures = _demo_fixtures()
    if not fixtures:
        raise AgentError(
            "Sample mode needs the stored responses in assets/demo/, and none were found."
        )

    asked = _words(question)
    slug = max(
        fixtures,
        key=lambda key: len(asked & _words(fixtures[key].get("question", ""))),
    )
    if not asked & _words(fixtures[slug].get("question", "")):
        slug = DEFAULT_DEMO if DEFAULT_DEMO in fixtures else next(iter(fixtures))

    fixture = fixtures[slug]
    output = fixture.get("output") or []
    text = ""
    for item in output:
        if item.get("type") == "message":
            text = "\n".join(
                part.get("text", "") for part in item.get("content") or []
            ).strip()

    return AgentReply(
        text=text,
        model=fixture.get("model", "sample"),
        tokens=fixture.get("tokens"),
        agent_name="",
        conversation_id="conv_demo_local",
        response_id=fixture.get("response_id"),
        output_items=output,
    )


def _to_reply(response) -> AgentReply:
    message = response.get("messages") if isinstance(response, dict) else None
    if isinstance(message, list):
        message = message[-1] if message else None

    text = getattr(message, "content", None)
    if isinstance(text, list):  # content blocks
        text = "\n".join(
            block.get("text", "") for block in text if isinstance(block, dict)
        )
    if not text:
        raise AgentError("The agent returned an empty answer. Try rephrasing the question.")

    metadata = getattr(message, "response_metadata", {}) or {}
    return AgentReply(
        text=text,
        model=metadata.get("model"),
        tokens=metadata.get("token_usage"),
        agent_name=getattr(message, "name", None),
        conversation_id=response.get("azure_ai_agents_conversation_id"),
        response_id=response.get("azure_ai_agents_previous_response_id"),
    )
