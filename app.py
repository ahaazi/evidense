"""Evidence — a minimal front end for the Azure AI Foundry clinical agent.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

import agent_client as backend
import citations
from parsing import documents_html, parse_answer, references_html, to_markdown

ASSETS = Path(__file__).parent / "assets"

# EXAMPLES = [
#     "What are the major drug interactions between paxlovid (nirmatrelvir/ritonavir) and common cardiovascular medications?",
#     "What is the recommended anticoagulation regimen for a pregnant patient with a mechanical heart valve?",
#     "Which antihypertensive drug class is first-line for a patient with hypertension, type 2 diabetes, and persistent microalbuminuria?",
#     "When should renal artery stenosis be suspected in a patient with worsening hypertension after initiating an ACE inhibitor?",
# ]

st.set_page_config(
    page_title="Evidense",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ----------------------------------------------------------------------------- setup
@st.cache_data
def load_css() -> str:
    return (ASSETS / "styles.css").read_text()


@st.cache_resource(show_spinner=False)
def get_agent():
    """Connect once per session. Returns (node, error_message)."""
    if backend.demo_mode():
        return None, None
    try:
        return backend.build_agent_node(), None
    except backend.AgentError as exc:
        return None, str(exc)


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

state = st.session_state
state.setdefault("thread", [])
state.setdefault("pending", None)
state.setdefault("conversation_id", None)

agent_node, connect_error = get_agent()
is_demo = backend.demo_mode()


# ----------------------------------------------------------------------------- render
def masthead() -> None:
    label = "sample data" if is_demo else backend.agent_label()
    st.markdown(
        f'<div class="masthead"><div class="wordmark">Evid<em>ence</em></div>'
        f'<div class="agent">{html.escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        '<div class="hero">'
        '<div class="eyebrow">Clinical evidence synthesis</div>'
        "<h1>Ask a question.<br>Read the <em>evidence</em>.</h1>"
        "<p>Answers are assembled from guideline documents in your knowledge base, "
        "with every claim traced back to the passage it came from.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if is_demo:
        st.markdown(
            '<div class="notice"><b>Sample mode.</b> No Azure project is configured, so '
            "the interface is showing stored answers — real responses, citations and all. "
            "Add <code>AZURE_AI_PROJECT_ENDPOINT</code> and <code>AGENT_NAME</code> to "
            "<code>.env</code> to query your own agent.</div>",
            unsafe_allow_html=True,
        )
    elif connect_error:
        st.markdown(
            f'<div class="notice"><b>Not connected.</b> {html.escape(connect_error)}</div>',
            unsafe_allow_html=True,
        )

    # st.markdown('<div class="prompts-label">Try one of these</div>', unsafe_allow_html=True)
    # for index, example in enumerate(EXAMPLES):
    #     if st.button(example, key=f"example-{index}"):
    #         state.pending = example
    #         st.rerun()


def render_meta(reply: backend.AgentReply, cited: int, retrieved: int) -> str:
    bits = []
    if reply.model:
        bits.append(reply.model)
    if cited:
        bits.append(f"{cited} citation{'s' if cited != 1 else ''}")
    if retrieved:
        bits.append(f"{retrieved} passages searched")
    if reply.tokens:
        bits.append(f"{reply.tokens:,} tokens")
    if reply.agent_name:
        bits.append(reply.agent_name)
    return "".join(f"<span>{html.escape(str(bit))}</span>" for bit in bits)


def render_turn(
    turn: dict,
    first: bool,
    citation_map: dict[str, int],
    passages: dict[int, citations.Passage],
) -> None:
    answer = turn["answer"]
    reply: backend.AgentReply = turn["reply"]
    evidence: citations.Evidence = turn["evidence"]

    if not first:
        st.markdown('<div class="turn-rule"></div>', unsafe_allow_html=True)

    cited = len(set(citation_map.values()))
    st.markdown(
        f'<h2 class="question">{html.escape(turn["question"])}</h2>'
        f'<div class="meta">{render_meta(reply, cited, evidence.retrieved)}</div>',
        unsafe_allow_html=True,
    )

    if reply.citation_error:
        st.markdown(
            '<div class="notice"><b>Sources unavailable.</b> The answer came through, but '
            "the passages behind its citations could not be read back. "
            f"<code>{html.escape(reply.citation_error[:200])}</code></div>",
            unsafe_allow_html=True,
        )

    for section in answer.sections:
        heading = ""
        if section.title:
            heading = (
                f'<div class="sec-label">{html.escape(section.number or "")}</div>'
                f'<h3 class="sec-title">{html.escape(section.title)}</h3>\n\n'
            )
        body = to_markdown(section.body, citation_map, passages)
        # Blank lines are required so the markdown parser closes each HTML block
        # and still renders the bullet lists inside the wrapper.
        st.markdown(
            f'<div class="answer">\n\n{heading}{body}\n\n</div>',
            unsafe_allow_html=True,
        )


def sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="eyebrow">Session</div>', unsafe_allow_html=True)
        if st.button("Start a new session", key="reset"):
            state.thread = []
            state.conversation_id = None
            state.pending = None
            st.rerun()

        if state.thread:
            st.markdown('<div class="prompts-label">Asked</div>', unsafe_allow_html=True)
            for turn in state.thread:
                question = turn["question"]
                short = question if len(question) <= 70 else question[:70].rstrip() + "…"
                st.markdown(f'<div class="ref-detail">{html.escape(short)}</div>',
                            unsafe_allow_html=True)

        if state.conversation_id:
            st.markdown(
                f'<div class="prompts-label">Conversation</div>'
                f'<div class="chunk">{html.escape(state.conversation_id)}</div>',
                unsafe_allow_html=True,
            )


def references(passages: list[citations.Passage]) -> None:
    """The evidence behind every citation on the page, numbered once."""
    documents = citations.documents_of(passages)
    count = len(passages)
    documents_count = len(documents)
    summary = (
        f"{count} passage{'s' if count != 1 else ''} · "
        f"{documents_count} document{'s' if documents_count != 1 else ''}"
    )
    st.markdown(
        f'<div class="refs-head"><span>References</span><span>{summary}</span></div>'
        f"{documents_html(documents)}{references_html(passages)}",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- flow
masthead()
sidebar()

# Read chat input before the hero gate below so a freshly submitted question
# hides the example prompts on this same render, instead of one rerun later.
asked = st.chat_input(
    "Ask a clinical question…" if not state.thread else "Ask a follow-up…"
)
if asked:
    state.pending = asked.strip()
    st.rerun()

if not state.thread and not state.pending:
    hero()

cited_passages, turn_citation_maps = citations.build_index(
    [(turn["answer"].body, turn["evidence"]) for turn in state.thread]
)
passages_by_number = {p.number: p for p in cited_passages if p.number}

for index, turn in enumerate(state.thread):
    render_turn(
        turn,
        first=index == 0,
        citation_map=turn_citation_maps[index],
        passages=passages_by_number,
    )

if state.pending:
    question = state.pending
    if state.thread:
        st.markdown('<div class="turn-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<h2 class="question">{html.escape(question)}</h2>',
        unsafe_allow_html=True,
    )

    try:
        if is_demo:
            with st.spinner("Searching the knowledge base…"):
                reply = backend.ask_demo(question)
        elif agent_node is None:
            raise backend.AgentError(connect_error or "The agent is not connected.")
        else:
            result = backend.StreamResult()
            st.write_stream(backend.ask_stream(agent_node, question, state.conversation_id, result))
            if result.error:
                raise result.error
            reply = result.reply
    except backend.AgentError as exc:
        state.pending = None
        st.markdown(
            f'<div class="notice"><b>The question did not go through.</b> '
            f"{html.escape(str(exc))}</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    state.conversation_id = reply.conversation_id or state.conversation_id
    state.thread.append(
        {
            "question": question,
            "reply": reply,
            "answer": parse_answer(reply.text),
            "evidence": citations.extract_evidence(reply.output_items),
        }
    )
    state.pending = None
    st.rerun()

if state.thread:
    if cited_passages:
        references(cited_passages)
    st.markdown(
        '<div class="disclaimer">Answers are generated from the documents in your '
        "knowledge base and can be incomplete or out of date. Check the cited passages "
        "before acting on anything clinical.</div>",
        unsafe_allow_html=True,
    )
