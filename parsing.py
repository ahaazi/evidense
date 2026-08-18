"""Turn the agent's raw markdown answer into the pieces the UI renders.

The agent returns one markdown string with two things tangled together:

  1. numbered top-level sections, e.g. ``1) Executive Summary``
  2. citation markers sitting next to the claims they support,
     e.g. ``【6:0†source】``

This module separates them, and swaps each marker for a numbered citation that
links to the passage behind it. Resolving markers to passages is
:mod:`citations`' job; this module only renders what it is handed.
"""

from __future__ import annotations

import html
import re

from citations import MARKER_RE, Document, Passage, preview_of

SECTION_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?([0-9]{1,2})[\).]\s+(\S.*?)\s*$", re.MULTILINE)
REFERENCES_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*)?(?:[0-9]{1,2}[\).]\s*)?(?:references|citations|sources)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
# What a line in a written-out reference list tends to look like.
ENTRY_RE = re.compile(r"^(?:[-*\u2022]|\(?\d{1,2}[\).]|\[\s*Source)\s*", re.IGNORECASE)
# A run of markers, plus whatever whitespace leads into it.
MARKER_RUN_RE = re.compile(r"[ \t]*(?:" + MARKER_RE.pattern + r"[ \t]*)+")

POPOVER_CHARS = 320


class Section:
    """One numbered part of an answer."""

    __slots__ = ("number", "title", "body")

    def __init__(self, number: str | None, title: str, body: str) -> None:
        self.number = number
        self.title = title
        self.body = body


class Answer:
    """A parsed answer: prose split into sections, with its markers noted."""

    __slots__ = ("sections", "raw", "body", "markers")

    def __init__(
        self, sections: list[Section], raw: str, body: str, markers: list[str]
    ) -> None:
        self.sections = sections
        self.raw = raw
        self.body = body
        """The prose actually rendered — what the citation markers are counted from."""
        self.markers = markers


def parse_answer(raw: str) -> Answer:
    """Split a raw agent answer into sections and collect its citation markers."""
    text = (raw or "").replace("\r\n", "\n").strip()
    body = _drop_written_reference_list(text)
    markers = [f"{f}:{c}" for f, c in MARKER_RE.findall(body)]
    return Answer(
        sections=_split_sections(body), raw=text, body=body, markers=markers
    )


def _drop_written_reference_list(text: str) -> str:
    """Drop a reference list the model wrote out in prose.

    References are built from the retrieval record rather than from what the
    model says it used, so a hand-written list at the end is a duplicate — and
    the less reliable of the two. What follows the heading has to actually read
    like a list of sources, so a section that merely discusses sources, or a
    heading in the middle of the answer, survives.
    """
    for match in reversed(list(REFERENCES_RE.finditer(text))):
        if match.start() < len(text) * 0.25:
            break
        lines = [line.strip() for line in text[match.end() :].split("\n") if line.strip()]
        if not lines:
            continue
        entries = sum(bool(ENTRY_RE.match(line)) for line in lines)
        if entries >= max(1, len(lines) * 0.6):
            return text[: match.start()].rstrip()
    return text


def _split_sections(body: str) -> list[Section]:
    matches = list(SECTION_RE.finditer(body))
    if not matches:
        return [Section(number=None, title="", body=body.strip())]

    sections: list[Section] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(number=None, title="", body=preamble))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections.append(
            Section(
                number=match.group(1),
                title=match.group(2).strip(),
                body=body[match.end() : end].strip(),
            )
        )
    return sections


# ----------------------------------------------------------------- prose
def to_markdown(
    text: str,
    citation_map: dict[str, int] | None = None,
    passages: dict[int, Passage] | None = None,
) -> str:
    """Normalise agent markdown and turn citation markers into numbered links.

    ``citation_map`` gives this turn's marker → reference number, and
    ``passages`` the passage behind each number, which becomes the preview a
    reader sees on hover. A marker with nothing behind it is rendered greyed
    out rather than dropped, so a gap in the evidence stays visible.
    """
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        # Agent uses '  • ' for second-level bullets; make it a real nested list.
        nested = re.match(r"^\s+[•●◦]\s+(.*)$", line)
        if nested:
            line = "  - " + nested.group(1)
        else:
            line = re.sub(r"^(\s*)[•●◦]\s+", r"\1- ", line)
        lines.append(line)

    out = "\n".join(lines)

    def render_run(match: re.Match[str]) -> str:
        numbers: list[int | None] = []
        for file_index, chunk_index in MARKER_RE.findall(match.group(0)):
            number = (citation_map or {}).get(f"{file_index}:{chunk_index}")
            if number not in numbers:  # the same source twice on one claim
                numbers.append(number)
        return "".join(_citation_html(number, passages) for number in numbers)

    return MARKER_RUN_RE.sub(render_run, out)


def _citation_html(number: int | None, passages: dict[int, Passage] | None) -> str:
    if number is None:
        return (
            '<span class="cite is-unmapped" '
            'title="This answer cited a passage that was not returned with it.">'
            "·</span>"
        )

    passage = (passages or {}).get(number)
    if passage is None:
        return f'<a class="cite" href="#ref-{number}">{number}</a>'

    document = passage.document
    where = " · ".join(part for part in (document.title, passage.locator) if part)
    excerpt = preview_of(passage.text, POPOVER_CHARS)

    return (
        '<span class="cite-wrap">'
        f'<a class="cite" href="#ref-{number}">{number}</a>'
        '<span class="cite-pop" role="note">'
        f'<span class="cite-pop-head">{html.escape(where)}</span>'
        f'<span class="cite-pop-body">{html.escape(excerpt)}</span>'
        "</span></span>"
    )


# ----------------------------------------------------------------- references
def references_html(passages: list[Passage]) -> str:
    """Render the reference list: one card per cited passage."""
    if not passages:
        return ""
    return f'<ol class="refs">{"".join(_reference_card(p) for p in passages)}</ol>'


def _reference_card(passage: Passage) -> str:
    document = passage.document
    number = passage.number

    meta = " · ".join(
        part
        for part in (document.collection, document.imprint, passage.locator)
        if part
    )
    quote = _quote_html(passage)
    link = ""
    if document.url:
        link = (
            f'<a class="ref-link" href="{html.escape(document.url, quote=True)}" '
            f'target="_blank" rel="noopener">{html.escape(document.host or "source")} ↗</a>'
        )
    elif document.blob_url:
        link = f'<span class="ref-file">{html.escape(document.key)}</span>'

    note = (
        '<span class="ref-note">excerpt cut short by the retrieval tool</span>'
        if passage.truncated
        else ""
    )
    footer = (
        f'<div class="ref-foot">{link}{note}</div>' if (link or note) else ""
    )

    anchor = f' id="ref-{number}"' if number else ""
    return (
        f'<li class="ref"{anchor}>'
        f'<span class="ref-num">{number if number else "–"}</span>'
        '<div class="ref-body">'
        f'<p class="ref-title">{html.escape(document.title)}</p>'
        + (f'<p class="ref-meta">{html.escape(meta)}</p>' if meta else "")
        + quote
        + footer
        + "</div></li>"
    )


def _quote_html(passage: Passage) -> str:
    """The retrieved passage itself — short by default, in full on request."""
    if not passage.text:
        return ""

    preview = html.escape(passage.preview)
    if not passage.has_more:
        return f'<span class="ref-quote">{preview}</span>'

    # Line breaks inside a paragraph are where the PDF wrapped, not where the
    # writer broke; only the paragraph breaks are worth keeping.
    paragraphs = [
        html.escape(" ".join(block.split()))
        for block in passage.text.split("\n\n")
        if block.strip()
    ]
    return (
        '<details class="ref-details">'
        f'<summary><span class="ref-quote">{preview}</span>'
        '<span class="ref-more">Read the full passage</span></summary>'
        f'<blockquote class="ref-quote is-full">{"<br><br>".join(paragraphs)}</blockquote>'
        "</details>"
    )


def documents_html(documents: list[Document]) -> str:
    """A one-line summary of which documents an answer drew on."""
    if not documents:
        return ""
    chips = "".join(
        (
            f'<a class="doc-chip" href="{html.escape(doc.url, quote=True)}" '
            f'target="_blank" rel="noopener">{html.escape(doc.title)}</a>'
            if doc.url
            else f'<span class="doc-chip">{html.escape(doc.title)}</span>'
        )
        for doc in documents
    )
    return f'<div class="doc-chips">{chips}</div>'
