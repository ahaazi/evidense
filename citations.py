"""Resolve the agent's citation markers to the passages they actually came from.

The agent answers with markers like ``【6:0†source】`` sitting next to each claim.
On their own they are opaque. The evidence behind them travels separately, in the
knowledge-base tool call that the same response carries: for every marker the tool
returns the document it came from and the verbatim passage that was retrieved.

This module joins the two halves back together, so a claim in the prose can be
traced to a named document, a page, and the exact text the model was reading:

    ``【6:0†source】`` -> Passage(document="Guideline for the pharmacological
                                treatment of hypertension in adults",
                                page="23", text="While BP treatment thresholds…")

Markers are numbered per *response*, so ``6:0`` in one turn is unrelated to
``6:0`` in the next. Resolution is therefore always scoped to the turn that
produced the marker; :func:`build_index` handles that bookkeeping and hands the
UI a stable, session-wide numbering.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlparse

REGISTRY_PATH = Path(__file__).parent / "assets" / "sources.json"

# 【6:0†source】 — the pair identifies one retrieved passage within one response.
MARKER_RE = re.compile(r"【\s*(\d+)\s*:\s*(\d+)\s*†[^】]*】")

_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_DANGLING_COMMENT_RE = re.compile(r"<!--[^>]*$")
_PAGE_NUMBER_RE = re.compile(r"PageNumber:\s*([^\s<>-]+)")
_PAGE_LABEL_RE = re.compile(r"Page(?:Footer|Header):\s*(.+?)\s*$", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[(.*?)\]\([^)]*\)", re.DOTALL)
_HEADING_RE = re.compile(r"^[ \t]{0,3}#+[ \t]*", re.MULTILINE)
_TAG_RE = re.compile(r"<[^>]+>")
_DANGLING_TAG_RE = re.compile(r"<[^>]*$")
# The retrieval tool appends this when it cuts its own output short.
_VISIBLE_MARKER_RE = re.compile(r"\s*Visible:\s*\d+%\s*-\s*\d+%\s*$")
_BLANKS_RE = re.compile(r"\n{3,}")
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
_SECTION_SUFFIX_RE = re.compile(r"_(?:text_)?sections?_(\d+)$")

PREVIEW_CHARS = 280


# ----------------------------------------------------------------- document metadata
@dataclass(frozen=True)
class Document:
    """A source document in the knowledge base."""

    key: str
    title: str
    collection: str = ""
    publisher: str = ""
    year: str = ""
    url: str = ""
    blob_url: str = ""

    @property
    def imprint(self) -> str:
        """Publisher and year, as they would read in a citation."""
        return " · ".join(part for part in (self.publisher, self.year) if part)

    @property
    def host(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.") if self.url else ""


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, str]]:
    """Read the document registry, ignoring ``_comment``-style keys."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if not key.startswith("_") and isinstance(value, dict)
    }


def _document_key(blob_url: str, uid: str) -> str:
    """The file name is what identifies a document; fall back to the uid."""
    if blob_url:
        name = unquote(urlparse(blob_url).path).rsplit("/", 1)[-1]
        if name:
            return name
    return uid.split("_")[0] if uid else "unknown"


def _title_from_filename(key: str) -> str:
    stem = re.sub(r"\.(pdf|docx?|txt|md|html?)$", "", key, flags=re.IGNORECASE)
    stem = re.sub(r"[-_]eng$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem or key


def _title_from_running_heads(labels: Iterable[str]) -> str:
    """Infer a document title from the running head repeated across its pages.

    Documents converted from PDF keep their page furniture as ``PageFooter``/
    ``PageHeader`` comments, and the running head is usually the title. Dates,
    footnote spill and stray sentences show up there too, so candidates have to
    look like a title before they are trusted.
    """
    counts: dict[str, int] = {}
    for label in labels:
        text = re.sub(r"\s+", " ", label).strip(" .;,")
        if not (8 <= len(text) <= 110):
            continue
        if not re.match(r"^[A-Za-z(]", text):  # footnote markers, bullets, numbers
            continue
        if re.match(r"^\d", text) or re.search(r"\b(19|20)\d{2}$", text):  # dates
            continue
        if text.count(".") > 2 or text.endswith("."):  # prose that spilled over
            continue
        counts[text] = counts.get(text, 0) + 1

    if not counts:
        return ""
    # Prefer the most repeated running head; break ties on the longer one.
    best = max(counts.items(), key=lambda item: (item[1], len(item[0])))
    return best[0]


def _resolve_document(
    key: str,
    blob_url: str,
    running_heads: Sequence[str],
    registry: dict[str, dict[str, str]],
) -> Document:
    entry = registry.get(key, {})
    title = entry.get("title") or _title_from_running_heads(running_heads)
    if not title:
        title = _title_from_filename(key)
    elif not entry and title.isupper():
        # Running heads are often set in caps; sentence case reads better.
        title = title[0] + title[1:].lower()
    return Document(
        key=key,
        title=title,
        collection=entry.get("collection", ""),
        publisher=entry.get("publisher", ""),
        year=entry.get("year", ""),
        url=entry.get("url", ""),
        blob_url=blob_url,
    )


# ----------------------------------------------------------------- passage cleaning
def _strip_comments(text: str) -> tuple[str, list[str], str]:
    """Pull page furniture out of a snippet.

    Returns the text without comments, the running heads it carried, and the
    first page number found in it.
    """
    heads: list[str] = []
    page = ""

    for body in _COMMENT_RE.findall(text):
        number = _PAGE_NUMBER_RE.search(body)
        if number and not page:
            page = number.group(1).strip(" .")
        label = _PAGE_LABEL_RE.search(body.strip())
        if label:
            heads.append(label.group(1))

    cleaned = _COMMENT_RE.sub("", text)
    # Conversions sometimes truncate mid-comment; drop the orphaned opener.
    cleaned = _DANGLING_COMMENT_RE.sub("", cleaned)
    return cleaned, heads, page


def _tables_to_text(text: str) -> str:
    """Flatten HTML tables into readable rows instead of dropping them."""
    if "<t" not in text.lower():
        return text
    out = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    out = re.sub(r"</\s*(?:tr|table)\s*>", "\n", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*(?:td|th)[^>]*>", " | ", out, flags=re.IGNORECASE)
    out = re.sub(r"</\s*(?:td|th)\s*>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"<\s*/?(?:table|thead|tbody|tfoot|tr)[^>]*>", "", out, flags=re.IGNORECASE)
    # Cells usually sit on their own source lines; pull each row back onto one.
    out = re.sub(r"\n[ \t]*\|", " |", out)
    out = re.sub(r"^[ \t]*\|[ \t]*", "", out, flags=re.MULTILINE)
    return out


def _figure_caption(match: re.Match[str]) -> str:
    """Keep a figure's description — it is often the whole content of the page."""
    caption = match.group(1).strip()
    return f"Figure — {caption}" if caption else ""


def clean_snippet(text: str) -> tuple[str, list[str], str]:
    """Turn a raw retrieved snippet into readable prose.

    Returns the cleaned text, the running heads it carried (used to name the
    document) and the page number it came from.
    """
    body, heads, page = _strip_comments(text or "")
    body = _IMAGE_RE.sub(_figure_caption, body)
    body = _tables_to_text(body)
    body = _TAG_RE.sub("", body)
    body = _DANGLING_TAG_RE.sub("", body)
    body = _VISIBLE_MARKER_RE.sub("", body)
    body = _HEADING_RE.sub("", body)
    # The converter escapes characters that would otherwise read as markdown;
    # nothing downstream treats this text as markdown, so put them back.
    body = _MARKDOWN_ESCAPE_RE.sub(r"\1", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = _BLANKS_RE.sub("\n\n", body)
    return body.strip(), heads, page


def preview_of(text: str, limit: int = PREVIEW_CHARS) -> str:
    """A short lead-in to a passage, cut at a word boundary."""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= limit:
        return flat
    head = flat[:limit].rsplit(" ", 1)[0].rstrip(" .,;:|")
    return head + "…"


# ----------------------------------------------------------------- passages
@dataclass
class Passage:
    """One retrieved chunk, tied to the marker the agent cited it with."""

    marker: str
    uid: str
    document: Document
    text: str
    page: str = ""
    section: str = ""
    raw: str = ""
    truncated: bool = False
    number: int | None = None

    @property
    def preview(self) -> str:
        return preview_of(self.text)

    @property
    def has_more(self) -> bool:
        """Whether the passage runs past what the preview shows."""
        return len(re.sub(r"\s+", " ", self.text).strip()) > PREVIEW_CHARS

    @property
    def locator(self) -> str:
        """Where in the document this passage sits."""
        if self.page:
            return f"p. {self.page}"
        return f"section {self.section}" if self.section else ""


@dataclass
class Evidence:
    """Everything one response retrieved, keyed by citation marker."""

    passages: dict[str, Passage] = field(default_factory=dict)
    retrieved: int = 0

    def get(self, marker: str) -> Passage | None:
        return self.passages.get(marker)

    def __bool__(self) -> bool:
        return bool(self.passages)


def _iter_tool_outputs(output_items: Any) -> Iterable[str]:
    """Yield the text output of every knowledge-base tool call in a response."""
    for item in output_items or []:
        get = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)
        if get("type") not in {"mcp_call", "function_call_output", "tool_call"}:
            continue
        output = get("output")
        if isinstance(output, str) and output.strip():
            yield output


def _decode_json_string(fragment: str) -> str:
    """Decode the body of a JSON string that was cut off before its closing quote."""
    fragment = fragment.rstrip()
    for end in range(len(fragment), 0, -1):
        try:
            return json.loads(f'"{fragment[:end]}"', strict=False)
        except ValueError:
            continue
    return fragment


def _salvage_record(fragment: str) -> dict[str, Any]:
    """Recover what is readable from a record the tool truncated mid-write.

    The retrieval tool caps the size of its output, so its last record usually
    stops partway through the passage. Reading the fields out by hand keeps that
    citation attached to its document instead of discarding it.
    """
    record: dict[str, Any] = {"truncated": True}
    for field_name in ("uid", "blob_url"):
        found = re.search(rf'"{field_name}"\s*:\s*"([^"]*)"', fragment)
        if found:
            record[field_name] = found.group(1)

    snippet = re.search(r'"snippet"\s*:\s*"', fragment)
    if snippet:
        record["snippet"] = _decode_json_string(fragment[snippet.end() :])
    else:  # not even a snippet field survived
        record["snippet"] = ""
        record["truncated"] = bool(record.get("uid"))
    return record


def _parse_tool_output(output: str) -> list[tuple[str, dict[str, Any]]]:
    """Split one tool output into ``(marker, record)`` pairs.

    The tool writes a marker followed by a JSON object per retrieved passage.
    Records are read one after another rather than by scanning for every marker,
    so a marker quoted inside a passage cannot invent a record of its own. The
    decoder is lenient because retrieved text carries raw control characters
    that strict JSON rejects, and anything that still will not decode is kept as
    a plain snippet — a change in the tool's shape degrades to "quote what we
    got" rather than losing the citation.
    """
    records: list[tuple[str, dict[str, Any]]] = []
    decoder = json.JSONDecoder(strict=False)
    position = 0

    while True:
        match = MARKER_RE.search(output, position)
        if not match:
            return records

        marker = f"{match.group(1)}:{match.group(2)}"
        start = match.end()
        start += len(output[start:]) - len(output[start:].lstrip())

        try:
            value, position = decoder.raw_decode(output, start)
        except ValueError:
            following = MARKER_RE.search(output, start)
            position = following.start() if following else len(output)
            value = _salvage_record(output[start:position])

        if not isinstance(value, dict):
            value = {"snippet": str(value)}
        records.append((marker, value))


def extract_evidence(
    output_items: Any, registry: dict[str, dict[str, str]] | None = None
) -> Evidence:
    """Build the marker → passage map for one agent response."""
    registry = load_registry() if registry is None else registry

    raw_records: list[tuple[str, dict[str, Any], str, list[str], str]] = []
    heads_by_document: dict[str, list[str]] = {}
    blob_by_document: dict[str, str] = {}

    for output in _iter_tool_outputs(output_items):
        for marker, record in _parse_tool_output(output):
            text, heads, page = clean_snippet(str(record.get("snippet", "")))
            uid = str(record.get("uid", ""))
            blob_url = str(record.get("blob_url", ""))
            if not (text or uid or blob_url):
                # Nothing survived here — usually the tail end of a truncated
                # output. Leaving it out makes the citation read as unresolved
                # rather than as a source with nothing behind it.
                continue
            key = _document_key(blob_url, uid)
            heads_by_document.setdefault(key, []).extend(heads)
            blob_by_document.setdefault(key, blob_url)
            raw_records.append((marker, record, text, [key, uid, blob_url], page))

    documents = {
        key: _resolve_document(key, blob_by_document.get(key, ""), heads, registry)
        for key, heads in heads_by_document.items()
    }

    passages: dict[str, Passage] = {}
    for marker, record, text, (key, uid, _blob), page in raw_records:
        section = ""
        section_match = _SECTION_SUFFIX_RE.search(uid)
        if section_match:
            section = section_match.group(1)
        passages[marker] = Passage(
            marker=marker,
            uid=uid or f"{key}#{marker}",
            document=documents[key],
            text=text,
            page=page,
            section=section,
            raw=str(record.get("snippet", "")),
            truncated=bool(record.get("truncated")),
        )

    return Evidence(passages=passages, retrieved=len(passages))


# ----------------------------------------------------------------- session numbering
def build_index(
    turns: Sequence[tuple[str, Evidence]],
) -> tuple[list[Passage], list[dict[str, int]]]:
    """Number every cited passage once, in the order a reader meets it.

    ``turns`` is the conversation so far as ``(answer text, evidence)`` pairs.
    Returns the numbered passages for the reference list, plus one
    marker → number map per turn for rendering that turn's prose.

    A passage keeps its number wherever it is cited again, so the reference
    list stays short and a number means the same thing all the way down the
    page. Markers a turn cannot resolve are looked up in earlier turns, which
    is what a follow-up answer that reuses the previous turn's retrieval needs.
    """
    ordered: list[Passage] = []
    number_by_uid: dict[str, int] = {}
    maps: list[dict[str, int]] = []

    for position, (text, evidence) in enumerate(turns):
        marker_map: dict[str, int] = {}
        for match in MARKER_RE.finditer(text or ""):
            marker = f"{match.group(1)}:{match.group(2)}"
            if marker in marker_map:
                continue

            passage = evidence.get(marker) if evidence else None
            if passage is None:  # a follow-up citing the previous turn's evidence
                for earlier_text, earlier in reversed(turns[:position]):
                    passage = earlier.get(marker) if earlier else None
                    if passage is not None:
                        break
            if passage is None:
                continue

            number = number_by_uid.get(passage.uid)
            if number is None:
                number = len(ordered) + 1
                number_by_uid[passage.uid] = number
                numbered = Passage(**{**passage.__dict__, "number": number})
                ordered.append(numbered)
            marker_map[marker] = number
        maps.append(marker_map)

    return ordered, maps


def documents_of(passages: Sequence[Passage]) -> list[Document]:
    """The distinct documents behind a set of passages, in citation order."""
    seen: dict[str, Document] = {}
    for passage in passages:
        seen.setdefault(passage.document.key, passage.document)
    return list(seen.values())
