# Evidence

A minimal Streamlit front end for an Azure AI Foundry clinical agent. Answers are
rendered as numbered sections with inline citations, and every citation resolves to
the passage it came from: the document, the page, and the verbatim text the model
was reading, listed once at the end of the page.

## Run it

```bash
pip install -r requirements.txt
cp env.example .env        # fill in AZURE_AI_PROJECT_ENDPOINT and AGENT_NAME
az login                   # or set AZURE_CLIENT_ID / AZURE_TENANT_ID / AZURE_CLIENT_SECRET
streamlit run app.py
```

Without a configured project the app starts in sample mode and replays stored
responses — real answers with their real retrieval attached — so the interface can
be reviewed before wiring up Azure. Force it with `DEMO_MODE=1`.

## Files

| File | Role |
| --- | --- |
| `app.py` | Layout, session thread, hero and answer rendering |
| `agent_client.py` | Azure auth, agent lookup, one `ask()` call, sample responses |
| `citations.py` | Resolves citation markers to the passages behind them |
| `parsing.py` | Splits the answer into sections; renders citations and references |
| `assets/sources.json` | Document metadata: title, publisher, year, public link |
| `assets/demo/*.json` | Stored real responses used by sample mode |
| `assets/styles.css` | Design tokens and all component styling |
| `.streamlit/config.toml` | Light theme matched to the stylesheet |

## How a citation becomes a reference

An answer arrives in two halves, and the app's main job is to join them.

**The prose** carries markers next to the claims they support:

```
Preferred oral agents … are methyldopa, beta-blockers … 【6:0†source】
```

On their own those markers say nothing. **The evidence** travels separately, in the
knowledge-base tool call of the same response, which returns one record per marker:

```json
【6:0†source】
{ "uid": "…_text_sections_63",
  "blob_url": "https://…/9789240033986-eng.pdf",
  "snippet": "While BP treatment thresholds for HTN in pregnancy…" }
```

The langchain agent node returns only the final message, so `agent_client.ask()`
fetches the response back by id (`responses.retrieve`) to get the tool calls with
it. That is a metadata read against a response that already exists — no model work
is repeated. `citations.extract_evidence()` then joins the halves, and the UI can
show a claim, the document it rests on, and the passage itself.

Details worth knowing:

- **Markers are numbered per response.** `6:0` in one turn is unrelated to `6:0` in
  the next, and one response can retrieve several times (`6:…`, then `9:…`).
  Resolution is always scoped to the turn that produced the marker.
- **References are numbered per passage, once per session.** A passage cited again
  in a later turn keeps its number, so a number means the same thing all the way
  down the page and the reference list only appears once, at the end.
- **A marker with nothing behind it renders greyed out** rather than being dropped,
  so a gap in the evidence stays visible instead of silently disappearing.
- **A reference list the model wrote out in prose is discarded.** References are
  built from the retrieval record, which is the more reliable of the two.
- **The retrieval tool truncates its own last record.** What is readable is
  recovered by hand and the reference is marked as cut short.

## Document metadata

Retrieval identifies a document only by its blob URL, so `assets/sources.json` maps
file names to a real citation:

```json
"9789240033986-eng.pdf": {
  "title": "Guideline for the pharmacological treatment of hypertension in adults",
  "publisher": "World Health Organization",
  "year": "2021",
  "url": "https://www.who.int/publications/i/item/9789240033986"
}
```

Add an entry to give a document a proper title and a link a reader can open. A
document that isn't listed still renders: its title is inferred from the running
head repeated across its own pages, falling back to the file name.

The `url` is deliberately the publisher's landing page rather than the blob URL —
the storage account rejects anonymous reads, so blob links are not openable from
the browser. If your container is reachable, add the blob (or a SAS) URL instead
and the reference will link straight to the PDF.

## Conversation continuity

`azure_ai_agents_conversation_id` is stored in session state and passed back on the
next question so follow-ups keep context. If the thread has expired, the call is
retried once as a fresh conversation. A follow-up that cites the previous turn's
retrieval without searching again still resolves: unknown markers are looked up in
earlier turns.
