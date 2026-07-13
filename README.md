# SentryQuery AI

An agentic RAG assistant that answers questions over indexed enterprise documents
with a two-agent LangGraph pipeline over Pinecone, and a Streamlit web UI. A
**Researcher** retrieves and drafts, and a **Critic** verifies every claim
against the retrieved sources.

## Demo

**RAG retrieval + runtime verification.** The Researcher retrieves from Pinecone
and grounds its answer, then the Critic verifies every claim against the exact
retrieved chunks, and the UI shows a confidence score and a green "verified" badge.

![Grounded answer with the Critic's green verified badge](assets/screenshots/rag-answer.png)

It surfaces the exact source chunks it consulted, expanded by PDF and page.

![Source chunks the agent consulted](assets/screenshots/rag-sources.png)

**The Critic catches an unsupported claim.** When an answer overreaches beyond
what the documents support, the Critic flags it and sends it back to the
Researcher for revision. The amber badge shows the answer was revised and names
what was still unsupported.

![The Critic flags an unsupported projection and forces a revision](assets/screenshots/critic-revision.png)

**Live web search.** For current information not in the filings, the agent
reaches for Tavily instead, and the UI is explicit that web answers are not
verified against the indexed documents.

![Live web search via Tavily with source links](assets/screenshots/tavily-web-search.png)

**Guard rail.** Off-topic questions are refused per the system prompt, without a
wasted tool call.

![Off-topic question refused](assets/screenshots/guardrail-refusal.png)

## Architecture

Documents are indexed once into Pinecone using OpenAI embeddings. On each query,
a **Researcher** agent (built via `create_agent` from `langchain.agents`,
LangChain's current agent constructor, which compiles a LangGraph state graph
internally) reasons about whether and how to call the retriever tool, rather
than following a fixed retrieve-then-answer pipeline. The agent can call the
retriever multiple times with refined queries if needed, and a system prompt
restricts it to grounded, on-topic answers. Its final answer is packaged into a
Pydantic `AnswerSchema` (answer, sources, tool_used, confidence) at a dedicated
synthesis step. The structured output is applied only there, never to the agent
itself, so its tool-calling loop stays intact.

### Multi-agent verification (Researcher + Critic)

The Researcher and a second **Critic** agent are wired as two distinct nodes in
an explicit LangGraph `StateGraph`. After the Researcher drafts an answer, the
Critic checks whether every claim is grounded in the **exact chunks the
Researcher retrieved**, captured in graph state and never re-queried (re-fetching
fresh text would quietly defeat the check). The Critic returns `APPROVE`, or
`REVISE` with a specific reason, and on `REVISE` the graph loops back to the
Researcher with that note, capped at a configurable number of passes
(`MAX_REVISIONS`, default 1).

This is the same groundedness check the eval harness grades offline, now
enforced at runtime: **the Critic enforces at request time what the eval harness
verifies in CI.** The Critic runs on a cheaper model (`gpt-4o-mini`) than the
Researcher's GPT-4o, since groundedness checking is a narrower verification task,
so the smaller model suffices at a fraction of the per-call cost. It runs at
`temperature=0` for deterministic, reproducible verdicts.

The Streamlit UI surfaces the final answer, the Critic's verdict (a green
"verified" badge, or amber when the answer was revised), the model's confidence,
and the exact source chunks the agent consulted, expanded by PDF and page.

## Stack

- LangChain agents (`create_agent` from `langchain.agents`) for the Researcher, compiled onto LangGraph
- LangGraph `StateGraph` for the Researcher + Critic multi-agent graph
- Pydantic for the structured `AnswerSchema` / `CriticVerdict` contracts
- LangChain for retriever tooling
- Pinecone as the vector database
- OpenAI `text-embedding-3-small` for embeddings (1536-d)
- GPT-4o for the Researcher, and `gpt-4o-mini` for the cheaper Critic
- Tavily for live web search (optional second agent tool)
- LangSmith for optional run tracing (env-toggleable, with a structured-logging fallback)
- Streamlit for the web UI

## Setup

```
python -m venv venv
source venv/bin/activate    # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
TAVILY_API_KEY=your_key
```

LangSmith keys are optional. Add them to trace runs (see [Observability](#observability)).

Pre-create a Pinecone index named `sentry-index` with dimension `1536` and
cosine similarity.

## Usage

Drop your PDFs into `./docs/` and index them once:

```
python sentry_query.py --ingest
```

Launch the web UI:

```
python -m streamlit run sentry_query.py
```

## Evaluation

A small eval harness lives in `evals/`. It runs each case in `evals/qa.json`
through the full Researcher → Critic graph and checks: (1) the answer contains an
expected keyword, (2) the agent used or skipped the retriever as expected, (3) the
packaged output validates against `AnswerSchema`, and (4) the Critic's verdict
matches on grounded cases. It then runs two **direct Critic checks**: an
ungrounded answer must be flagged `REVISE`, and a grounded one `APPROVE`. That
REVISE check is the case that fails without the Critic and passes with it, the
same groundedness check the Critic enforces at runtime.

```
python evals/eval.py
```

The harness grew alongside the system. The single-agent baseline passed **7/7**
on keyword + tool-use checks, and the current harness passes **9/9** after adding
schema-validation, Critic-verdict, and the two direct Critic checks. The extra
points are new *correctness dimensions* (structured validity, groundedness) that
the original code could not satisfy, not a change in answer accuracy. The
harness output is the only performance number claimed here.

## Observability

Every pipeline run emits a structured log record (tool routing, Critic verdict,
revision count, confidence), so there is always basic observability with no
setup.

For full tracing of the two-agent graph, set LangSmith keys in `.env` (see
`.env.example`):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=sentryquery
```

With these set, each run traces the Researcher's tool calls, the drafted schema,
the Critic's verdict, and any revision loop as nested runs in LangSmith. If they
are unset, the app runs identically with tracing off and never hard-fails on a
missing tracing key. The Streamlit sidebar shows the current tracing state.

![Observability sidebar showing the LangSmith tracing state](assets/screenshots/observability.png)