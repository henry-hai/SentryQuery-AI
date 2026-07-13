# SentryQuery AI

An agentic AI assistant that answers questions over indexed enterprise documents
using a LangGraph ReAct agent over Pinecone, with a Streamlit web UI.

## Demo

**RAG retrieval** — the agent calls the Pinecone retriever and grounds its answer
in the indexed documents.

![RAG answer grounded in indexed documents](assets/screenshots/rag-answer.png)

It also surfaces the exact source chunks it consulted, expanded by PDF and page.

![Source chunks the agent consulted](assets/screenshots/rag-sources.png)

**Live web search** — for questions that need current information, the agent
reaches for the Tavily web-search tool instead.

![Live web search via Tavily](assets/screenshots/tavily-web-search.png)

**Guard rail** — off-topic questions are refused per the system prompt, without
the agent wasting a tool call.

![Off-topic question refused](assets/screenshots/guardrail-refusal.png)

## Architecture

Documents are indexed once into Pinecone using OpenAI embeddings. On each query,
a **Researcher** agent (built via `create_agent` from `langchain.agents`,
LangChain's current agent constructor, which compiles a LangGraph state graph
internally) reasons about whether and how to call the retriever tool — rather
than following a fixed retrieve-then-answer pipeline. The agent can call the
retriever multiple times with refined queries if needed, and a system prompt
restricts it to grounded, on-topic answers. Its final answer is packaged into a
Pydantic `AnswerSchema` (answer, sources, tool_used, confidence) at a dedicated
synthesis step — the structured output is applied only there, never to the agent
itself, so its tool-calling loop stays intact.

### Multi-agent verification (Researcher + Critic)

The Researcher and a second **Critic** agent are wired as two distinct nodes in
an explicit LangGraph `StateGraph`. After the Researcher drafts an answer, the
Critic checks whether every claim is grounded in the **exact chunks the
Researcher retrieved** — captured in graph state, never re-queried (re-fetching
fresh text would quietly defeat the check). The Critic returns `APPROVE`, or
`REVISE` with a specific reason, and on `REVISE` the graph loops back to the
Researcher with that note, capped at a configurable number of passes
(`MAX_REVISIONS`, default 1).

This is the same groundedness check the eval harness grades offline, now
enforced at runtime: **the Critic enforces at request time what the eval harness
verifies in CI.** The Critic runs on a cheaper model (`gpt-4o-mini`) than the
Researcher's GPT-4o — groundedness checking is a narrower verification task, so
the smaller model suffices at a fraction of the per-call cost — at
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
- GPT-4o for the Researcher; `gpt-4o-mini` for the cheaper Critic
- Tavily for live web search (optional second agent tool)
- Streamlit for the web UI

## Setup

```
python -m venv venv
source venv/bin/activate    # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file with:

```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
TAVILY_API_KEY=your_key
```

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

A small eval harness lives in `evals/`. It runs each Q&A pair in `evals/qa.json`
through the agent and checks (1) the answer contains an expected keyword and
(2) the agent correctly used (or skipped) the retriever for the given question.
Off-topic prompts are expected to be refused per the system prompt.

```
python evals/eval.py
```
