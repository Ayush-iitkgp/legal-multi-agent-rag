## Legal Multi‑Agent RAG for Contract Analysis

Interactive multi‑agent Retrieval‑Augmented Generation (RAG) system that helps users query and analyze a small corpus of legal contracts (NDA, Vendor Services Agreement, SLA, DPA).
The system runs as a console app and returns grounded answers with clause‑level citations and risk flags.

---

## 1. Problem overview

The goal is to assist a legal/BD user at Acme Corp in understanding and stress‑testing a vendor contract package.
Given natural‑language questions (e.g. *“What is the uptime commitment?”*, *“Which law governs the DPA?”*, *“Are there conflicting governing laws?”*), the system should:

- **Ingest** the provided contracts in `problem-statement/data/*.txt`
- **Retrieve** the most relevant clauses across all agreements
- **Analyze** them via specialized agents
- **Respond** with:
  - A concise natural‑language answer
  - Explicit clause citations
  - Highlighted legal / financial risks where applicable

The app is intentionally console‑based and multi‑turn to keep the focus on RAG design and agent orchestration rather than UI.

---

## 2. Architecture overview

- **Tech stack**
  - Python 3.10
  - LangChain / LangGraph
  - OpenAI (chat + embeddings)
  - ChromaDB vector store

- **Key modules**
  - `src/app.py` – CLI loop and conversation lifecycle
  - `src/graph.py` – LangGraph definition and multi‑agent orchestration
  - `src/agents/core.py` – individual agent prompts / behaviors
  - `src/ingest/core.py` – document loading and section‑based chunking
  - `src/retrieval/vectorstore.py` – vector store build/load helpers
  - `src/llm/openai_api.py` & `src/llm/factory.py` – LLM and embedding factory
  - `src/config.py` – configuration (paths, models, retrieval parameters)

### 2.1 Multi‑agent orchestration (Chain‑of‑Thought)

The system is implemented as a **LangGraph** with explicit, meaningful agents:

- **Router (Orchestrator)** – `node_router`
  - Analyzes the user question using `analyze_query` (classifier agent).
  - Decides which **document types** are likely relevant, using filename‑based types such as:
    - `nda_acme_vendor`
    - `vendor_services_agreement`
    - `service_level_agreement`
    - `data_processing_agreement`
  - Stores the routing decision in `GraphState.routed_doc_types`.

- **Clause Extractor (Retriever)** – `node_clause_extractor`
  - Builds or loads the Chroma vector store over pre‑chunked sections.
  - Runs a **map‑reduce style retrieval**:
    - Map: issues multiple queries (base question + focus‑area‑augmented variants).
    - Reduce:
      - Aggregates all retrieved chunks.
      - Filters to the routed `document_type` set from the Router.
      - De‑duplicates by `(source, section_index)` so each clause appears once.
  - Populates `GraphState.clauses` (and `retrieved`) with the final clause set.

- **Legal Analyst (Answer Agent)** – `node_legal_analyst`
  - Consumes the extracted clauses.
  - Drafts a grounded answer with inline references `[1], [2], …` using `draft_answer`.
  - Appends the generated answer to the conversation history.

- **Legal Auditor (Risk Agent)** – `node_legal_auditor`
  - Consumes the same clause set.
  - Uses a risk‑focused prompt to:
    - Identify material legal / financial risks to Acme.
    - Pay attention to **cross‑document inconsistencies** (e.g. conflicting governing laws, caps vs. uncapped liability, breach timelines).
  - Returns a short bullet list with severity and clause references.

- **Answer Composer** – `node_answer_composer`
  - Combines:
    - The Legal Analyst’s answer
    - The Legal Auditor’s risk bullets
  - Produces a final, user‑facing string:
    - Question
    - Answer
    - Risk flags

The graph flow is:

```text
router → clause_extractor → legal_analyst → legal_auditor → answer_composer → END
```

---

## 3. RAG design and ingestion

### 3.1 Document corpus

The legal corpus lives in `problem-statement/data`:

- `nda_acme_vendor.txt`
- `vendor_services_agreement.txt`
- `service_level_agreement.txt`
- `data_processing_agreement.txt`

`src.config.DATA_DIR` points to this directory, and `load_raw_docs` recursively loads all `*.txt` files from there.

### 3.2 Chunking strategy (section / Markdown‑style)

For legal text, fixed‑size token windows can split obligations mid‑clause.
Instead, this project uses **section‑based chunking** implemented in `simple_clause_chunk`:

- Splits on **structural headers**, treating each as the start of a new chunk:
  - Numbered headings: `"1. Scope of Services"`, `"3. Data Breach Notification"`, etc.
  - Markdown headings: `#`, `##`, `###` (supported for future documents).
  - Short ALL‑CAPS headings: `TERMINATION`, `GOVERNING LAW`, etc.
- Each chunk therefore corresponds roughly to a **single clause or section** and its explanatory text.
- Every `Document` chunk is tagged with:
  - `source` – original filename
  - `document_type` – filename stem (e.g. `data_processing_agreement`)
  - `section_index` – per‑document section number

This yields clauses like “NDA – Term and Termination”, “DPA – Data Breach Notification (72‑hour requirement)”, etc., which are directly consumable by the agents and easy to cite.

### 3.3 Embedding and LLM models

- **Embedding model**
  - Default: OpenAI `text-embedding-3-small`
  - Rationale: good semantic performance on dense legal text with efficient cost profile.

- **LLM model**
  - Default: `gpt-4.1-mini` via OpenAI.
  - Used for:
    - Query analysis / routing.
    - Answer drafting.
    - Risk assessment.
  - Temperature is set low (e.g. `0.1`) in `make_chat_model` to prioritize determinism and grounding over creativity.

### 3.4 Retrieval mechanism

- Vector store: **ChromaDB**
  - Stored under `storage/chroma` as configured in `INDEX_DIR`.
  - Automatically built on first run if not present (from the ingested chunks).
- Retrieval:
  - Uses `vs.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})`.
  - Top‑k (default 8) chosen to balance recall across multiple contracts with prompt length.
  - Clause Extractor runs multiple retrievals for different query variants and then reduces.

### 3.5 (Optional) Re‑ranking

No heavy re‑ranking model is used. Instead:

- Map‑reduce retrieval + strict metadata filtering by `document_type`.
- De‑duplication by `(source, section_index)`.

This provides a lightweight, deterministic “re‑ranking” effect without additional models.

---

## 4. Setup instructions

### 4.1 Prerequisites

- Python **3.10**
- An OpenAI API key with access to:
  - `gpt-4.1-mini`
  - `text-embedding-3-small`

### 4.2 Environment and dependencies

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install .
```

Or, if you prefer to use `pyproject.toml` directly:

```bash
pip install -e .
```

Set your OpenAI credentials (for example in `.env` or shell env):

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 5. Running the console app

From the project root (with the virtualenv activated):

```bash
python -m src.app
```

You should see:

```text
Legal Multi-Agent RAG CLI
Type your question about the contracts, or /quit to exit.
```

Then you can ask questions such as:

- “What is the notice period for terminating the NDA?”
- “What is the uptime commitment in the SLA?”
- “Which law governs the Vendor Services Agreement?”
- “Do confidentiality obligations survive termination of the NDA?”
- “Which agreement governs data breach notification timelines?”
- “Are there conflicting governing laws across agreements?”

For each query, the system will:

- Maintain conversation history.
- Route to the relevant agreements.
- Extract and analyze clauses.
- Print a final answer and a risk summary.

---

## 6. Design choices & trade‑offs

- **Multi‑agent vs. single‑agent**
  - Chosen: explicit Router, Clause Extractor, Legal Analyst, Legal Auditor.
  - Trade‑off: slightly more complexity vs. much clearer separation of concerns and easier debugging / observability.

- **Section‑based chunking vs. fixed token windows**
  - Chosen: section / header‑based chunking.
  - Benefits: avoids splitting obligations mid‑clause; improves interpretability and citation quality.
  - Trade‑off: sections can be uneven in length, but the legal documents are relatively small, so this is acceptable.

- **OpenAI models vs. local (Ollama)**
  - Chosen: OpenAI for simplicity and stronger legal‑text performance out of the box.
  - Trade‑off: external dependency and API cost vs. higher quality and less infra work.
  - The `MODEL_BACKEND` config is structured so an Ollama backend could be added later.

- **Simple retrieval + metadata filtering vs. learned re‑ranker**
  - Chosen: base dense retrieval + deterministic filtering by `document_type`.
  - Trade‑off: no heavy re‑ranking or cross‑encoder cost; relies on good embeddings and prompts, which is adequate for this small corpus.

---

## 7. Evaluation approach

Basic evaluation is manual / scenario‑based, using the sample queries from the assignment:

- Coverage questions:
  - “What is the notice period for terminating the NDA?”
  - “What is the uptime commitment in the SLA?”
  - “Which law governs the Vendor Services Agreement?”
  - “Which agreement governs data breach notification timelines?”
- Risk and inconsistency questions:
  - “Are there conflicting governing laws across agreements?”
  - “Is liability capped for breach of confidentiality?”
  - “Is Vendor XYZ’s liability capped for data breaches?”
  - “Are there any clauses that could pose financial risk to Acme Corp?”

For each query we check:

- **Grounding** – Are answers supported by the correct clauses?
- **Citation quality** – Do the referenced clauses `[1], [2], …` correspond to relevant sections?
- **Risk sensitivity** – Does the Legal Auditor flag key exposures (uncapped liability, conflicting laws, breach timelines)?

Due to the small corpus size, a lightweight manual checklist is sufficient and easy to run end‑to‑end.

**Limitations of this evaluation**

- No automated metrics (e.g. retrieval‑hit rate, answer correctness scores).
- No large‑scale test set; focused on a curated handful of high‑signal questions.
- Human judgment required to mark answers as acceptable or not.

---

## 8. Known limitations & future work

- **Small, static corpus only**
  The system assumes a small set of text contracts in `problem-statement/data`. Handling PDFs at scale, versioning, or dynamic uploads would require an ingestion service.

- **No formal re‑ranking / calibration**
  Retrieval is dense + heuristic filtering; a cross‑encoder or reranker could improve robustness on edge cases.

- **Single‑LLM backend**
  Only OpenAI is wired right now. A configurable backend (OpenAI vs. Ollama) and per‑agent model choice would make the system more portable.

- **Evaluation depth**
  Evaluation is qualitative and based on a handful of stress‑test queries; a production‑ready system would need a richer labeled set and automated checks.

Despite these trade‑offs, the current design demonstrates a clear, modular multi‑agent RAG architecture that is easy to extend and reason about for legal contract analysis.
