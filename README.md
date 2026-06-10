
# 🤖 Asynchronous Agentic AI Streaming Backend

An asynchronous, enterprise-grade AI backend microservice engineered with **FastAPI** and **Python** to orchestrate stateful, multi-step LLM workflows. Powered by **LangChain** and **LangGraph**, the system utilizes a serverless **Neon PostgreSQL** database with the **`pgvector`** extension for highly optimized semantic vector space retrieval, delivering real-time response tokens natively via **Server-Sent Events (SSE)**.

---

## 🚀 Live Deployment & Environment Setup

The core execution engine is actively deployed and running live on **Render**. You can interface with the live streaming gateway or spin it up locally using the infrastructure baselines below:

* **Production Cloud Base URL:** `https://learning-langchain.onrender.com`
* **Local Container Development URL:** `http://localhost:8000`

---

## 🔌 Production-Grade API Endpoint Verification Guide

This backend service is fully optimized for asynchronous task execution. It exposes structural REST endpoints for metadata extraction, binary file streaming ingest channels, and real-time stateful conversational agents.

### 🗺️ System Endpoint Reference Matrix

| Context Phase | HTTP Method | Route Target | Content-Type Payload | Expected Response Type |
| :--- | :--- | :--- | :--- | :--- |
| **System** | `GET` | `/health` | None | `application/json` |
| **System** | `GET` | `/db-test` | None | `application/json` |
| **Phase 1** | `POST` | `/extract-features` | Query String Matrix | `application/json` (Structured Schema) |
| **Phase 3** | `POST` | `/ingest-pdf-file` | `multipart/form-data` | `application/json` (Ingest Telemetry) |
| **Phase 3** | `POST` | `/rag-ask` | `application/json` | `application/json` (Grounded Context) |
| **Phase 4** | `POST` | `/agent-chat` | `application/json` | `text/event-stream` (Real-Time SSE) |

> 📌 *The verification profiles below are pre-configured to point directly to the live Render production cluster environment.*

---

## 🛠️ Execution & Testing Profiles (cURL)

#### 1. Global Core Health Gateway
Verifies that the FastAPI microservice runtime engine is live, stable, and listening on the container cluster.

```bash
curl -X GET "[https://learning-langchain.onrender.com/health](https://learning-langchain.onrender.com/health)" \
     -H "Content-Type: application/json"

```

##### 📥 Expected Output:

```json
{
  "status": "healthy",
  "service": "agentic-ai-core"
}

```

---

#### 2. Vector Database Extension Handshake Check

Instructs the SQLAlchemy connection pooling manager to perform a diagnostic pass to the PostgreSQL layer, confirming that the **`pgvector`** core kernel extension is validated and active.

```bash
curl -X GET "[https://learning-langchain.onrender.com/db-test](https://learning-langchain.onrender.com/db-test)" \
     -H "Content-Type: application/json"

```

##### 📥 Expected Output:

```json
{
  "status": "connected",
  "database": "postgresql",
  "pgvector_extension": "active"
}

```

---

#### 3. Structured Architectural Feature Extractor

Passes unstructured text blocks or raw system logs as query strings. The backend binds `gemini-2.5-flash` using a strict structural **Pydantic JSON schema** layout to enforce runtime validation guarantees.

```bash
curl -X POST "[https://learning-langchain.onrender.com/extract-features?text_input=We%20built%20our%20entire%20microservice%20architecture%20using%20Python%20and%20orchestrated%20it%20via%20Docker%20containers.%20However,%20we%20are%20hitting%20an%20unfortunate%20socket.gaierror%20DNS%20resolution%20timeout%20when%20trying%20to%20reach%20the%20database](https://learning-langchain.onrender.com/extract-features?text_input=We%20built%20our%20entire%20microservice%20architecture%20using%20Python%20and%20orchestrated%20it%20via%20Docker%20containers.%20However,%20we%20are%20hitting%20an%20unfortunate%20socket.gaierror%20DNS%20resolution%20timeout%20when%20trying%20to%20reach%20the%20database)." \
     -H "Content-Type: application/json"

```

##### 📥 Expected Output:

```json
{
  "programming_languages": ["Python"],
  "infrastructure_tools": ["Docker"],
  "architectural_bottleneck": "The microservice is experiencing a socket.gaierror DNS resolution timeout when attempting to connect to the external database host engine."
}

```

---

#### 4. Binary Multipart PDF Ingestion Engine

Handles complex binary file uploads. The service loads the document stream into volatile memory buffers via `pypdf`, dynamically extracts and chunks content structural boundaries, maps the fragments to **768-dimensional array vectors** using `gemini-embedding-001`, and commits them to the database.

```bash
curl -X POST "[https://learning-langchain.onrender.com/ingest-pdf-file](https://learning-langchain.onrender.com/ingest-pdf-file)" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/absolute/path/to/your/technical_resume.pdf"

```

##### 📥 Expected Output:

```json
{
  "status": "success",
  "inserted_chunks": 14,
  "filename": "technical_resume.pdf"
}

```

---

#### 5. Grounded Retrieval-Augmented Generation (RAG) Engine

Performs an optimized mathematical **cosine similarity distance vector lookup** against the incoming query string. This pipeline feeds retrieved technical context fragments into an LLM prompt wrapper, forcing a precise system evaluation grounded exclusively by factual records.

```bash
curl -X POST "[https://learning-langchain.onrender.com/rag-ask](https://learning-langchain.onrender.com/rag-ask)" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What are the core technical projects listed in Shaun Hossain portfolio documentation?"
     }'

```

##### 📥 Expected Output:

```json
{
  "answer": "According to the verified system documentation, Shaun Hossain has successfully architected an Enterprise Resource Planning (ERP) System at MononSoft Ltd, alongside multiple high-performance web solutions including an advanced eCommerce Platform featuring server-side rendering, a YouTube replication platform, and a comprehensive Tutor Finder network utility layer.",
  "context_sources": ["technical_resume.pdf"],
  "retrieved_fragments_count": 3
}

```

---

#### 6. Stateful Autonomous Agent Pipeline (Real-Time Streaming)

Triggers an advanced **LangGraph loop workflow**. The request runs through specialized security **regex guardrail hooks** to stop injection strings, hydrates chronological conversation history logs from the database, autonomously selects and executes background diagnostics tools, and streams response text pieces back sequentially using **Server-Sent Events (SSE)**.

```bash
curl -N -X POST "[https://learning-langchain.onrender.com/agent-chat](https://learning-langchain.onrender.com/agent-chat)" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "enterprise_verification_session_99",
       "user_prompt": "Run a system diagnostic loop on my platform extensions and write a short, motivational quote about clean code architecture."
     }'

```

##### 📥 Expected Stream Logs:

```text
data: {"event": "status", "message": "Hydrating conversation memory..."}

data: {"event": "status", "message": "Analyzing request intent..."}

data: {"event": "tool_start", "tool": "GetSystemDiagnostics", "message": "Executing background system tool: GetSystemDiagnostics"}

data: {"event": "tool_end", "tool": "GetSystemDiagnostics"}

data: {"event": "status", "message": "Generating final response..."}

data: {"event": "token", "text": "System Check complete: pgvector is validated and active on the host node.\n\n"}

data: {"event": "token", "text": "\"Clean code is not written by chance; it is the masterpiece of an architect who values clarity over cleverness.\""}

data: {"event": "done", "session_id": "enterprise_verification_session_99"}

```

---

### 💡 Core Operational Architectural Notes:

* **The Absolute Rule of the `-N` Parameter:** When testing the streaming agent `/agent-chat` route, adding the `-N` (or `--no-buffer`) argument is **mandatory**. By default, cURL buffers network data chunks internally; the `-N` flag turns off this buffer, printing the incoming Server-Sent Events (SSE) stream tokens to the terminal in absolute real time.
* **Cold Instantiation Cycles:** Since this system runs on a serverless container architecture via Render's free tier pool, initial API requests will experience a 30-50 second delay if the underlying container has entered an idle sleep cycle. Once the instance finishes spinning up, all subsequent query routes execute instantly.


