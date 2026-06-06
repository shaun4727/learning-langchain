### Lesson 4: Advanced Retrieval (Retrieval-Augmented Generation - RAG)

#### The Architectural "Why"

In Lesson 3, we successfully set up our multi-container PostgreSQL database with the native `pgvector` extension. Now, we will implement **Retrieval-Augmented Generation (RAG)** to connect your FastAPI server and Google's Gemini model to private, domain-specific technical knowledge (such as resumes, internal documentation, or codebases).

An LLM's static knowledge is frozen at its last training cutoff date. RAG solves this by implementing an updated, production-grade two-phase architecture using the latest production models:

1. **The Ingestion Pipeline:** Breaks large technical files or binary PDF uploads into smaller semantic chunks, transforms those chunks into high-dimensional floating-point arrays called **Embeddings** using Google's active `gemini-embedding-001` model, and saves them directly into a vectorized PostgreSQL table. To remain compatible with standard database schemas without sacrificing performance, we leverage **Matryoshka Representation Learning (MRL)** to safely compress the vectors down to 768 dimensions natively at the API boundary.
2. **The Retrieval Pipeline:** When a user queries your API, the server turns that query into a live embedding vector using the exact same dimensionality constraints, runs a mathematical **Cosine Similarity** calculation inside the database kernel to isolate the top relevant chunks, and drops those exact facts directly into the high-performance `gemini-2.5-flash` system prompt.

---

### Phase Structural Lifecycle

| Phase Sequence | Engine | Structural Mechanism | Technical Objective |
| --- | --- | --- | --- |
| **1. Text/PDF Ingestion** | `pypdf` + `python-multipart` | Extracts raw binary streams from file uploads and handles page-by-page paragraph grouping. | Prepares raw documents for tokenization and vector processing. |
| **2. Ingestion Embedding** | `GoogleGenerativeAIEmbeddings` | Text splitters slice data $\rightarrow$ `gemini-embedding-001` generates stabilized 768-dimensional float vectors. | Converts unstructured text blocks into mathematically indexable coordinates. |
| **3. Storage Phase** | `SQLAlchemy 2.0` + `pgvector` | Native SQL inserts execute over an asynchronous pipeline. | Saves raw text and vectors side-by-side inside the database kernel. |
| **4. Vector Retrieval** | `pgvector` operator (`<=>`) | Executes an optimized Cosine Distance query across database indices. | Pulls the top $N$ most semantically relevant text fragments matching the client query. |
| **5. Grounded Synthesis** | FastAPI + Gemini | Context injection compiles live text fragments straight into the prompt template. | Forces `gemini-2.5-flash` to respond strictly using the retrieved database facts. |

---

### Step 1: Upgrading Dependencies

To support native vector handling alongside real-time multi-part binary file uploads (like PDF resumes), update your **`requirements.txt`** file to match this stabilized version matrix:

```text
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
langchain-core>=0.3.0
langchain-google-genai>=1.0.1
pydantic>=2.0.0
asyncpg>=0.29.0
sqlalchemy>=2.0.0
greenlet>=3.0.0
pgvector>=0.2.0
pypdf>=4.0.0
python-multipart>=0.0.9

```

---

### Step 2: Defining the Database Schema Model

Create a new file in your backend folder named **`models.py`**. This script explicitly defines how database columns map down to actual text elements and mathematical vector fields. We configure the vector field for **768 dimensions**, matching our embedding engine configuration.

```python
from sqlalchemy import Column, Integer, Text, String
from pgvector.sqlalchemy import Vector
from database import Base

class KnowledgeChunk(Base):
    """Database model for storing chunked engineering documents along with their vector embeddings."""
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    source_file = Column(String(255), nullable=True)
    # Target dimensionality locked to 768 using Matryoshka compression flags
    embedding = Column(Vector(768), nullable=False)

```

---

### Step 3: Updating `main.py` for Model Discovery

Open **`main.py`** and ensure your top import statements reflect the correct LangChain Google integration classes, and that your `startup_event` routine builds the extensions inside Postgres during container initialization.

```python
import os
import io
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pypdf import PdfReader

from database import engine, Base, get_db
import models  # Registers models into SQLAlchemy's global compilation metadata

app = FastAPI(title="Agentic AI Backend - RAG Engine", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        # Register pgvector extension inside the core kernel instance
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Automatically compile and construct tables mapped via SQLAlchemy models
        await conn.run_sync(Base.metadata.create_all)

```

---

### Step 4: Injecting Ingestion, PDF Processing, & RAG Query Routes

Append the complete, updated Pydantic validation schemas and operational RAG pipelines to the bottom of your **`main.py`**:

```python
# =====================================================================
# PHASE 3: ADVANCED RAG (INGESTION & RETRIEVAL) ROUTING
# =====================================================================

class KnowledgeIngestionRequest(BaseModel):
    content: str = Field(..., description="The raw technical text or documentation content to ingest.")
    source_file: Optional[str] = Field("manual.txt", description="The origin source name of the document.")

class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="The technical question you want to ask the grounded LLM model.")


@app.post("/ingest-knowledge")
async def ingest_knowledge(payload: KnowledgeIngestionRequest, db: AsyncSession = Depends(get_db)):
    """Ingestion Engine: Converts plain technical text chunks into 768-dimension vectors and saves them."""
    try:
        embeddings_engine = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            output_dimensionality=768
        )
        
        chunks = [chunk.strip() for chunk in payload.content.split("\n\n") if chunk.strip()]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text chunks discovered in payload.")

        vectors = embeddings_engine.embed_documents(chunks)

        for chunk_text, vector in zip(chunks, vectors):
            db_chunk = models.KnowledgeChunk(
                content=chunk_text,
                source_file=payload.source_file,
                embedding=vector
            )
            db.add(db_chunk)
            
        await db.commit()
        return {"status": "success", "inserted_chunks": len(chunks)}
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Knowledge ingestion breakdown: {str(e)}")


@app.post("/ingest-pdf-file")
async def ingest_pdf_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Binary File Ingestion Engine: Extracts actual text from uploaded PDFs, vectorizes, and saves it."""
    try:
        file_content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(file_content))
        
        extracted_text = ""
        for page in pdf_reader.pages:
            text_content = page.extract_text()
            if text_content:
                extracted_text += text_content + "\n\n"
        
        chunks = [chunk.strip() for chunk in extracted_text.split("\n\n") if chunk.strip()]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text could be extracted from the PDF file.")
            
        embeddings_engine = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            output_dimensionality=768
        )
        
        vectors = embeddings_engine.embed_documents(chunks)

        for chunk_text, vector in zip(chunks, vectors):
            db_chunk = models.KnowledgeChunk(
                content=chunk_text,
                source_file=file.filename,
                embedding=vector
            )
            db.add(db_chunk)
            
        await db.commit()
        return {"status": "success", "inserted_chunks": len(chunks), "filename": file.filename}
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"PDF parsing or ingestion failed: {str(e)}")


@app.post("/rag-ask")
async def rag_ask_question(payload: RAGQueryRequest, db: AsyncSession = Depends(get_db)):
    """Retrieval-Augmented Query Engine: Syncs query vector layouts and forces a grounded response via Gemini 2.5."""
    try:
        embeddings_engine = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            output_dimensionality=768
        )
        
        query_vector = embeddings_engine.embed_query(payload.question)

        stmt = (
            select(models.KnowledgeChunk)
            .order_by(models.KnowledgeChunk.embedding.cosine_distance(query_vector))
            .limit(3)
        )
        result = await db.execute(stmt)
        matched_chunks = result.scalars().all()

        if not matched_chunks:
            return {"answer": "No technical documentation has been ingested into the memory store yet.", "context_used": []}

        context_block = "\n---\n".join([chunk.content for chunk in matched_chunks])
        context_sources = [chunk.source_file for chunk in matched_chunks]

        system_prompt = (
            "You are an elite enterprise software systems engineer expert.\n"
            "Answer the user's question using ONLY the verified documentation context blocks provided below.\n"
            "If the context does not contain the answer, state clearly that the documentation does not cover this topic.\n\n"
            f"--- START SYSTEM VERIFIED CONTEXT ---\n{context_block}\n--- END SYSTEM VERIFIED CONTEXT ---\n\n"
            f"User Question: {payload.question}"
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        response = await llm.ainvoke(system_prompt)

        return {
            "answer": response.content,
            "context_sources": list(set(context_sources)),
            "retrieved_fragments_count": len(matched_chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing pipeline failure: {str(e)}")

```

---

### Actionable Task

1. Save the updated versions of `requirements.txt`, `main.py`, and your `models.py` file inside your workspace directory.
2. Force a clean image rebuild to install `pypdf` and `python-multipart` into your API container environment:

```bash
docker compose down --volumes
docker compose up --build -d

```

3. Open your terminal logs (`docker compose logs api -f`) and verify that your system successfully launches with the text: `Application startup complete.`

Once your logs confirm the system is online, respond back with **Finished**. We will then proceed to test your live endpoints by uploading your technical resume directly through the brand new file system layer!


# # Problems and it's solution

Here is the comprehensive summary of all the technical issues encountered during the implementation and testing of Lesson 4, along with their root causes and structural solutions.

### Lesson 4 Troubleshooting & Resolution Log

| # | Error / Problem Symptom | Root Cause | Provided Solution / Action Item |
| --- | --- | --- | --- |
| **1** | `curl: (56) Recv failure: Connection reset by peer` | The FastAPI/Uvicorn backend server process crashed abruptly mid-request, causing the network socket to drop before returning an HTTP response. | Run `docker compose logs api --tail=50` to inspect the container trace. Verify that environment variables (`GEMINI_API_KEY`) are bound correctly in `docker-compose.yml`, and restart port states via `docker compose down && docker compose up --build -d`. |
| **2** | `404 NOT_FOUND` for `models/text-embedding-004` during knowledge ingestion. | The legacy `text-embedding-004` model was permanently deprecated and decommissioned by Google, causing endpoint network failures. | Migrated the embedding engine to an active model (`models/gemini-embedding-001`). Because the new model defaults to 3072 dimensions while your database was built for 768, we explicitly passed `output_dimensionality=768` to employ Matryoshka Representation Learning and prevent database schema errors. |
| **3** | `500 Internal Server Error` stating: `name 'GoogleGenAIEmbeddings' is not defined` | A Python `NameError` inside the `/rag-ask` endpoint. The code attempted to instantiate a class that was either not imported or mistyped. | Corrected the class name to LangChain's official Google registry format: **`GoogleGenerativeAIEmbeddings`**. Updated both the top imports and the instantiation lines inside `/ingest-knowledge` and `/rag-ask`. |
| **4** | `404 NOT_FOUND` for `models/gemini-1.5-flash` inside the `/rag-ask` endpoint. | A model version discrepancy. The Phase 1 route (`/extract-features`) was updated to use `gemini-2.5-flash`, but the Phase 3 route (`/rag-ask`) was still pointing to the deprecated `gemini-1.5-flash` engine. | Updated the LLM initialization layer inside the `/rag-ask` endpoint at the bottom of `main.py` to target the active model: `ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)`. |
| **5** | RAG pipeline fails to return resume skills; returns empty or non-applicable context blocks. | Conceptual mismatch. Passing a string like `"Adjustable Resume- Shaun Hossain.pdf"` in a standard text payload only stores that string as metadata. The system had no logic to access, open, or read the local binary file context. | Implemented a structural backend upgrade: Installed the `pypdf` parsing engine and coded a dedicated **`/ingest-pdf-file`** endpoint that accepts an `UploadFile` stream, extracts text dynamically page-by-page, and pushes the generated chunks into the vector store. |
| **6** | `RuntimeError: Form data requires "python-multipart" to be installed.` (Container crashes on startup). | FastAPI handles incoming file streams utilizing the standard `multipart/form-data` encoding. To parse this, it requires the external package `python-multipart`, which was missing from the image environment. | Appended `python-multipart>=0.0.9` and `pypdf>=4.0.0` directly to the `requirements.txt` file. Executed a clean cache wipe and cluster rebuild using `docker compose down && docker compose up --build -d` to securely bake the packages into the application layer. |

### Golden Rules for Future Reference

1. **Dimension Matching:** Whenever you alter an embedding model configuration, ensure the `output_dimensionality` on the query side (`/rag-ask`) matches the exact vector size configuration of the ingestion side (`/ingest-knowledge`), as well as your PostgreSQL database column configuration.
2. **Context Isolation:** If your RAG pipeline states it cannot find an answer, always trace the database layer first using SQL in pgAdmin (`SELECT * FROM knowledge_chunks;`) to verify that the raw text chunks actually exist inside the kernel storage layer.