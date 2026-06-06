### Lesson 4: Advanced Retrieval (Retrieval-Augmented Generation - RAG)

#### The Architectural "Why"

In Lesson 3, we successfully set up our multi-container PostgreSQL database with the native `pgvector` extension. Now, we will implement **Retrieval-Augmented Generation (RAG)** to connect your FastAPI server and Google's Gemini model to private, domain-specific technical knowledge.

An LLM's static knowledge is frozen at its last training cutoff date, and it has no inherent awareness of private engineering files, internal company codebases, or proprietary API documentations. RAG solves this by implementing a two-phase architecture:

1. **The Ingestion Pipeline:** Breaks large technical files into smaller semantic chunks, transforms those chunks into high-dimensional floating-point arrays called **Embeddings** using Google's embedding model, and saves them directly into a vectorized PostgreSQL table.
2. **The Retrieval Pipeline:** When a user queries your API, the server turns that query into a live embedding vector, runs a lightning-fast mathematical **Cosine Similarity** calculation inside the database kernel to isolate the top relevant chunks, and drops those exact facts directly into the LLM's system prompt. This guarantees that Gemini's answers are fully grounded in accurate technical data, entirely eliminating random hallucinations.

---

### Phase Structural Lifecycle

| Phase Sequence | Engine | Structural Mechanism | Technical Objective |
| --- | --- | --- | --- |
| **1. Ingestion Phase** | `langchain-google-genai` | Text splitters slice data $\rightarrow$ `text-embedding-004` generates 768-dimensional float vectors. | Converts unstructured text blocks into mathematically indexable coordinates. |
| **2. Storage Phase** | `SQLAlchemy 2.0` + `pgvector` | Native SQL inserts execute over an asynchronous pipeline. | Saves raw text and vectors side-by-side inside the database kernel. |
| **3. Vector Retrieval** | `pgvector` operator (`<->` or `<=>`) | Executes an optimized Cosine Distance query across database indices. | Pulls the top $N$ most semantically relevant text fragments matching the client query. |
| **4. Grounded Synthesis** | FastAPI + Gemini | Context injection compiles live text fragments straight into the prompt template. | Forces Gemini to respond strictly using the retrieved database facts. |

---

### Step 1: Upgrading Dependencies

To allow SQLAlchemy to map native vector column data types inside Python, we must append the `pgvector` partner package to our service environment dependencies.

Update your **`requirements.txt`** file to append the `pgvector` python utility library:

```text
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
langchain-core>=0.3.0
langchain-google-genai>=1.0.0
pydantic>=2.0.0
asyncpg>=0.29.0
sqlalchemy>=2.0.0
greenlet>=3.0.0
pgvector>=0.2.0

```

---

### Step 2: Defining the Database Schema Model

Create a new file in your backend folder named **`models.py`**. This script imports our central declarative base and explicitly defines how database columns map down to actual text elements and mathematical vector fields. We will configure it for 768 dimensions, which perfectly matches Google's modern `text-embedding-004` model layout.

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
    # Google's text-embedding-004 model generates vectors with exactly 768 dimensions
    embedding = Column(Vector(768), nullable=False)

```

---

### Step 3: Updating `main.py` for Model Discovery

To ensure that SQLAlchemy discovers our table schema definition and automatically initializes the database tables inside Postgres when the container boots up, we must import `models` right inside our main routing gateway.

Open **`main.py`** and alter the top imports and your `startup_event` routine to reflect this specific configuration block:

```python
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenAIEmbeddings

from database import engine, Base, get_db
import models  # <--- CRITICAL: Registers models into SQLAlchemy's global compilation metadata

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

# Keep your existing endpoints (/health, /db-test, etc.) below this...

```

---

### Step 4: Injecting Ingestion & RAG Query Routes

Now, let's build the operational RAG pipelines. We will append two new endpoints to the bottom of your **`main.py`** file:

1. `/ingest-knowledge`: An endpoint that accepts technical documents, passes them to Google's embedding model, and saves them into PostgreSQL.
2. `/rag-ask`: An endpoint that executes a live vector query and passes the context to Gemini for a grounded answer.

Append the following code to the bottom of your **`main.py`**:

```python
# Pydantic validation schemas for data transfer boundaries
class KnowledgeIngestionRequest(BaseModel):
    content: str = Field(..., description="The raw technical text or documentation content to ingest.")
    source_file: Optional[str] = Field("manual.txt", description="The origin source name of the document.")

class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="The technical question you want to ask the grounded LLM model.")


@app.post("/ingest-knowledge")
async def ingest_knowledge(payload: KnowledgeIngestionRequest, db: AsyncSession = Depends(get_db)):
    """Ingestion Engine: Converts plain technical text into vectors and saves it to PostgreSQL."""
    try:
        # Initialize Google GenAI Embedding model instances natively
        embeddings_engine = GoogleGenAIEmbeddings(model="models/text-embedding-004")
        
        # Simple strategic semantic paragraph chunking
        chunks = [chunk.strip() for chunk in payload.content.split("\n\n") if chunk.strip()]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text chunks discovered in payload.")

        # Batch compute embeddings over the external network boundary
        # Note: LangChain's current embeddings object wraps synchronous network calls under the hood,
        # so we run it directly while keeping our database transactions strictly asynchronous.
        vectors = embeddings_engine.embed_documents(chunks)

        # Map chunks and vectors side-by-side into model entities
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


@app.post("/rag-ask")
async def rag_ask_question(payload: RAGQueryRequest, db: AsyncSession = Depends(get_db)):
    """Retrieval-Augmented Query Engine: Extracts context via vector math and forces a grounded response."""
    try:
        embeddings_engine = GoogleGenAIEmbeddings(model="models/text-embedding-004")
        
        # 1. Transform client question text string into a single embedding vector match array
        query_vector = embeddings_engine.embed_query(payload.question)

        # 2. Execute an asynchronous Cosine Distance vector lookup query inside the database kernel
        # The .cosine_distance() function uses the specialized pgvector <=> math operator under the hood
        stmt = (
            select(models.KnowledgeChunk)
            .order_by(models.KnowledgeChunk.embedding.cosine_distance(query_vector))
            .limit(3)
        )
        result = await db.execute(stmt)
        matched_chunks = result.scalars().all()

        if not matched_chunks:
            return {"answer": "No technical documentation has been ingested into the memory store yet.", "context_used": []}

        # 3. Consolidate extracted rows into a unified context block
        context_block = "\n---\n".join([chunk.content for chunk in matched_chunks])
        context_sources = [chunk.source_file for chunk in matched_chunks]

        # 4. Compile a highly defensive system prompt targeting absolute grounding
        system_prompt = (
            "You are an elite enterprise software systems engineer expert.\n"
            "Answer the user's question using ONLY the verified documentation context blocks provided below.\n"
            "If the context does not contain the answer, state clearly that the documentation does not cover this topic.\n\n"
            f"--- START SYSTEM VERIFIED CONTEXT ---\n{context_block}\n--- END SYSTEM VERIFIED CONTEXT ---\n\n"
            f"User Question: {payload.question}"
        )

        # 5. Route the prompt payload straight out to the high-performance Gemini model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
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

1. Save the updated versions of `requirements.txt` and `main.py`, and create the new `models.py` file inside your working project directory.
2. Force a clean build to install the new `pgvector` library package and re-orchestrate the application state matrix:
```bash
docker compose down --volumes
docker compose up --build

```


3. Verify that your containers boot up cleanly and show that `api-1` successfully finishes application startup initialization routines.

Once the application is live, report back with **Finished**. In the next turn, we will run deep network verification steps by feeding our system mock enterprise documentation and observing how it utilizes vector distance calculations to answer questions!