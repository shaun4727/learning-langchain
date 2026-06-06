import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional 

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from database import engine, Base, get_db
from sqlalchemy import text, select 
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile, File
from pypdf import PdfReader
import io

import models

app = FastAPI(title="Agentic AI Backend", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# PHASE 1: STRUCTURED DATA EXTRACTION SCHEMAS & ROUTES
# =====================================================================

class TechnicalFeatureExtractor(BaseModel):
    """Schema for extracting technical architectural features from structural logs or project text."""
    programming_languages: List[str] = Field(
        description="List of programming languages explicitly mentioned or used (e.g., Python, TypeScript)."
    )
    infrastructure_tools: List[str] = Field(
        description="DevOps, container, or runtime environment orchestration technologies mentioned (e.g., Docker, Kubernetes)."
    )
    architectural_bottleneck: str = Field(
        description="A clear, single-sentence summary of the core engineering bottleneck or bug described."
    )


@app.post("/extract-features", response_model=TechnicalFeatureExtractor)
async def extract_features(text_input: str):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is missing.")
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        structured_llm = llm.with_structured_output(
            TechnicalFeatureExtractor, 
            method="json_schema"
        )
        
        result = await structured_llm.ainvoke(text_input)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# PHASE 2: CORE INFRASTRUCTURE LIFECYCLE & DIAGNOSTICS
# =====================================================================

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        # Register pgvector extension inside the core kernel instance
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Automatically compile and construct tables mapped via SQLAlchemy models
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agentic-ai-core"}


@app.get("/db-test")
async def test_database_connection(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
        extension_exists = result.scalar()
        
        if extension_exists:
            return {"status": "connected", "database": "postgresql", "pgvector_extension": "active"}
        raise HTTPException(status_code=500, detail="Database connected but pgvector extension missing.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database handshake failed: {str(e)}")

# =====================================================================
# PHASE 3: ADVANCED RAG (INGESTION & RETRIEVAL) ROUTING
# =====================================================================

class KnowledgeIngestionRequest(BaseModel):
    content: str = Field(..., description="The raw technical text or documentation content to ingest.")
    source_file: Optional[str] = Field("manual.txt", description="The origin source name of the document.")

class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="The technical question you want to ask the grounded LLM model.")




@app.post("/ingest-pdf-file")
async def ingest_pdf_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    try:
        # 1. Read the raw binary stream of the uploaded file into memory
        file_content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(file_content))
        
        # 2. Extract text from all pages of the PDF document
        extracted_text = ""
        for page in pdf_reader.pages:
            text_content = page.extract_text()
            if text_content:
                extracted_text += text_content + "\n\n"
        
        # 3. Use your existing logic to split text into chunks
        chunks = [chunk.strip() for chunk in extracted_text.split("\n\n") if chunk.strip()]
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text could be extracted from the PDF.")
            
        embeddings_engine = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            output_dimensionality=768
        )
        
        vectors = embeddings_engine.embed_documents(chunks)

        for chunk_text, vector in zip(chunks, vectors):
            db_chunk = models.KnowledgeChunk(
                content=chunk_text,
                source_file=file.filename, # Automatically grabs the actual file name
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
    """Retrieval-Augmented Query Engine: Extracts context via vector math and forces a grounded response."""
    try:
        # FIXED: Updated class name, model target, and dimensionality configuration
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