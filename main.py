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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
    


# =====================================================================
# PHASE 4: AUTONOMOUS AGENT LAYER & TOOL DEFINITIONS
# =====================================================================

class SearchKnowledgeBase(BaseModel):
    """Search the vectorized system knowledge store for engineering documentation, skills, projects, or resume details."""
    query: str = Field(..., description="The semantic search query targeting historical document fragments.")

class GetSystemDiagnostics(BaseModel):
    """Fetch live infrastructure environment runtime states, database health, and extension configurations."""
    confirm: bool = Field(..., description="Set to True to trigger a live database connection handshake check.")

class AgentChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique thread identifier tracking a single distinct conversation conversation.")
    user_prompt: str = Field(..., description="The textual query or follow-up instruction passed to the agent.")


@app.post("/agent-chat")
async def agent_reasoning_engine(payload: AgentChatRequest, db: AsyncSession = Depends(get_db)):
    """Stateful Autonomous Agent: Restores session threads from PostgreSQL, reasons, runs tools, and saves logs."""
    try:
        # 1. Hydrate Short-Term Memory: Fetch chronological history for this session thread
        stmt = (
            select(models.ChatMessage)
            .where(models.ChatMessage.session_id == payload.session_id)
            .order_by(models.ChatMessage.timestamp.asc())
        )
        result = await db.execute(stmt)
        historical_rows = result.scalars().all()

        # 2. Compile Context Message Array for LangChain and Gemini
        message_list = [
            SystemMessage(content=(
                "You are an elite autonomous system agent executing software diagnostic loops.\n"
                "You have access to historical logs and operational tools to satisfy parameters perfectly.\n"
                "Review the context of past turns (if any) to address follow-up references cleanly."
            ))
        ]

        # Map historical database records directly into structured LangChain message schemas
        for row in historical_rows:
            if row.role == "user":
                message_list.append(HumanMessage(content=row.content))
            elif row.role == "model":
                message_list.append(AIMessage(content=row.content))

        # Append the current live user prompt to the very tail of the sequence
        message_list.append(HumanMessage(content=payload.user_prompt))

        # 3. Initialize reasoning engine and bind existing functional tools
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        llm_with_tools = llm.bind_tools([SearchKnowledgeBase, GetSystemDiagnostics])
        
        # Execute reasoning pass over the entire historical message block
        ai_msg = await llm_with_tools.ainvoke(message_list)
        
        final_answer = ""
        executed_tools_list = []

        # 4. Handle Operational Tool Execution Logic
        if not ai_msg.tool_calls:
            final_answer = ai_msg.content
            response_source = "direct_llm_with_memory"
        else:
            response_source = "agent_tool_execution_with_memory"
            tool_results = []
            
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                executed_tools_list.append(tool_name)
                
                if tool_name == "SearchKnowledgeBase":
                    embeddings_engine = GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-001",
                        output_dimensionality=768
                    )
                    query_vector = embeddings_engine.embed_query(tool_args["query"])
                    
                    vector_stmt = (
                        select(models.KnowledgeChunk)
                        .order_by(models.KnowledgeChunk.embedding.cosine_distance(query_vector))
                        .limit(3)
                    )
                    vector_res = await db.execute(vector_stmt)
                    matched_chunks = vector_res.scalars().all()
                    
                    context_text = "\n---\n".join([c.content for c in matched_chunks]) if matched_chunks else "No content found."
                    tool_results.append(f"Tool [SearchKnowledgeBase] Output:\n{context_text}")
                    
                elif tool_name == "GetSystemDiagnostics":
                    try:
                        res = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
                        ext = res.scalar()
                        status = f"Active connected. pgvector status: {ext}"
                    except Exception as e:
                        status = f"Database connectivity error: {str(e)}"
                    tool_results.append(f"Tool [GetSystemDiagnostics] Output: {status}")

            # Synthesize final answer using the telemetry outputs alongside history
            combined_tool_context = "\n\n".join(tool_results)
            synthesis_prompt = (
                f"Executed Tool Telemetry Output blocks:\n{combined_tool_context}\n\n"
                "Synthesize your final comprehensive answer now based on this output and the thread context."
            )
            
            # Temporarily append the telemetry results to guide the final pass
            message_list.append(AIMessage(content=f"[System Executed Tools: {executed_tools_list}]"))
            message_list.append(HumanMessage(content=synthesis_prompt))
            
            synthesis_res = await llm.ainvoke(message_list)
            final_answer = synthesis_res.content

        # 5. Commit State Changes: Save the current exchange logs down to PostgreSQL
        user_log = models.ChatMessage(session_id=payload.session_id, role="user", content=payload.user_prompt)
        agent_log = models.ChatMessage(session_id=payload.session_id, role="model", content=final_answer)
        
        db.add(user_log)
        db.add(agent_log)
        await db.commit()

        return {
            "session_id": payload.session_id,
            "response_source": response_source,
            "executed_tools": executed_tools_list,
            "answer": final_answer
        }
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Agent Stateful Reasoning Core Breakout: {str(e)}")


