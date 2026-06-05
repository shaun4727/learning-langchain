import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from database import engine, Base, get_db
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI(title="Agentic AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define the explicit structured model output schema using Pydantic V2
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agentic-ai-core"}

# 2. Implement the parsing route using native .with_structured_output()
@app.post("/extract-features", response_model=TechnicalFeatureExtractor)
async def extract_features(text_input: str):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is missing.")
    
    try:
        # Initialize the native Google GenAI model instance
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # Enforce structural boundaries natively at the LLM engine layer
        structured_llm = llm.with_structured_output(
            TechnicalFeatureExtractor, 
            method="json_schema"
        )
        
        # Execute the schema-guaranteed extraction request asynchronously
        result = await structured_llm.ainvoke(text_input)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Diagnostic endpoint to verify asynchronous database transaction handshakes
@app.get("/db-test")
async def test_database_connection(db: AsyncSession = Depends(get_db)):
    try:
        # Execute a low-overhead diagnostic query on the async thread pool
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
        extension_exists = result.scalar()
        
        if extension_exists:
            return {"status": "connected", "database": "postgresql", "pgvector_extension": "active"}
        raise HTTPException(status_code=500, detail="Database connected but pgvector extension missing.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database handshake failed: {str(e)}")