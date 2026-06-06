from sqlalchemy import Column, Integer, Text, String, DateTime, func
from pgvector.sqlalchemy import Vector
from database import Base

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    source_file = Column(String(255), nullable=True)
    embedding = Column(Vector(768), nullable=False)




class ChatMessage(Base):
    """Database model for storing short-term session conversation history logs dynamically."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True) # Indexed for high-speed chronological retrieval
    role = Column(String(50), nullable=False) # Stores 'user' or 'model'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())