from sqlalchemy import Column, Integer, Text, String
from pgvector.sqlalchemy import Vector
from database import Base

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    source_file = Column(String(255), nullable=True)
    embedding = Column(Vector(768), nullable=False)