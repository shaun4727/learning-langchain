import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# 1. Prioritize the unified connection string (Production / Render)
DATABASE_URL = os.getenv("DATABASE_URL")

# 2. Fall back to building it manually if DATABASE_URL is missing (Local Docker Compose)
if not DATABASE_URL:
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', '')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'postgres')
    DATABASE_URL = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"
else:
    # Auto-patch Neon's default 'sslmode' query parameter for asyncpg compatibility
    if "sslmode=require" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("sslmode=require", "ssl=require")

# Initialize the high-performance async database engine
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# Generate an asynchronous session maker factory
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    """Abstract base class for all relational and vector database models."""
    pass

# Dependency injection generator for FastAPI routes
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()