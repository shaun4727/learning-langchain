### Lesson 3: PostgreSQL & pgvector Setup

#### The Architectural "Why"

In Phase 1, we learned how to use LLMs as deterministic extraction filters. However, an LLM's internal memory is strictly limited by its **context window boundary**. If you pass a 500-page enterprise technical manual or an entire production log history directly into a Gemini prompt, you will hit context limitations, incur massive financial token costs, and degrade response speed.

To scale our applications to handle massive domain knowledge, we must separate **Storage** from **Compute**.

We will use **PostgreSQL** coupled with the native **`pgvector`** extension as our relational vector database. Instead of storing just raw text strings, we convert text paragraphs into mathematical arrays of floating-point numbers called **Embeddings** (which represent the deep semantic meaning of the words).

When a user submits a query, we transform that query into an embedding vector, execute an optimized mathematical vector distance calculation inside PostgreSQL (like Cosine Similarity or Inner Product), and pull only the most relevant text slices to send to Gemini.

Using native PostgreSQL for vector storage—rather than a standalone specialized vector database—allows us to run relational joins, apply ACID transaction compliance, and manage strict data relational integrity in a single, unified database instance.

---

### Step 1: Modifying the Multi-Container Infrastructure

We need to evolve our infrastructure from a single FastAPI container to a coordinated, multi-service network. We will inject a PostgreSQL instance pre-bundled with the `pgvector` compilation binaries, along with an orchestration management panel (`pgAdmin`) to verify data health visually.

| Configuration Component | Target Configuration Change | Architectural Purpose |
| --- | --- | --- |
| **`docker-compose.yml`** | Append `db` (Postgres 16 + pgvector) and `pgadmin` services. Create a bridge network and a persistent database volume. | Guarantees data persistence across container rebuilds and links the containers into an isolated network. |
| **`requirements.txt`** | Add `asyncpg` and `greenlet`. | Installs high-performance, asynchronous non-blocking PostgreSQL drivers for FastAPI. |
| **`.env`** | Add database credentials (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`). | Centralizes database credentials securely away from the application code layer. |

---

### Step 2: The Infrastructure File Updates

**1. `requirements.txt**`
Update your file to append the asynchronous Postgres drivers and standard object-relational mapping (ORM) dependencies:

```text
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
langchain-core>=0.3.0
langchain-google-genai>=1.0.0
pydantic>=2.0.0
asyncpg>=0.29.0
sqlalchemy>=2.0.0
greenlet>=3.0.0

```

**2. `.env**`
Append your database orchestration credentials to your existing local environment file:

```env
GEMINI_API_KEY=AIzaSyYourActualGeminiApiKeyHere

# PostgreSQL Configuration Boundary
POSTGRES_USER=shaun_admin
POSTGRES_PASSWORD=secure_monon_pass
POSTGRES_DB=agentic_vector_store
POSTGRES_HOST=db
POSTGRES_PORT=5432

```

**3. `docker-compose.yml**`
Replace your current configuration with this multi-service architecture topology:

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    env_file:
      - .env
    depends_on:
      db:
        condition: service_healthy
    networks:
      - agentic_network

  db:
    image: pgvector/pgvector:pg16
    container_name: agentic_postgres_db
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - agentic_network

  pgadmin:
    image: dpage/pgadmin4
    container_name: agentic_pgadmin_ui
    ports:
      - "5050:80"
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@mononsoft.local
      PGADMIN_DEFAULT_PASSWORD: root
    depends_on:
      - db
    networks:
      - agentic_network

volumes:
  postgres_data:

networks:
  agentic_network:
    driver: bridge

```

---

### Step 3: Setting Up Asynchronous Database Session Management

Create a new file in your backend folder named `database.py`. This script initializes the asynchronous database engine and establishes a context manager to handle database connections safely.

```python
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

# Construct the asynchronous connection string matrix
DATABASE_URL = f"postgresql+asyncpg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

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

```

---

### Step 4: Modifying `main.py` for Database Lifecycle and Vector Registration

Now, update `main.py` to activate the `pgvector` extension dynamically on database boot and expose a clean database diagnostic endpoint.

```python
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from database import engine, Base, get_db

app = FastAPI(title="Agentic AI Backend - Data Cluster", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables and register pgvector native binary structures
@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        # Register the pgvector extension natively inside the core database instance
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Compile tables mapped via SQLAlchemy models (to be fully utilized in Lesson 4)
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agentic-ai-core"}

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

```

---

### Actionable Task

1. Save the updated layout versions of `requirements.txt`, `.env`, `docker-compose.yml`, and `main.py`, and create the new `database.py` file.
2. Spin up your multi-container environment in your terminal window:
```bash
docker compose up --build

```


3. Look closely at the terminal console text trail. You should see three key milestones:
* The `db` healthcheck evaluating successfully to a healthy status.
* The `api` container booting up without throwing any `asyncpg` or connection errors.


4. Open your browser and navigate to the diagnostic endpoint: `http://localhost:8000/db-test`.

When the endpoint returns `{"status": "connected", "database": "postgresql", "pgvector_extension": "active"}`, reply with **Finished**. Let me know if the Docker container network initialization hits any speed bumps along the way!