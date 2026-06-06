### Lesson 6: Persistent Conversation Memory & Session State Management

#### The Architectural "Why"

In Lesson 5, you successfully transformed your RAG pipeline into an **Autonomous Agent** capable of executing dynamic tools based on user intent. However, that agent has one glaring limitation: it is entirely **stateless**. Every time you hit the `/agent-chat` endpoint, the system treats the incoming request as a brand-new interaction. If you tell the agent your name or ask a follow-up question like *"Can you elaborate on that first point?"*, it will fail because it retains zero awareness of the immediate past.

To build a production-grade chat experience (like a real-world web application dashboard), an agent requires **Persistent Conversation Memory**.

We will implement a relational state tracking matrix inside PostgreSQL. Instead of storing history in volatile local server memory (which wipes out whenever Docker containers scale or restart), every text exchange will be tagged with a unique `session_id`, safely written to the database kernel, and hydrated on-the-fly into the Gemini context window during execution.

---

### Memory Layer Lifecycle

| Sequence Stage | System Component | Technical Mechanism | Operational Objective |
| --- | --- | --- | --- |
| **1. Session Hydration** | PostgreSQL + SQLAlchemy | Queries `chat_history` filtered by incoming `session_id` ordered chronologically. | Reconstructs the exact short-term memory of the conversation. |
| **2. Context Window Assembly** | LangChain Core | Maps DB records into an array of `HumanMessage` and `AIMessage` objects. | Feeds historical context alongside new prompts into Gemini. |
| **3. Reasoning & Tool Execution** | `gemini-2.5-flash` | Evaluates the entire conversational chain to decide tool execution paths. | Allows follow-up context to influence dynamic tool parameters. |
| **4. State Persistence** | Async Database Pipeline | Commits both the raw user input and the final agent answer as new database rows. | Secures the dialogue log persistently before ending the request stream. |

---

### Step 1: Extending the Database Schema Model

Open your **`models.py`** file. We need to add a new declarative table schema called `ChatMessage` to handle message strings, execution roles (`user` or `model`), and persistent string-based session keys.

Append this class definition to your **`models.py`**:

```python
from sqlalchemy import Column, Integer, Text, String, DateTime, func
from pgvector.sqlalchemy import Vector
from database import Base

# Keep your existing KnowledgeChunk class above...

class ChatMessage(Base):
    """Database model for storing short-term session conversation history logs dynamically."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True) # Indexed for high-speed chronological retrieval
    role = Column(String(50), nullable=False) # Stores 'user' or 'model'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())

```

---

### Step 2: Upgrading the Agent Communication Payloads

Open **`main.py`**. To transition our chat system from standard query strings to structured transactional payloads, we must first upgrade our top imports and replace the old Pydantic schema with a state-aware schema.

1. Add these classes to the top import block of your **`main.py`**:

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

```

2. Replace your previous basic prompt requests with this structured `AgentChatRequest` body class:

```python
class AgentChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique thread identifier tracking a single distinct conversation conversation.")
    user_prompt: str = Field(..., description="The textual query or follow-up instruction passed to the agent.")

```

---

### Step 3: Upgrading the `/agent-chat` Endpoint

Now, let's replace your previous stateless agent implementation inside **`main.py`**. The updated code handles the complete memory lifecycle: it fetches old messages, structures them, executes tools if necessary, and saves the new transaction logs back to Postgres.

Replace your entire `@app.post("/agent-chat")` endpoint with this stateful architecture:

```python
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

```

---

### Actionable Task

1. Save your updated code files (`models.py` and `main.py`).
2. Because we altered our structural SQL database models by introducing the `chat_history` table schema, we must tell Docker to wipe the temporary transactional volumes and force clean initial compilations:

```bash
docker compose down --volumes
docker compose up --build -d

```

3. Run a sequential execution thread verification block in an alternate terminal to watch short-term memory operate in real-time. We will use a shared `session_id` called `"dev_session_01"`:

* **Turn 1: Instruct the agent with custom parameter facts:**
```bash
curl -X POST "http://localhost:8000/agent-chat" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "dev_session_01", "user_prompt": "Hello! I am Shaun Hossain, a developer testing this interface."}'

```


*(The agent will reply directly acknowledging you).*
* **Turn 2: Run a follow-up query requiring short-term memory access (No name passed here):**
```bash
curl -X POST "http://localhost:8000/agent-chat" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "dev_session_01", "user_prompt": "What was my name again? Check if my system extensions are active too."}'

```



4. Check your terminal output response. Because the agent parses your tracking session thread history, it will accurately state your name (**Shaun Hossain**) while simultaneously calling the `GetSystemDiagnostics` tool to check your `pgvector` extensions!

Once your stateful logs execute correctly, reply with **Finished** to conclude Lesson 6.