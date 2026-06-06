### Lesson 5: Autonomous Agents & Dynamic Tool Execution (Function Calling)

#### The Architectural "Why"

In Lesson 4, you built a deterministic **Retrieval-Augmented Generation (RAG)** pipeline. It followed a strict, hardcoded loop: take a question $\rightarrow$ generate a query embedding $\rightarrow$ search PostgreSQL $\rightarrow$ synthesize an answer. While powerful, this structure is rigid. If a user asks a question that requires a direct database calculation, a health check, or a structured feature extraction, a standard RAG pipeline will blindly try to look up vector matches anyway, often resulting in irrelevant context.

**Lesson 5 introduces Autonomous Agents.** Instead of hardcoding the execution path, we hand the LLM a collection of specialized "Tools" (Python functions). When a request hits the API, the Gemini reasoning engine acts as a brain: it analyzes the input, determines its own intent, dynamically selects the correct tool to call, processes the results, and loops until it can deliver an optimal answer.

---

### Agentic Loop Lifecycle

| Sequence Stage | Component | Technical Mechanism | Operational Objective |
| --- | --- | --- | --- |
| **1. Intent Analysis** | `gemini-2.5-flash` | Processes user prompt against schemas of all registered tools. | Decides whether to use a tool or reply directly. |
| **2. Tool Selection** | Native Tool Binding | Generates a structured `tool_calls` payload containing function names and exact arguments. | Emits an execution directive instead of raw text. |
| **3. Core Execution** | FastAPI Application Layer | Intercepts the directive, runs the corresponding Python function (e.g., SQL lookup). | Bridges the LLM's reasoning with real-world infrastructure. |
| **4. Final Synthesis** | Grounded Context Loop | Feeds the raw tool outputs back into the LLM context window. | Formulates a finalized, factually accurate response for the user. |

---

### Step 1: Defining the Agent's Tool Schemas

To let Gemini understand what tools are available and what parameters they require, we define them using Pydantic schemas. We will create two tools for our agent:

1. `SearchKnowledgeBase`: Allows the agent to query your vector database dynamically when asked about documents or resumes.
2. `GetSystemDiagnostics`: Allows the agent to check database connectivity and infrastructure extensions instantly.

Append these schemas to the bottom of your **`main.py`** file:

```python
# =====================================================================
# PHASE 4: AUTONOMOUS AGENT LAYER & TOOL DEFINITIONS
# =====================================================================

class SearchKnowledgeBase(BaseModel):
    """Search the vectorized system knowledge store for engineering documentation, skills, projects, or resume details."""
    query: str = Field(..., description="The semantic search query targeting historical document fragments.")

class GetSystemDiagnostics(BaseModel):
    """Fetch live infrastructure environment runtime states, database health, and extension configurations."""
    confirm: bool = Field(..., description="Set to True to trigger a live database connection handshake check.")

```

---

### Step 2: Injecting the Dynamic Agent Endpoint

Now, we will implement the routing mechanism. When a user sends a prompt to `/agent-chat`, the system binds these schemas to `gemini-2.5-flash`. If Gemini decides a tool is needed, our code executes the logic immediately using your active database configurations.

Append this complete endpoint function to the very bottom of your **`main.py`**:

```python
@app.post("/agent-chat")
async def agent_reasoning_engine(user_prompt: str, db: AsyncSession = Depends(get_db)):
    """Autonomous Agent Endpoint: Dynamically reasons, executes functional code tools, and synthesizes answers."""
    try:
        # 1. Initialize the core model and bind the structural tool schemas
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        llm_with_tools = llm.bind_tools([SearchKnowledgeBase, GetSystemDiagnostics])
        
        # 2. Execute initial reasoning pass
        ai_msg = await llm_with_tools.ainvoke(user_prompt)
        
        # If the model does not require any tools, return its direct text response immediately
        if not ai_msg.tool_calls:
            return {"response_source": "direct_llm", "answer": ai_msg.content}
            
        # 3. Process tool directives emitted by the model
        tool_results = []
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "SearchKnowledgeBase":
                # Execute the exact vector search logic built in Lesson 4 dynamically
                embeddings_engine = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    output_dimensionality=768
                )
                query_vector = embeddings_engine.embed_query(tool_args["query"])
                
                stmt = (
                    select(models.KnowledgeChunk)
                    .order_by(models.KnowledgeChunk.embedding.cosine_distance(query_vector))
                    .limit(3)
                )
                result = await db.execute(stmt)
                matched_chunks = result.scalars().all()
                
                context_text = "\n---\n".join([c.content for c in matched_chunks]) if matched_chunks else "No content found."
                tool_results.append(f"Tool [SearchKnowledgeBase] Output:\n{context_text}")
                
            elif tool_name == "GetSystemDiagnostics":
                # Execute live database handshake check dynamically
                try:
                    res = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
                    ext = res.scalar()
                    status = f"Active connected. pgvector status: {ext}"
                except Exception as e:
                    status = f"Database connectivity error: {str(e)}"
                tool_results.append(f"Tool [GetSystemDiagnostics] Output: {status}")

        # 4. Synthesize final response by feeding tool telemetry back into the model
        combined_tool_context = "\n\n".join(tool_results)
        synthesis_prompt = (
            "You are an elite autonomous system agent executing software diagnostic loops.\n"
            "You formulated a plan and executed system tools. Review your execution outputs below and write a final, comprehensive answer to the user.\n\n"
            f"Executed Tool Telemetry:\n{combined_tool_context}\n\n"
            f"Original User Request: {user_prompt}"
        )
        
        final_response = await llm.ainvoke(synthesis_prompt)
        return {
            "response_source": "agent_tool_execution",
            "executed_tools": [tc["name"] for tc in ai_msg.tool_calls],
            "answer": final_response.content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Reasoning Core Breakout: {str(e)}")

```

---

### Actionable Task

1. Save your updated `main.py` file. Uvicorn will automatically reload your application.
2. Open an alternate terminal window to test how your system routes calls based on context:

* **Test Case A (Triggers System Diagnostics Tool):**
```bash
curl -X 'POST' 'http://localhost:8000/agent-chat?user_prompt=check%20if%20my%20database%20and%20pgvector%20extensions%20are%20online'

```


* **Test Case B (Triggers Knowledge Base Vector Tool):**
```bash
curl -X 'POST' 'http://localhost:8000/agent-chat?user_prompt=look%20up%20what%20skills%20are%20listed%20on%20my%20resume'

```


* **Test Case C (Direct Reply - No Tools Triggered):**
```bash
curl -X 'POST' 'http://localhost:8000/agent-chat?user_prompt=write%20a%20one%20sentence%20greeting%20to%20a%20developer'

```



3. Observe how the `"response_source"` and `"executed_tools"` values change based on the nature of your request.

Once your endpoints execute successfully and you analyze how the tool routing behaves under different prompts, report back with **Finished** to complete the final structural milestone of this architectural core!


# # Explain lesson 5

Yes, you have the exact right idea! Instead of your frontend or client application having to guess whether to query a vector database or check a system health status, you route everything through a single chat interface.

However, there is an important technical distinction in how this works under the hood: the system does not actually make external HTTP network requests to your other endpoints (like `/db-test` or `/rag-ask`). Instead, the `/agent-chat` endpoint acts as an intelligent **central gateway**. It uses the Gemini reasoning engine to look at your prompt, match it against your registered Python functions (tools), and execute the exact database or vector math logic natively within the same call.

### How the Agent Routes Your Intent

| If You Type This in Chat... | Gemini Automatically Detects... | Internal Function Executed | The Action It Performs |
| --- | --- | --- | --- |
| *"What tools are listed on my resume?"* | **Document Retrieval Intent** | `SearchKnowledgeBase` | Generates a query embedding and runs a cosine distance vector match in PostgreSQL. |
| *"Is the database connection working?"* | **Infrastructure Diagnostics Intent** | `GetSystemDiagnostics` | Triggers a live async database connection handshake check natively. |
| *"Explain what a fullstack developer does."* | **General Knowledge Intent** | *No Tool Required* | Gemini answers directly from its own training knowledge without hitting the database. |

### Why this is a Massive Upgrade for Your Application

1. **Cleaner Frontend Architecture:** In your Next.js frontend, you no longer need complex state handling, multiple pages, or conditional buttons to decide which API endpoint to hit. You only need **one text input field** connected to `/agent-chat`.
2. **Contextual Intelligence:** Because Gemini receives the raw output of these executed tools *before* replying, it can combine information. For example, if you ask *"Check if the database is online, and if it is, tell me the first skill listed in my resume,"* the agent will automatically execute **both** tools sequentially in a single turn and synthesize a complete, unified answer.

You can save your file and start experimenting with these different phrasing test cases to watch how dynamically the backend switches its execution paths!