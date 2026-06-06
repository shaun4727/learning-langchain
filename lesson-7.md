### Lesson 7: Real-Time Token Streaming & Async Event Yields (Server-Sent Events)

#### The Architectural "Why"

In Lesson 6, you built a robust, stateful autonomous agent that retains memory across conversations by saving and loading chat histories directly from a PostgreSQL database kernel. However, our communication pattern is still synchronous from a user experience (UX) perspective. When a client application hits the `/agent-chat` endpoint, the connection hangs until the agent completes its entire lifecycle: loading history $\rightarrow$ analyzing intent $\rightarrow$ executing a vector math tool $\rightarrow$ waiting for the final LLM response synthesis.

For complex requests, this can take several seconds, leaving users staring at a frozen loading spinner on your Next.js frontend.

**Lesson 7 introduces Asynchronous Event Streaming via Server-Sent Events (SSE).** Instead of returning a single bulk JSON object at the very end of processing, our FastAPI endpoint will immediately open an active HTTP connection channel and return a `StreamingResponse`. The backend will transform into an **Asynchronous Event Generator**, throwing off structured data yields in real-time as they happen:

1. A status event notifying the UI that the agent is actively analyzing intent.
2. A tool-execution event notifying the UI exactly which background tool (e.g., database lookup) has been engaged.
3. A real-time token stream that pushes individual words or text fragments straight into the browser the absolute millisecond Gemini outputs them.

---

### Async Streaming Lifecycle

| Sequence Stage | System Component | Communication Mechanism | Frontend UX Impact |
| --- | --- | --- | --- |
| **1. Stream Handshake** | FastAPI `StreamingResponse` | HTTP header `Content-Type: text/event-stream` | Immediate connection resolution; page loading state terminates. |
| **2. Status Telemetry** | Async Yield Loop | Emits an explicit JSON string block tagged as `status` or `tool_start`. | UI displays dynamic indicators like *"Agent is querying vector index..."* |
| **3. Generative Pipeline** | `llm.astream()` Chunks | Iterates through an asynchronous context cursor stream chunk-by-chunk. | Words type out on the screen in real-time with zero human-perceivable latency. |
| **4. Structural Wrap-up** | Background DB Commit | Persists historical role messages into the `chat_history` table after stream termination. | Cleans up system sockets cleanly without blocking the client screen. |

---

### Step 1: Upgrading Top-Level Imports

Open your **`main.py`** file. We need to import the native response engine that allows FastAPI to stream long-lived data segments instead of basic static models, alongside Python's standard `json` utility tool to serialize our messaging events.

Add these items to your existing top import declarations:

```python
import json
from fastapi.responses import StreamingResponse

```

---

### Step 2: Implementing the Asynchronous Streaming Architecture

We will now rewrite the `@app.post("/agent-chat")` endpoint. Because a streaming channel must continuously yield raw text line data chunks using the SSE protocol standard (`data: <payload>\n\n`), we will implement an internal **Asynchronous Generator Function** (`async def event_generator()`).

This internal generator handles the analytical planning, captures tool execution checkpoints, streams back text tokens line-by-line, and gracefully writes the final consolidated interaction history back to PostgreSQL.

Replace your entire previous `/agent-chat` route code inside **`main.py`** with this production streaming engine:

```python
@app.post("/agent-chat")
async def agent_reasoning_engine(payload: AgentChatRequest, db: AsyncSession = Depends(get_db)):
    """Stateful Streaming Agent: Resolves conversational history and streams tool execution logs and response tokens in real-time via SSE."""
    
    async def event_generator():
        try:
            # 1. Initial Handshake Telemetry
            yield f"data: {json.dumps({'event': 'status', 'message': 'Hydrating conversation memory...'})}\n\n"
            
            # Fetch chronological history for this session thread
            stmt = (
                select(models.ChatMessage)
                .where(models.ChatMessage.session_id == payload.session_id)
                .order_by(models.ChatMessage.timestamp.asc())
            )
            result = await db.execute(stmt)
            historical_rows = result.scalars().all()

            # Compile structural LangChain historical sequence matrix
            message_list = [
                SystemMessage(content=(
                    "You are an elite autonomous system agent executing software diagnostic loops.\n"
                    "Review past conversation history turns to satisfy follow-up instructions smoothly."
                ))
            ]
            for row in historical_rows:
                if row.role == "user":
                    message_list.append(HumanMessage(content=row.content))
                elif row.role == "model":
                    message_list.append(AIMessage(content=row.content))

            # Append current live prompt execution vector
            message_list.append(HumanMessage(content=payload.user_prompt))

            # 2. Analyze Intent & Determine Tool Requirements
            yield f"data: {json.dumps({'event': 'status', 'message': 'Analyzing request intent...'})}\n\n"
            
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            llm_with_tools = llm.bind_tools([SearchKnowledgeBase, GetSystemDiagnostics])
            
            # Primary reasoning loop execution pass
            ai_msg = await llm_with_tools.ainvoke(message_list)
            
            final_answer_text = ""
            executed_tools_list = []

            # 3. Dynamic Tool Branch Routing
            if ai_msg.tool_calls:
                tool_results = []
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    executed_tools_list.append(tool_name)
                    
                    # Notify the frontend UI exactly which system capability is being engaged
                    yield f"data: {json.dumps({'event': 'tool_start', 'tool': tool_name, 'message': f'Executing background system tool: {tool_name}'})}\n\n"
                    
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

                    yield f"data: {json.dumps({'event': 'tool_end', 'tool': tool_name})}\n\n"

                # Inject tool tracking observations into synthesis prompt sequence
                combined_tool_context = "\n\n".join(tool_results)
                synthesis_prompt = (
                    f"Executed Tool Telemetry Output blocks:\n{combined_tool_context}\n\n"
                    "Synthesize your final comprehensive answer now based on this output and the thread context."
                )
                message_list.append(AIMessage(content=f"[System Executed Tools: {executed_tools_list}]"))
                message_list.append(HumanMessage(content=synthesis_prompt))

            # 4. Asynchronous Token Generation Pass
            yield f"data: {json.dumps({'event': 'status', 'message': 'Generating final response...'})}\n\n"
            
            # Switch loop to astream chunk generation to extract tokens live
            async for chunk in llm.astream(message_list):
                if chunk.content:
                    final_answer_text += chunk.content
                    # Yield raw string token fragments immediately to the client network pipe
                    yield f"data: {json.dumps({'event': 'token', 'text': chunk.content})}\n\n"

            # 5. Post-Stream Memory Log Persistence
            user_log = models.ChatMessage(session_id=payload.session_id, role="user", content=payload.user_prompt)
            agent_log = models.ChatMessage(session_id=payload.session_id, role="model", content=final_answer_text)
            
            db.add(user_log)
            db.add(agent_log)
            await db.commit()
            
            # Sign off connection cleanly
            yield f"data: {json.dumps({'event': 'done', 'session_id': payload.session_id})}\n\n"

        except Exception as e:
            await db.rollback()
            yield f"data: {json.dumps({'event': 'error', 'detail': f'Streaming Pipeline Breakout: {str(e)}'})}\n\n"

    # Return structural content stream interface back to the server gateway
    return StreamingResponse(event_generator(), media_type="text/event-stream")

```

---

### Actionable Task

1. Save your newly rewritten `main.py` script file. Your Uvicorn server watch process inside Docker will reload automatically.
2. Open an alternate terminal window to run a network transaction request. To inspect streaming text updates directly in a shell environment without buffering, we supply the native **`-N` (no-buffer)** parameter to our `curl` execution utility:

```bash
curl -N -X POST "http://localhost:8000/agent-chat" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "stream_session_01", "user_prompt": "Check if my database extensions are online and write a short poem about coding."}'

```

3. Analyze your terminal output feedback layout closely.

Instead of sitting blank for three seconds and printing a bulk block, you will watch separate text blocks output instantly: first the memory hydration status, followed by the database diagnostic events, and finally individual code words streaming sequentially down your terminal pane!

Once your streaming endpoint prints logs successfully and you observe how individual events stream over the network socket channel, respond with **Finished** to advance to our final structural milestone: **Lesson 8**.