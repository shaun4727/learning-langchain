
## 🚀 Live Deployment & Testing

The FastAPI backend for this curriculum is actively deployed and running live on **Render**. You can interact with the streaming Agentic AI engine directly from your local terminal using the configuration below.

* **Production Base URL:** `https://learning-langchain.onrender.com`
* **Streaming Endpoint:** `/agent-chat`

---

### 🛠️ Verification via cURL

To test the agent's tool-calling, autonomous decision-making, and streaming capabilities, run the following command in your terminal:

```bash
curl -N -X POST "[https://learning-langchain.onrender.com/agent-chat](https://learning-langchain.onrender.com/agent-chat)" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "stream_session_01", 
       "user_prompt": "Check if my database extensions are online and write a short poem about coding."
     }'


### Agentic AI Architecture Curriculum

| Phase | Lesson | Core Concepts & Architecture | Official Documentation | Target Actionable Task |
| --- | --- | --- | --- | --- |
| **Phase 1: Foundation** | 1. FastAPI Microservice & Environment | Pydantic validation, CORS, Docker configuration, and Prettier integration. | [FastAPI Docs](https://fastapi.tiangolo.com/) | Initialize a containerized FastAPI application with a basic health-check endpoint. |
|  | 2. LangChain Core & Structured Output | `ChatModels`, `ChatPromptTemplates`, and `with_structured_output` for predictable API responses. | [LangChain Core](https://python.langchain.com/docs/concepts/core/) | Build an endpoint that forces the LLM to return a strictly typed JSON response. |
| **Phase 2: RAG & pgvector** | 3. PostgreSQL & pgvector Setup | Vector spaces, connection pooling, and `langchain-postgres` initialization. | [pgvector Integration](https://python.langchain.com/docs/integrations/vectorstores/pgvector/) | Spin up a PostgreSQL+pgvector database via Docker and securely connect it to FastAPI. |
|  | 4. Advanced Retrieval Pipelines | Embeddings, document ingestion, and integrating vector similarity search into a retriever. | [Retrievers](https://python.langchain.com/docs/concepts/retrievers/) | Ingest sample documents and create an endpoint that returns mathematically relevant context. |
| **Phase 3: LangGraph** | 5. StateGraph & Workflow Fundamentals | `StateGraph`, Nodes, Edges, and defining the global state schema. | [LangGraph Core](https://langchain-ai.github.io/langgraph/concepts/low_level/) | Build a linear, multi-step graph that generates, critiques, and refines text. |
|  | 6. Agentic Behavior & Tool Calling | Binding tools to LLMs, `ToolNode`, and conditional edges for autonomous decision-making. | [Tool Calling](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/) | Create an agent that autonomously decides whether to query pgvector or execute a calculation. |
|  | 7. Memory & Persistence | `MemorySaver`, thread IDs, and conversational context persistence within LangGraph. | [Memory Concepts](https://langchain-ai.github.io/langgraph/concepts/memory/) | Implement cross-turn memory so the agent remembers previous user interactions perfectly. |
| **Phase 4: Integration** | 8. Frontend Implementation | Consuming streaming API endpoints in Next.js, advanced styling, and client-side state. | [Next.js Fetching](https://nextjs.org/docs) | Build the chat UI and connect it to the Agentic FastAPI backend. |

Please review the roadmap above. I will wait for your approval before we start Lesson 1.