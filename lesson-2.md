### Lesson 2: LangChain Core & Structured Output

#### The Architectural "Why"

Traditional LLM integrations rely on prose prompts asking the model to *"Return valid JSON"*. This introduces a high risk of parsing failures because the model can easily wrap responses in markdown backticks (e.g., ````json ... ````), hallucinate unauthorized fields, or shift data types when under heavy load.

To build deterministic multi-agent applications, we require absolute structural guarantees. We will use LangChain's `.with_structured_output()` feature combined with Gemini's native `json_schema` response configuration mode. This bypasses standard text-parsing post-processors completely: the Gemini engine natively forces its internal token generation steps to adhere strictly to your specified schema. If the model outputs data that mismatches your specifications, validation flags it immediately at the boundary before bad or unstructured data can pollute your downstream services.

---

#### Step 1: Upgrading Dependencies & Environment Variables

We must update your `requirements.txt` to add LangChain's core engine alongside the Google GenAI integration package. We will also introduce a `.env` file to securely feed your Gemini API key to the Docker daemon.

| Configuration File | Target Configuration Change | Purpose |
| --- | --- | --- |
| `requirements.txt` | Add `langchain-core`, `langchain-google-genai`, and `pydantic>=2.0` | Installs the modern LangChain Expression Language (LCEL) ecosystem and Gemini drivers. |
| `.env` | Add `GEMINI_API_KEY=your_api_key_here` | Injects your secret credentials securely into the runtime environment without hardcoding values in code. |
| `docker-compose.yml` | Map `env_file: - .env` | Instructs the docker engine to automatically bind local environment variables into the running container lifecycle. |

---

#### Step 2: The Code Implementation

**1. `requirements.txt**`
Update your local file to match this complete library configuration:

```text
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
langchain-core>=0.3.0
langchain-google-genai>=1.0.0
pydantic>=2.0.0

```

**2. `.env**`
Create a new file named `.env` in your root backend folder (at the same level as `main.py`):

```env
GEMINI_API_KEY=AIzaSyYourActualGeminiApiKeyHere

```

**3. `docker-compose.yml**`
Modify your docker-compose file to pipe the `.env` configuration file directly down to the running application service context:

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

```

**4. `main.py**`
We will now define a targeted data schema (`TechnicalFeatureExtractor`) and set up an asynchronous extraction endpoint (`/extract-features`) using LCEL components.

*Official Documentation Reference:* [LangChain ChatGoogleGenerativeAI Structured Output Docs](https://reference.langchain.com/python/langchain-google-genai/chat_models/ChatGoogleGenerativeAI/with_structured_output)

```python
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI

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

```

---

### Actionable Task

1. Save the updated versions of `requirements.txt`, `.env`, `docker-compose.yml`, and `main.py` inside your local workspace.
2. Spin up the container environment in your terminal window to trigger container compilation and install the newly declared dependencies:
```bash
docker compose up --build

```


3. Once the server registers a clean initialization, open your browser and head over to the Swagger docs UI at `http://localhost:8000/docs`.
4. Click into the new `POST /extract-features` block, select **Try it out**, and paste this specific string exactly into the `text_input` query field:
> *"We are attempting to transition our unified microservice framework from Python 3.10 to Python 3.12. However, the legacy deployment automation script written in bash crashed because it attempts to compile ancient Docker Compose images which fail to pull down safely due to an invalid authentication layer block on our local domain socket permissions."*


5. Run the request and confirm that the response output block cleanly emits structured JSON data conforming perfectly to your Pydantic properties.

Once the payload displays successfully without markdown blocks, reply with the word **Finished**. Let me know if any dependencies trigger compile conflicts along the way!