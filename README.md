# AI_Learning_Repo

Practical AI learning projects covering LLM applications, LangChain, RAG,
tool-using agents, machine learning, FastAPI, Gradio, and Docker deployment.

## Prerequisites

- Python 3.10 or newer
- Git
- An OpenAI API key for projects 1, 3, 4, 5, 6, 7, 9, and 10
- A Groq API key for project 2
- Docker Desktop for projects 8, 9, and 10
- AWS credentials and an Amazon Bedrock model ID for project 11

Create and activate a virtual environment before running a local Python project:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Store API keys in a local `.env` file where the project uses `python-dotenv`.
Never commit secrets to the repository.

## Project Summary

| # | Project | Summary |
|---|---|---|
| 1 | [AI Summarizer](AI_Mini_Projects/1%20-%20AI_Summarizer) | Fetches a website and creates a short summary with an OpenAI model and Gradio. |
| 2 | [Arena AI](AI_Mini_Projects/2%20-%20Arena_AI) | Compares answers from OpenAI and Groq models for the same prompt and collects a vote. |
| 3 | [AI LangChain Summarizer](AI_Mini_Projects/3%20-%20AI_Langchain_Summarizer) | Builds a website summarizer with a LangChain prompt, model, and output parser chain. |
| 4 | [AI Memory Chat](AI_Mini_Projects/4%20-%20AI_Memory_Chat) | Maintains conversation history so the assistant can use previous chat turns. |
| 5 | [AI Smart Shop Agent](AI_Mini_Projects/5%20-%20AI_Smart_Shop_Agent) | Uses an LLM tool-calling loop to look up product prices through a shopping assistant. |
| 6 | [AI PDF Chat](AI_Mini_Projects/6%20-%20AI_PDF_Chat) | Embeds an uploaded PDF and answers questions with RAG and page references. |
| 7 | [AI Sessions Chat](AI_Mini_Projects/7%20-%20AI_Sessions_Chat) | Indexes web pages and answers questions using vector search and retrieved context. |
| 8 | [AI QuickBite ETA](AI_Mini_Projects/8%20-%20AI_QuickBite-ETA) | Predicts food-delivery arrival time with scikit-learn and serves it through FastAPI. |
| 9 | [AI ScalerGPT](AI_Mini_Projects/9%20-%20AI_Scaler_GPT) | Runs a Docker Compose RAG chatbot over Markdown and text notes using ChromaDB. |
| 10 | [AI DeskBuddy](AI_Mini_Projects/10%20-AI_DeskBuddy) | Runs an agent, private tools service, and Redis-backed conversation memory with Docker Compose. |
| 11 | [AI Travel Assistant Agent](AI_Mini_Projects/11%20-%20AI_Travel_Assistant_Agent) | Chains weather, packing, cost-estimation, and calculator tools to plan a trip against a budget. |

## Running the Projects

Run each command from the corresponding project directory.

### 1. AI Summarizer

Install `gradio`, `openai`, `requests`, `beautifulsoup4`, and `python-dotenv`,
set `OPENAI_API_KEY` in `.env`, then run:

```bash
python pro3_summarizer_app.py
```

Open the Gradio URL printed in the terminal and enter a website URL.

### 2. Arena AI

Install `openai`, `gradio`, and `python-dotenv`. Set both
`OPENAI_API_KEY` and `GROQ_API_KEY` in `.env`, then run:

```bash
python pro4_arena_app.py
```

Open the Gradio URL to send one prompt to both models.

### 3. AI LangChain Summarizer

Install `langchain-openai`, `langchain-core`, `requests`, `beautifulsoup4`, and
`python-dotenv`. Set `OPENAI_API_KEY`, then run:

```bash
python pro1_summarizer_langchain.py
```

### 4. AI Memory Chat

Install `langchain-openai`, `langchain-core`, and `python-dotenv`. Set
`OPENAI_API_KEY`, then run:

```bash
python pro2_memory_chat.py
```

Type `quit` or `exit` to end the conversation.

### 5. AI Smart Shop Agent

Install `openai`, `gradio`, and `python-dotenv`. Set `OPENAI_API_KEY`, then run:

```bash
python pro3_shop_assistant_app.py
```

Ask the assistant for the price of shoes, a hat, a bag, shorts, or pants.

### 6. AI PDF Chat

Install the dependencies listed at the top of `pro7_pdf_chat.py`, set
`OPENAI_API_KEY`, then run:

```bash
python pro7_pdf_chat.py
```

Open the Gradio URL, upload a PDF, and ask questions about its contents.

### 7. AI Sessions Chat

Install the dependencies listed at the top of `pro8_ai_sessions_chat.py`, then run:

```bash
python pro8_ai_sessions_chat.py
```

Set `OPENAI_API_KEY`, load one or more URLs, and ask questions about the indexed pages.

### 8. AI QuickBite ETA

Docker builds the model during image creation and starts the FastAPI service:

```bash
docker build -t quickbite-eta:v1 .
docker run --rm -p 8000:8000 --name eta-service quickbite-eta:v1
```

Open <http://localhost:8000/docs> to test `POST /predict`. Stop the service with
`Ctrl+C`.

### 9. AI ScalerGPT

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, ingest the sample notes, and
start the services:

```bash
docker compose run --rm app python ingest.py
docker compose up --build
```

Use the API at <http://localhost:8000/docs>. Stop the services with:

```bash
docker compose down
```

### 10. AI DeskBuddy

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, and start all services:

```bash
docker compose up --build
```

Use the agent API at <http://localhost:9000/docs>. Check service status with
`docker compose ps -a` and view logs with `docker compose logs <service>`.
Stop the services with:

```bash
docker compose down
```

### 11. AI Travel Assistant Agent

This project uses Amazon Bedrock through the Strands Agents SDK. Install the
dependencies required by the project, configure AWS credentials, and provide a
`config.py` on the Python import path with a `MODEL_ID` value. Then run:

```bash
python travel_assistant.py
```

You can also provide a request directly:

```bash
python travel_assistant.py "I'm going to Manali for 4 days, budget 15000"
```

The agent checks weather, suggests packing items, estimates the trip cost, and
compares the estimate with the supplied budget.

## Development Notes

- Run commands from the project directory so local imports and files resolve correctly.
- Keep virtual environments, `.env` files, generated models, vector databases, and caches out of Git.
- Rebuild Docker images after changing application code or dependencies.
- Consult the project-level README files for detailed API examples and troubleshooting.
