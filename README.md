# GithubChat

A RAG assistant to allow you to chat with any github repo. 
Learn fast. The default repo is AdalFlow github repo.

[![Watch the video](https://img.youtube.com/vi/PvZTSmTK8b0/maxresdefault.jpg)](https://youtu.be/PvZTSmTK8b0)
*Click the image above to watch the demo video*



## Project Structure
```
.
├── frontend/           # React frontend application
├── src/                # Python backend code
├── api.py              # FastAPI server
├── app.py              # Streamlit application
└── pyproject.toml      # Python dependencies
```

## Backend Setup

1. Install dependencies:
```bash
poetry install
```

2. Set up OpenAI API key:

Create a `.streamlit/secrets.toml` file in your project root:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

Add your OpenAI API key to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

## Running the Applications

### Streamlit UI
Run the streamlit app:
```bash
poetry run streamlit run app.py
```

### FastAPI Backend
Run the API server:
```bash
poetry run uvicorn api:app --reload
```
The API will be available at http://localhost:8000

### React Frontend
1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
pnpm install
```


3. Start the development server:
```bash
pnpm run dev
```
The frontend will be available at http://localhost:3000

## API Endpoints

### POST /query
Analyzes a GitHub repository based on a query.
```json
// Request
{
  "repo_url": "https://github.com/username/repo",
  "query": "What does this repository do?"
}

// Response
{
  "rationale": "Analysis rationale...",
  "answer": "Detailed answer...",
  "contexts": [...]
}
```

## ROADMAP
- [x] Clearly structured RAG that can prepare a repo, persit from reloading, and answer questions.
  - `DatabaseManager` in `src/data_pipeline.py` to manage the database.
  - `RAG` class in `src/rag.py` to manage the whole RAG lifecycle.

### On the RAG backend
- [ ] Conditional retrieval. Sometimes users just want to clarify a past conversation, no extra context needed.
- [ ] Create an evaluation dataset  
- [ ] Evaluate the RAG performance on the dataset  
- [ ] Auto-optimize the RAG model

### On the React frontend

- [ ] Support the display of the whole conversation history instead of just the last message.
- [ ] Support the management of multiple conversations.
