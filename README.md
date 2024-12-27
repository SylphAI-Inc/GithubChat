# RAG Code Assistant

A Retrieval-Augmented Generation (RAG) system for analyzing and understanding code repositories. The system provides both a command-line interface and a web UI for interacting with your codebase.

## Features

- Code-aware responses using RAG with Qdrant vector database
- Memory for maintaining conversation context
- Support for multiple programming languages
- Interactive web interface
- Command-line interface

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Set up Qdrant:

Option 1: Using Docker (recommended):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Option 2: Install locally:
```bash
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xvz
./qdrant
```

3. Set up OpenAI API key:

Create a `.streamlit/secrets.toml` file in your project root:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

Add your OpenAI API key to `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

## Running the Application

### Web Interface

1. Run the demo version (with test data):
```bash
poetry run streamlit run app.py
```

2. Run the repository analysis version:
```bash
poetry run streamlit run app_repo.py
```

### Command Line Interface

Run the RAG system directly:
```bash
poetry run python rag.py
```

## Usage Examples

1. **Demo Version (app.py)**
   - Ask about Alice (software engineer)
   - Ask about Bob (data scientist)
   - Ask about the company cafeteria
   - Test memory with follow-up questions

2. **Repository Analysis (app_repo.py)**
   - Enter your repository path
   - Click "Load Repository"
   - Ask questions about classes, functions, or code structure
   - View implementation details in expandable sections

## Vector Database

The system uses Qdrant as the vector database for efficient similarity search:
- Documents are automatically uploaded to Qdrant collection "code_chunks"
- Embeddings are created using OpenAI's text-embedding-3-small model
- Retrieval is optimized for code implementation queries
- Filters are applied to prioritize code over documentation

## Security Note

- Never commit your `.streamlit/secrets.toml` file
- Add it to your `.gitignore`
- Keep your API key secure

## Example Queries

- "What does the RAG class do?"
- "Show me the implementation of the Memory class"
- "How is data processing handled?"
- "Explain the initialization process"

## Troubleshooting

If you encounter issues:

1. Ensure Docker is running before starting Qdrant
   - Install Docker Desktop from https://www.docker.com/products/docker-desktop if not installed
   - Start Docker Desktop
   - Wait for Docker to be fully running before starting Qdrant

2. Verify your OpenAI API key is correctly set in `.streamlit/secrets.toml`
   - The key should be in quotes
   - The file should be in the `.streamlit` directory
   - The format should be exactly: `OPENAI_API_KEY = "your-key-here"`

3. Make sure all ports are available:
   - Qdrant uses port 6333
   - Streamlit uses port 8503
   - If either port is in use, you may need to stop other services or change the ports