from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import adalflow as adal
from src.rag import RAG
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from pydantic_models import *

def load_environment():
    """Load environment variables from .env file if available, otherwise use system environment variables."""
    try:
        # Try to load from .env file for local development
        load_dotenv()
        print("Loaded environment variables from .env file")
    except FileNotFoundError:
        # In production, env variables should be set in the environment
        print("No .env file found, using system environment variables")
    except Exception as e:
        print(f"Note: Error loading .env file: {e}")

# Load environment variables
load_environment()

# Check for required environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is required. Either:\n"
        "1. Create a .env file with OPENAI_API_KEY=your-key-here (for local development)\n"
        "2. Set the environment variable in your deployment platform (for production)"
    )

# Initialize FastAPI app
app = FastAPI(
    title="GithubChat API", 
    description="API for querying GitHub repositories using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG component
try:
    # Set up adalflow environment
    os.environ["OPENAI_API_KEY"] = openai_api_key
    adal.setup_env()
    rag = RAG()
    print("Successfully initialized RAG component")
except Exception as e:
    print(f"Error initializing RAG component: {e}")
    raise RuntimeError(f"Failed to initialize RAG component: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """
    Query a GitHub repository with RAG
    
    Args:
        request: QueryRequest containing repo_url and query
        
    Returns:
        QueryResponse containing the answer, rationale, and relevant contexts
    """
    try:
        # Prepare retriever for the repository
        rag.prepare_retriever(request.repo_url)
        
        # Get response and retrieved documents
        response, retrieved_documents = rag(request.query)
        
        # Format response
        return QueryResponse(
            rationale=response.rationale if hasattr(response, 'rationale') else "",
            answer=response.answer if hasattr(response, 'answer') else response.raw_response,
            contexts=[
                Document(
                    text=doc.text,
                    meta_data=DocumentMetadata(
                        file_path=doc.meta_data.get('file_path', ''),
                        type=doc.meta_data.get('type', ''),
                        is_code=doc.meta_data.get('is_code', False),
                        is_implementation=doc.meta_data.get('is_implementation', False),
                        title=doc.meta_data.get('title', '')
                    )
                )
                for doc in retrieved_documents[0].documents
            ] if retrieved_documents and retrieved_documents[0].documents else []
        )
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)  # Log the error
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "GithubChat API",
        "description": "API for querying GitHub repositories using RAG",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 