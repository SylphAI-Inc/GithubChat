import streamlit as st
import os
import tempfile
from rag import RAG
from adalflow.core.types import Document
from adalflow.components.data_process import TextSplitter, ToEmbeddings
from adalflow.core.db import LocalDB
import adalflow as adal
import glob
import re

def extract_class_definition(content: str, class_name: str) -> str:
    """Extract a complete class definition from the content."""
    lines = content.split('\n')
    class_start = -1
    indent_level = 0
    
    # Find the class definition start
    for i, line in enumerate(lines):
        if f"class {class_name}" in line:
            class_start = i
            # Get the indentation level of the class
            indent_level = len(line) - len(line.lstrip())
            break
    
    if class_start == -1:
        return content
    
    # Collect the entire class definition
    class_lines = [lines[class_start]]
    current_line = class_start + 1
    
    while current_line < len(lines):
        line = lines[current_line]
        # If we hit a line with same or less indentation, we're out of the class
        if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
            break
        class_lines.append(line)
        current_line += 1
    
    return '\n'.join(class_lines)

def process_repository_files(repo_path: str):
    """Process all code files in the repository."""
    # File extensions to look for, prioritizing code files
    code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']
    doc_extensions = ['.md', '.txt', '.rst', '.json', '.yaml', '.yml']
    
    documents = []
    st.write(f"Scanning repository at: {repo_path}")
    
    # Process code files first, with special handling for implementation files
    for ext in code_extensions:
        files = glob.glob(f"{repo_path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Skip virtual environment directories and UI files
            if '.venv' in file_path or 'node_modules' in file_path:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    # Determine if this is an implementation file
                    is_implementation = (
                        not relative_path.startswith('test_') and
                        not relative_path.startswith('app_') and
                        'test' not in relative_path.lower()
                    )
                    
                    st.write(f"Processing: {relative_path}")
                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": True,
                            "is_implementation": is_implementation
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                st.error(f"Error reading {file_path}: {str(e)}")
    
    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{repo_path}/**/*{ext}", recursive=True)
        for file_path in files:
            if '.venv' in file_path or 'node_modules' in file_path:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, repo_path)
                    st.write(f"Processing: {relative_path}")
                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": False,
                            "is_implementation": False
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                st.error(f"Error reading {file_path}: {str(e)}")
    
    return documents

# Initialize RAG system with repository data
@st.cache_resource
def init_rag(_repo_path: str):
    """Initialize RAG with repository data. Using _repo_path for cache key."""
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    # Create a temporary directory for the database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "repo_db")
    
    # Process repository files
    with st.spinner("Processing repository files..."):
        documents = process_repository_files(_repo_path)
        if not documents:
            st.error("No documents found in the repository!")
            return None
    
    # Initialize database with repository documents
    with st.spinner("Creating embeddings..."):
        data_transformer = adal.Sequential(
            TextSplitter(split_by="word", chunk_size=400, chunk_overlap=200),
            ToEmbeddings(
                embedder=adal.Embedder(
                    model_client=adal.OpenAIClient(),
                    model_kwargs={
                        "model": "text-embedding-3-small",
                        "dimensions": 256,
                        "encoding_format": "float",
                    }
                ),
                batch_size=100
            )
        )
        
        db = LocalDB("repo_db")
        db.register_transformer(transformer=data_transformer, key="split_and_embed")
        db.load(documents)
        db.transform(key="split_and_embed")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.save_state(filepath=db_path)
    
    # Create RAG instance with repository database using code analysis prompt
    return RAG(index_path=db_path, prompt_type="code_analysis")

def extract_class_name_from_query(query: str) -> str:
    """Extract class name from a query about a class."""
    # Common patterns for asking about classes
    patterns = [
        r'class (\w+)',
        r'the (\w+) class',
        r'what does (\w+) do',
        r'how does (\w+) work',
        r'show me (\w+)',
        r'explain (\w+)',
    ]
    
    query = query.lower()
    words = query.split()
    
    # First try to find class name using patterns
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            # Return the first match, capitalized
            return matches[0].capitalize()
    
    # If no pattern match, look for words that might be class names (capitalized words)
    for word in words:
        # Skip common words
        if word in ['the', 'class', 'show', 'me', 'how', 'does', 'what', 'is', 'are', 'explain']:
            continue
        # Return any word that starts with a capital letter in the original query
        original_words = prompt.split()
        for original_word in original_words:
            if original_word.lower() == word and original_word[0].isupper():
                return original_word
    
    return None

st.title("Repository Code Assistant")
st.caption("Analyze and ask questions about your code repository")

# Repository path input
repo_path = st.text_input(
    "Repository Path",
    value=os.getcwd(),
    help="Enter the full path to your repository"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None

# Initialize RAG with repository path
if st.button("Load Repository"):
    if os.path.exists(repo_path):
        st.session_state.rag = init_rag(repo_path)
        if st.session_state.rag:
            st.success(f"Repository loaded successfully from: {repo_path}")
    else:
        st.error("Invalid repository path!")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    if st.session_state.rag:
        st.session_state.rag.memory.current_conversation.dialog_turns.clear()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "context" in message:
            with st.expander(f"View source from {message.get('file_path', 'unknown')}"):
                st.code(message["context"], language=message.get("language", "python"))

# Chat input
if st.session_state.rag and (prompt := st.chat_input(
    "Ask about the code (e.g., 'Show me the implementation of the RAG class', 'How is memory handled?')"
)):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing code..."):
            response, docs = st.session_state.rag(prompt)
            
            # Show relevant context first, then the explanation
            if docs and docs[0].documents:
                # Try to find implementation code first
                implementation_docs = [
                    doc for doc in docs[0].documents 
                    if doc.meta_data.get("is_implementation", False)
                ]
                
                # Use implementation if found, otherwise use first document
                doc = implementation_docs[0] if implementation_docs else docs[0].documents[0]
                context = doc.text
                file_path = doc.meta_data.get("file_path", "unknown")
                file_type = doc.meta_data.get("type", "python")
                
                with st.expander(f"View source from {file_path}"):
                    st.code(context, language=file_type)
                
                # Now show the explanation
                st.write(response)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "context": context,
                    "file_path": file_path,
                    "language": file_type
                })
            else:
                st.write(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
elif not st.session_state.rag:
    st.info("Please load a repository first!") 