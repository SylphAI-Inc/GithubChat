import os
import sys
from rag import RAG
import streamlit as st
from typing import Optional
from adalflow import Sequential
from logging_config import (
    log_info,
    log_success,
    log_warning,
    log_error,
    log_debug
)
from data_base_manager import DatabaseManager
from adalflow.core.types import Document, List

from data_pipeline import (
    extract_class_definition,
    extract_class_name_from_query,
    documents_to_adal_documents,
    create_pipeline
)


@st.cache_resource
def init_rag(_repo_path: str) -> Optional[RAG]:
    """
    Initialize RAG with repository data.
    :param _repo_path: Path to the repository.
    """
    try:
        # Load API key
        open_ai_api_key = os.getenv("OPENAI_API_KEY")
        if not open_ai_api_key:
            log_error(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
            st.error(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            sys.exit(1)

        log_info("OpenAI API key set successfully.")

        repo_name = os.path.basename(os.path.normpath(_repo_path))

        # Initialize database manager to prevent redundant computation by checking
        # for pickled repos
        # We'll handle logging separately
        db_manager = DatabaseManager(repo_name=repo_name)

        # Load or create the database
        try:
            needs_transform = db_manager.load_or_create_db()
        except Exception as e:
            log_error(
                f"An error occurred while loading or creating the database: {e}")
            st.error("An error occurred while loading or creating the database.")
            sys.exit(1)

        if needs_transform:
            with st.spinner("Processing repository files..."):
                # Load documents from the source directory and transform them to adal.Documents
                documents: List[Document] = documents_to_adal_documents(
                    db_manager.source_dir)

                if not documents:
                    log_warning(f"No documents found in the repository: {
                                db_manager.source_dir}")
                    st.warning("No documents found in the repository.")
                    sys.exit(0)  # Exit gracefully if no documents to process

                # Create transformation pipeline
                pipeline: Sequential = create_pipeline()

                with st.spinner("Creating embeddings..."):
                    # Transform documents and save to the database
                    try:
                        db_manager.transform_documents_and_save(
                            documents=documents, pipeline=pipeline)
                        log_success(
                            "Documents transformed and saved successfully.")
                    except Exception as e:
                        log_error(
                            f"An error occurred during transformation and saving: {e}")
                        st.error(
                            "An error occurred during transformation and saving.")
                        sys.exit(1)

        # Initialize RAG instance with database
        try:
            rag_instance = RAG(db=db_manager.db)
            log_success("RAG instance initialized with database.")
            return rag_instance
        except Exception as e:
            log_error(f"Failed to initialize RAG: {e}")
            st.error("An error occurred while initializing RAG.")
            return None

    except Exception as e:
        log_error(f"Failed to initialize RAG: {e}")
        st.error("An error occurred during RAG initialization.")
        sys.exit(1)


# Function: Display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "context" in message:
                with st.expander(f"View source from {message.get('file_path', 'unknown')}"):
                    st.code(message["context"], language=message.get(
                        "language", "python"))


# Function: Handle chat input
def handle_chat_input():
    if st.session_state.rag and (prompt := st.chat_input(
        "Ask about the code (e.g., 'Show me the implementation of the RAG class', 'How is memory handled?')"
    )):
        log_info(f"User submitted prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Analyze prompt and provide response
        try:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing code..."):
                    response, retriever_output = st.session_state.rag(prompt)
                    log_success(f"RAG response generated for prompt: {prompt}")

                    # Display response and relevant context
                    if retriever_output and retriever_output.documents:
                        display_context_and_response(
                            retriever_output, prompt, response)
                    else:
                        st.write(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
        except Exception as e:
            log_error(f"Error generating response: {e}")
            st.error("An error occurred while processing your request.")


# Function: Display relevant context and response
def display_context_and_response(retriever_output, prompt, response):

    # print(f"Doc Indices {len(retriever_output.doc_indices)}: {
    #       retriever_output.doc_indices}")
    # print("=" * 20)
    # print(f"Doc Scores {len(retriever_output.doc_scores)}: {
    #       retriever_output.doc_scores}")
    # print("=" * 20)
    # print(f"Documents {len(retriever_output.documents)}: {
    #       retriever_output.documents}")
    # print("=" * 20)

    implementation_docs = [
        doc for doc in retriever_output.documents if doc.meta_data.get("is_implementation", False)
    ]

    doc = implementation_docs[0] if implementation_docs else retriever_output.documents[0]
    context = doc.text
    file_path = doc.meta_data.get("file_path", "unknown")
    file_type = doc.meta_data.get("type", "python")

    class_name = extract_class_name_from_query(prompt)

    if class_name and file_type == "python":
        class_context = extract_class_definition(context, class_name)
        if class_context != context:
            context = class_context
            log_debug(f"Extracted class definition for {class_name}")

    with st.expander(f"View source from {file_path}"):
        st.code(context, language=file_type)

    st.write(response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "context": context,
        "file_path": file_path,
        "language": file_type
    })


# Streamlit UI Rendering
def main():
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

    # Event: Load repository
    if st.button("Load Repository"):
        st.session_state.rag = init_rag(repo_path)

        if st.session_state.rag:
            st.success(f"Repository loaded successfully from: {repo_path}")

    # Event: Clear chat
    if st.button("Clear Chat"):
        log_info("Clearing chat messages and conversation history.")
        st.session_state.messages = []
        if st.session_state.rag:
            st.session_state.rag.memory.current_conversation.dialog_turns.clear()
        log_success("Chat messages and conversation history cleared.")

    # Display chat messages
    display_chat_messages()

    # Handle chat input
    handle_chat_input()


if __name__ == "__main__":
    main()
