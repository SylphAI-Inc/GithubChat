import os
import sys
import hashlib
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

# There's one potential edge case that needs to be taken into account for the documents_to_adal_documents
# if a user modifies the underlying source_dir without changing the string argument, Streamlit won't
# re-run `documents_to_adal_documents` because it sees the same function input. This can be manually
# handled by:
# 1) passing a "last modified" timestamp as an argument to the cached function
# or
# 2) Cleaning the cache when you want to force a re-read.

# It's also important to note once again that the st.cache_resource or any form of streamlit caching
# is NOT going to handle caching behaviors like the API calls to create vector embeddings of the code base.
# I've already resolved this in a previous commit.


def initialize_session_State():
    state = st.session_state
    if "messages" not in state:
        st.session_state.messages = []
    if "rag" not in state:
        state.rag = None
    if "all_context_docs" not in state:
        state.current_doc_index = 0
    if "last_retriever_output" not in state:
        state.last_retriever_output = None


def generate_docset_hash(doc_set: List[Document]):
    lines = []

    for doc in doc_set:
        md = doc.meta_data
        part = f"{md.get('file_path', '')}::{md.get('code_state', '')}::{
            md.get('line_count', '')}"
        lines.append(part)

    joined = "\n".join(lines)

    hash_str = hashlib.md5(joined.encode("utf-8").hexdigest()[:8])
    return hash_str


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

        # Initialize database manager
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
                documents: List[Document] = documents_to_adal_documents(
                    db_manager.source_dir)
                if not documents:
                    log_warning(f"No documents found in the repository: {
                                db_manager.source_dir}")
                    st.warning("No documents found in the repository.")
                    sys.exit(0)  # Exit gracefully if no documents to process

                pipeline: Sequential = create_pipeline()

                with st.spinner("Creating embeddings..."):
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


# def display_chat_messages():
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
#             if "context" in message:
#                 with st.expander(f"View source from {message.get('file_path', 'unknown')}"):
#                     st.code(message["context"], language=message.get(
#                         "language", "python"))


def handle_chat_input():
    """
    Checks for user input. If a prompt is submitted, calls RAG and updates
    session state with the new retrieval results.
    """
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

                    # If we have docs, store them in session state so we can re-render them even when
                    # there's no new prompt
                    st.session_state.last_retriever_output = retriever_output
                    st.session_state.last_prompt = prompt
                    st.session_state.last_response = response

                    if retriever_output and retriever_output.documents:
                        # Pass the new doc list, so we can decide if we reset or not
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


def display_context_and_response(retriever_output, prompt, response):
    """
    Displays relevant context and response, including pagination and detailed metadata logging,
    without resetting pagination unless the doc list changes.
    """

    # 1. Gather docs
    all_docs = retriever_output.documents

    # 2. Log metadata for debugging
    contexts = []
    file_paths = []
    file_extensions = []
    categories = []
    code_states = []  # 1 for testing, 0 for implementation, -1 for non-code
    titles = []
    file_sizes = []
    line_counts = []

    for doc in all_docs:
        metadata = doc.meta_data
        context_text = doc.text

        file_path = metadata.get("file_path", "unknown")
        file_extension = metadata.get("file_extension", "unknown")
        category = metadata.get("category", "unknown")
        code_state = metadata.get("code_state", -1)
        title = metadata.get("title", "No Title")
        file_size = metadata.get("file_size", 0)
        line_count = metadata.get("line_count", 0)

        contexts.append(context_text)
        file_paths.append(file_path)
        file_extensions.append(file_extension)
        categories.append(category)
        code_states.append(code_state)
        titles.append(title)
        file_sizes.append(file_size)
        line_counts.append(line_count)

    print("=" * 20)
    print(f"Doc Indices: {retriever_output.doc_indices}")
    print(f"Doc Scores: {retriever_output.doc_scores}")
    print(f"file_paths: {file_paths}")
    print(f"file_extensions: {file_extensions}")
    print(f"categories: {categories}")
    print(f"code_states: {code_states}")
    print(f"titles: {titles}")
    print(f"file_sizes: {file_sizes}")
    print(f"line_counts: {line_counts}")
    print("=" * 20)

    # 3. If no docs, just display response
    if not all_docs:
        st.write(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        return

    # 4. Decide whether to replace docs in session state or not
    #    We only overwrite if the newly retrieved docs are different from what's stored
    #    A simple test is to compare lengths or compare the IDs of the docs.
    old_docs = st.session_state.get("all_context_docs", [])

    # Quick check: if lengths differ or set of file_paths differ, they're probably new
    old_file_paths = [doc.meta_data.get("file_path") for doc in old_docs]
    new_file_paths = file_paths

    docs_changed = (len(all_docs) != len(old_docs)) or (
        set(old_file_paths) != set(new_file_paths))

    if docs_changed:
        st.session_state.all_context_docs = all_docs
        st.session_state.current_doc_index = 0
        log_debug("New document list detected; resetting doc index to 0.")
    else:
        log_debug("Document list appears unchanged; preserving current doc index.")

    # 5. Helper for pagination
    def clamp_index(i):
        return max(0, min(i, len(st.session_state.all_context_docs) - 1))

    def update_doc_index(delta):
        st.session_state.current_doc_index = clamp_index(
            st.session_state.current_doc_index + delta)
        log_debug(f"Doc index updated to {st.session_state.current_doc_index}")

    # 6. Show pagination controls if multiple docs
    if len(st.session_state.all_context_docs) > 1:
        pagination_key = f"pagination_{st.session_state.last_prompt}"
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("⬅️ Previous", key=f"{pagination_key}_prev_{st.session_state.current_doc_index}"):
                update_doc_index(-1)
        with cols[1]:
            st.write(
                f"Document {st.session_state.current_doc_index + 1} "
                f"of {len(st.session_state.all_context_docs)}"
            )
        with cols[2]:
            if st.button("Next ➡️", key=f"{pagination_key}_next_{st.session_state.current_doc_index}"):
                update_doc_index(1)

    # 7. Retrieve the correct doc based on current index
    idx = st.session_state.current_doc_index
    if 0 <= idx < len(st.session_state.all_context_docs):
        doc = st.session_state.all_context_docs[idx]
    else:
        # In case something is out-of-sync
        st.error("Document index out of range.")
        log_error("Document index out of range.")
        return

    context = doc.text
    metadata = doc.meta_data

    file_path = metadata.get("file_path", "unknown")
    file_type = metadata.get("type", "python")
    file_size = metadata.get("file_size", 0)
    line_count = metadata.get("line_count", 0)
    category = metadata.get("category", "unknown")
    code_state = metadata.get("code_state", -1)

    # 8. Display Document Metadata
    st.markdown("### 📄 Document Metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**File Path:** `{file_path}`")
        st.write(f"**File Type:** `{file_type}`")
    with col2:
        if file_size:
            st.write(f"**File Size:** {file_size} bytes")
        if line_count:
            st.write(f"**Line Count:** {line_count}")

    # 9. Possibly extract class definition
    class_name = extract_class_name_from_query(prompt)
    if class_name and file_type.lower() == "python":
        class_context = extract_class_definition(context, class_name)
        if class_context != context:
            context = class_context
            log_debug(f"Extracted class definition for {class_name}")

    # 10. Display code context with category / code_state tags
    state_description = (
        "Test File" if code_state == 1 else
        "Implementation File" if code_state == 0 else
        "Non-Code File"
    )

    with st.expander(
        f"View source from `{file_path}` "
        f"[Category: {category.capitalize()}, State: {state_description}]",
        expanded=True
    ):
        st.code(context, language=file_type.lower())

    # 11. Display the response
    st.markdown("### 🤖 Assistant's Response")
    st.write(response)

    # 12. Update chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "context": context,
        "file_path": file_path,
        "language": file_type.lower(),
        "category": category,
        "code_state": code_state
    })


def main():
    initialize_session_State()

    st.title("📂 Repository Code Assistant")
    st.caption("Analyze and ask questions about your code repository")

    repo_path = st.text_input(
        "📁 Repository Path",
        value=os.getcwd(),
        help="Enter the full path to your repository"
    )

    # Ensure session state variables exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "all_context_docs" not in st.session_state:
        st.session_state.all_context_docs = []
    if "current_doc_index" not in st.session_state:
        st.session_state.current_doc_index = 0

    # Event: Load repository
    if st.button("🔄 Load Repository"):
        st.session_state.rag = init_rag(repo_path)

        if st.session_state.rag:
            st.success(f"✅ Repository loaded successfully from: `{repo_path}`")
            # Reset docs & chat upon new repository
            st.session_state.all_context_docs = []
            st.session_state.current_doc_index = 0
            st.session_state.messages = []
            st.session_state.last_retriever_output = None
            st.session_state.last_prompt = ""
            st.session_state.last_response = ""

    # Event: Clear chat
    if st.button("🧹 Clear Chat"):
        log_info("Clearing chat messages and conversation history.")
        st.session_state.messages = []
        if st.session_state.rag:
            try:
                st.session_state.rag.memory.current_conversation.dialog_turns.clear()
                log_success("Chat messages and conversation history cleared.")
            except AttributeError:
                log_warning(
                    "RAG instance does not have a 'memory' attribute or 'current_conversation'.")
        else:
            log_warning(
                "RAG instance not initialized. No chat history to clear.")
        st.success("✅ Chat cleared successfully.")

    # display_chat_messages()
    handle_chat_input()

    # If there's an existing retriever output (docs) from a previous prompt,
    # we need to re-display them so the user sees the pagination UI on button clicks.
    if st.session_state.last_retriever_output:
        # We call this again with the stored docs, prompt, and response, so that any button clicks
        # will re-run this code and preserve UI between state transitions.
        display_context_and_response(
            st.session_state.last_retriever_output,
            st.session_state.last_prompt,
            st.session_state.last_response
        )


if __name__ == "__main__":
    main()
