import streamlit as st
import os
import tempfile
from rag import RAG
from test_rag import initialize_test_database
from logging_config import logger  # Import centralized logger


@st.cache_resource
def init_rag():
    """Initialize RAG with test data."""
    logger.info("Initializing RAG with test data.")
    try:
        # Set OpenAI API key from Streamlit secrets
        logger.debug("Setting OpenAI API key from Streamlit secrets.")
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

        # Create a temporary directory for the database
        temp_dir = tempfile.mkdtemp()
        print(f"temp_dir: {temp_dir}")
        logger.info("Temporary directory created at %s", temp_dir)
        db_path = os.path.join(temp_dir, "test_db")
        print(f"db_path: {db_path}")

        # Initialize test database with example data
        logger.info("Initializing test database at %s", db_path)
        db = initialize_test_database(db_path)

        # Create RAG instance with test database using general QA prompt
        logger.info(
            "Creating RAG instance with index_path=%s and prompt_type='general_qa'", db_path)
        rag_instance = RAG(db=db, prompt_type="general_qa")

        logger.info("RAG instance created successfully.")
        return rag_instance
    except Exception as e:
        logger.exception("Failed to initialize RAG: %s", e)
        st.error("An error occurred during RAG initialization.")
        return None


logger.info("Launching RAG Chat Interface")
st.title("RAG Chat Interface")
st.caption("Test data includes information about Alice (software engineer), Bob (data scientist), and the company cafeteria.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Session state for messages initialized as an empty list.")

# Initialize RAG
logger.debug("Calling init_rag() to initialize RAG instance.")
rag = init_rag()

# Clear chat button
if st.button("Clear Chat"):
    logger.info(
        "User clicked 'Clear Chat' button. Clearing chat history and resetting RAG memory.")
    st.session_state.messages = []
    if rag:
        logger.debug("Clearing current conversation in RAG memory.")
        rag.memory.current_conversation.dialog_turns.clear()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "context" in message:
            file_path = message.get("file_path", "sample")
            logger.debug("Expanding context for file: %s", file_path)
            with st.expander(f"View source from {file_path}"):
                st.code(message["context"],
                        language=message.get("language", "text"))

# Chat input
prompt = st.chat_input(
    "What would you like to know about Alice, Bob, or the cafeteria?")
if rag and prompt:
    logger.info("User submitted prompt: %s", prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    logger.debug("Appended user message to session state.")

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            logger.debug("Querying RAG with prompt: %s", prompt)
            try:
                response, retriever_output = rag(prompt)
                logger.info("RAG response generated successfully.")
                st.write(response)

                if retriever_output and retriever_output.documents:
                    context = retriever_output.documents[0].text
                    file_path = retriever_output.documents[0].meta_data.get(
                        "title", "sample")
                    logger.debug(
                        "Retrieved context from document titled '%s'.", file_path)
                    with st.expander(f"View source from {file_path}"):
                        st.code(context, language="text")

                    # Add assistant message with context to chat history
                    logger.debug(
                        "Appending assistant response with context to session state.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "context": context,
                        "file_path": file_path,
                        "language": "text"
                    })
                else:
                    logger.debug("No relevant documents returned by RAG.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
            except Exception as e:
                logger.exception("Error during RAG response generation: %s", e)
                st.error("An error occurred while processing your request.")
elif not rag:
    logger.warning("RAG instance not initialized. Prompt submission disabled.")
    st.info("RAG initialization failed. Please check logs for details.")
