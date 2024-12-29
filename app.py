import streamlit as st
import os
import tempfile
from rag import RAG
from test_rag import initialize_test_database

# Initialize RAG system with test data
@st.cache_resource
def init_rag():
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    # Create a temporary directory for the database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_db")
    
    # Initialize test database with example data
    initialize_test_database(db_path)
    
    # Create RAG instance with test database using general QA prompt
    return RAG(index_path=db_path, prompt_type="general_qa")

st.title("RAG Chat Interface")
st.caption("Test data includes information about Alice (software engineer), Bob (data scientist), and the company cafeteria.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG
rag = init_rag()

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    # Reset RAG memory
    rag.memory.current_conversation.dialog_turns.clear()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "context" in message:
            with st.expander("View relevant context"):
                st.write(message["context"])

# Chat input
if prompt := st.chat_input("What would you like to know about Alice, Bob, or the cafeteria?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, docs = rag(prompt)
            st.write(response)
            
            # Show relevant context
            if docs and docs[0].documents:
                context = docs[0].documents[0].text
                with st.expander("View relevant context"):
                    st.write(context)
                
                # Add assistant message with context to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "context": context
                })
            else:
                # Add assistant message without context to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                }) 