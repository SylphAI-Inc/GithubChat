import os
from rag import RAG
import tempfile
from data_pipeline import create_sample_documents
from adalflow.core.types import List, Document
from data_pipeline import create_pipeline
from adalflow.core.db import LocalDB

from logging_config import (
    log_info,
    log_init,
    log_success,
    log_error
)


def initialize_test_database(db_path: str):
    documents: List[Document] = create_sample_documents()

    log_info("Starting transformation of documents.")
    # Get the data transformer
    data_transformer = create_pipeline()

    try:
        # Initialize the local database
        db = LocalDB("code_db")
        log_init("LocalDB", {"name": "code_db"})

        # Register and apply the transformer
        db.register_transformer(
            transformer=data_transformer, key="split_and_embed")
        db.load(documents)
        db.transform(key="split_and_embed")
        log_success("Documents transformed successfully.")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Save the transformed data to the database
        db.save_state(filepath=db_path)
        log_success(f"Database saved to {db_path}")
        return db

    except Exception as e:
        log_error(f"Failed to transform and save documents: {e}")


def main():
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "Your OpenAI API key"

    # Create a temporary directory for the database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_db")

    # Initialize test database
    initialize_test_database(db_path)

    # Create RAG instance
    rag = RAG(index_path=db_path)

    # Test conversation flow with memory
    test_conversation = [
        "Who is Alice and what does she do?",
        "What about Bob? What's his expertise?",
        "Can you tell me more about what the previous person works on?",
        "What was her favorite project?",  # Tests memory of Alice
        # Tests memory of both
        "Between these two people, who has more experience with RAG systems?",
        # Tests memory and context combination
        "Do they ever meet? Where might they have lunch together?"
    ]

    print("Starting conversation test with memory...\n")
    for i, query in enumerate(test_conversation, 1):
        print(f"\n----- Query {i} -----")
        print(f"User: {query}")
        try:
            # Get conversation history before the response
            print("\nCurrent Conversation History:")
            history = rag.memory()
            if history:
                print(history)
            else:
                print("(No history yet)")

            # Correct method invocation
            response, docs = rag(query)
            print(f"\nAssistant: {response}")

            # Show most relevant document used
            if docs:
                most_relevant = docs[0].documents[0].text.strip()
                print(f"\nMost relevant context used: \n{
                      most_relevant[:200]}...")

        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
