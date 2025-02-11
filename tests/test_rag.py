import os
from src.rag import RAG
import adalflow as adal
from adalflow.utils import get_adalflow_default_root_path
import tempfile
from src.data_pipeline import (
    create_sample_documents,
    transform_documents_and_save_to_db,
)


def initialize_test_database(db_path: str):
    """Initialize database with sample documents."""
    documents = create_sample_documents()
    transform_documents_and_save_to_db(documents, db_path)
    return db_path


def main():

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_db")

    initialize_test_database(db_path)

    # Create RAG instance
    rag = RAG(index_path=db_path)

    # Test conversation flow with memory
    test_conversation = [
        "Who is Alice and what does she do?",
        "What about Bob? What's his expertise?",
        "Can you tell me more about what the previous person works on?",
        "What was her favorite project?",  # Tests memory of Alice
        "Between these two people, who has more experience with RAG systems?",  # Tests memory of both
        "Do they ever meet? Where might they have lunch together?",  # Tests memory and context combination
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

            response, docs = rag(query)
            print(f"\nAssistant: {response}")

            # Show most relevant document used
            if docs:
                most_relevant = docs[0].documents[0].text.strip()
                print(f"\nMost relevant context used: \n{most_relevant[:200]}...")

        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "=" * 50)


if __name__ == "__main__":
    from adalflow.utils import setup_env

    setup_env()
    main()
