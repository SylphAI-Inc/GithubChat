import os
import sys
from rag import RAG
from typing import Optional
from adalflow import Sequential
from adalflow.eval.base import BaseEvaluator, EvaluationResult
from adalflow.eval import RetrieverRecall
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

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

questions = [
    "Does the repository provide a requirements.txt or equivalent for dependencies?",
    "Are dependency versions specified for reproducibility?",
    "Are environment setup instructions clear and executable?",
    "Does the repository mention the required Python version or runtime environment?",
    "Are there instructions for setting up the environment, such as using conda, virtualenv, or Docker?",
    "Are version tags or release notes available for stable releases?",
    "Does the repository provide a clear entry point for running experiments or training?",
    "Are there examples or documentation for running baseline experiments?",
    "Are instructions for running scripts or workflows detailed and easy to follow?",
    "Are training commands or configurations explicitly provided?",
    "Is the file organization intuitive and documented in the README.md?",
    "Are major functions and modules explained in the code or documentation?",
    "Is there a script or module for loading datasets?",
    "Are there placeholders or instructions for adding custom datasets?",
    "Does the repository provide a configuration file or system for hyperparameters?",
    "Are hyperparameter descriptions detailed in the documentation?",
    "Are training parameters like epochs and batch sizes configurable?",
    "Does the repository include example commands or templates for usage?",
    "Can a user reproduce experiments or workflows with minimal troubleshooting?",
    "Is the purpose and structure of the repository explained clearly?"
]

repos = [
    "https://github.com/NovaSky-AI/SkyThought",
    "https://github.com/allenai/s2orc?tab=readme-ov-file",
    "https://github.com/SylphAI-Inc/AdalFlow",
    "https://github.com/microsoft/graphrag"
]

prefix = "/.adalflow/repositories/"

repo_paths = [
    "AdalFlow",
    "GithubChat",
    "SkyThought",
    "graphrag"
]


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
            sys.exit(1)

        if needs_transform:
            # Load documents from the source directory and transform them to adal.Documents
            documents: List[Document] = documents_to_adal_documents(
                db_manager.source_dir)

            if not documents:
                log_warning(f"No documents found in the repository: {
                            db_manager.source_dir}")
                sys.exit(0)  # Exit gracefully if no documents to process

            # Create transformation pipeline
            pipeline: Sequential = create_pipeline()

            # Transform documents and save to the database
            try:
                db_manager.transform_documents_and_save(
                    documents=documents, pipeline=pipeline)
                log_success(
                    "Documents transformed and saved successfully.")
            except Exception as e:
                log_error(
                    f"An error occurred during transformation and saving: {e}")
                sys.exit(1)

        # Initialize RAG instance with database
        try:
            rag_instance = RAG(db=db_manager.db)
            log_success("RAG instance initialized with database.")
            return rag_instance
        except Exception as e:
            log_error(f"Failed to initialize RAG: {e}")
            return None

    except Exception as e:
        log_error(f"Failed to initialize RAG: {e}")
        sys.exit(1)


def rag_query(rag: RAG, prompt: str):
    """
    Queries the RAG pipeline with the given prompt and returns the response with relevant context.

    Args:
      rag: The RAG instance.
      prompt: The user's query.

    Returns:
      A dictionary containing the response, context, file path, and language, 
      or an error message if RAG is not initialized or the prompt is empty.
    """
    if not rag or not prompt:
        log_error("RAG instance not initialized or prompt is empty.")
        return "An error occurred while processing your request."

    try:
        response, retriever_output = rag(prompt)
        log_success(f"RAG response generated for prompt: {prompt}")

        print(f"Doc Indices {len(retriever_output.doc_indices)}: {
              retriever_output.doc_indices}")
        print("=" * 20)
        print(f"Doc Scores {len(retriever_output.doc_scores)}: {
              retriever_output.doc_scores}")
        print("=" * 20)
        print(f"Documents {len(retriever_output.documents)}")
        print("=" * 20)

        implementation_docs = [
            doc for doc in retriever_output.documents if doc.meta_data.get("is_implementation", False)
        ]

        print(f"Implementation Documents: {len(implementation_docs)}")

        contexts = []
        file_paths = []
        file_types = []
        is_codes = []
        titles = []
        file_sizes = []
        line_counts = []

        for doc in implementation_docs:
            metadata = doc.meta_data
            context = doc.text
            file_path = metadata.get("file_path", "unknown")
            file_type = metadata.get("type")
            is_code = metadata.get("is_code")
            title = metadata.get("title")
            file_size = metadata.get("file_size")
            line_count = metadata.get("line_count")

            contexts.append(context)
            file_paths.append(file_path)
            file_types.append(file_type)
            is_codes.append(is_code)
            titles.append(title)
            file_sizes.append(file_size)
            line_counts.append(line_count)

        print(f"contexts{contexts}")
        print(f"file_paths: {file_paths}")
        print(f"file_types: {file_types}")
        print(f"titles: {titles}")
        print(f"file_sizes: {file_sizes}")
        print(f"line_counts: {line_counts}")

        class_name = extract_class_name_from_query(prompt)

        if class_name and file_type == "python":
            class_context = extract_class_definition(context, class_name)
            if class_context != context:
                context = class_context
                log_debug(f"Extracted class definition for {class_name}")

        return {
            "contexts": contexts,
            "file_paths": file_paths,
            "file_sizes": file_sizes,
            "line_counts": line_counts,
            "response": response,
            "context": context
        }

    except Exception as e:
        log_error(f"Error generating response: {e}")
        return "An error occurred while processing your request."


def main():
    """
    Main function to handle user queries and interact with the RAG pipeline.
    """
    repo_path = "/mnt/c/Users/ddIdk/Desktop/github/github-chat/GithubChat"
    rag = init_rag(repo_path)

    if not rag:
        print("Failed to load repository.")
        return

    for query in questions:
        prompt = query
        result = rag_query(rag, prompt)
        retrieved_contexts = result.get("contexts", [])
        file_paths = result.get("file_paths", [])
        file_sizes = result.get("file_sizes", [])
        line_counts = result.get("line_counts", [])
        response = result.get("response", "")

        # Format the contexts into readable strings
        formatted_contexts = []
        for i, context in enumerate(retrieved_contexts):
            formatted_context = (
                f"Context {i+1}:\n"
                f"File Path: {file_paths[i]}\n"
                f"File Size: {file_sizes[i]} bytes\n"
                f"Line Count: {line_counts[i]} lines\n"
                f"Content:\n{context}\n"
                f"{'-'*40}\n"
            )
            formatted_contexts.append(formatted_context)

        # Save the formatted contexts to a file
        with open("retrieved_contexts.txt", "w", encoding="utf-8") as file:
            file.writelines(formatted_contexts)

        # Store the contexts in an array for use in Python
        python_array = [
            {
                "file_path": file_paths[i],
                "file_size": file_sizes[i],
                "line_count": line_counts[i],
                "content": retrieved_contexts[i]
            }
            for i in range(len(retrieved_contexts))
        ]

        # Print confirmation
        print(f"Saved {len(retrieved_contexts)
                       } contexts to 'retrieved_contexts.txt'")
        print("Contexts are also stored in a Python array for further use.")

        # Optional: Print the Python array if needed
        print("\nPython Array:\n", python_array)


if __name__ == "__main__":
    main()
