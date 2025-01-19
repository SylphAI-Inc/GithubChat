import os
import subprocess
import re
# import glob
from pathlib import Path
from typing import Optional


import adalflow as adal
from pipeline_transformers import L2Norm
from adalflow import Sequential
from adalflow.core.types import List, Document
from adalflow.components.data_process import TextSplitter, ToEmbeddings
from config import configs

# Import custom logging functions and utilities
from logging_config import (
    log_info,
    log_success,
    log_warning,
    log_error,
    log_debug,
    LoggerUtility
)


def extract_class_definition(content: str, class_name: str) -> str:
    """
    Extract a complete class definition from the content.

    Args:
        content (str): The source code containing the class.
        class_name (str): The name of the class to extract.

    Returns:
        str: The extracted class definition or the original content if not found.
    """
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
        log_warning(f"Class '{class_name}' not found in the content.")
        return content

    # Collect the entire class definition
    class_lines = [lines[class_start]]
    current_line = class_start + 1

    while current_line < len(lines):
        line = lines[current_line]
        # If we hit a line with same or less indentation, we're out of the class
        if line.strip() and (len(line) - len(line.lstrip()) <= indent_level):
            break
        class_lines.append(line)
        current_line += 1

    extracted_class = '\n'.join(class_lines)
    log_info(f"Extracted class '{class_name}' definition.")
    return extracted_class


def extract_class_name_from_query(query: str) -> str:
    """
    Extract class name from a query about a class, with fallback for capitalized words.

    Args:
        query (str): The input query string.

    Returns:
        str or None: The extracted class name or None if not found.
    """
    log_info(f"Extracting class name from query: {query}")

    # Patterns for explicit class queries
    patterns = [
        r'class (\w+)',
        r'the (\w+) class',
        r'what does (\w+) do',
        r'how does (\w+) work',
        r'show me (\w+)',
        r'explain (\w+)',
    ]

    # List of common words to skip during fallback
    common_words = {
        'the', 'class', 'show', 'me', 'how', 'does', 'what',
        'is', 'are', 'explain', 'a', 'an', 'to', 'in', 'and',
        'or', 'on', 'with', 'for', 'of', 'by', 'at'
    }

    # Try matching the query against the patterns
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            class_name = matches[0].capitalize()
            log_debug(f"Class name '{
                      class_name}' extracted using pattern '{pattern}'")
            return class_name

    # Fallback: Extract capitalized words, ignoring common words
    words = query.split()
    for word in words:
        if word[0].isupper() and word.lower() not in common_words:
            log_debug(f"Class name '{
                      word}' extracted as a fallback (capitalized word)")
            return word

    # No match found
    log_warning(f"No class name found in query: {query}")
    return None


def download_github_repo(repo_url: str, local_path: str) -> str:
    """
    Downloads a GitHub repository to a specified local path.

    Args:
        repo_url (str): The URL of the GitHub repository to clone.
        local_path (str): The local directory where the repository will be cloned.

    Returns:
        str: The output message from the `git` command.
    """
    log_info(f"Starting download of repository: {repo_url}")
    try:
        # Check if Git is installed
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log_success("Git is installed.")

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)
        log_info(f"Cloning into directory: {local_path}")

        # Clone the repository
        result = subprocess.run(
            ["git", "clone", repo_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        log_success("Repository cloned successfully.")
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg = f"Error during cloning: {e.stderr.decode('utf-8')}"
        log_error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        log_error(error_msg)
        return error_msg


def documents_to_adal_documents(
    path: str,
    code_extensions: Optional[List[str]] = None,
    doc_extensions: Optional[List[str]] = None,
    ignored_paths: Optional[List[str]] = None
) -> List[Document]:
    """
    Recursively reads all documents from a given directory path, ignoring specified directories,
    and returns a list of Document objects with detailed logging.

    Args:
        path (str): The root directory path to read documents from.
        code_extensions (List[str], optional): List of code file extensions to include.
            Defaults to ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'].
        doc_extensions (List[str], optional): List of document file extensions to include.
            Defaults to ['.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.toml'].
        ignored_paths (List[str], optional): List of directory names to ignore.
            Defaults to ['.git', '__pycache__', '.vscode', '.venv', 'node_modules', '.streamlit'].

    Returns:
        List[Document]: A list of Document objects representing the files.

    Raises:
        TypeError: If `path` is not a string.
        Exception: If traversal fails.
    """
    if not isinstance(path, str):
        log_error(f"Invalid path type: {type(path)}. Expected str.")
        raise TypeError("Path must be a string.")

    # Set default extensions if not provided
    code_extensions = code_extensions or [
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']
    doc_extensions = doc_extensions or [
        '.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.toml']
    ignored_paths = ignored_paths or [
        '.git', '__pycache__', '.vscode', '.venv', 'node_modules', '.streamlit']

    documents: List[Document] = []
    root_path = Path(path)

    log_info(f"Starting to read documents from: {root_path}")

    def process_file(file_path: str, relative_path: str, is_code: bool) -> None:
        """
        Process an individual file and add it to the documents list.

        Args:
            file_path (str): The full path to the file.
            relative_path (str): The path relative to the root directory.
            is_code (bool): Indicates whether the file is a code file.
        """
        try:
            path_obj = Path(file_path)
            encoding = 'utf-8'
            try:
                content = path_obj.read_text(encoding=encoding)
                log_debug(
                    f"Read file {file_path} with encoding {encoding}.")
            except UnicodeDecodeError:
                log_warning(f"UnicodeDecodeError for file {
                            file_path}. Trying 'latin1' encoding.")
                encoding = 'latin1'
                content = path_obj.read_text(encoding=encoding)
                log_debug(
                    f"Read file {file_path} with fallback encoding {encoding}.")
            # add other encodings later!

            is_implementation = (
                not relative_path.startswith('test_')
                and not relative_path.startswith('app_')
                and 'test' not in relative_path.lower()
            )

            extension = path_obj.suffix[1:]  # Remove the dot
            file_stat = os.stat(file_path)
            metadata = {
                "file_path": relative_path,
                "type": extension,
                "is_code": is_code,
                "is_implementation": is_implementation,
                "title": path_obj.name,
                "file_size": file_stat.st_size,
                "line_count": len(content.splitlines()),
            }

            doc = Document(
                text=content,
                meta_data=metadata
            )
            documents.append(doc)
            log_debug(f"Processed file {file_path} into Document with title '{
                      doc.meta_data['title']}'")
        except Exception as e:
            log_error(f"Error processing file {file_path}: {e}")

    try:
        # Define the combined list of extensions to include
        include_extensions = code_extensions + doc_extensions

        LoggerUtility.traverse_and_log(
            base_path=root_path,
            ignored_paths=ignored_paths,
            include_extensions=include_extensions,
            process_file_callback=process_file
        )

        log_info(f"Completed processing. Total documents collected: {
                 len(documents)}")
    except Exception as e:
        log_error(f"Error during traversal and processing: {e}")
        raise e

    return documents


def create_pipeline() -> Sequential:
    """
    Creates and returns the data transformation pipeline.

    Returns:
        adal.Sequential: The sequential data transformer pipeline.
    """
    log_info("Preparing data transformation pipeline.")

    splitter = TextSplitter(**configs["text_splitter"])

    embedder = adal.Embedder(
        model_client=configs["embedder"]["model_client"](),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )

    batch_embed = ToEmbeddings(
        embedder=embedder, batch_size=configs["embedder"]["batch_size"]
    )

    normalize = L2Norm()

    data_transformer = adal.Sequential(splitter, batch_embed, normalize)

    log_success("Data transformation pipeline is ready.")
    return data_transformer


def create_sample_documents() -> List[Document]:
    """
    Create some sample documents for testing.

    Returns:
        List[Document]: A list of sample `Document` objects.
    """
    log_info("Creating sample documents for testing.")
    sample_texts = [
        """Alice is a software engineer who loves coding in Python. 
        She specializes in machine learning and has worked on several NLP projects.
        Her favorite project was building a chatbot for customer service.""",

        """Bob is a data scientist with expertise in deep learning.
        He has published papers on transformer architectures and attention mechanisms.
        Recently, he's been working on improving RAG systems.""",

        """The company cafeteria serves amazing tacos on Tuesdays.
        They also have a great coffee machine that makes perfect lattes.
        Many employees enjoy their lunch breaks in the outdoor seating area."""
    ]

    sample_docs = [
        Document(text=text, meta_data={"title": f"doc_{i}"})
        for i, text in enumerate(sample_texts)
    ]
    log_success(f"Created {len(sample_docs)} sample documents.")
    return sample_docs


# Need to add in DatabaseManager class instance to load in the documents when running in single file mode.
# def main():
#     """Main function to process repositories and transform documents."""
#     setup_env()
#     logger.info("Starting data pipeline script.")

#     repo_url = "https://github.com/microsoft/LMOps"
#     local_path = os.path.join(get_adalflow_default_root_path(), "LMOps")

#     # Download repository
#     result = download_github_repo(repo_url, local_path)
#     logger.info("Repository clone result: %s", result)

#     # Process documents
#     target_path = os.path.join(local_path, "prompt_optimization")
#     documents = documents_to_adal_documents(target_path)

#     # Transform with cache check
#     db_path = os.path.join(
#         get_adalflow_default_root_path(), "db_microsoft_lmops")
#     transform_with_cache_check(documents, db_path)


# if __name__ == "__main__":
#     main()
