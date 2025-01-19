from logging_config import logger, LoggerUtility
import os
import sys
from pathlib import Path
from adalflow.core.component import Component
from adalflow.core.types import List, Document
from adalflow.utils import EntityMapping
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Change to this ordering of imports if stuff breaks
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
# from logging_config import logger, LoggerUtility


class DocumentToAdalDocument(Component):
    """
    A component that, given a path, recursively reads all documents in that directory
    and returns a List[Document].
    """

    def __init__(self,
                 code_extensions: List[str] = None,
                 doc_extensions: List[str] = None,
                 ignored_paths: List[str] = None):
        """
        Initializes the DocumentToAdalDocument component with optional configurations.
        Args:
            code_extensions (List[str], optional): List of code file extensions to include.
                Defaults to ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'].
            doc_extensions (List[str], optional): List of document file extensions to include.
                Defaults to ['.md', '.txt', '.rst', '.json', '.yaml', '.yml', '.toml'].
            ignored_paths (List[str], optional): List of directory names to ignore.
                Defaults to ['.git', '__pycache__', '.vscode', '.venv', 'node_modules', '.streamlit', '__init__'].
        """
        super().__init__()
        self.code_extensions = code_extensions or ['.py', '.js', '.ts',
                                                   '.java', '.cpp', '.c', '.go', '.rs']
        self.doc_extensions = doc_extensions or ['.md', '.txt', '.rst',
                                                 '.json', '.yaml', '.yml', '.toml']
        self.ignored_paths = ignored_paths or ['.git', '__pycache__', '.vscode', '.venv',
                                               'node_modules', '.streamlit', "__init__"]

    def call(self, path: str) -> List[Document]:
        """
        Recursively reads all documents from the given directory path, ignoring specified directories,
        and returns a list of Document objects.
        Args:
            path (str): The root directory path to read documents from.
        Returns:
            List[Document]: A list of Document objects representing the files.
        Raises:
            TypeError: If `path` is not a string.
            Exception: If traversal fails.
        """
        if not isinstance(path, str):
            logger.error(f"Invalid path type: {type(path)}. Expected str.")
            raise TypeError("Path must be a string.")
        documents: List[Document] = []
        root_path = Path(path)
        logger.info(f"Reading all documents from repository: {root_path}")

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
                # Attempt to read file with UTF-8 encoding
                try:
                    content = path_obj.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"UnicodeDecodeError for file {
                                   file_path}. Trying 'latin1' encoding.")
                    content = path_obj.read_text(encoding='latin1')
                is_implementation = (
                    not relative_path.startswith('test_')
                    and not relative_path.startswith('app_')
                    and 'test' not in relative_path.lower()
                )
                extension = path_obj.suffix[1:]  # Remove the dot
                title = path_obj.name
                doc = Document(
                    text=content,
                    meta_data={
                        "file_path": relative_path,
                        "type": extension,  # extension without dot
                        "is_code": is_code,
                        "is_implementation": is_implementation,
                        "title": title,
                    },
                    # Initialize other required fields if necessary, e.g.,
                    # vector=[], parent_doc_id=None, order=0, score=None
                )
                documents.append(doc)
                logger.debug(f"Processed file {file_path} into Document with title '{
                             doc.meta_data['title']}'")
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        try:
            # Collect all relevant extensions
            extensions = self.code_extensions + self.doc_extensions
            LoggerUtility.traverse_and_log(
                root_path, self.ignored_paths, extensions, process_file
            )
            logger.info(f"Processed {len(documents)
                                     } documents from path {root_path}.")
        except Exception as e:
            logger.error(f"Error during traversal and processing: {e}")
            raise e
        return documents

    def __call__(self, path: str) -> List[Document]:
        """
        Allows the component to be called directly like a function.
        Args:
            path (str): The root directory path to read documents from.
        Returns:
            List[Document]: A list of Document objects representing the files.
        """
        return self.call(path)


EntityMapping.register("DocumentToAdalDocument", DocumentToAdalDocument)
__all__ = ["DocumentToAdalDocument"]
