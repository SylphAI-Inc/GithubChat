import os
from typing import List

from adalflow import Sequential
from adalflow.core.db import LocalDB
from adalflow.core.types import Document

from adalflow.utils import get_adalflow_default_root_path

from logging_config import (
    log_info,
    log_success,
    log_warning,
    log_error,
    # log_debug
)


class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    """

    def __init__(self, repo_name: str):
        """
        Initialize the DatabaseManager with the repository name and logger.

        :param repo_name: Name of the repository.
        :param logger: Logger instance for logging messages.
        """

        # Generate directory structure using private method
        paths = self._create_repo(repo_name)
        self.db_file_path = paths["db_file_path"]
        self.source_dir = paths["source_dir"]

        self.db = None

    def _create_repo(self, repo_name: str) -> dict:
        """
        Create a persistent directory structure for a repository within the home directory.

        :param repo_name: Name of the repository.
        :return: Dictionary with paths to the source directory and database file.
        :raises Exception: If directory creation fails.
        """
        log_info(f"Creating repository structure for {repo_name}")

        try:
            # Root directory for adalflow
            root_path = get_adalflow_default_root_path()
            log_info(f"Root path for adalflow: {root_path}")

            # Ensure root directory exists
            os.makedirs(root_path, exist_ok=True)

            # Source directory for repository data
            source_dir = os.path.join(root_path, "repositories", repo_name)
            os.makedirs(source_dir, exist_ok=True)
            log_info(f"Source directory created at {source_dir}")

            # Path to the database file
            db_dir = os.path.join(root_path, "databases")
            os.makedirs(db_dir, exist_ok=True)
            db_file_path = os.path.join(db_dir, f"{repo_name}.pkl")
            log_info(f"Database file path: {db_file_path}")

            return {
                "source_dir": source_dir,
                "db_file_path": db_file_path,
            }

        except Exception as e:
            log_error(f"Failed to create repository structure: {e}")
            raise

    def _create_local_db(self) -> LocalDB:
        """
        Create a new LocalDB at the specified file path.

        :return: An instance of LocalDB.
        """
        log_info(f"Creating database at {self.db_file_path}")
        try:
            db = LocalDB("code_db")
            self.db = db
            log_success(f"LocalDB created successfully at {self.db_file_path}")
            return db
        except Exception as e:
            log_error(f"Failed to create database at {self.db_file_path}: {e}")
            raise

    def load_database(self) -> LocalDB:
        """
        Load an existing LocalDB from the specified file path.

        :return: An instance of LocalDB.
        """
        log_info(f"Loading existing database from {self.db_file_path}")
        try:
            db = LocalDB.load_state(filepath=self.db_file_path)
            self.db = db
            log_success(f"Database loaded successfully from f{
                        self.db_file_path}")
            return db
        except Exception as e:
            log_error(f"Failed to load database from {self.db_file_path}: {e}")
            raise

    def load_or_create_db(self) -> bool:
        """
        Load an existing LocalDB or create a new one if it doesn't exist.

        :return: Tuple containing the LocalDB instance and a boolean indicating if transformations are needed.
        """
        if os.path.exists(self.db_file_path):
            try:
                self.load_database()
                log_info(
                    "Existing database loaded. Skipping redundant transformations.")
                return False
            except Exception:
                log_warning("Loading failed. Creating a new database.")
        else:
            log_warning("Database not found. Creating a new one.")

        # Create a new database if it doesn't exist or loading fails
        self._create_local_db()
        return True

    def transform_documents_and_save(self, documents: List[Document], pipeline: Sequential):
        """
        Transform documents using the provided pipeline and save them to the database.

        :param documents: A list of Document objects to be transformed and saved.
        :param pipeline: A Sequential transformation pipeline to process the documents.
        :raises Exception: If any step in the transformation or saving process fails.
        """
        log_info(f"Transforming documents and saving to database at {
                 self.db_file_path}")
        try:
            # Register the transformer pipeline with the database
            self.db.register_transformer(
                transformer=pipeline, key="split_and_embed")
            log_info("Transformer 'split_and_embed' registered successfully.")

            # Load documents into the database
            self.db.load(documents)
            log_info("Documents loaded into the database.")

            # Transform documents using the registered transformer
            self.db.transform(key="split_and_embed")
            log_success("Documents transformed successfully.")

            # Save the transformed state of the database
            self.db.save_state(filepath=self.db_file_path)
            log_info(f"Database saved successfully at {self.db_file_path}")
        except Exception as e:
            log_error(f"Failed to transform and save documents: {e}")
            raise
