import adalflow as adal
from adalflow.core.types import ModelClientType, Document, List
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import re
import glob
from adalflow.utils import get_adalflow_default_root_path
from config import configs
from adalflow.core.db import LocalDB
from qdrant_client import QdrantClient, models


def initialize_qdrant(collection_name: str, vector_size: int) -> QdrantClient:
    """Initialize Qdrant client and create collection if needed.
    
    Args:
        collection_name (str): Name of the collection to create/use
        vector_size (int): Size of the vectors (embedding dimensions)
        
    Returns:
        QdrantClient: Initialized Qdrant client
    """
    client = QdrantClient(url="http://localhost:6333")
    
    # Create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        )
    
    return client


def upload_to_qdrant(client: QdrantClient, collection_name: str, documents: List[Document]):
    """Upload documents to Qdrant.
    
    Args:
        client (QdrantClient): Qdrant client
        collection_name (str): Name of the collection to upload to
        documents (List[Document]): List of documents to upload
    """
    points = []
    for i, doc in enumerate(documents):
        points.append(
            models.PointStruct(
                id=i,
                vector=doc.vector,
                payload={
                    "text": doc.text,
                    "file_path": doc.meta_data.get("file_path", ""),
                    "type": doc.meta_data.get("type", ""),
                    "is_code": doc.meta_data.get("is_code", True),
                    "is_implementation": doc.meta_data.get("is_implementation", False)
                }
            )
        )
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )


def prepare_data_pipeline():
    splitter = TextSplitter(**configs["text_splitter"])
    embedder = adal.Embedder(
        model_client=configs["embedder"]["model_client"](),
        model_kwargs=configs["embedder"]["model_kwargs"],
    )
    embedder_transformer = ToEmbeddings(
        embedder=embedder, batch_size=configs["embedder"]["batch_size"]
    )
    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer


def download_github_repo(repo_url, local_path):
    """
    Downloads a GitHub repository to a specified local path.

    Args:
        repo_url (str): The URL of the GitHub repository to clone.
        local_path (str): The local directory where the repository will be cloned.

    Returns:
        str: The output message from the `git` command.
    """
    try:
        # Check if Git is installed
        print(f"local_path: {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Clone the repository
        result = subprocess.run(
            ["git", "clone", repo_url, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        return f"Error during cloning: {e.stderr.decode('utf-8')}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def read_all_documents(path: str):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.

    Returns:
        list: A list of strings, where each string is the content of a file.
    """
    documents = []
    pathes = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
                    pathes.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return [
        adal.Document(text=doc, meta_data={"title": path})
        for doc, path in zip(documents, pathes)
    ]


def transform_documents_and_save_to_db(documents: List[Document], db_path: str, use_qdrant: bool = True):
    """
    Transforms a list of documents and saves them to a database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.
        use_qdrant (bool): Whether to also store in Qdrant for vector search.
    """
    # Get the data transformer
    data_transformer = prepare_data_pipeline()

    # Save the documents to a local database
    db = LocalDB("code_db")
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)

    # If using Qdrant, also store the transformed documents there
    if use_qdrant:
        transformed_docs = db.get_transformed_data("split_and_embed")
        vector_size = configs["embedder"]["model_kwargs"]["dimensions"]
        qdrant_client = initialize_qdrant("code_chunks", vector_size)
        upload_to_qdrant(qdrant_client, "code_chunks", transformed_docs)

    return db


def chat_with_adalflow_lib():
    """
    (1) Download repo: https://github.com/SylphAI-Inc/AdalFlow
    (2) Read all documents in the repo
    (3) Transform the documents using the data pipeline
    (4) Save the transformed documents to a local database
    """
    # Download the repository
    repo_url = "https://github.com/SylphAI-Inc/AdalFlow"
    local_path = os.path.join(get_adalflow_default_root_path(), "AdalFlow")
    download_github_repo(repo_url, local_path)
    # Read all documents in the repository
    documents = read_all_documents(local_path)
    # Transform the documents using the data pipeline
    db_path = os.path.join(get_adalflow_default_root_path(), "db_adalflow")
    transform_documents_and_save_to_db(documents, db_path)


from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.db import LocalDB


if __name__ == "__main__":
    from adalflow.utils import get_logger

    adal.setup_env()

    chat_with_adalflow_lib()

    # # get_logger()
    # repo_url = "https://github.com/microsoft/LMOps"
    # from adalflow.utils import get_adalflow_default_root_path

    # local_path = os.path.join(get_adalflow_default_root_path(), "LMOps")

    # # download_github_repo(repo_url, local_path)

    # target_path = os.path.join(local_path, "prompt_optimization")

    # documents = read_all_documents(target_path)
    # print(len(documents))
    # print(documents[0])
    # # transformed_documents = prepare_data_pipeline()(documents[0:2])
    # # print(len(transformed_documents))
    # # print(transformed_documents[0])

    # # save to local db
    # # from adalflow.core.db import LocalDB

    # db = LocalDB("microsft_lomps")
    # key = "split_and_embed"
    # print(prepare_data_pipeline())
    # db.register_transformer(transformer=prepare_data_pipeline(), key=key)
    # db.load(documents)
    # db.transform(key=key)
    # transformed_docs = db.transformed_items[key]
    # print(len(transformed_docs))
    # print(transformed_docs[0])
    # db_path = os.path.join(get_adalflow_default_root_path(), "db_microsft_lomps")
    # db.save_state(filepath=db_path)
    # db = load_db(db_path)
