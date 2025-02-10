from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    repo_url: str
    query: str

class DocumentMetadata(BaseModel):
    file_path: str
    type: str
    is_code: bool = False
    is_implementation: bool = False
    title: str = ""

# Consider extending what metadata is tracked in DoumentMetaData
# or create an additional class. I think it could be important to track other
# metadata metrics, such as line count and file size.

# Justification: 
# During the EDA stages of model improvement, tracking repo sizes, file sizes, line counts
# could reveal relationships between performance.

class Document(BaseModel):
    text: str
    meta_data: DocumentMetadata

class QueryResponse(BaseModel):
    rationale: str
    answer: str
    contexts: List[Document]


__all__ = [
    "QueryReqest",
    "QueryResponse",
    "Document",
    "DocumentMetadata"
]