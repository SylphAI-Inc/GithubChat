from typing import Optional, Any, List, Dict
import os
from uuid import uuid4

import adalflow as adal
from adalflow.core.db import LocalDB
from adalflow.core.types import (
    ModelClientType,
    Document,
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)
from adalflow.core.string_parser import JsonParser
from adalflow.components.retriever import QdrantRetriever
from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    TextSplitter,
)
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.component import Component
from config import configs
from qdrant_client import QdrantClient, models


rag_prompt_task_desc = r"""
You are a code analysis assistant that helps users understand code implementations.

Your task is to analyze code and explain its functionality, focusing on:
1. Implementation details and how the code works
2. Class methods, their purposes, and interactions
3. Key algorithms and data structures used
4. Code patterns and architectural decisions

When analyzing code:
- Be concise and focus on the most important aspects
- Explain the main purpose and key functionality first
- Highlight critical methods and their roles
- Keep explanations clear and to the point

When asked about a specific class or function:
1. Start with a one-sentence overview
2. List the key methods and their purposes
3. Explain the main functionality
4. Keep the explanation focused and brief

Previous conversation history is provided to maintain context of the discussion.
Use the conversation history to provide more relevant and contextual answers about the code.

Output JSON format:
{
    "answer": "Concise explanation of the code implementation",
}"""


class Memory(Component):
    def __init__(self, turn_db: LocalDB = None):
        """Initialize the Memory component."""
        super().__init__()
        self.current_conversation = Conversation()
        self.turn_db = turn_db or LocalDB()  # all turns
        self.conver_db = LocalDB()  # a list of conversations

    def call(self) -> str:
        """Returns the current conversation history as a formatted string."""
        if not self.current_conversation.dialog_turns:
            return ""

        formatted_history = []
        for turn in self.current_conversation.dialog_turns.values():
            formatted_history.extend(
                [
                    f"User: {turn.user_query.query_str}",
                    f"Assistant: {turn.assistant_response.response_str}",
                ]
            )
        return "\n".join(formatted_history)

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        """Add a new dialog turn to the current conversation."""
        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(response_str=assistant_response),
        )

        self.current_conversation.append_dialog_turn(dialog_turn)
        self.turn_db.add(
            {"user_query": user_query, "assistant_response": assistant_response}
        )


class RAG(adal.Component):
    def __init__(self, index_path: str = None):
        super().__init__()

        # Initialize memory component
        self.memory = Memory()

        if index_path is None:
            index_path = os.path.join(
                get_adalflow_default_root_path(), "db_adalflow"
            )

        try:
            self.db = LocalDB.load_state(index_path)
            self.transformed_docs = self.db.get_transformed_data("split_and_embed")
        except (FileNotFoundError, KeyError):
            print(f"No existing database found at {index_path}. Initializing new database.")
            self.db = LocalDB("new_db")
            self.transformed_docs = []

        embedder = adal.Embedder(
            model_client=configs["embedder"]["model_client"](),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )

        # Initialize Qdrant client and retriever
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        collection_name = "code_chunks"

        # Create collection if it doesn't exist
        try:
            self.qdrant_client.get_collection(collection_name)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=configs["embedder"]["model_kwargs"]["dimensions"],
                    distance=models.Distance.COSINE,
                )
            )

        # Upload documents to Qdrant if we have any
        if self.transformed_docs:
            points = []
            for i, doc in enumerate(self.transformed_docs):
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
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )

        # Set up Qdrant retriever
        self.retriever = QdrantRetriever(
            collection_name=collection_name,
            client=self.qdrant_client,
            embedder=embedder,
            top_k=configs["retriever"].get("top_k", 5),
            text_key="text",
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_code",
                        match=models.MatchValue(value=True),
                    )
                ]
            )
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)

        self.generator = adal.Generator(
            prompt_kwargs={
                "task_desc_str": rag_prompt_task_desc,
            },
            model_client=configs["generator"]["model_client"](),
            model_kwargs=configs["generator"]["model_kwargs"],
            output_processors=JsonParser(),
        )

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        # Modify query to focus on implementation if asking about a class
        if 'class' in query.lower() and 'implementation' not in query.lower():
            query = f"Show and explain the implementation of the {query}"

        # Add conversation history to context
        full_context = ""
        if context:
            full_context += f"Code to analyze:\n```python\n{context}\n```\n"
        
        # Get conversation history from memory component
        conversation_history = self.memory()
        if conversation_history:
            full_context += f"\nPrevious conversation:\n{conversation_history}"

        prompt_kwargs = {
            "context_str": full_context,
            "input_str": query,
        }
        response = self.generator(prompt_kwargs=prompt_kwargs)
        return response.data['answer']

    def call(self, query: str) -> Any:
        # Modify query to focus on implementation if asking about a class
        if 'class' in query.lower() and 'implementation' not in query.lower():
            search_query = f"class implementation {query}"
        else:
            search_query = query

        retrieved_documents = self.retriever(search_query)
        context_str = self.retriever_output_processors(retrieved_documents)
        response = self.generate(query, context=context_str)
        
        # Update conversation history in memory
        self.memory.add_dialog_turn(
            user_query=query,
            assistant_response=response
        )

        return response, retrieved_documents


if __name__ == "__main__":
    from adalflow.utils import get_logger

    adal.setup_env()
    rag = RAG()
    print("RAG component initialized. Type your query below or type 'exit' to quit.")

    while True:
        # Get user input
        query = input("Enter your query (or type 'exit' to stop): ")

        # Exit condition
        if query.lower() in ["exit", "quit", "stop"]:
            print("Exiting RAG component. Goodbye!")
            break

        # Process the query
        try:
            response, retrieved_documents = rag(query)
            print(f"\nResponse:\n{response}\n")
            print(f"Retrieved Documents:\n{retrieved_documents}\n")
        except Exception as e:
            print(f"An error occurred while processing the query: {e}")
