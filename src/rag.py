from typing import Any, List, Tuple, Optional
from uuid import uuid4
import os
from datetime import datetime

import adalflow as adal
from adalflow.core.types import (
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
)
from adalflow.core.component import DataComponent
from config import configs
from src.data_pipeline import DatabaseManager
from adalflow.utils import printc
from dataclasses import dataclass, field


class Memory(DataComponent):
    """Simple conversation management with a list of dialog turns."""

    def __init__(self):
        super().__init__()
        self.current_conversation = Conversation()

    def call(self) -> List[DialogTurn]:

        all_dialog_turns = self.current_conversation.dialog_turns

        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(response_str=assistant_response),
        )

        self.current_conversation.append_dialog_turn(dialog_turn)


@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc": "Rationale for the answer."})
    answer: str = field(default="", metadata={"desc": "Answer to the user query."})

    __output_fields__ = ["rationale", "answer"]


system_prompt = r"""
You are a code assistant which answer's user question on a Github Repo. 
You will receive user query, relevant context, and past conversation history.
Think step by step."""

# history is a list of dialog turns
RAG_TEMPLATE = r"""<START_OF_SYS_PROMPT>
{{system_prompt}}
{{output_format_str}}
<END_OF_SYS_PROMPT>
{# OrderedDict of DialogTurn #}
{% if conversation_history %}
<START_OF_CONVERSATION_HISTORY>
{% for key, dialog_turn in conversation_history.items() %}
{{key}}.
User: {{dialog_turn.user_query.query_str}}
You: {{dialog_turn.assistant_response.response_str}}
{% endfor %}
<END_OF_CONVERSATION_HISTORY>
{% endif %}
{% if contexts %}
<START_OF_CONTEXT>
{% for context in contexts %}
{{loop.index }}.
File Path: {{context.meta_data.get('file_path', 'unknown')}}
Content: {{context.text}}
{% endfor %}
<END_OF_CONTEXT>
{% endif %}
<START_OF_USER_PROMPT>
{{input_str}}
<END_OF_USER_PROMPT>
"""


class RAG(adal.Component):
    __doc__ = """RAG with one repo.
    If you want to load a new repo. You need to call prepare_retriever(repo_url_or_path) first."""

    def __init__(self):

        super().__init__()

        # Initialize embedder, generator, and db_manager
        self.memory = Memory()

        self.embedder = adal.Embedder(
            model_client=configs["embedder"]["model_client"](),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )

        self.initialize_db_manager()

        # Get the appropriate prompt template
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)

        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": data_parser.get_output_format_str(),
                "conversation_history": self.memory(),
                "system_prompt": system_prompt,
                "contexts": None,
            },
            model_client=configs["generator"]["model_client"](),
            model_kwargs=configs["generator"]["model_kwargs"],
            output_processors=data_parser,
        )
        self.previous_retrieved_documents = None

    def initialize_db_manager(self):
        self.db_manager = DatabaseManager()
        self.transformed_docs = []

    def prepare_retriever(self, repo_url_or_path: str):
        r"""Run prepare_retriever once for each repo."""
        self.initialize_db_manager()
        self.transformed_docs = self.db_manager.prepare_database(repo_url_or_path)
        print(f"len(self.transformed_docs): {len(self.transformed_docs)}")
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=self.embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )

    def is_clarification_query(self, query: str) -> bool:
        """
        Determines if the current query is a clarification of a previous query.
        """
        if not self.memory():
            return False

        clarification_prompt = f"""
        You are a clarification detector. Analyze if the query is a follow-up or clarification of the previous conversation.
        Your response should include:
        - A rationale explaining your reasoning
        - A clear True/False answer

        Output your response in this format:
            {{
            "rationale": "Your step-by-step reasoning here",
            "answer": "True or False"
            }}

        Conversation History:
        {self.memory()}

        Query:
        {query}
        """
        response = self.generator(
            prompt_kwargs={
                "conversation_history": self.memory(),
                "system_prompt": clarification_prompt,
            },
        )

        is_clarification = "true" in response.data.answer.lower()
        return is_clarification

    def call(self, query: str) -> Tuple[Any, Any]:
        previous_context = (
            self.previous_retrieved_documents[0].documents
            if self.previous_retrieved_documents
            else None
        )

        is_clarification = self.is_clarification_query(query)

        if is_clarification and self.previous_retrieved_documents:
            retrieved_documents = self.previous_retrieved_documents
        else:
            retrieved_documents = self.retriever(query)
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]
            self.previous_retrieved_documents = retrieved_documents

        printc(f"retrieved_documents: {retrieved_documents[0].documents}")
        printc(f"memory: {self.memory()}")

        prompt_kwargs = {
            "input_str": query,
            "contexts": retrieved_documents[0].documents,
            "conversation_history": self.memory(),
        }
        response = self.generator(
            prompt_kwargs=prompt_kwargs,
        )

        # for debug
        prompt_str = self.generator.get_prompt(**prompt_kwargs)
        printc(f"prompt_str: {prompt_str}")

        final_response = response.data

        self.memory.add_dialog_turn(user_query=query, assistant_response=final_response)

        return final_response, retrieved_documents

    def _format_doc_paths(self, documents: List[Any]) -> str:
        """Helper to format document paths for logging"""
        return "\n  ".join(
            f"- {doc.meta_data.get('file_path', 'unknown')}"
            for doc in documents
        )


if __name__ == "__main__":
    from adalflow.utils import get_logger

    adal.setup_env()
    # repo_url = "https://github.com/SylphAI-Inc/AdalFlow"
    repo_url = "https://github.com/SylphAI-Inc/GithubChat"
    rag = RAG()
    rag.prepare_retriever(repo_url)
    print(
        f"RAG component initialized for repo: {repo_url}. Type your query below or type 'exit' to quit."
    )

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
            rag.memory.add_dialog_turn(user_query=query, assistant_response=response)
            print(f"\nResponse:\n{response}\n")
            print(f"Retrieved Documents:\n{retrieved_documents}\n")
        except Exception as e:
            print(f"An error occurred while processing the query: {e}")
