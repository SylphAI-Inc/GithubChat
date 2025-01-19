# rag.py

import json
import warnings
from uuid import uuid4
from typing import Optional, Any
from dotenv import load_dotenv

from adalflow.core.db import LocalDB
from adalflow.core.types import (
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)

from adalflow import (
    Embedder,
    Generator,
    setup_env
)

from adalflow.core.component import Component
from adalflow.core.string_parser import JsonParser
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.data_process import RetrieverOutputToContextStr

from config import configs, prompts
from logging_config import (
    logger,
    log_init,
    log_info,
    log_success,
    log_warning,
    log_error
)

load_dotenv()

# Optionally, comment out the warning filter during debugging
warnings.filterwarnings('error', category=RuntimeWarning)


class Memory(Component):
    """
    Stores the current conversation history and keeps track of individual dialog turns.

    This component allows you to retrieve a formatted string of the current conversation history
    and add new user-assistant turns as they happen.
    """

    def __init__(self, turn_database: LocalDB = None):
        """Initialize the conversation memory with an optional turn database."""
        super().__init__()
        self.current_conversation = Conversation()
        self.turn_database = turn_database or LocalDB()      # Stores all turns
        # Potentially store entire conversation objects here
        self.conversation_database = LocalDB()
        log_success("Memory component initialized.")

    def __call__(self) -> str:
        """
        Returns the current conversation history as a formatted string.
        Useful for providing context to an LLM or any other interface.
        """
        log_info("Retrieving current conversation history from Memory.")
        if not self.current_conversation.dialog_turns:
            logger.debug("No dialog turns available in current conversation.")
            return ""

        formatted_history = []
        for turn in self.current_conversation.dialog_turns.values():
            formatted_history.append(f"User: {turn.user_query.query_str}")
            formatted_history.append(
                f"Assistant: {turn.assistant_response.response_str}")

        logger.debug("Conversation history retrieved successfully.")
        return "\n".join(formatted_history)

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        """Add a new dialog turn to the current conversation."""
        log_info("Adding a dialog turn to the Memory component.")
        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(
                response_str=assistant_response),
        )

        # Append to the in-memory conversation
        self.current_conversation.append_dialog_turn(dialog_turn)

        # Also log it in the turn_database
        self.turn_database.add({
            "user_query": user_query,
            "assistant_response": assistant_response
        })
        logger.debug(
            "Dialog turn added successfully: user_query=%s, assistant_response=%s",
            user_query,
            assistant_response
        )


class RAG(Component):
    """
    Retrieval-Augmented Generation (RAG) component.

    Responsible for:
    1. Retrieving relevant documents or chunks using a FAISS-based retriever.
    2. Combining them into context (pre-processing if needed).
    3. Generating an answer using an LLM (through Generator).
    4. Storing the conversation in memory.
    """

    def __init__(self, db: LocalDB = None, prompt_type: str = "code_analysis"):
        """Initialize the RAG component."""
        super().__init__()
        log_info(f"Initializing RAG with prompt_type={prompt_type}")

        # Initialize the conversation memory
        self.memory = Memory()

        if db is None:
            log_warning("No database provided. Initializing a new LocalDB.")
            self.db = LocalDB("new_db")
            self.transformed_documents = []
        else:
            self.db = db
            try:
                self.transformed_documents = self.db.get_transformed_data(
                    "split_and_embed")
                log_success(
                    "Loaded transformed documents from the provided database.")
            except Exception as e:
                log_error(f"Failed to load transformed documents: {e}")
                self.transformed_documents = []

        # Try to initialize the embedder, retriever, and generator
        try:
            embedder = Embedder(
                model_client=configs["embedder"]["model_client"](),
                model_kwargs=configs["embedder"]["model_kwargs"],
            )
            log_init("Embedder", configs["embedder"])

            # FAISS-based retriever
            self.retriever = FAISSRetriever(
                **configs["retriever"],
                embedder=embedder,
                documents=self.transformed_documents,
                document_map_func=lambda doc: doc.vector,
            )
            log_init("FAISS Retriever", configs["retriever"])

            self.retriever_output_processor = RetrieverOutputToContextStr(
                deduplicate=True)
            log_info("Retriever output processor initialized with deduplication.")

            # Prompt and generator
            prompt_template = prompts.get(
                prompt_type, prompts["code_analysis"])
            self.generator = Generator(
                prompt_kwargs={"task_desc_str": prompt_template},
                model_client=configs["generator"]["model_client"](),
                model_kwargs=configs["generator"]["model_kwargs"],
                output_processors=JsonParser(),
            )
            log_init("Generator", configs["generator"])

        except RuntimeWarning as rw:
            log_error(f"RuntimeWarning during RAG initialization: {rw}")
            raise
        except Exception as e:
            log_error(f"Error during RAG initialization: {e}")
            raise

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        """
        Generate a response using the loaded LLM model and optionally a context string
        derived from the conversation + retrieved documents.
        """
        log_info(f"Generating response for query: {query}")
        if not self.generator:
            log_error("Generator not set. Cannot generate a response.")
            raise ValueError("Generator is not set")

        # Force mention "implementation" if user asks about a class but doesn't mention "implementation"
        if "class" in query.lower() and "implementation" not in query.lower():
            logger.debug("Adjusting query to include implementation details.")
            query = f"Show and explain the implementation of the {query}"

        combined_context = ""
        if context:
            logger.debug("Adding code context to combined_context.")
            combined_context += f"Code to analyze:\n```python\n{
                context}\n```\n"

        # Also retrieve any stored conversation history
        conversation_history = self.memory()
        if conversation_history:
            logger.debug("Appending conversation history to combined_context.")
            combined_context += f"\nPrevious conversation:\n{
                conversation_history}"

        # Prepare the final prompt
        prompt_kwargs = {
            "context_str": combined_context,
            "input_str": query,
        }
        logger.debug("Prompt prepared:\n%s",
                     json.dumps(prompt_kwargs, indent=2))

        try:
            response = self.generator(prompt_kwargs=prompt_kwargs)
            answer = response.data.get("answer", "No answer provided")
            log_success(f"Response generated successfully: {answer}")
            logger.debug("Full response data: %s", response.data)
            return answer
        except RuntimeWarning as rw:
            log_error(f"RuntimeWarning during response generation: {rw}")
            raise
        except Exception as e:
            log_error(f"Error during response generation: {e}")
            raise

    def __call__(self, query: str) -> Any:
        """
        Process a user query by retrieving relevant documents, generating a response,
        and updating memory with the new turn.
        """
        log_info(f"Processing query: {query}")

        # If user asks about a class, ensure search includes the word 'implementation'
        if "class" in query.lower() and "implementation" not in query.lower():
            search_query = f"class implementation {query}"
            logger.debug(f"Search query adjusted to: {search_query}")
        else:
            search_query = query

        # The retriever currently returns a list with one element (RetrieverOutput).
        # We only handle a single output here, so we take the [0] index.
        try:
            retriever_output = self.retriever(search_query)[0]
            logger.debug(f"RetrieverOutput obtained for search query: {
                         search_query}")
        except Exception as e:
            log_error(f"Error retrieving documents: {e}")
            raise

        # Map the retrieved doc indices to actual Documents in memory
        try:
            # The retriever_output has a list of doc_indices. We then fetch from self.transformed_documents.
            retriever_output.documents = [
                self.transformed_documents[idx] for idx in retriever_output.doc_indices
            ]
            logger.debug("Mapped document indices to actual Document objects.")
        except IndexError as ie:
            log_error(f"Doc index out of range: {ie}")
        except Exception as e:
            log_error(f"Error mapping documents: {e}")

        # Build a context string from the retrieved documents
        try:
            context_str = self.retriever_output_processor(retriever_output)
            logger.debug("Built context string from retrieved documents.")
        except Exception as e:
            log_error(f"Error processing retriever outputs: {e}")
            context_str = ""

        # Generate a final answer using the context
        try:
            response = self.generate(query, context=context_str)
            log_success(f"Response generated for query '{query}': {response}")
        except Exception as e:
            log_error(f"Error during response generation: {e}")
            response = "An error occurred while generating the response."

        # Update memory with this latest turn
        try:
            self.memory.add_dialog_turn(
                user_query=query, assistant_response=response)
            log_success(
                "Updated conversation memory with the latest dialog turn.")
        except Exception as e:
            log_error(f"Error updating memory: {e}")

        return response, retriever_output


if __name__ == "__main__":
    setup_env()
    log_info("Starting RAG in standalone mode.")
    try:
        rag_component = RAG()
        log_success("RAG initialized. Type your query or 'exit' to quit.")
    except Exception as e:
        log_error(f"Failed to initialize RAG: {e}")
        exit(1)

    while True:
        try:
            query_input = input("Enter your query (or type 'exit' to stop): ")
            if query_input.lower() in ["exit", "quit", "stop"]:
                log_info("Exiting RAG. Goodbye!")
                break

            logger.debug(f"User entered query: {query_input}")
            response, retriever_output = rag_component(query_input)
            print(f"\nResponse:\n{response}\n")
            logger.debug(f"Assistant response: {response}")
            print(f"Retriever Output:\n{retriever_output}\n")
        except Exception as e:
            log_error(f"An error occurred while processing the query: {e}")
            print(f"An error occurred while processing the query: {e}")


__all__ = ["Memory", "RAG"]
