import pickle
import numpy as np
from typing import List
from copy import deepcopy

from adalflow.core.types import Document
from adalflow.core.component import Component
from adalflow.utils import EntityMapping

import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Should be picklable and serializable


class L2Norm(Component):
    """
    A component that applies L2 normalization to each Document's vector.

    - **Input**: List[Document]
    - **Output**: List[Document] with normalized vectors
    """

    def __init__(self):
        """
        Initializes the L2Norm component and calls the parent class's initializer.
        """
        super().__init__()
        logger.info("Initialized L2Norm component.")

    def call(self, input_data: List[Document]) -> List[Document]:
        """
        Normalize the 'vector' field of each Document in the list to have an L2 norm of 1.

        Args:
            input_data (List[Document]): A list of Document instances to normalize.

        Returns:
            List[Document]: The same list with each Document's vector normalized.

        Raises:
            TypeError: If input_data is not a List[Document] or contains non-Document items.
            ValueError: If any vector does not meet expected dimensions.
        """
        # Immutability in Sequential pipeline, since parts are mutated
        output = deepcopy(input_data)

        if not isinstance(input_data, list):
            logger.error(
                f"L2Norm received an invalid type: {type(input_data)}. Expected List[Document].")
            raise TypeError("L2Norm can only handle a List[Document].")

        for idx, doc in enumerate(output):
            if not isinstance(doc, Document):
                logger.error(
                    f"L2Norm received a non-Document item at index {idx}: {type(doc)}.")
                raise TypeError(
                    f"Expected all items in the list to be Document instances, but got {type(doc)} at index {idx}.")

            if not isinstance(doc.vector, list):
                logger.error(
                    f"Document at index {idx} has an invalid vector type: {type(doc.vector)}. Expected List[float].")
                raise TypeError(
                    f"Document.vector must be a List[float], but got {type(doc.vector)} at index {idx}.")

            try:
                # Higher precision before truncating to prevent some overflow
                vec = np.array(doc.vector, dtype=np.float64)
                # np.linalg.norm computes in float 64 by default, ensuring we avoid overflow prior to truncation
                normalized = self.normalize_l2(vec).astype(np.float32)
                # For rest of the pipeline's compatability, convert back to a List. Later on check if we can convert over portions
                # of relevant code to numpy arrays to leverage numpy vectorization (primarily for embedding generation and FAISS index on GPU)
                doc.vector = normalized.tolist()
                logger.debug(
                    f"Document ID {doc.id} vector normalized successfully.")
            except ValueError as ve:
                logger.error(
                    f"Normalization failed for Document ID {doc.id}: {ve}")
                raise ve

        return output

    def __call__(self, path: str) -> List[Document]:
        """
        Allows the component to be called directly like a function.

        Args:
            path (str): The root directory path to read documents from.

        Returns:
            List[Document]: A list of Document objects representing the files.
        """
        return self.call(path)

    @staticmethod
    def normalize_l2(x: np.ndarray) -> np.ndarray:
        """
        Normalize a NumPy array to have an L2 unit norm.

        Args:
            x (np.ndarray): A 1D NumPy array representing the vector.

        Returns:
            np.ndarray: The L2-normalized vector.

        Raises:
            ValueError: If the input array is not 1D.
        """
        if x.ndim != 1:
            logger.error(
                f"normalize_l2 received an array with invalid number of dimensions: {x.ndim}. Expected 1D.")
            raise ValueError("Expected a 1D array for L2 normalization.")

        norm = np.linalg.norm(x)  # default computes as float64 for norms

        if norm == 0:
            norm = 1e-12  # Avoid division by zero
        return (x / norm).astype(np.float32)  # Now truncate to float32


EntityMapping.register("L2Norm", L2Norm)

__all__ = ["L2Norm"]
