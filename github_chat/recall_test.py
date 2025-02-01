from adalflow.eval.base import EvaluationResult
from adalflow.eval import RetrieverRecall
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

# retrieved_contexts = [
#     "Apple is founded before Google.",
#     "Feburary has 28 days in common years. Feburary has 29 days in leap years. Feburary is the second month of the year."
# ]

# retrieved_contexts = [
#     "Apple was founded in 1976 by Steve Jobs.",
#     "Bananas are a type of fruit. They are high in potassium.",
#     "The Eiffel Tower is in Paris. It is a famous landmark."
# ]

# gt_contexts = [
#     [
#         "Apple is founded in 1976.",
#         "Google is founded in 1998.",
#         "Apple is founded before Google"
#     ],
#     ["Feburary has 28 days in common years", "Feburary has 29 days in leap years"]
# ]

# gt_contexts = [
#     ["Apple was founded in 1976.", "Steve Jobs founded Apple.",
#         "1976 was the founding year of Apple."],
#     ["Bananas are fruits.", "They contain potassium.",
#         "Potassium is abundant in bananas."],
#     ["Eiffel Tower is in Paris.", "Famous landmarks in Paris include the Eiffel Tower."]
# ]

# retriever_recall = RetrieverRecall()

# thing = retriever_recall.compute(
#     retrieved_contexts, gt_contexts)

# docs = [document for document in]
# # print(f"Recall: {avg_recall}, Recall List: {recall_list}")


# def _compute_single_item(retrieved_context: str, gt_context: List[str]) -> float:
#     relevant_retrieved = sum(
#         1 for context in gt_context if context in retrieved_context)
#     return relevant_retrieved / len(gt_context)


# print(_compute_single_item(retrieved_contexts[0], gt_context=gt_contexts[0]))


# ==========================

class SemanticRetrieverRecall(RetrieverRecall):
    def __init__(self, embedder, threshold=0.85):
        super().__init__()
        self.embedder = embedder
        self.threshold = threshold

    def _compute_single_item(
        self, retrieved_context: List[str], gt_context: List[str]
    ) -> float:
        # Compute embeddings
        retrieved_embeddings = self.embedder(retrieved_context)
        gt_embeddings = self.embedder(gt_context)

        recalled = 0
        for gt_emb in gt_embeddings:
            similarities = cosine_similarity([gt_emb], retrieved_embeddings)
            if np.max(similarities) >= self.threshold:
                recalled += 1
        return recalled / len(gt_context)

    def compute(
        self,
        retrieved_contexts: List[List[str]],
        gt_contexts: List[List[str]],
    ) -> EvaluationResult:
        if len(retrieved_contexts) != len(gt_contexts):
            raise ValueError(
                "The number of retrieved context lists and ground truth context lists should be the same."
            )
        recall_list = []
        for retrieved_context, gt_context in zip(retrieved_contexts, gt_contexts):
            recall = self._compute_single_item(retrieved_context, gt_context)
            recall_list.append(recall)

        avg_score = sum(recall_list) / len(recall_list)
        return EvaluationResult(
            avg_score, recall_list, additional_info={
                "type": f"SemanticRetrieverRecall@{len(retrieved_contexts[0])}"}
        )
