# from typing import List

# # Convert to work with AdalFlow's GitHub Chat.


# def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
#     """
#     Calculate Precision@K.
#     :param retrieved: List of retrieved contexts.
#     :param relevant: List of ground truth relevant contexts.
#     :param k: Number of top contexts to consider.
#     :return: Precision@K value.
#     """
#     retrieved_at_k = retrieved[:k]
#     relevant_count = sum(
#         1 for context in retrieved_at_k if context in relevant)
#     return relevant_count / k


# def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
#     """
#     Calculate Recall@K.
#     :param retrieved: List of retrieved contexts.
#     :param relevant: List of ground truth relevant contexts.
#     :param k: Number of top contexts to consider.
#     :return: Recall@K value.
#     """
#     retrieved_at_k = retrieved[:k]
#     relevant_count = sum(
#         1 for context in retrieved_at_k if context in relevant)
#     return relevant_count / len(relevant) if relevant else 0.0


# def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
#     """
#     Calculate Mean Reciprocal Rank (MRR).
#     :param retrieved: List of retrieved contexts.
#     :param relevant: List of ground truth relevant contexts.
#     :return: MRR value.
#     """
#     for rank, context in enumerate(retrieved, start=1):
#         if context in relevant:
#             return 1 / rank
#     return 0.0
