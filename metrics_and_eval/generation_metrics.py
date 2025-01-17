
# from typing import List

# # Convert to work with AdalFlow's GitHub Chat.


# def generation_precision_at_k(generated_answers: List[str], ground_truth_answers: List[str], k: int) -> float:
#     """
#     Calculate Generation Precision@K.
#     :param generated_answers: List of generated answers.
#     :param ground_truth_answers: List of ground truth answers.
#     :param k: Number of top answers to consider.
#     :return: Precision@K value.
#     """
#     generated_at_k = generated_answers[:k]
#     ground_truth_set = set(ground_truth_answers)
#     correct = sum(1 for ans in generated_at_k if ans in ground_truth_set)
#     return correct / k if k > 0 else 0.0


# def generation_recall_at_k(generated_answers: List[str], ground_truth_answers: List[str], k: int) -> float:
#     """
#     Calculate Generation Recall@K.
#     :param generated_answers: List of generated answers.
#     :param ground_truth_answers: List of ground truth answers.
#     :param k: Number of top answers to consider.
#     :return: Recall@K value.
#     """
#     generated_at_k = generated_answers[:k]
#     ground_truth_set = set(ground_truth_answers)
#     correct = sum(1 for ans in generated_at_k if ans in ground_truth_set)
#     return correct / len(ground_truth_set) if ground_truth_set else 0.0
