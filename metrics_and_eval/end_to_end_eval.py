# import json
# from typing import List, Dict, Any, Callable, Union

# from retrieval_metrics import precision_at_k, recall_at_k, mean_reciprocal_rank
# from generation_metrics import generation_precision_at_k, generation_recall_at_k
# from bare_bones_rag import generate_answer, retrieve_contexts

# # Convert to work with AdalFlow's GitHub Chat.


# def evaluate_retrieval(
#     queries: List[str],
#     ground_truths: List[List[str]],
#     top_k: int,
#     retrieve_fn: Callable[[str, int], List[Dict[str, Any]]]
# ) -> Dict[str, float]:
#     """
#     Evaluate retrieval metrics over multiple queries using precision@k, recall@k, and MRR.
#     :param queries: List of queries.
#     :param ground_truths: List of lists of relevant contexts per query.
#     :param top_k: Number of top contexts to retrieve.
#     :param retrieve_fn: Function to retrieve contexts for a query.
#     :return: Dictionary with aggregated retrieval metrics.
#     """
#     precisions = []
#     recalls = []
#     mrrs = []

#     for query, relevant in zip(queries, ground_truths):
#         # Ensure ground truths are treated as sets for efficient comparisons
#         relevant_set = set(relevant)
#         retrieved = retrieve_fn(query, top_k=top_k)

#         precisions.append(precision_at_k(retrieved, relevant_set, k=top_k))
#         recalls.append(recall_at_k(retrieved, relevant_set, k=top_k))
#         mrrs.append(mean_reciprocal_rank(retrieved, relevant_set))

#     return {
#         f"Precision@{top_k}": sum(precisions) / len(queries) if queries else 0.0,
#         f"Recall@{top_k}": sum(recalls) / len(queries) if queries else 0.0,
#         "MRR": sum(mrrs) / len(queries) if queries else 0.0,
#     }

# # Need and evaluate generation


# def evaluate_rag(
#     questions: List[str],
#     ground_truth_answers: List[Union[List[str], str]],
#     k_values: List[int],
#     top_k_contexts: int = CONFIG.get("DEFAULT_TOP_K_CONTEXTS"),
#     retrieve_fn: Callable[[str, int], List[str]] = retrieve_contexts,
#     generate_fn: Callable[[str, int], Dict[str, Any]] = generate_answer
# ) -> Dict[str, Any]:
#     """
#     Evaluate Retrieval-Augmented Generation (RAG) performance using retrieval and generation metrics.
#     :param queries: List of queries to evaluate.
#     :param ground_truth_answers: Ground truth answers for each query.
#     :param k_values: List of k values for evaluation (e.g., [1, 5, 10]).
#     :param top_k_contexts: Number of contexts to retrieve for each query.
#     :param retrieve_fn: Function to retrieve contexts for a query.
#     :param generate_fn: Function to generate answers given a query and context.
#     :return: Dictionary containing evaluation metrics for RAG.
#     """
#     main_logger.info("Starting RAG evaluation...")

#     # Validate input lengths
#     if not questions:
#         raise ValueError("No questions provided for evaluation.")
#     if not ground_truth_answers:
#         raise ValueError("No ground truth answers provided for evaluation.")
#     if len(questions) != len(ground_truth_answers):
#         raise ValueError(
#             "Mismatch between the number of questions and ground truth answers.")

#     # Ensure ground_truth_answers is a list of sets for efficient membership checks
#     ground_truth_answers = [
#         set(gt) if isinstance(gt, list) else {gt}
#         for gt in ground_truth_answers
#     ]

#     # Initialize metrics dictionary
#     metrics: Dict[str, float] = {f"precision@{k}": 0.0 for k in k_values}
#     metrics.update({f"recall@{k}": 0.0 for k in k_values})
#     total_questions = len(questions)

#     # Precompute the maximum k value
#     max_k = max(k_values)

#     # Iterate over each question and its corresponding ground truth answers
#     for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers)):
#         main_logger.info(f"Evaluating question {
#                          i + 1}/{total_questions}: {question}")

#         # Retrieve top_k_contexts for the current question
#         try:
#             retrieved_contexts: List[str] = retrieve_fn(
#                 question, top_k=top_k_contexts)
#             # retrieved_contexts shape: [top_k_contexts]
#         except Exception as e:
#             main_logger.error(f"Error retrieving contexts for question '{
#                               question}': {str(e)}")
#             continue

#         # Generate answers using the QA pipeline
#         try:
#             # batch_inputs: List[Dict[str, str]] = [{"question": question, "context": cxt} for cxt in retrieved_contexts]
#             # raw_answers: List[Dict[str, Any]] = qa_pipeline(batch_inputs)
#             # raw_answers shape: [top_k_contexts], each dict contains 'answer' and 'score'

#             gen_result = generate_fn(question, top_k=top_k_contexts)
#             generated_answer = gen_result["best_answer"]
#             all_generated_answers = [ans["answer"]
#                                      for ans in gen_result["all_answers"]]
#             # all_generated_answers shape: [top_k_contexts]

#             # Sort answers by their confidence score in descending order
#             raw_answers_sorted: List[Dict[str, Any]] = sorted(
#                 raw_answers, key=lambda x: x["score"], reverse=True
#             )
#         except Exception as e:
#             main_logger.error(f"Error generating answers for question '{
#                               question}': {str(e)}")
#             continue

#         # precision and recall for each k
#         for k in k_values:
#             current_k = min(k, len(all_generated_answers))
#             if current_k == 0:
#                 continue  # avoid division by zero

#             top_k_generated = all_generated_answers[:current_k]

#             # Calculate generation precision@k and recall@k using generation_metrics
#             gen_precision = generation_precision_at_k(
#                 top_k_generated, list(ground_truth), k=current_k)
#             gen_recall = generation_recall_at_k(
#                 top_k_generated, list(ground_truth), k=current_k)

#             metrics[f"generation_precision@{k}"] += gen_precision
#             metrics[f"generation_recall@{k}"] += gen_recall

#         # Calculate retrieval precision@k, recall@k, and MRR
#         try:
#             retrieval_metrics = evaluate_retrieval(
#                 queries=[question],
#                 ground_truths=[list(ground_truth)],
#                 top_k=top_k_contexts,
#                 retrieve_fn=retrieve_fn
#             )
#             for k in k_values:
#                 metrics[f"precision@{k}"] += retrieval_metrics.get(
#                     f"Precision@{top_k_contexts}", 0.0)
#                 metrics[f"recall@{k}"] += retrieval_metrics.get(
#                     f"Recall@{top_k_contexts}", 0.0)
#             metrics["MRR"] += retrieval_metrics.get("MRR", 0.0)
#         except Exception as e:
#             main_logger.error(f"Error evaluating retrieval for question '{
#                               question}': {str(e)}")
#             continue

#         # Normalize metrics by the total number of questions
#         for k in k_values:
#             metrics[f"precision@{k}"] /= total_questions
#             metrics[f"recall@{k}"] /= total_questions
#             metrics[f"generation_precision@{k}"] /= total_questions
#             metrics[f"generation_recall@{k}"] /= total_questions

#         metrics["MRR"] /= total_questions

#         main_logger.info("RAG evaluation completed.")
#         return metrics


# def main():
#     evaluation_data_path = "metrics/evaluation_data.json"

#     try:
#         with open(evaluation_data_path, 'r') as f:
#             data = json.load(f)
#         queries = data.get("questions")
#         ground_truths = data.get("ground_truth_answers", [])
#     except Exception as e:
#         main_logger.error(f"Failed to load evaluation data: {e}")
#         return

#     if not queries or not ground_truths:
#         main_logger.error(
#             "Evaluation data is missing queries or ground truth answers.")
#         return

#     k_values = CONFIG.get("K_VALUES_FOR_METRICS")
#     top_k_contexts = CONFIG.get("DEFAULT_TOP_K_CONTEXTS", 5)

#     try:
#         retrieval_metrics = evaluate_retrieval(
#             queries=queries,
#             ground_truths=ground_truths,
#             top_k=top_k_contexts,
#             retrieve_fn=retrieve_contexts
#         )
#     except Exception as e:
#         main_logger.error(f"Error during retrieval evaluation: {e}")
#         return

#     main_logger.info("Retrieval Evaluation Completed.")

#     print("=== Retrieval Metrics ===")
#     for metric, value in retrieval_metrics.items():
#         print(f"{metric}: {value:.4f}")

#     # Evaluate RAG metrics
#     try:
#         rag_metrics = evaluate_rag(
#             questions=queries,
#             ground_truth_answers=ground_truths,
#             k_values=k_values,
#             top_k_contexts=top_k_contexts,
#             retrieve_fn=retrieve_contexts,
#             generate_fn=generate_answer
#         )
#     except Exception as e:
#         main_logger.error(f"Error during RAG evaluation: {e}")
#         return

#     print("\n=== RAG Metrics ===")
#     for metric, value in rag_metrics.items():
#         print(f"{metric}: {value:.4f}")

#     # Save metrics
#     try:
#         with open('metrics/retrieval_metrics.json', 'w') as f:
#             json.dump(retrieval_metrics, f, indent=4)
#         with open('metrics/rag_metrics.json', 'w') as f:
#             json.dump(rag_metrics, f, indent=4)
#         main_logger.info("Metrics successfully saved.")
#     except Exception as e:
#         main_logger.error(f"Failed to save metrics: {e}")


# if __name__ == "__main__":
#     main()
