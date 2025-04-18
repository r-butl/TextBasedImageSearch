import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

database = np.loadtxt("random_vectors.txt")

query_data = np.loadtxt("test_input.txt")

queries = query_data[:, :10] 
ground_truth_indices = query_data[:, 10].astype(int)

reciprocal_ranks = []

for i, (vector, true_index) in enumerate(zip(queries, ground_truth_indices)):
    vector = vector.reshape(1, -1)
    similarities = cosine_similarity(vector, database)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    rank = np.where(sorted_indices == true_index)[0][0] + 1
    rr = 1 / rank
    reciprocal_ranks.append(rr)

    print(f"Query {i+1}")
    print("True index:", true_index)
    print("Similarity scores (sorted):")
    for idx in sorted_indices:
        print(f"Index {idx}: {similarities[idx]:.4f}")
    print(f"Correct index ranked at position: {rank}")
    print(f"Reciprocal Rank: {rr:.4f}")

mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
print(reciprocal_ranks)
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
