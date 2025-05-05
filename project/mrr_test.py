from data_controller import EmbeddingDataset
import torch
import os
from model import Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

def mrr(model, dataset):
    model.eval()
    reciprocal_ranks = []

    # Extract all image embeddings and compute predicted image embeddings from text using batching
    batch_size = 64
    image_embeddings = []
    query_embeddings = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            text_embs, image_embs = batch
            image_embeddings.append(image_embs.numpy())
            predicted_image_embs = model(text_embs).detach().cpu().numpy()
            query_embeddings.append(predicted_image_embs)

    image_embeddings = np.concatenate(image_embeddings, axis=0)
    query_embeddings = np.concatenate(query_embeddings, axis=0)

    # For each query (predicted image embedding), compute similarity with all true image embeddings
    for i, (query, true_index) in enumerate(tqdm(zip(query_embeddings, range(len(image_embeddings))), total=len(query_embeddings), desc="Querying")):
        query = query.reshape(1, -1)
        similarities = cosine_similarity(query, image_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        rank = np.where(sorted_indices == true_index)[0][0] + 1
        rr = 1 / rank
        reciprocal_ranks.append(rr)

    mrr_value = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"Mean Reciprocal Rank (MRR): {mrr_value:.4f}")
    std_dev_final = np.std(reciprocal_ranks)
    print(f"Final Standard Deviation of Reciprocal Ranks: {std_dev_final:.4f}")
    return mrr_value

def main():
    image_dir = os.path.abspath("./formatted_data/test/image_embeddings")
    text_dir = os.path.abspath("./formatted_data/test/text_embeddings")

    dataset = EmbeddingDataset(image_dir, text_dir)

    input_shape, output_shape = dataset.get_feature_sizes()
    layers = [ 1024, 2048, 1024]

    network = Model(input_shape, output_shape, layers=layers)

    # Load model weights
    model_weights_path = os.path.abspath("best_model.pt")
    if os.path.exists(model_weights_path):
        network.load_state_dict(torch.load(model_weights_path))
        network.eval()  # Set the model to evaluation mode
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

    mrr(network, dataset)

if __name__ == "__main__":
    main()
