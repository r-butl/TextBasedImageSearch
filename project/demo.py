import os
import torch
import numpy as np
from model import Model
from data_controller import EmbeddingDataset
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sentence embedding model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set paths
base_path = "/research2/lrbutler/TextBasedImageSearch/data/"
image_embedding_dir = os.path.join(base_path, "formatted_data/test/image_embeddings")
image_file_dir = os.path.join(base_path, "raw_data/train_images")
text_embedding_dir = os.path.join(base_path, "formatted_data/test/text_embeddings")
model_weights_path = "best_model.pt"

# Load dataset and model
dataset = EmbeddingDataset(image_embedding_dir, text_embedding_dir)
input_shape, output_shape = dataset.get_feature_sizes()

model = Model(input_size=384, output_size=512, layers=[1024, 768, 512, 384])
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

# Load image embeddings and filenames
image_files = sorted([f[:-4] for f in os.listdir(image_embedding_dir) if f.endswith(".npy")])
image_embeddings = [np.load(os.path.join(image_embedding_dir, f"{fname}.npy")) for fname in image_files]
image_embeddings = np.array(image_embeddings)
print(f"Loaded {len(image_embeddings)} image embeddings.")

# Encode sentence
def encode_text_to_embedding(text):
    text_features = text_model.encode(text)
    text_tensor = torch.tensor(text_features).unsqueeze(0).float().to(device)
    with torch.no_grad():
        predicted_embedding = model(text_tensor)
    return predicted_embedding.cpu().numpy()

# Find and display top K similar images
def find_similar_images(sentence_embedding, top_k=10):
    similarities = cosine_similarity(sentence_embedding.reshape(1, -1), image_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1][:top_k]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, idx in enumerate(sorted_indices):
        img_id = image_files[idx]
        img_path = os.path.join(image_file_dir, f"{img_id}.jpg")
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Rank {i+1}\nSim: {similarities[idx]:.2f}")
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    query = input("Enter a sentence to search for similar images: ")
    embedding = encode_text_to_embedding(query)
    find_similar_images(embedding)
