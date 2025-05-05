import pandas as pd
import os
from save_model2.model import Model
import torch
from data_controller import EmbeddingDataset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Load the model
layers = [512, 1024, 2048, 2048, 1024]
input_shape = 384
output_shape = 768
model = Model(input_shape, output_shape, layers=layers)
model_weights_path ="./save_model2/best_model.pt"

if os.path.exists(model_weights_path):
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()  # Set the model to evaluation mode
    print("Model weights loaded successfully.")
else:
    raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

def predict(model, text_emb):
    predicted_image_embs = model(text_emb).detach().cpu().numpy()
    return predicted_image_embs

# Load the dataset
image_dir = os.path.abspath("/home/bchen1/data/formatted_data/test/image_embeddings")
text_dir = os.path.abspath("/home/bchen1/data/formatted_data/test/text_embeddings")
dataset = EmbeddingDataset(image_dir, text_dir)

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

# Meta data
dataset_file_path = '/home/bchen1/data/dataset'
train = 'TextCaps_0.1_train.json'
val = 'TextCaps_0.1_val.json'

train_data = pd.read_json(os.path.join(dataset_file_path, train))['data']
val_data = pd.read_json(os.path.join(dataset_file_path, val))['data']

# Full look up
lookup_table = pd.concat([train_data, val_data])
lookup_table = pd.DataFrame(list(lookup_table))

# Class remapping
class_categories = {"Person": "Human", "Woman": "Human", "Man": "Human", "Girl": "Human", "Boy": "Human", "Human face": "Human", "Human arm": "Human", "Human leg": "Human", "Clothing": "Apparel", "Shirt": "Apparel", "Dress": "Apparel", "Footwear": "Apparel", "Hat": "Apparel", "Glove": "Apparel", "Scarf": "Apparel", "Sunglasses": "Apparel", "Jewelry": "Apparel", "Food": "Food & Drink", "Drink": "Food & Drink", "Beer": "Food & Drink", "Wine": "Food & Drink", "Dairy": "Food & Drink", "Snack": "Food & Drink", "Fruit": "Food & Drink", "Vegetable": "Food & Drink", "Pizza": "Food & Drink", "Cookie": "Food & Drink", "Furniture": "Furniture & Home", "Chair": "Furniture & Home", "Bed": "Furniture & Home", "Table": "Furniture & Home", "Lamp": "Furniture & Home", "Refrigerator": "Furniture & Home", "Oven": "Furniture & Home", "Shelf": "Furniture & Home", "Mirror": "Furniture & Home", "Tool": "Tools & Appliances", "Screwdriver": "Tools & Appliances", "Blender": "Tools & Appliances", "Microwave oven": "Tools & Appliances", "Washing machine": "Tools & Appliances", "Printer": "Tools & Appliances", "Building": "Architecture", "House": "Architecture", "Window": "Architecture", "Door": "Architecture", "Tower": "Architecture", "Castle": "Architecture", "Skyscraper": "Architecture", "Vehicle": "Transportation", "Car": "Transportation", "Bus": "Transportation", "Train": "Transportation", "Airplane": "Transportation", "Helicopter": "Transportation", "Motorcycle": "Transportation", "Boat": "Transportation", "Bicycle": "Transportation", "Sports equipment": "Sports & Leisure", "Ball": "Sports & Leisure", "Racket": "Sports & Leisure", "Scoreboard": "Sports & Leisure", "Skateboard": "Sports & Leisure", "Toy": "Sports & Leisure", "Guitar": "Sports & Leisure", "Piano": "Sports & Leisure", "Mobile phone": "Electronics", "Laptop": "Electronics", "Computer monitor": "Electronics", "Keyboard": "Electronics", "Mouse": "Electronics", "Tablet": "Electronics", "Television": "Electronics", "Calculator": "Electronics", "Tree": "Nature & Animals", "Flower": "Nature & Animals", "Plant": "Nature & Animals", "Mammal": "Nature & Animals", "Bird": "Nature & Animals", "Dog": "Nature & Animals", "Cat": "Nature & Animals", "Fish": "Nature & Animals", "Insect": "Nature & Animals", "Poster": "Art & Display", "Picture frame": "Art & Display", "Sculpture": "Art & Display", "Painting": "Art & Display", "Billboard": "Art & Display", "Crown": "Art & Display", "Box": "Containers & Packaging", "Bottle": "Containers & Packaging", "Jar": "Containers & Packaging", "Cup": "Containers & Packaging", "Bowl": "Containers & Packaging", "Plate": "Containers & Packaging", "Tin can": "Containers & Packaging", "Bag": "Containers & Packaging", "Suitcase": "Containers & Packaging"}

model.eval()

# For each query (predicted image embedding), compute similarity with all true image embeddings
# All 5 captions are present in the dataset
for i, (query, true_index) in enumerate(tqdm(zip(query_embeddings, range(len(image_embeddings))), total=len(query_embeddings), desc="Querying")):
    query = query.reshape(1, -1)
    similarities = cosine_similarity(query, image_embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    # Retrieve the meta data from the top 10 results
    top_10_indices = sorted_indices[:10]
    top_10_filenames = [dataset.data_id[idx] for idx in top_10_indices]
    top_10_metadata = lookup_table[lookup_table['image_id'].isin(top_10_filenames)].drop_duplicates(subset='image_id')[['image_id', 'image_classes']]

    # Retrieve the meta data from the query
    query_filename = dataset.data_id[true_index]
    query_metadata = lookup_table[lookup_table['image_id'] == query_filename].drop_duplicates(subset='image_id')[['image_id', 'image_classes']]

    # Original (fine-grained) level comparison
    query_classes = set()
    for classes in query_metadata['image_classes']:
        query_classes.update(classes)

    retrieved_classes = set()
    for classes in top_10_metadata['image_classes']:
        retrieved_classes.update(classes)

    intersection_fine = query_classes & retrieved_classes
    union_fine = query_classes | retrieved_classes
    precision_fine = len(intersection_fine) / len(retrieved_classes) if retrieved_classes else 0
    recall_fine = len(intersection_fine) / len(query_classes) if query_classes else 0
    jaccard_fine = len(intersection_fine) / len(union_fine) if union_fine else 0

    # Coarse-level abstraction via class_categories mapping
    def map_to_coarse(classes, mapping):
        return set(mapping.get(cls, cls) for cls in classes)

    query_coarse = map_to_coarse(query_classes, class_categories)
    retrieved_coarse = map_to_coarse(retrieved_classes, class_categories)

    intersection_coarse = query_coarse & retrieved_coarse
    union_coarse = query_coarse | retrieved_coarse
    precision_coarse = len(intersection_coarse) / len(retrieved_coarse) if retrieved_coarse else 0
    recall_coarse = len(intersection_coarse) / len(query_coarse) if query_coarse else 0
    jaccard_coarse = len(intersection_coarse) / len(union_coarse) if union_coarse else 0

    # Store results
    if 'metrics_fine' not in globals():
        metrics_fine = []
        metrics_coarse = []

    metrics_fine.append((precision_fine, recall_fine, jaccard_fine))
    metrics_coarse.append((precision_coarse, recall_coarse, jaccard_coarse))

# After the for loop, summarize and print the results
if 'metrics_fine' in globals():
    fine_avg = np.mean(metrics_fine, axis=0)
    coarse_avg = np.mean(metrics_coarse, axis=0)

    print("\n--- Fine-Grained Level ---")
    print(f"Average Precision: {fine_avg[0]:.4f}")
    print(f"Average Recall: {fine_avg[1]:.4f}")
    print(f"Average Jaccard: {fine_avg[2]:.4f}")

    print("\n--- Coarse-Grained Level ---")
    print(f"Average Precision: {coarse_avg[0]:.4f}")
    print(f"Average Recall: {coarse_avg[1]:.4f}")
    print(f"Average Jaccard: {coarse_avg[2]:.4f}")