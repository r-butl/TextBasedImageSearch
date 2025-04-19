import torch
import timm
import requests
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm


# Load a pre-trained sentence encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model = SentenceTransformer('all-MiniLM-L6-v2')

image_model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
image_model.eval()
image_model.to(device)

def extract_image_features(image, model):
    image_input = load_image(image)
    with torch.no_grad():
        features = model(image_input) 

    return features.squeeze().cpu().numpy()

def extract_text_features(text, model):
    return model.encode(text)

def load_image(img_path, max_size=512, shape=(518,518)):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
        
    in_transform = transforms.Compose([
                        transforms.Resize(shape),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image.to(device)

def main():
    target_dir = "../data/raw_data/"
    save_dir_base = "../data/formatted_data/"

    # Load and concatenate train and val meta data
    train_meta = pd.read_json(os.path.join(target_dir, 'TextCaps_0.1_train.json'))
    val_meta = pd.read_json(os.path.join(target_dir, 'TextCaps_0.1_val.json'))
    meta_data = pd.concat([train_meta, val_meta], ignore_index=True)
    meta_data = meta_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 
    total_len = len(meta_data)
    train_end = int(0.75 * total_len)
    val_end = int(0.85 * total_len)

    splits = {
        'train': meta_data.iloc[:train_end],
        'validate': meta_data.iloc[train_end:val_end],
        'test': meta_data.iloc[val_end:]
    }

    for split_name, split_data in splits.items():
        save_dir = os.path.join(save_dir_base, split_name)
        image_embedding_directory = os.path.join(save_dir, 'image_embeddings')
        text_embedding_directory = os.path.join(save_dir, 'text_embeddings')

        os.makedirs(image_embedding_directory, exist_ok=True)
        os.makedirs(text_embedding_directory, exist_ok=True)

        image_dir = 'train_images'
        load_dir = os.path.join(target_dir, image_dir)

        completed_files = pd.DataFrame([f[:-4] for f in os.listdir(image_embedding_directory) if f.endswith('.npy')], columns=['file_id'])

        for _, row in tqdm(split_data.iterrows(), total=len(split_data)):
            data = row['data']
            if data['image_id'] not in completed_files['file_id'].values:
                target_image_path = os.path.join(load_dir, data['image_id'] + '.jpg')

                caption = data['caption_str']

                image_features = extract_image_features(target_image_path, image_model)
                np.save(os.path.join(image_embedding_directory, f"{data['image_id']}.npy"), image_features)

                text_features = extract_text_features(caption, text_model)
                np.save(os.path.join(text_embedding_directory, f"{data['image_id']}.npy"), text_features)
            else:
                print(f"skipping {data['image_id']}")

    
if __name__ == "__main__":
    main()