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

# Load a pre-trained sentence encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model = SentenceTransformer('all-MiniLM-L6-v2')

image_model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
image_model.eval()
image_model.to(device)

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

def extract_image_features(image, model):
    image_input = load_image(image)
    with torch.no_grad():
        features = model(image_input) 

    return features.squeeze().cpu().numpy()

def extract_text_features(text, model):
    return model.encode(text)

def main():

    target_dir = "../data/raw_data/"
    save_dir = "../data/formatted_data/"

    image_embedding_directory = os.path.join(save_dir, 'image_embeddings')
    if not os.path.exists(image_embedding_directory):
        os.makedirs(image_embedding_directory)
    
    text_embedding_directory = os.path.join(save_dir, 'text_embeddings')
    if not os.path.exists(text_embedding_directory):
        os.makedirs(text_embedding_directory)

    mode = 'train' # ['train', 'val', 'test]

    if mode == 'train':
        file = 'TextCaps_0.1_train.json'
        directory = 'train_images'
    elif mode == 'val':
        file = 'TextCaps_0.1_val.json'
        directory = 'train_images'
    elif mode == 'test':
        file = 'TextCaps_0.1_test.json'
        directory = 'test_images'

    meta_data = pd.read_json(os.path.join(target_dir, file))
    load_dir = os.path.join(target_dir, directory)
    
    print(meta_data['data'][0]['image_id'])
    for i, row in meta_data.iterrows():
        data = row['data']
        target_image_path = os.path.join(load_dir, data['image_id'] + '.jpg')
        caption = data['caption_str']

        # Create the image feature
        image_features = extract_image_features(target_image_path, image_model)
        np.save(os.path.join(image_embedding_directory, f"{data['image_id']}.npy"), image_features)

        # create the text feature
        text_features = extract_text_features(caption, text_model)
        np.save(os.path.join(text_embedding_directory, f"{data['image_id']}.npy"), text_features)
  
if __name__ == "__main__":
    main()