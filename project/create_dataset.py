import torch
import timm
import requests
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model = SentenceTransformer('all-MiniLM-L6-v2')  # Uses PyTorch

model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
model.eval()
model.to(device)

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

def extract_dinov2_features(image_path, model):
    image_input = load_image(image_path)
    with torch.no_grad():
        features = model(image_input) 

    return features.squeeze().cpu().numpy()




# image_dir = "/research2/lrbutler/TextBasedImageSearch/data/raw_data/train_images/"
# save_dir = "/home/bchen1/csci611/TextBasedImageSearch/image_feature_exploration/image_features_embedding/"

# image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
# for image_file in image_files:
#     image_path = os.path.join(image_dir, image_file)
#     features = extract_dinov2_features(image_path, model)

#     feature_file_name = f"{os.path.splitext(image_file)[0]}_dinov2_embed.npy"
#     feature_save_path = os.path.join(save_dir, feature_file_name)

#     np.save(feature_save_path, features)