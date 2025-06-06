from torch.utils.data import Dataset
import os
import numpy as np
import torch

class EmbeddingDataset(Dataset):
    def __init__(self, image_dir, text_dir, cache=True, return_data_id=False):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.npy')
        ])
        self.text_paths = sorted([
            os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith('.npy')
        ])

        self.return_data_id = return_data_id

        self.data_id = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]

        assert len(self.image_paths) == len(self.text_paths), "Mismatched image and text embeddings"

        self.cache = cache

        if self.cache:
            print("Caching dataset into memory...")
            self.image_data = [torch.tensor(np.load(p), dtype=torch.float32) for p in self.image_paths]
            self.text_data = [torch.tensor(np.load(p), dtype=torch.float32) for p in self.text_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cache:

            if self.return_data_id:
                return self.text_data[idx], self.image_data[idx], self.data_id[idx]
            else:
                return self.text_data[idx], self.image_data[idx]
        else:
            image = torch.tensor(np.load(self.image_paths[idx]), dtype=torch.float32)
            text = torch.tensor(np.load(self.text_paths[idx]), dtype=torch.float32)

            if self.return_data_id:
                return text, image, self.data_id[idx]
            else:
                return text, image

    def get_feature_sizes(self):
        if self.cache:
            return self.text_data[0].shape[0],  self.image_data[0].shape[0]
        else:
            sample_image = np.load(self.image_paths[0])
            sample_text = np.load(self.text_paths[0])
            return sample_text.shape[0], sample_image.shape[0]