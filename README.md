# 🖼️ Text-Based Image Search

This project implements a machine learning system that allows users to retrieve relevant images based on natural language text queries. The system aligns text and image feature embeddings using a trained neural network and retrieves the closest-matching images based on cosine similarity.

## 🔍 Overview

Given a text query (e.g., "a cat sitting on a couch"), the system:

1. Encodes the query using a pretrained text embedding model (MiniLM-L6-v2).
2. Uses a trained feedforward neural network to map the text embedding into image embedding space.
3. Compares the predicted image embedding against a database of image embeddings (extracted using Dinov2).
4. Returns the most semantically relevant images using cosine similarity.

## 🧠 Model Architecture

- **Text Embedding**: MiniLM-L6-v2
- **Image Embedding**: Dinov2
- **Neural Network**: 5-layer feedforward network
- **Loss Function**: Cosine Similarity Loss
- **Optimization**: Adam with ReduceLRonPlateau
- **Training Dataset**: [TextCaps](https://textvqa.org/textcaps/) (28k images with 140k captions)

## 📊 Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**  
- **Precision / Recall / Jaccard Similarity** (evaluated on both fine-grained and coarse-grained class sets)

> 📈 Achieved recall of 0.84 (coarse-grained), showing strong ability to retrieve semantically relevant images.

---

## 🚀 Getting Started

### 🔎 Hyperparameter Search
Run to find the best layer sizes and training config:
```bash
python hyperparameter_search.py
```
### 🏋️‍♂️ Train the Model
Train using the best configuration:
```bash
python train.py
```
### 🧪 Test the Model
Evaluate using cosine similarity & MRR:
```bash
python test.py
```

## 📎 References & Docs

- 📝 [Final Report](https://docs.google.com/document/d/1B-J3qEIiFqq9X2RtMSx1UHk0HzMy2SpJRlLScrQFPYU/edit?tab=t.0#heading=h.gh4ewt1fh2kz)
- 📄 [Project Proposal](https://docs.google.com/document/d/1Mo-P9GfpyhX9XlX8lM1zaRSUbBbvL0LRXpTFck77eYk/edit?tab=t.0)
- 📚 [TextCaps Dataset](https://textvqa.org/textcaps/)
- 🧠 [Dinov2](https://github.com/facebookresearch/dinov2)
- 🔤 [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 👨‍💻 Contributors

- Lucas Butler  
- **Boxi Chen**  
- Anthony Pecoraro  
- Hayat White

