import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_n_retrievals(sketch, image_embeddings, images, model, top_n=5):
    sketch_embedding = model.sketch_tower(sketch.to(device)).cpu().detach().numpy()
    similarities = cosine_similarity(sketch_embedding, image_embeddings)
    top_indices = np.argsort(similarities, axis=1)[0][-top_n:][::-1]
    return [images[i] for i in top_indices]
