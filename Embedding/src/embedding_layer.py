import numpy as np
from gensim.models import Word2Vec

# embedding layer
class EmbeddingLayer:
    def __init__(self, model_path):
        self.model = Word2Vec.load(model_path)
        
    def embed_batch(self, batched_tokens):
        embeddings = []
        for tokens in batched_tokens:
            embeddings.append([self.model.wv.get_vector(token) for token in tokens])
        
        return np.array(embeddings)