import PyPDF2
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

text = "This is a test sentence"

def get_embeddings(text):
    embeddings = model.encode(text)
    return embeddings

embeddings = get_embeddings(text)
print(embeddings)



