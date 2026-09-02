import math
from sentence_transformers import SentenceTransformer


def embed_texts(texts: list[str], model="intfloat/multilingual-e5-large"):
    sentence_model = SentenceTransformer(model)
    return sentence_model.encode(texts)


def embed_query(query, model="intfloat/multilingual-e5-large"):
    return embed_texts([query], model=model)[0]

def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot_product / (norm_a * norm_b)