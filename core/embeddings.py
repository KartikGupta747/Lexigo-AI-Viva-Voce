import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)

    faiss_index.add(embeddings)

    return faiss_index, embeddings