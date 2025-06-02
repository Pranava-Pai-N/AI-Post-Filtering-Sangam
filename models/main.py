import pickle
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("models/scheme_embeddings.pkl", "rb") as f:
    corpus_embeddings = pickle.load(f)

with open("models/schemes.json", "r", encoding="utf-8") as f:
    corpus_metadata = json.load(f)

def recommend_or_match(
    query: str,
    user_schemes: List[Dict[str, str]],
    threshold: float = 0.70,
    top_n: int = 5
) -> Union[Dict[str, Union[bool, float, str]], Dict[str, Union[bool, List[Dict[str, str]]]]]:
    
    query_emb = model.encode([query])

    if user_schemes:
        input_texts = [s['title'] + " - " + s['description'] for s in user_schemes]
        input_embs = model.encode(input_texts)
        sim_scores = cosine_similarity(query_emb, input_embs)[0]
        max_idx = sim_scores.argmax()

        if sim_scores[max_idx] >= threshold:
            return {
                "matched": True,
                "scheme_id": user_schemes[max_idx]['scheme_id'],
            }

    sims = cosine_similarity(query_emb, corpus_embeddings)[0]
    top_idxs = sims.argsort()[-top_n:][::-1]
    top_recs = [corpus_metadata[i] for i in top_idxs]

    return {
        "matched": False,
        "recommendations": top_recs
    }
