"""
A deliberately simple RAG pipeline: TF-IDF retrieval over a tiny corpus,
"generation" that stitches the top chunk into a sentence. This exists only
so `anchor demo` produces real numbers with zero setup — swap this out for
your own retriever + LLM call and everything downstream (metrics, judge,
tracing) works unchanged, because it only ever talks to the Chunk/answer
contract, never to this file.
"""

from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from anchor.schema import Chunk

CORPUS = [
    ("c1", "The Constitution of Nepal establishes Nepal as a federal democratic republic."),
    ("c2", "Nepal's federal structure divides the country into seven provinces."),
    ("c3", "The Supreme Court of Nepal is the highest judicial authority in the country."),
    ("c4", "Citizens of Nepal have the right to information under Article 27."),
    ("c5", "The Prime Minister of Nepal is the head of government, appointed per Article 76."),
    ("c6", "Local level governments in Nepal handle municipal services and local infrastructure."),
    ("c7", "The National Assembly is the upper house of Nepal's federal parliament."),
    ("c8", "Fundamental rights in Nepal's constitution include equality, freedom, and justice."),
]


def make_pipeline(top_k: int = 3):
    ids, texts = zip(*CORPUS)
    vectorizer = TfidfVectorizer().fit(texts)
    matrix = vectorizer.transform(texts)

    def pipeline(query: str) -> Tuple[List[Chunk], str]:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix)[0]
        ranked = sorted(zip(ids, texts, sims), key=lambda x: -x[2])[:top_k]

        chunks = [Chunk(id=i, text=t, score=float(s)) for i, t, s in ranked]

        # Naive "generation": paraphrase-free stitch of the top chunk, plus
        # one deliberately unsupported sentence to prove the judge catches it.
        answer = chunks[0].text if chunks else "I don't know."
        if "province" in query.lower():
            answer += " There are exactly seven provinces, each with its own elected assembly and governor."
            # ^ "governor" claim is not actually in the corpus — intentional
            #   hallucination so the demo output isn't a trivial 1.0 score.

        return chunks, answer

    return pipeline
