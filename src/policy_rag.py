"""Part D (1/2) - Policy search ("RAG"): find the right policy text and answer ONLY from it.

Run from the project root:   python src/policy_rag.py

RAG = Retrieval Augmented Generation. Steps:
  1. CHUNK   split every policy file into small overlapping pieces of words
  2. INDEX   turn each chunk into numbers with TF-IDF (rare words count more)
  3. SEARCH  turn the question into numbers too, find the most similar chunks (cosine similarity)
  4. ANSWER  pick the sentence that best matches the question, or say "not found"
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "policies"
NOT_FOUND = "I could not find that information in the policy documents."

STOP = {"the", "a", "an", "is", "are", "of", "to", "for", "in", "on", "how", "what", "when", "who", "do", "does",
        "i", "my", "be", "can", "after", "from", "it", "or", "and", "at", "by", "which", "why", "if", "than",
        "that", "much", "many", "must", "there", "will", "has", "have", "with", "was", "were", "this", "not"}


def stems(text):
    """Very simple word stemming: lowercase, drop stop words, keep the first 5 letters."""
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


class PolicyRAG:
    def __init__(self, chunk_size=40, overlap=10):
        self.docs = {p.name: p.read_text(encoding="utf-8").strip() for p in sorted(POLICIES.glob("*.txt"))}
        self.chunks = []
        step = chunk_size - overlap
        for name, text in self.docs.items():
            words = text.split()
            for start in range(0, len(words), step):
                self.chunks.append({"source": name, "text": " ".join(words[start:start + chunk_size])})
                if start + chunk_size >= len(words):
                    break
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 3),
                 "text": self.chunks[i]["text"]} for i in best]

    def answer(self, question, k=3):
        hits = self.retrieve(question, k)
        sources = list(dict.fromkeys(h["source"] for h in hits))
        # candidate sentences come from the documents that were retrieved
        sentences = [s.strip() for src in sources for s in re.split(r"(?<=\.)\s+", self.docs[src]) if s.strip()]
        q = stems(question)
        # a word that appears in few sentences is more informative -> weight 1/frequency
        freq = {w: sum(w in stems(s) for s in sentences) for w in q}
        best, best_score, best_shared = None, 0.0, 0
        for s in sentences:
            shared = q & stems(s)
            score = sum(1.0 / freq[w] for w in shared)
            if score > best_score:
                best, best_score, best_shared = s, score, len(shared)
        # grounding rule: need at least 2 matching words, otherwise admit we do not know
        if best is None or best_shared < 2:
            return {"answer": NOT_FOUND, "sources": [], "retrieved": hits}
        return {"answer": best, "sources": sources, "retrieved": hits}


def evaluate(rag):
    """Run the 14 test questions in data/raw/policy_eval.csv and score them."""
    tests = pd.read_csv(ROOT / "data" / "raw" / "policy_eval.csv", keep_default_na=False)
    rows = []
    for _, t in tests.iterrows():
        r = rag.answer(t.question)
        unanswerable = t.expected_doc == "NONE"
        rows.append({"question": t.question,
                     "right_document": (t.expected_doc in r["sources"]) if not unanswerable else None,
                     "fact_found": (t.expected_fact in r["answer"]) if not unanswerable else None,
                     "correctly_refused": (r["answer"] == NOT_FOUND) if unanswerable else None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rag = PolicyRAG()
    print("chunks:", len(rag.chunks))
    print("\nEXAMPLE:", rag.answer("What is the late fee for a failed payment?")["answer"])
    result = evaluate(rag)
    print("\n" + result.to_string(index=False))
    print("\nright document:", int(result.right_document.sum()), "/ 12",
          "| fact found:", int(result.fact_found.sum()), "/ 12",
          "| correctly refused:", int(result.correctly_refused.sum()), "/ 2")
