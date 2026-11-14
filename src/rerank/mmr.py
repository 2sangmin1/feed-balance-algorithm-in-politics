import numpy as np

class MMRReranker:
    """Maximal Marginal Relevance: 관련성과 다양성의 균형으로 재랭킹
    lam=1.0 → 원래 추천기와 동일, lam이 낮을수록 다양성 우선
    """

    def __init__(self, base, X, lam: float = 0.7, pool: int = 50):
        self.base = base
        self.X = X
        self.lam = lam
        self.pool = pool

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        cand = self.base.recommend(history, k=self.pool)
        rel = self.base.scores(history)[cand]
        sim = (self.X[cand] @ self.X[cand].T).toarray()

        selected = [int(np.argmax(rel))]
        remaining = [i for i in range(len(cand)) if i != selected[0]]
        while len(selected) < k and remaining:
            mmr = [self.lam * rel[i] - (1 - self.lam) * sim[i, selected].max() for i in remaining]
            best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)
        return [cand[i] for i in selected]
