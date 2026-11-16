import numpy as np

class CalibratedReranker:
    """Calibrated Recommendations (Steck, 2018)
    유저 기록의 주제 분포 p와 추천 목록의 주제 분포 q의 KL 발산을 줄이도록 탐욕적 재랭킹
    목적함수: (1 - lam) * 관련성 합 - lam * KL(p || q~)
    """

    def __init__(self, base, topics, lam: float = 0.5, pool: int = 50, alpha: float = 0.01):
        self.base = base
        self.lam = lam
        self.pool = pool
        self.alpha = alpha
        cats, self.item_cat = np.unique(np.asarray(topics), return_inverse=True)
        self.n_cats = len(cats)

    def _dist(self, items) -> np.ndarray:
        counts = np.bincount(self.item_cat[items], minlength=self.n_cats).astype(float)
        return counts / counts.sum()

    def _kl(self, p: np.ndarray, q: np.ndarray) -> float:
        q = (1 - self.alpha) * q + self.alpha * p  # q가 0이 되는 것 방지 (Steck 2018)
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        cand = self.base.recommend(history, k=self.pool)
        rel = self.base.scores(history)[cand]
        p = self._dist(history)

        selected, rel_sum = [], 0.0
        remaining = list(range(len(cand)))
        while len(selected) < k and remaining:
            chosen = [cand[j] for j in selected]
            vals = [
                (1 - self.lam) * (rel_sum + rel[i]) - self.lam * self._kl(p, self._dist(chosen + [cand[i]]))
                for i in remaining
            ]
            best = remaining[int(np.argmax(vals))]
            selected.append(best)
            remaining.remove(best)
            rel_sum += rel[best]
        return [cand[i] for i in selected]
