import numpy as np

def group_approval(X, topics, bias: float = 0.3) -> np.ndarray:
    """주제 선호 집단별 기사 호감도 근사 (집단 수 x 기사 수)
    집단 중심 벡터와의 유사도 + 자기 선호 주제 보너스 (SimUser와 같은 구조)
    """
    topics = np.asarray(topics)
    groups = np.unique(topics)
    A = np.zeros((len(groups), X.shape[0]))
    for gi, g in enumerate(groups):
        centroid = np.asarray(X[topics == g].mean(axis=0)).ravel()
        centroid /= np.linalg.norm(centroid) or 1.0
        A[gi] = np.asarray(X @ centroid).ravel() + bias * (topics == g)
    return A

def bridging_scores(A: np.ndarray) -> np.ndarray:
    """가장 덜 호응하는 집단의 호감도 = 브리징 점수 (Ovadya & Thorburn, 2023 아이디어 근사)
    한 집단만 좋아하는 기사는 낮고, 모든 집단이 고르게 호응하는 기사는 높음
    """
    s = A.min(axis=0)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)

class BridgingReranker:
    """(1 - lam) * 관련성 + lam * 브리징 점수로 재정렬"""

    def __init__(self, base, bridge: np.ndarray, lam: float = 0.5, pool: int = 50):
        self.base = base
        self.bridge = bridge
        self.lam = lam
        self.pool = pool

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        cand = np.array(self.base.recommend(history, k=self.pool))
        rel = self.base.scores(history)[cand]
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-12)
        score = (1 - self.lam) * rel + self.lam * self.bridge[cand]
        return cand[np.argsort(-score)[:k]].tolist()
