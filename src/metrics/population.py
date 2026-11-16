import numpy as np
import pandas as pd

def population_metrics(result: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """조건·라운드별 집단 수준 지표
    homogenization: 유저 쌍 간 피드 자카드 유사도 평균 (Chaney et al., 2018)
    coverage: 해당 라운드까지 한 번이라도 추천된 기사 비율 (Kaminskas & Bridge, 2016)
    """
    rows = []
    for cond, g in result.groupby("condition"):
        seen = set()
        for r, gr in g.groupby("round"):
            feeds = gr["feed"].tolist()
            M = np.zeros((len(feeds), n_items), dtype=np.float32)
            for i, f in enumerate(feeds):
                M[i, f] = 1
                seen.update(f)
            inter = M @ M.T
            sizes = M.sum(axis=1)
            union = sizes[:, None] + sizes[None, :] - inter
            iu = np.triu_indices(len(feeds), k=1)
            rows.append({
                "condition": cond,
                "round": r,
                "homogenization": float((inter[iu] / union[iu]).mean()),
                "coverage": len(seen) / n_items,
            })
    return pd.DataFrame(rows)
