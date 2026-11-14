# 관련 연구 정리

## 피드백 루프 시뮬레이션
- **Chaney, Stewart, Engelhardt (RecSys 2018)** — How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility
  - 추천 → 행동 → 재학습 루프를 시뮬레이션, 유저 행동 동질화는 증가하나 만족도 향상은 없음
  - 반영: 집단 수준 동질화 지표(유저 간 피드 유사도) 추가

## 재랭킹 기법
- **Steck (RecSys 2018)** — Calibrated Recommendations
  - 유저 관심사 비율대로 추천 비율을 맞추는 사후 재랭킹, λ로 정확도와 트레이드오프 조절
  - 반영: 3주차 캘리브레이션 구현 (장르 → 주제)
- **Kaya & Bridge (RecSys 2019)** — A Comparison of Calibrated and Intent-Aware Recommendations
  - 캘리브레이션 vs 의도 인식(xQuAD 등) 비교, 의도 인식 방식이 트레이드오프 우수
  - 반영: 시간 남으면 추가 비교 기법 후보

## 평가 지표
- **Kaminskas & Bridge (TiiS 2016)** — Diversity, Serendipity, Novelty, and Coverage
  - beyond-accuracy 지표의 표준 정의 정리
  - 반영: ILD는 표준 지표로 유지, 커버리지 지표 추가

## 브리징
- **Ovadya & Thorburn (2023)** — Bridging Systems
  - 서로 다른 집단 모두에게 호응받는 콘텐츠를 우대하는 랭킹
  - 반영: 선호 주제별 유저 집단 간 고른 클릭을 받은 기사에 가산점
  - 한계: 브리징을 단일 지표로 정량화하는 것에 대한 비판 존재 → 논문 한계 섹션

## 국내 연구
- 네이버 뉴스 동질화 사례 연구 (질적 분석)
- 유튜브 추천 알고리즘 필터버블 분석 (설문, 텍스트 마이닝)
- 카카오 다음뉴스 다양성 지수 (기업 자체 측정)
- 차별점: 한국어 뉴스 기반 재랭킹 기법의 시뮬레이션 비교는 확인된 범위에서 부재 → RISS/DBpia 추가 확인 필요
