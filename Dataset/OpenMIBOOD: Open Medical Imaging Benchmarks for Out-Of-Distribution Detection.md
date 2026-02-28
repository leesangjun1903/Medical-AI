# OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection 

## 개요
**OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection** 논문은 의료 영상 분야에서 인공지능(AI) 모델이 훈련 데이터와 다른(분포 밖, OOD) 데이터를 얼마나 잘 탐지하는지 평가할 수 있는 벤치마크를 제시합니다[1][2][3]. 의료 AI의 신뢰성과 안전성을 높이기 위해 OOD 탐지는 매우 중요한 과제입니다.

---

## 핵심 내용

### 1. 벤치마크 구조 및 데이터셋
- **총 14개 데이터셋**: 미세병리학, 내시경, MRI 등 다양한 의료 영상 분야에서 수집[1][2][3].
- **3가지 주요 벤치마크**:
  - **ID (In-Distribution)**: 훈련 데이터와 같은 분포의 데이터.
  - **cs-ID (Covariate-Shifted In-Distribution)**: 라벨은 같지만 입력 특성 분포가 변한 데이터.
  - **Near-OOD**: 훈련 데이터와 유사하지만 분포 밖에 있는 데이터.
  - **Far-OOD**: 의미적으로 완전히 다른 분포의 데이터[1][2][4].

### 2. 실험 방법 및 평가
- **모델 훈련**: 여러 신경망(ANN) 아키텍처를 사용, SGD/Adam 최적화기와 교차 엔트로피 손실 함수 활용[1].
- **OOD 탐지 방법**: 24개의 사후(post-hoc) 방식 OOD 탐지 방법을 비교 분석. 주로 모델의 출력(logit)이나 피처(feature) 정보를 활용[1][2][4].
- **대표적 방법**: Mahalanobis 거리 기반 MDSEns 등 다양한 방법이 포함됨[1].

### 3. 성능 평가 지표
- **AUROC**(ROC 곡선 아래 면적), **AUPR**(정밀도-재현율 곡선 아래 면적) 등 다양한 정량적 지표로 OOD 탐지 성능을 평가[1][4].

### 4. 주요 결과 및 시사점
- 자연 이미지용 OOD 탐지 방법이 의료 영상에선 성능이 다르게 나타남. 의료 영상 특성에 맞는 OOD 탐지법 개발 필요[1][2][4].
- 피처 기반 OOD 탐지법이 로그잇/확률 기반 방법보다 의료 영상에서 더 우수한 경향[4].
- 공개된 벤치마크와 코드로 누구나 재현 및 비교 실험 가능[1][3].

---

## 결론
OpenMIBOOD는 의료 영상 AI의 신뢰성 향상과 안전성 확보를 위한 **표준 벤치마크**를 제공하며, 의료 AI 분야에서 OOD 탐지 연구의 기준점 역할을 합니다[1][2][3].

---

### 참고
- 논문 결과물 및 코드: [OpenMIBOOD GitHub](https://github.com/remic-othr/OpenMIBOOD)[3]

[1] https://www.themoonlight.io/ko/review/openmibood-open-medical-imaging-benchmarks-for-out-of-distribution-detection
[2] https://www.themoonlight.io/en/review/openmibood-open-medical-imaging-benchmarks-for-out-of-distribution-detection
[3] https://arxiv.org/abs/2503.16247
[4] https://voxel51.com/blog/the-best-of-cvpr-2025-series-day-3
[5] https://www.themoonlight.io/ko/review/enhancing-out-of-distribution-detection-in-medical-imaging-with-normalizing-flows
[6] https://www.threads.net/@byeongki_j/post/DC1hOdUTGw_/ai%EB%A1%9C-%EB%85%BC%EB%AC%B8%EC%A0%95%EB%A6%AC-30%EC%B4%88-%EC%BB%B7-%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95%EC%82%AC%EC%8B%A4-%EB%AA%A8%EB%93%A0-%EB%AC%B8%EC%84%9C-%EB%8B%A4-%EA%B0%80%EB%8A%A5%EB%B0%95%EC%82%AC%EA%B3%BC%EC%A0%95%EC%97%90-%EA%B0%80%EC%9E%A5-%EB%B6%80%EC%A1%B1%ED%95%9C-%EA%B2%83%EC%9D%80-%ED%95%99%EC%A0%90%EB%8F%84-gpu%EB%8F%84-%EC%95%84%EB%8B%88%EA%B3%A0-%EC%8B%9C%EA%B0%84%EC%9D%B4%EC%97%88%EC%96%B4%ED%8A%B9%ED%9E%88-%EC%88%98%EC%8B%AD%EA%B0%9C%EC%94%A9-%EB%85%BC%EB%AC%B8%EC%9D%B4-%EC%8F%9F
[7] https://arxiv.org/html/2503.16247v1
[8] https://openreview.net/forum?id=oUg5rC_95OM
[9] https://www.semanticscholar.org/paper/8923d232ef0a4a922a81cc5d5b4aeed8eb538491
[10] https://www.semanticscholar.org/paper/899fc919576b86220a49c0477dbe9dc7872e0c17
[11] https://www.semanticscholar.org/paper/5124d924e42553e6bf72ac32a401e3d8325d11c5
[12] http://link.springer.com/journal/11814
[13] https://www.semanticscholar.org/paper/b14f1374f1eaccca801dee46599fab8dd1ac2cac
[14] https://www.semanticscholar.org/paper/348421f787048c5c37a05501c20d143c7154a29d
[15] https://www.semanticscholar.org/paper/7f5b9627e8fee7983103716d006aa2bdd46b15b6
[16] https://www.semanticscholar.org/paper/44ca0c08080a494ffafbe07b3b1b7bc62591955b
[17] https://www.semanticscholar.org/paper/70a46402acd0fefc737a9ae1162c57e803453bd9
[18] https://www.semanticscholar.org/paper/9b50bb73a84fbd5baf40908054898db51f5e38de
[19] https://github.com/remic-othr/OpenMIBOOD
[20] https://pubmed.ncbi.nlm.nih.gov/35468060/
