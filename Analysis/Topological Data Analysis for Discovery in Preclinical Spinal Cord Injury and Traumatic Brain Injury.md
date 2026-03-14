
# Topological Data Analysis for Discovery in Preclinical Spinal Cord Injury and Traumatic Brain Injury

**저자:** Jessica L. Nielson, Jesse Paquette, Aiwen W. Liu, … Gunnar E. Carlsson, Geoffrey T. Manley, Michael S. Beattie, Jacqueline C. Bresnahan & Adam R. Ferguson  
**학술지:** *Nature Communications*, 6:8581 (2015)  
**DOI:** 10.1038/ncomms9581

---

## 1. 핵심 주장과 주요 기여 (간결 요약)

복잡한 신경학적 장애에서의 데이터 기반 발견(data-driven discovery)은 대규모·이질적 데이터 세트로부터 의미 있는 증후군적(syndromic) 지식을 추출하여 정밀 의학(precision medicine)의 잠재력을 향상시킬 수 있다.

본 논문은 VISION-SCI 리포지토리에서 추출한 전임상 외상성 뇌손상(TBI) 및 척수손상(SCI) 데이터 세트에 위상적 데이터 분석(TDA)을 적용하였다.

**핵심 발견 3가지:**
1. 조직병리학적, 기능적, 건강 결과의 상호 관계를 직접 시각화함으로써 TDA는 증후군적 네트워크 전반에서 새로운 패턴을 감지하였으며, SCI와 동시 발생하는 TBI 간의 상호작용과 미공개 다기관 전임상 약물 시험에서의 약물 부작용을 밝혀냈다.
2. TDA는 또한 흉추 SCI 후 어떤 약물보다 수술 중 고혈압이 장기 회복을 더 잘 예측한다는 것을 발견하였다.
3. TDA 기반 데이터 기반 발견은 기초연구 및 임상 문제(결과 평가, 신경중환자 치료, 치료 계획, 신속 정밀 진단 등)에 대한 의사결정 지원에 큰 잠재적 응용 가능성을 가진다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

빅데이터 시대에 생물의학 연구자들은 대량의 데이터에 직면해 있으며, CNS 손상 분야에서는 조직학적, 생리학적, 생행동학적 결과부터 치료 시험의 건강기록까지 방대한 정보를 수집하고 있으나, 계산 보조 없이 치료적 발견을 관리·해석하기 어렵다.

기초연구에서 임상 치료로의 전환은 빅데이터 통합 문제로 개념화될 수 있으며, 수천 편의 논문이 SCI·TBI를 특성화하려 했으나 이 복잡한 장애를 아직 완전히 이해하지 못하고, 소수의 치료법만이 임상시험을 통과하여 환자 치료 기준이 되었다.

일반적인 CNS 손상 연구는 방대한 양의 데이터를 생성하지만 소수의 변수만 분석하며, 바그래프와 회복 곡선 등에 의한 시각화 시도는 다중비교 문제에 의한 위양성(type-1 error)과 다변량 정보의 낭비(type-2 error)에 취약하다.

### 2.2 제안하는 방법 (Mapper 기반 TDA + SVD 렌즈)

#### (a) 전체 파이프라인

데이터 세트의 기능적·조직학적 결과는 모든 결과의 이변량 상관행렬(bivariate correlation matrix)로부터 TDA로 분석되었고, 주성분 및 2차 메트릭 특이값 분해(SVD) 렌즈로 처리하여 증후군 공간(syndrome space)을 생성하였다. 이후 TDA가 증후군 공간을 여러 번 재표본하여 피험자를 노드로 연결하고 겹치는 피험자를 에지로 연결하여 강건한 네트워크 토폴로지를 생성하였다.

#### (b) 수학적 핵심: Mapper Algorithm + SVD Lens

**단계 1: 거리 행렬(Distance Matrix)**

데이터 행렬 $X \in \mathbb{R}^{n \times p}$ ($n$: 피험자 수, $p$: 결과 변수 수)에서 피험자 간 비유사도(dissimilarity)를 측정한다. 논문은 상관 기반 거리(correlation-based distance)를 사용:

$$d(x_i, x_j) = 1 - \rho(x_i, x_j)$$

여기서 $\rho(x_i, x_j)$는 피험자 $i$와 $j$ 간 Pearson 상관계수이다.

**단계 2: SVD Lens (필터 함수)**

데이터는 주성분(principal) 및 2차 메트릭(secondary metric) 특이값 분해(SVD) 렌즈로 처리되어 증후군 공간을 생성한다.

데이터 행렬의 SVD 분해:

$$X = U \Sigma V^T$$

여기서:
- $U \in \mathbb{R}^{n \times n}$: 좌측 특이벡터 (피험자 공간)
- $\Sigma \in \mathbb{R}^{n \times p}$: 특이값 대각행렬
- $V \in \mathbb{R}^{p \times p}$: 우측 특이벡터 (변수 공간)

필터 함수(lens function)는 $f: X \to \mathbb{R}^k$ (보통 $k=2$)로 정의되며, 1차 및 2차 SVD 성분을 사용:

$$f(x_i) = \left( u_{i1}\sigma_1, \; u_{i2}\sigma_2 \right)$$

**단계 3: Mapper 구성**

위상 공간 $X$와 $Z$에 대해, 연속 함수 $f: X \to Z$가 주어졌을 때, $Z$의 유한 열린 커버 $\mathcal{U} = \{U_\alpha\}\_{\alpha \in A}$에 대해, $X$ 위의 풀백 커버(pullback cover)를 $f^*({\mathcal{U}}) = \{f^{-1}(U_\alpha)\}_{\alpha \in A}$로 정의하고, Mapper 구성은 이 풀백 커버의 너브 복합체(nerve complex)로 정의된다.

$$\mathcal{M}(\mathcal{U}, f) := \mathcal{N}(f^\*(\mathcal{U}))$$

구체적으로:
1. 렌즈 $f$의 값 범위를 겹침이 있는 구간(bins)으로 분할: $\{U_1, U_2, \ldots, U_m\}$, 겹침 비율 $p \in (0,1)$
2. 각 구간 $U_i$의 역상(preimage) $f^{-1}(U_i)$ 내에서 클러스터링 수행 (예: single-linkage clustering)
3. 생성된 클러스터를 노드로, 공통 피험자를 가진 노드 쌍을 에지로 연결

**단계 4: 열지도(Heat Map) 시각화**

각 노드에 특정 결과 변수의 평균값을 색상으로 매핑하여 네트워크 토폴로지 위에서 다변량 결과 패턴을 시각적으로 탐색한다.

### 2.3 모델 구조 및 데이터

본 논문에서는 4가지 주요 데이터 세트에 TDA를 적용:

| 데이터 세트 | 설명 | 피험자 수 |
|---|---|---|
| **TBI-SCI 복합 모델** | 일측성 SCI + 동측/대측 TBI 쥐 모델 | ~49마리 |
| **경추 SCI 약물 시험** | 반절제, 낙하-타격, 충격-구동 반-타박 | 다수 |
| **흉추 SCI 약물 시험** | MP/Minocycline 치료 시험 | 다수 |
| **OSU MASCIS 시험** | 다기관 전임상 데이터 (1994-1996) | N=334 |

TBI-SCI 복합 모델에서 각 손상 그룹은 네트워크 토폴로지의 구별되는 영역을 차지하며, 특히 SCI+TBI 동측(ipsi) 그룹은 기능적 회복이 sham 대조군과 유사하여 sham 근처에 위치하지만, 병리학적으로는 SCI 단독 또는 SCI+TBI 대측과 차이가 없었다.

### 2.4 성능 향상 및 주요 결과

**기존 방법 대비 장점:**

TDA는 기하학적 토폴로지의 수학적 개념을 데이터에 적용하여, 회귀분석이나 GLM 등 전통적 모수적 접근법이 노이즈로 간주할 수 있는 관계를 발견할 수 있다.

단변량 분석에서는 병변 위치의 효과가 미미하고 통계적 유의성이 다양했지만, TDA는 전체 종단점의 앙상블을 사용하여 전체 증후군 공간을 렌더링했을 때 극적인 다차원적 효과를 발견하였다.

**약물 효과에 대한 발견:**

전두부 개방 운동 및 그루밍 과제에서 기능적 결손에 대한 데이터 기반 발견은, 12.5mm 낙하-타격 타박을 받은 피험자 중 Minocycline 또는 Methylprednisolone을 투여받은 피험자에서 무투약 대조군에 비해 운동신경 보존과 병변 중심부의 조직 면적이 감소했음을 밝혔다.

TDA를 전임상 치료 시험에 적용한 결과, methylprednisolone(MP)과 minocycline 치료의 경추·흉추 SCI 간 재현 불가능한 효능이 드러났으며, 수술 중 고혈압이 흉추 SCI 이후 더 나쁜 신경학적 회복을 예측한다는 새로운 발견을 하였다.

**교차 검증(Cross-validation):**

교차 검증 테스트에서 methylprednisolone(MP1 및 MCP)의 유해 효과에 대해, BBB 기능 회복(p=0.24)이나 조직 보존(p=0.20)에 약물의 유의미한 영향은 없었다.

### 2.5 한계

논문에서 직접 언급되거나 방법론적으로 추론 가능한 한계는 다음과 같다:

1. **파라미터 의존성:** Mapper 알고리즘은 필터 함수의 선택, 겹침 비율(overlap parameter $p$), 구간(bin) 수, 클러스터링 알고리즘 등 다수의 하이퍼파라미터에 결과가 민감할 수 있다.

2. **모델-프리 특성의 양면성:** TDA의 주요 한계는 그 장점에서도 비롯되는데, 모든 메트릭 관련 정보를 무시하므로 다른 범주의 데이터를 구별하는 능력이 제한될 수 있다.

3. **전임상 데이터의 제한:** 쥐 모델에서 얻은 데이터를 인간 임상에 바로 적용하기 어려우며, 데이터 세트 크기가 상대적으로 작다.

4. **통계적 유의성 검증:** TDA가 발견한 패턴의 통계적 유의성을 정량적으로 검증하기 위한 체계적인 프레임워크가 부족하다. 교차 검증에서 일부 결과가 유의하지 않았다.

5. **재현성:** Mapper 출력의 안정성(stability)과 재현성에 대한 이론적 보장이 제한적이다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 TDA 접근법은 **모델-프리(model-free)** 특성을 가지며, 이것이 일반화 성능과 직결된다:

### 3.1 일반화에 유리한 특성

- **위상적 불변성(Topological Invariance):** 위상적 특징은 본질적으로 이동, 진폭 및 주파수 스케일링 등의 변환에 강건하고 불변하다. 이러한 특성은 서로 다른 실험 조건이나 다른 종(species)의 데이터에서도 구조적 패턴이 보존될 가능성을 높인다.

- **비선형 차원 축소:** SVD 렌즈와 결합된 Mapper는 비선형적 관계까지 포착하는 차원 축소 방법으로서, 전통적 PCA나 선형 모델보다 복잡한 데이터 구조에서 일반화할 가능성이 높다.

- **다변량 통합:** 단변량 효과가 미미했던 결과를 전체 종단점 앙상블을 통해 극적 다차원적 효과로 발견한 사례는, 개별 변수보다 전체적 데이터 형상(shape)에서 일반화 가능한 패턴을 추출할 수 있음을 시사한다.

### 3.2 일반화 향상을 위한 과제

- **파라미터 민감도 해결:** 겹침 비율 $p$, 구간 수 $m$, 해상도(resolution) 등 Mapper의 하이퍼파라미터에 대한 **자동 선택** 또는 **안정성 분석** 방법론이 필요하다. Persistent homology의 관점에서, 여러 스케일에서의 위상적 특징의 지속성(persistence)을 다음과 같이 추적할 수 있다:

$$\text{Persistence}(H_k) = \{(b_i, d_i)\}_{i=1}^{N_k}$$

여기서 $b_i$는 $k$-차원 위상적 특징의 생성 시점, $d_i$는 소멸 시점이다. 지속성이 긴 특징($d_i - b_i$가 큰 특징)이 진정한 구조적 패턴을 반영한다.

- **통계적 검증 강화:** Mapper 네트워크의 통계적 유의성을 검정하기 위한 순열 검정(permutation test)이나 부트스트랩(bootstrap) 기반 방법을 결합할 필요가 있다.

- **데이터 규모 확장:** 전임상에서 임상으로의 일반화를 위해 더 큰 표본 크기와 다양한 손상 모델로의 확장이 필수적이다.

- TDA의 장단점을 고려할 때, 다른 방법론과 결합하여 TDA의 잠재력을 최대한 발휘하는 것이 권장된다.

### 3.3 SVD 렌즈의 일반화 관련 수식

SVD 렌즈에서 주성분이 설명하는 분산(variance explained)은:

$$\text{VE}_k = \frac{\sigma_k^2}{\sum_{i=1}^{\min(n,p)} \sigma_i^2}$$

이때, 상위 $k$개의 성분으로 데이터의 주요 분산을 포착함으로써 **노이즈 차원**을 제거하고 일반화 가능한 저차원 표현을 얻는다. 그러나 최적의 $k$ 선택은 데이터에 의존하며, scree plot이나 정보이론적 기준(AIC, BIC 등)을 활용할 수 있다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

1. **정밀 의학 패러다임 확장:** 정밀 의학은 분석 도구와 데이터 시각화를 적용하여 SCI·TBI 같은 복잡한 장애의 이해와 치료를 향상시키고자 하며, 본 연구는 TDA를 통해 근본적인 증후군적 손상 패턴의 발견과 전임상 약물 시험의 정밀 치료 타겟팅 평가를 위한 접근법을 적용하였다. 이는 후속 연구에서 **환자 계층화(patient stratification)**와 **바이오마커 발견**에 TDA를 적용하는 선례가 되었다.

2. **임상 확장:** 이후 연구인 TRACK-TBI Pilot 다기관 연구(586명의 급성 TBI 환자)에서 TDA를 적용하여 환자의 자연적 하위 그룹을 식별하는 데 활용하였다. 이는 전임상에서 임상으로의 실질적 확장을 보여준다.

3. **혈압 관리와 신경학적 회복:** 급성 SCI 후 급성 저혈압은 신경성 쇼크로 인해 흔하며, 1990년대의 소규모 연구에서 MAP 증강이 회복과 관련된다고 제안되었고 이것이 현재의 임상 지침의 기초가 되었으나, MAP 목표치에 대해 논란이 있다. Nielson 등의 TDA 발견(수술 중 고혈압이 회복 예측)은 이 분야의 후속 연구를 촉진하였다.

### 4.2 향후 연구 시 고려할 점

1. **다중 스케일 분석:** 단일 스케일에서의 TDA가 아닌, 여러 해상도(resolution)에서의 안정적 위상적 특징을 Persistent Homology로 포착하는 **다중 스케일 접근법**이 필요하다.

2. **딥러닝과의 결합:** TDA에서 추출한 위상적 특징을 딥러닝 모델의 입력 특징으로 사용하는 하이브리드 접근이 최근 활발히 연구되고 있다.

3. **인과 추론(Causal Inference):** TDA는 상관적 패턴을 발견하지만, 인과 관계를 직접 추론하지 않는다. 인과 추론 방법론과의 결합이 필요하다.

4. **전임상-임상 전환(Translational Gap):** 쥐 모델에서 발견된 패턴이 인간에 적용 가능한지 체계적 검증이 필요하다.

5. **재현성과 공유:** VISION-SCI와 같은 개방형 리포지토리의 확장과 분석 파이프라인의 표준화가 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 내용 | Nielson et al. (2015)와의 비교 |
|---|---|---|---|
| **Topological network analysis of patient similarity for precision management of acute blood pressure in SCI** (eLife) | 2021 | 데이터 축소 기법(dissimilarity matrices)과 피험자 중심 위상적 네트워크 분석을 결합하여 SCI 후 회복의 예측 인자를 식별 | Nielson et al.의 혈압-회복 발견을 **임상 데이터로 확장**하여 검증 |
| **Promises and pitfalls of TDA for brain connectivity analysis** (NeuroImage) | 2021 | 간질과 조현병에서 persistent homology의 변별력을 테스트했으나, 조현병 분류에서 TDA는 랜덤 수준에 가까운 성능을 보였다 | TDA의 **한계**를 실증적으로 보여주어 Nielson et al.의 접근에도 주의 필요성 시사 |
| **TDA as a New Tool for EEG Processing** (Frontiers in Neuroscience) | 2021 | TDA는 전통적 방법과 다른 각도에서 데이터를 분석하며, 쌍방향 관계를 넘어 풍부한 상호작용을 모델링하고, EEG 시계열의 다양한 동적 특성을 구분할 수 있다 | 신경과학에서 TDA의 **EEG 적용**으로 범위 확장; Mapper 외에도 persistent homology 중심 |
| **Topological data analysis in biomedicine: A review** (J. Biomed. Inform.) | 2022 | 대수적 토폴로지에 기반한 TDA 방법론이 데이터의 "형태(shape)"와 관련된 특징을 기술하고 활용하여 전통적 분석이 놓칠 수 있는 통찰을 발견할 수 있으며, 정밀 의학, 구조생물학, 세포 표현형 분석에 적용된다 | Nielson et al.의 **방법론적 유산**을 포괄적으로 정리한 리뷰 |
| **Mapper algorithm comprehensive review (2007-2025)** (Int. J. Data Sci. Anal.) | 2025 | Mapper 알고리즘은 고차원 데이터의 단순화된 그래프 표현을 구성하여 기저 구조적 패턴을 발견하는 TDA 기법이다 | Mapper의 **이론적 발전과 다양한 분야 적용** 현황을 종합적으로 정리 |
| **Data-driven prediction of SCI recovery** (medRxiv preprint) | 2024 | SCI 회복 예측에서 머신러닝(ML)은 회복 궤적 예측을 향상시키는 유망한 접근이지만, 임상 실무 통합에는 효능과 적용 가능성에 대한 체계적 이해가 필요하다 | Nielson et al.의 TDA 이후 다양한 ML 기법(랜덤 포레스트, SVM, 딥러닝 등)이 SCI 예측에 적용되어 **방법론적 다양화** 진행 |

### 최신 연구 흐름의 핵심 차이점

1. **Persistent Homology의 부상:** Nielson et al.은 주로 Mapper 알고리즘을 사용했으나, 2020년 이후 연구에서는 **Persistent Homology**가 더 이론적으로 견고한 방법으로 주목받고 있다.

2. **TDA + 머신러닝 하이브리드:** 모델 성능 평가, 대수적 토폴로지를 사용한 신경망 가중치 진화 분석, 딥러닝 모델 해석에서의 TDA 효과 입증 등 TDA와 딥러닝의 결합 연구가 증가하고 있다.

3. **임상 데이터 규모 확대:** 전임상 소규모 데이터에서 수백~수천 명 규모의 임상 데이터(예: TRACK-TBI 586명)로 확장되고 있다.

4. **자동화 도구 발전:** giotto-tda, KeplerMapper 등 TDA 파이프라인의 소프트웨어 생태계가 성숙하여 접근성이 향상되었다.

---

## 참고 출처

1. Nielson, J.L. et al. (2015). "Topological data analysis for discovery in preclinical spinal cord injury and traumatic brain injury." *Nature Communications*, 6:8581. [https://www.nature.com/articles/ncomms9581](https://www.nature.com/articles/ncomms9581)
2. PMC 전문: [https://pmc.ncbi.nlm.nih.gov/articles/PMC4634208/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4634208/)
3. PubMed: [https://pubmed.ncbi.nlm.nih.gov/26466022/](https://pubmed.ncbi.nlm.nih.gov/26466022/)
4. ResearchGate: [https://www.researchgate.net/publication/283280759](https://www.researchgate.net/publication/283280759)
5. Kentucky Knowledge Repository: [https://uknowledge.uky.edu/physiology_facpub/79/](https://uknowledge.uky.edu/physiology_facpub/79/)
6. Nielson, J.L. et al. (2017). "Uncovering precision phenotype-biomarker associations in traumatic brain injury using topological data analysis." *PLOS One*. [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169490](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169490)
7. Torres et al. (2021). "Topological network analysis of patient similarity for precision management of acute blood pressure in spinal cord injury." *eLife*. [https://elifesciences.org/articles/68015](https://elifesciences.org/articles/68015)
8. Gracia-Tabuenca et al. (2021). "Promises and pitfalls of topological data analysis for brain connectivity analysis." *NeuroImage*. [https://www.sciencedirect.com/science/article/pii/S105381192100522X](https://www.sciencedirect.com/science/article/pii/S105381192100522X)
9. Xu, Drougard & Roy (2021). "Topological Data Analysis as a New Tool for EEG Processing." *Frontiers in Neuroscience*, 15:761703. [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.761703/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.761703/full)
10. Skaf & Laubenbacher (2022). "Topological data analysis in biomedicine: A review." *Journal of Biomedical Informatics*, 130:104082. [https://www.sciencedirect.com/science/article/pii/S1532046422000983](https://www.sciencedirect.com/science/article/pii/S1532046422000983)
11. Topological data analysis - Wikipedia. [https://en.wikipedia.org/wiki/Topological_data_analysis](https://en.wikipedia.org/wiki/Topological_data_analysis)
12. Comprehensive review of the Mapper algorithm (2025). *Int. J. Data Science and Analytics*, Springer. [https://link.springer.com/article/10.1007/s41060-025-00971-0](https://link.springer.com/article/10.1007/s41060-025-00971-0)
13. Singh, Mémoli & Carlsson (2007). "Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition." *Eurographics Symposium on Point-Based Graphics*. [https://research.math.osu.edu/tgda/mapperPBG.pdf](https://research.math.osu.edu/tgda/mapperPBG.pdf)
14. Brüningk et al. (2024). "Data-driven prediction of spinal cord injury recovery." *medRxiv*. [https://www.medrxiv.org/content/10.1101/2024.05.03.24306807v1](https://www.medrxiv.org/content/10.1101/2024.05.03.24306807v1)
15. Bui et al. (2025). "Understanding Conventional Deep Learning Models Through the Lens of Topological Data Analysis Using the Mapper Algorithm." *RTIS 2024*, LNNS, vol 1421, Springer. [https://link.springer.com/chapter/10.1007/978-3-031-92545-0_8](https://link.springer.com/chapter/10.1007/978-3-031-92545-0_8)
16. Diva-portal: "Illustrations of Data Analysis Using the Mapper Algorithm." [https://www.diva-portal.org/smash/get/diva2:900997/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:900997/FULLTEXT01.pdf)

---

> **참고:** 본 분석에서 SVD 렌즈와 Mapper 알고리즘의 수학적 정의는 논문 원문과 Singh, Mémoli & Carlsson (2007)의 Mapper 원 논문, 그리고 관련 TDA 문헌을 기반으로 재구성하였습니다. 논문 원문에서 구체적 수식이 명시적으로 제시되지 않은 부분(예: 거리 함수의 구체적 형태)은 방법론 서술에 근거하여 표준적인 형태로 기술하였으므로, 이 부분은 100% 확정적이지 않을 수 있습니다.
