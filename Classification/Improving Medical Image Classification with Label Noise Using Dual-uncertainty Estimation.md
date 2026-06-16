# Improving Medical Image Classification with Label Noise Using Dual-uncertainty Estimation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
의료 영상 분류에서 발생하는 두 가지 고유한 레이블 노이즈 유형(불일치 노이즈, 단일 타겟 노이즈)을 **이중 불확실성 추정(Dual-uncertainty Estimation)** 으로 동시에 처리함으로써 노이즈에 강건한 의료 영상 분류 모델을 훈련할 수 있다.

### 주요 기여
| 기여 항목 | 내용 |
|---|---|
| 노이즈 유형 정의 | 의료 영상에서의 두 가지 레이블 노이즈를 체계적으로 분류 |
| 이중 불확실성 프레임워크 | iDUP + MC-Dropout 기반 이중 불확실성 추정 |
| 부스팅 기반 커리큘럼 학습 | 클래스 불균형을 고려한 강건한 학습 절차 제안 |
| 데이터셋 공개 | 10명 이상의 안과 전문의 어노테이션이 포함된 Kaggle DR+ 데이터셋 공개 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

#### (1) 두 가지 레이블 노이즈 유형

```
[불일치 레이블 노이즈 (Disagreement Label Noise)]
- 여러 의사의 상이한 판단에서 발생
- 다수결 투표 후에도 오진 가능

[단일 타겟 레이블 노이즈 (Single-target Label Noise)]
- 단일 의사 혹은 다수결 결과 자체가 틀린 경우
- 완전히 잘못된 진단 기록에서 발생
```

#### (2) 기존 방법의 한계
- **클래스 불균형 환경**에서 노이즈 샘플과 소수 클래스의 정상 하드 샘플을 혼동하여 제거
- 의료 영상 특유의 **비대칭(클래스 의존적) 노이즈** 처리 미흡
- 메타러닝 기반 방법은 별도의 **클린 데이터셋** 필요

---

### 2-2. 제안하는 방법 (수식 포함)

#### Step 1: 불일치 불확실성 (UoD - Uncertainty of Disagreement)

먼저 샘플 $x_i$에 대한 경험적 등급 분포(히스토그램)를 계산합니다:

$$p_i^c = \frac{\sum_j \mathbf{1}_{y_i^j = g^c}}{n_i} \tag{1}$$

여기서 $n_i$는 샘플 $x_i$에 대한 어노테이션 수, $g^c$는 $c$번째 질환 등급입니다.

**Direct Uncertainty Prediction (DUP)** 기반 UoD:

$$UoD_i = U_1(x_i) = U(P_i) = 1 - \sum_{j=1}^{k}(p_i^j)^2 \tag{2}$$

**개선된 iUoD (어노테이션 수의 분산 반영):**

$$iUoD_i = \left[\min\left(\sum_j \mathbf{1}_{y_i^j = g^c}\right)\right]^\eta \cdot UoD_i \tag{3}$$

- $\eta$: 하이퍼파라미터 (어노테이션 수 불균형에 대한 가중치)
- $\min(\sum_j \mathbf{1}_{y_i^j = g^c})$: 가장 적은 투표를 받은 카테고리의 투표 수

> **iUoD의 필요성 예시:** $Y_1 = [0,1]$, $Y_2 = [0,0,0,1,1,1]$인 경우 DUP는 동일한 UoD = 0.5를 부여하지만, $Y_2$는 더 많은 의사가 참여했으므로 더 신뢰할 수 있는 판단으로 해석 가능 → iUoD가 이를 보정

#### Step 2: 단일 타겟 불확실성 (UoSL - Uncertainty of Single-target Label)

MC-Dropout을 통해 $T$회 확률적 순전파를 수행하여 평균 예측 엔트로피를 계산:

$$UoSL_i = -\sum_c m_{i,c} \log m_{i,c} \tag{4}$$

$$m_i = \frac{1}{T}\sum_t p_i^t$$

- $p_i^t$: $t$번째 순전파에서의 클래스 확률 벡터
- $m_i$: $T$회 순전파의 평균 예측값

#### Step 3: 부스팅 기반 커리큘럼 학습 (Boosting Training)

**샘플 가중치 설정:**

$$w_i = \begin{cases} 1 - nUoSL_i, & nUoSL_i > t_{UoSL} \\ 1, & nUoSL_i \leq t_{UoSL} \end{cases} \tag{8}$$

**결합 손실 함수:**

$$\mathcal{L}_{FL}(\hat{X}) = -\sum_{i=1}^{\hat{z}}(1 - p_i \cdot q_i)^\gamma \cdot \log(p_i) \tag{5}$$

$$\mathcal{L}_{wCE}(X) = -\sum_{i=1}^{z} w_i \cdot (q_i \cdot \log p_i + (1-q_i) \cdot \log(1-p_i)) \tag{6}$$

$$\mathcal{L} = \alpha \times \mathcal{L}_{FL}(\hat{X}) + \beta \times \mathcal{L}_{wCE}(X) \tag{7}$$

- $\mathcal{L}_{FL}$: Focal Loss (클린 샘플 $\hat{X}$에 적용, 클래스 불균형 완화)
- $\mathcal{L}_{wCE}$: 가중 Cross-Entropy (모든 샘플 $X$에 적용)
- $\beta = \left(\frac{epoch_i}{epoch_{all}}\right)^2$: 커리큘럼 학습 진행에 따라 점진적으로 증가

---

### 2-3. 모델 구조

```
┌─────────────────────────────────────────────────────────────┐
│                 Dual-uncertainty Framework                   │
│                                                             │
│  [다중 어노테이션 샘플]                                        │
│       ↓                                                     │
│  iDUP → iUoD 계산 → 임계값 필터링 (t_UoD > 0.5 제거)         │
│                                                             │
│  [단일 타겟 레이블 샘플]                                       │
│       ↓                                                     │
│  MC-Dropout (T회 순전파) → UoSL 계산 → 가중치 부여            │
│                                                             │
│  [부스팅 커리큘럼 학습]                                        │
│  초기: 클린 샘플 $\hat{X}$ 로 Focal Loss 학습                  │
│  점진적: 모든 샘플에 가중 CE Loss 적용                         │
└─────────────────────────────────────────────────────────────┘
```

**백본 아키텍처:**
- **ResNet-101** (Dropout 레이어: 각 Basic Layer 이후, 확률 0.3 삽입)
- **9-Layer CNN** (각 Conv 레이어 및 첫 번째 FC 레이어 이후 Dropout)

---

### 2-4. 성능 향상

#### ISIC 2019 (피부 병변, 합성 비대칭 노이즈)

| 방법 | 노이즈 10% (B) | 노이즈 20% (B) | 노이즈 40% (B) |
|---|---|---|---|
| Cross-Entropy | 82.63 | 80.73 | 78.27 |
| MixUp | 81.99 | 80.83 | 79.49 |
| Label Smoothing | 81.24 | 80.56 | 79.65 |
| CurriculumNet | 82.65 | 81.12 | 79.38 |
| Co-teaching | 80.25 | 78.80 | 77.23 |
| O2U-Net | 77.14 | 76.37 | 74.04 |
| **Ours** | **83.26** | **82.90** | **81.01** |

(AUC %, ResNet-101 기준, B=Best epoch)

#### Gleason 2019 (전립선암, 의사 불일치 노이즈)

| 방법 | F1 Score (Best) |
|---|---|
| Majority-vote | 81.88 |
| Annotator confusion estimation | 81.08 |
| **Majority-vote + Ours** | **83.22** |

#### Kaggle DR+ (망막 질환, 다중 레이블 노이즈)

| 방법 | F1 Score |
|---|---|
| Original + Re-sampling | 45.42 |
| Majority-vote + Re-sampling | 51.05 |
| **Ours** | **55.91** |

---

### 2-5. 한계점

1. **Long-tail 분포 처리 미흡:** 매우 불균형한 클래스 분포(Long-tailed)에서의 완전한 해결책 미제시
2. **Open-label 인식 미지원:** 훈련 시 보지 못한 새로운 클래스(Open-set) 처리 불가
3. **하드 샘플 손실 위험:** UoD가 높은 소수 클래스 샘플이 노이즈로 오인되어 제거될 위험 존재
4. **유사 클래스 간 구별 어려움:** UoSL이 인접 클래스 간 노이즈 탐지에 일반화 성능 저하 (예: NPDRI ↔ NPDRII, UoSL = 0.51)
5. **임계값 민감성:** $t_{UoD}$, $t_{UoSL}$, $\eta$ 등 하이퍼파라미터 튜닝 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 위한 핵심 설계 요소

#### (A) 샘플 재가중치 전략 (Sample Re-weighting)
기존 방법들이 불확실성 높은 샘플을 **제거(hard removal)** 하는 방식과 달리, 본 논문은 **재가중치(re-weighting)** 방식을 채택:

$$w_i = 1 - nUoSL_i \quad \text{(노이즈 의심 샘플에 낮은 가중치 부여)}$$

이 방식은 소수 클래스의 하드 샘플이 완전히 배제되는 것을 방지하여 **소수 클래스에 대한 일반화** 능력을 보존합니다.

#### (B) Focal Loss를 통한 클래스 불균형 처리

$$\mathcal{L}_{FL}(\hat{X}) = -\sum_{i=1}^{\hat{z}}(1 - p_i \cdot q_i)^\gamma \cdot \log(p_i)$$

$(1-p_i \cdot q_i)^\gamma$ 항이 쉬운 샘플(이미 잘 분류되는)에 낮은 가중치를 부여하여 **어려운 샘플에 집중 학습**을 유도합니다.

#### (C) 커리큘럼 학습의 일반화 효과
$\beta = \left(\frac{epoch_i}{epoch_{all}}\right)^2$로 점진적으로 모든 샘플을 학습에 포함시키는 방식은 **조기 과적합(early overfitting)** 을 방지하여 일반화 성능을 향상시킵니다.

#### (D) 다양한 도메인 검증
세 가지 서로 다른 의료 영상 모달리티(피부경, 병리 슬라이드, 안저 사진)에서의 실험을 통해 도메인 횡단 일반화 가능성을 시사합니다.

### 3-2. 일반화 관련 실험 결과 해석

**안정성 지표 (Best - Last epoch 차이):**
- CurriculumNet의 40% 노이즈에서 Best-Last 차이: **6.37%** (불안정)
- 본 논문 방법의 40% 노이즈에서 Best-Last 차이: **2.35%** (상대적으로 안정)

→ 더 작은 Best-Last 차이는 훈련 안정성과 일반화 성능의 지표

**어노테이션 품질 vs 데이터 양 분석 (Gleason 2019):**
> "어노테이션의 질이 훈련 샘플의 수보다 모델 성능에 더 크게 기여한다"
(pathologist 1: 6,506 샘플, pathologist 6: 2,060 샘플 → 유사 성능)

이는 불확실성 기반 필터링이 **데이터 효율적 일반화**에 기여함을 시사합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 향후 연구에 미치는 영향

#### (A) 의료 AI 신뢰성 향상
불확실성 정량화를 훈련 과정에 통합함으로써 **설명 가능한 AI(XAI)** 와 **신뢰할 수 있는 AI** 연구의 기반을 마련합니다. 모델이 "얼마나 확신하는가"를 수치화하여 임상 의사결정 지원에 활용 가능합니다.

#### (B) 의료 데이터 어노테이션 패러다임 변화
단순한 다수결 투표를 넘어서, **불확실성 가중 어노테이션 집계** 방식의 연구를 촉진합니다.

#### (C) 반지도 학습과의 융합 가능성
UoSL로 식별된 노이즈 의심 샘플을 **레이블 없는 데이터로 재활용**하는 반지도 학습 접근법으로의 확장이 가능합니다.

#### (D) 연합 학습(Federated Learning)과의 시너지
여러 의료기관의 어노테이터 불일치를 처리하는 프레임워크로서, **프라이버시 보호 연합 학습**에서의 노이즈 처리에 응용 가능합니다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법 | 핵심 아이디어 | 본 논문과의 관계 |
|---|---|---|---|
| **DivideMix** (Li et al., 2020, arXiv:2002.07394) | GMM + MixMatch | GMM으로 클린/노이즈 분리 후 반지도 학습 | 본 논문이 비교 베이스라인으로 활용; GMM은 불균형 데이터에 취약 |
| **Dual T** (Yao et al., 2020, arXiv:2006.07805) | 전이 행렬 추정 | 이중 전이 행렬로 추정 오차 감소 | 모델 기반 접근 vs 본 논문의 샘플 기반 접근 |
| **Noise-robust Loss** (Song et al., 2020, arXiv:2007.08199) | 손실 함수 설계 | 노이즈 강건 손실 함수 서베이 | 본 논문이 인용한 서베이; 본 논문의 방향성과 상호보완 |
| **Karimi et al.** (2020, Medical Image Analysis) | 의료 특화 분석 | 의료 영상에서의 노이즈 레이블 기술 분석 | 본 논문이 의료 특화 후속 연구로 위치 |

---

### 4-3. 앞으로 연구 시 고려할 점

#### (1) Long-tail 분포와의 통합
의료 데이터의 현실적 분포(희귀 질환 = 극소수 샘플)를 고려한 **Long-tail 학습과 노이즈 레이블 처리의 통합 프레임워크** 개발이 필요합니다.

$$\mathcal{L}_{LT-Noise} = \mathcal{L}_{balanced} + \lambda \cdot \mathcal{L}_{uncertainty}$$

#### (2) 동적 임계값 설정
현재 고정된 $t_{UoD} = 0.5$를 학습 과정 중 **적응적으로 조정**하는 메커니즘 연구:
$$t_{UoD}^{(epoch)} = f(epoch, \text{class distribution}, \text{noise rate estimate})$$

#### (3) 기초 모델(Foundation Model)과의 결합
대규모 사전학습 모델(예: MedCLIP, BioViL 등)의 특징 표현을 활용하여 불확실성 추정의 정확도를 향상시키는 연구가 필요합니다.

#### (4) Open-set 노이즈 처리
훈련 클래스 외의 카테고리에서 유입된 노이즈(Open-set noise)를 구별하는 메커니즘:
$$UoSL_{open} = \text{Energy Score}(f(x_i)) \text{ or } \text{OOD Detection}$$

#### (5) 어노테이션 비용 최적화
불확실성 점수를 **능동 학습(Active Learning)** 에 활용하여 재어노테이션이 가장 필요한 샘플을 우선 선정하는 연구.

#### (6) 멀티모달 의료 데이터로의 확장
영상 + 임상 기록 + 유전체 데이터 등 **멀티모달 의료 데이터**에서의 교차 모달 불확실성 추정 연구.

---

## 참고 자료 (본 논문에서 직접 인용된 주요 문헌)

1. **Ju et al. (2021)** - "Improving Medical Image Classification with Label Noise Using Dual-uncertainty Estimation," arXiv:2103.00528v2 *(분석 대상 논문)*
2. **Li et al. (2020)** - "DivideMix: Learning with Noisy Labels as Semi-supervised Learning," arXiv:2002.07394
3. **Gal & Ghahramani (2016)** - "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," ICML 2016
4. **Raghu et al. (2019)** - "Direct Uncertainty Prediction for Medical Second Opinions," ICML 2019
5. **Kendall & Gal (2017)** - "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?," NeurIPS 2017
6. **Han et al. (2018)** - "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels," NeurIPS 2018
7. **Guo et al. (2018)** - "CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images," ECCV 2018
8. **Lin et al. (2017)** - "Focal Loss for Dense Object Detection," ICCV 2017
9. **Xue et al. (2019)** - "Robust Learning at Noisy Labeled Medical Images: Applied to Skin Lesion Classification," ISBI 2019
10. **Song et al. (2020)** - "Learning from Noisy Labels with Deep Neural Networks: A Survey," arXiv:2007.08199
11. **Karimi et al. (2020)** - "Deep Learning with Noisy Labels: Exploring Techniques and Remedies in Medical Image Analysis," Medical Image Analysis, vol. 65
12. **Yao et al. (2020)** - "Dual T: Reducing Estimation Error for Transition Matrix in Label-Noise Learning," arXiv:2006.07805
13. **He et al. (2016)** - "Deep Residual Learning for Image Recognition," CVPR 2016

> **⚠️ 정확도 관련 고지:** 본 분석은 제공된 PDF 원문(arXiv:2103.00528v2)에 기반하여 작성되었으며, 논문에 명시되지 않은 사항(예: 2021년 이후 후속 연구 비교)은 논문 내 인용 문헌 범위 내에서만 기술하였습니다. 2020년 이후 최신 연구 비교는 논문 내 참조된 문헌 기준으로 작성되었으며, 논문 발표(2021년 3월) 이후의 연구 동향은 본 PDF에 포함되지 않아 직접 확인이 불가능함을 명시합니다.
