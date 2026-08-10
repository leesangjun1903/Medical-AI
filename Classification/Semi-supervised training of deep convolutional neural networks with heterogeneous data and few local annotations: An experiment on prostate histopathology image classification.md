# Semi-supervised training of deep convolutional neural networks with heterogeneous data and few local annotations: An experiment on prostate histopathology image classification

## 논문 정보
**제목:** Semi-supervised training of deep convolutional neural networks with heterogeneous data and few local annotations: An experiment on prostate histopathology image classification

**저자:** Niccolò Marini, Sebastian Otálora, Henning Müller, Manfredo Atzori

**게재지:** Medical Image Analysis 73 (2021) 102165

---

## 1. Executive Summary (10문장 이내)

이 논문은 전립선암 조직병리 이미지의 Gleason 등급 분류(Gleason grading/scoring)를 위한 반지도학습(semi-supervised learning) 기반 Teacher/Student CNN 학습 패러다임을 제안한다.  
디지털 병리 이미지는 수집 기관과 장비에 따라 외관이 크게 달라지고(inter-dataset heterogeneity), 전문가의 픽셀 단위 주석이 매우 희소하다는 이중 도전 과제가 존재한다.  
저자들은 고용량 Teacher 모델(ResNeXt50)이 약한 주석 또는 미주석 데이터에 대해 의사 레이블(pseudo-label)을 생성하고, 소용량 Student 모델(DenseNet121)이 이를 활용하여 학습하는 순차적 구조를 설계하였다.  
두 가지 접근법(semi-supervised, semi-weakly supervised)과 세 가지 학생 훈련 변형(Variant I, II, III)을 체계적으로 비교 평가하였다.  
공개 데이터셋인 TMAZ(886개 TMA 코어, 강한 주석)와 TCGA-PRAD(449개 WSI, 약한 주석)라는 이질적인 두 데이터셋에서 동시에 성능을 평가하였다.  
제안 방법은 완전지도학습 베이스라인 대비 패치 수준($\kappa$: 0.5667 → 0.6127), TMA 코어 수준($\kappa$: 0.7186 → 0.7645), WSI 수준($\kappa$: 0.2293 → 0.4529) 모두에서 통계적으로 유의미한 성능 향상을 달성하였다.  
특히 WSI 수준에서 약 2배의 성능 향상은 이질적 외부 데이터셋에 대한 일반화 능력이 크게 개선되었음을 의미한다.  
이 방법은 기존 최고 성능 논문인 Arvaniti et al.(2018) 및 Bulten et al.(2020)보다 TMAZ에서 우수한 결과를 보였다.  
결론적으로, 의사 레이블 데이터의 활용은 국소 주석 부족 문제와 데이터 이질성 문제를 동시에 완화하는 효과적인 전략임을 실험적으로 입증하였다.

---

### 1-1. 연구의 목적과 필요성

**연구 목적:**
> "이질적인 여러 데이터셋에 일반화되는 CNN 분류 모델을 적은 수의 국소 주석만으로 학습시키는 방법을 제안한다." (Abstract, p.1)

**필요성의 세 가지 축:**

| 문제 | 구체적 내용 | 출처 |
|------|-----------|------|
| 데이터 이질성 | 기관별 염색 프로토콜 비표준화로 WSI 외관이 크게 상이함 | Introduction, p.2 |
| 주석 희소성 | 픽셀 단위 강한 주석(strong label)은 전문 병리학자가 필요하여 비용이 높음 | p.1-2 |
| 일반화 실패 | 단일 기관 데이터로 학습한 모델은 외부 데이터셋에서 성능이 급격히 저하됨 | p.2 |

> **용어 설명:**
> - **강한 주석(Strong label/annotation):** 이미지의 각 픽셀에 대해 전문가가 직접 부여하는 세밀한 레이블. 매우 정확하지만 생성 비용이 높음
> - **약한 주석(Weak label/annotation):** 이미지 전체에 대한 진단명(예: Gleason score)처럼 위치 정보 없이 부여되는 레이블. 생성 비용이 낮지만 정보가 부족함
> - **Gleason score:** 전립선암의 조직 구조 이상 정도를 1~5등급(Gleason Pattern, GP)으로 평가하고, 가장 흔한 두 패턴의 합으로 계산하는 진단 지표 (p.3)

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|---------|------|------|
| 1 | Teacher/Student 패러다임으로 국소 주석 없이 이질적 데이터셋에 일반화 가능 | TMAZ, TCGA-PRAD 모두에서 베이스라인 대비 향상 | Abstract, Table 6,7 |
| 2 | Semi-supervised 접근이 Semi-weakly supervised보다 약간 우수 | 교사 모델이 강한 주석만으로 학습 시 의사 레이블 노이즈가 적음 | p.13, Table 6,7 |
| 3 | Student Variant III(혼합 훈련)가 두 데이터셋 모두에서 균형 잡힌 성능 | TMAZ 및 TCGA-PRAD 동시 고성능 달성 | p.14, Fig.4,5 |
| 4 | 의사 레이블 수(K값)와 성능 사이에 최적점 존재(단조증가 아님) | K 증가 시 노이즈 증가로 성능 저하 가능 | p.13, Fig.4,5 |
| 5 | 기존 최고 성능(Arvaniti 2018, Bulten 2020) 대비 TMAZ에서 우수 | Gleason grading κ=0.6613 vs 0.55, Gleason scoring κ=0.8124 vs 0.75 | Table 8, p.13 |
| 6 | 인터-데이터셋 이질성은 오버피팅의 주요 원인 | 단일 데이터셋 훈련 시 외부 데이터셋에서 큰 성능 저하 | p.13-14 |

---

### 2-1. 상세 설명

#### 해결하고자 하는 문제 (p.1-2)

1. **의사 레이블 희소성:** 대규모 공개 디지털 병리 데이터(TCGA 등)에는 픽셀 단위 주석이 없음
2. **이질적 데이터 일반화:** 기관마다 다른 스캐너, 염색 조건으로 인해 단일 기관 학습 모델은 외부 데이터에서 성능 급락
3. **계산 효율:** 추론(inference) 시 경량 모델 필요

---

#### 제안하는 방법 및 수식

**Teacher/Student 패러다임 구조:**

$$\text{Teacher} \xrightarrow{\text{pseudo-label}} \text{Student}$$

**의사 레이블 생성:**

$$\hat{y}_i = \arg\max_c P(\text{class}=c \mid x_i; \theta_T)$$

- $\hat{y}_i$: 샘플 $x_i$에 대한 의사 레이블
- $P(\text{class}=c \mid x_i; \theta_T)$: Teacher 모델 파라미터 $\theta_T$에 의한 softmax 확률
- $c$: 클래스 (Benign, GP3, GP4, GP5)

> **용어 설명:**
> - **Softmax:** 다중 클래스 분류에서 각 클래스에 속할 확률의 합이 1이 되도록 정규화하는 함수. $P(c) = \frac{e^{z_c}}{\sum_j e^{z_j}}$

**평가 지표 - 이차 가중 Cohen's Kappa (p.10):**

$$\kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i-j)^2}{(N-1)^2}$$

- $\kappa$: Cohen's Kappa 계수 (완벽 일치=1, 우연 수준=0)
- $w_{i,j}$: 이차 가중치 (실제 클래스와 예측 클래스 간 거리가 멀수록 더 크게 패널티)
- $O_{i,j}$: 실제 클래스 $i$, 예측 클래스 $j$인 샘플 수
- $E_{i,j}$: 우연에 의해 기대되는 클래스 $i$, $j$ 샘플 수
- $N$: 전체 클래스 수 (Gleason pattern: $N=4$, Gleason score: $N=5$)
- $i, j$: 정렬된 클래스 인덱스

> **용어 설명:**
> - **Cohen's Kappa ($\kappa$):** 두 평가자(또는 모델과 전문가) 간의 합의 수준을 우연에 의한 합의를 보정하여 측정하는 지표. 이차 가중(quadratic weighted) 버전은 예측이 실제 값에서 멀수록 더 강하게 패널티를 부여

**손실 함수 (p.8):**

$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log \hat{p}_c$$

- $\mathcal{L}$: Cross-entropy loss
- $y_c$: 클래스 $c$의 실제 레이블 (one-hot)
- $\hat{p}_c$: 클래스 $c$의 예측 확률
- $C$: 클래스 수 (4)

**Blue-Ratio (BR) 패치 선택 (p.7, Chang et al., 2012):**

$$BR = \frac{B \times 255}{(R + G + B + \epsilon)} - G$$

- $R, G, B$: 픽셀의 RGB 채널 값
- $\epsilon$: 분모가 0이 되는 것을 방지하는 작은 상수
- BR이 높을수록 핵(nuclei) 밀도가 높은 패치로 선택 우선순위가 높음

> **용어 설명:**
> - **Blue-Ratio (BR):** H&E 염색 이미지에서 핵(nucleus) 밀도가 높은 조직 영역을 자동으로 식별하기 위한 색상 기반 점수. 핵은 헤마톡실린에 의해 파란색으로 염색되므로 파란색 성분이 많은 패치가 진단에 유용함

---

#### 모델 구조 (p.8)

| 구성 요소 | Teacher | Student |
|---------|---------|---------|
| 아키텍처 | ResNeXt50_32x4d | DenseNet121 |
| 파라미터 수 | ~22M | ~7M |
| 사전학습 데이터 | YFCC100M (Instagram 10억 이미지) | ImageNet (100만 이미지) |
| 출력 노드 | 4 (Benign, GP3, GP4, GP5) | 4 (동일) |
| 역할 | 의사 레이블 생성 | 최종 분류 수행 |

> **용어 설명:**
> - **ResNeXt:** 잔차 연결(residual connection)에 다중 경로(grouped convolution)를 결합한 CNN 아키텍처. "32x4d"는 32개 그룹, 그룹당 4채널을 의미
> - **DenseNet:** 각 층이 이전 모든 층의 출력을 입력으로 받는 밀집 연결 구조의 CNN. 파라미터 효율이 높음

**두 가지 Teacher/Student 접근법:**

```
[Semi-supervised Learning]
TMAZ(강한주석) → Teacher 학습 → 의사레이블 생성 → Student 학습

[Semi-weakly supervised Learning]
TCGA-PRAD(약한주석) → Teacher 사전학습
                    ↓
TMAZ(강한주석) → Teacher 미세조정 → 의사레이블 생성 → Student 학습
```

**세 가지 Student Training Variants (p.6, Fig.2):**

| Variant | 훈련 데이터 | 특징 |
|---------|-----------|------|
| I (파란선) | 의사 레이블만 | TCGA-PRAD에 특화, TMAZ에 약함 |
| II (초록선) | 의사레이블 사전학습 → TMAZ 미세조정 | TMAZ에 강하지만 TCGA-PRAD에 약함 |
| III (자홍선) | 의사레이블 + TMAZ 동시 훈련 | 두 데이터셋 모두 균형 잡힌 성능 |
| Baseline (회색점선) | TMAZ만 | 완전지도학습 기준선 |

---

#### 성능 향상 및 한계

**성능 향상 요약 (Table 6, 7, 8):**

| 평가 수준 | 베이스라인 κ | 최고 κ (방법) | 향상폭 |
|---------|-----------|------------|------|
| Patch-level (TMAZ) | 0.5667±0.0285 | **0.6127±0.0133** (Semi-sup, Variant II) | +0.046 |
| TMA Core-level (TMAZ) | 0.7186±0.0306 | **0.7645±0.0231** (Semi-sup, Variant II) | +0.046 |
| WSI-level (TCGA-PRAD) | 0.2293±0.1350 | **0.4529±0.0512** (Semi-sup, Variant I) | +0.224 |

**한계:**
- WSI 수준 κ=0.45는 병리학자 간 합의 수준에 크게 못 미침 (p.13)
- 다수결 투표(majority voting) 방식의 집계에서 기질(stroma) 패치가 오분류 유발 (p.14)
- 성능이 사용된 의사 레이블 수(K)에 민감하며 단조증가하지 않음 (p.13)
- TCGA-PRAD에서 큰 신뢰구간: 대부분 모델에서 불안정한 성능 (Fig.4,5,C)
- 두 데이터셋에서 Variant II가 교차 성능 저하(trade-off) 발생 (p.14)

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|-----|---------|
| Teacher/Student 패러다임 전반 | Fig.2 (p.7), Section 2.2 (p.6) |
| Semi-supervised 성능 결과 | Fig.4 (p.9), Table 6 (p.11) |
| Semi-weakly supervised 성능 결과 | Fig.5 (p.10), Table 7 (p.11) |
| 기존 방법 대비 성능 비교 | Table 1 (p.4), Table 8 (p.13) |
| 혼동 행렬 최고 성능 시각화 | Fig.6 (p.12) |
| 의사 레이블 시각화 | Fig.7 (p.14) |
| Blue-Ratio 패치 선택 | Fig.3 (p.7), Section 2.3.1 (p.6-7) |
| Cohen's Kappa 수식 | p.10, Section 2.3.5 |
| 오버피팅 논의 | p.13-15, Fig.4C, Fig.5C |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 연구 주제
- **저자 보고:** "This paper presents a novel approach to train CNNs that generalize to heterogeneous datasets originating from various sources and without local annotations." (Abstract, p.1)
- **내 해석:** 이 연구는 디지털 병리의 핵심 병목인 '주석 병목(annotation bottleneck)'과 '도메인 이동(domain shift)' 문제를 teacher-student 구조로 동시에 해결하려는 실용적 접근이다.

### 방법 (수식 포함)
- **저자 보고:** "The teacher model computes a probability array. It labels the sample as the class with the highest probability. The pseudo-labels are used as if a human expert had performed the annotation." (p.6)
- **내 해석:** 이는 Lee (2013)의 Pseudo-Label 기법의 확장이지만, 고용량 사전학습 모델(SNS 10억 이미지 학습)을 교사로 활용함으로써 저품질 의사 레이블 문제를 완화하는 핵심 차별점이 있다.

### 결과
- **저자 보고 (Table 6):**
  - Patch-level: $\kappa = 0.6127 \pm 0.0133$ (from $\kappa = 0.5667 \pm 0.0285$)
  - WSI-level: $\kappa = 0.4529 \pm 0.0512$ (from $\kappa = 0.2293 \pm 0.1350$)
- **내 해석:**
  - WSI 수준의 약 2배 성능 향상($\Delta\kappa \approx 0.224$)은 인상적이나, 베이스라인 $\kappa=0.2293$의 표준편차($\pm 0.1350$)가 극도로 크다는 점에서 베이스라인 자체가 불안정함을 의미한다. ⚠️ 따라서 이 향상폭의 임상적 의미는 신중하게 해석해야 한다.
  - Variant I이 TCGA-PRAD에서 최고 성능을 보이는 것은 해당 도메인 내 과적합이 아닌 데이터 분포 매칭 효과로 볼 수 있다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

### ⚠️ 통계적 취약점

| 항목 | 문제점 | 위치 |
|-----|-------|------|
| TCGA-PRAD 베이스라인 표준편차 | $\kappa = 0.2293 \pm \mathbf{0.1350}$ — SD가 평균의 59%로 매우 불안정 | Table 6 |
| Semi-weakly Variant II TCGA 성능 | $\kappa = 0.4121 \pm \mathbf{0.0963}$ — 다른 방법 대비 SD 2배 | Table 7 |
| 10회 반복의 충분성 | 10회 반복이 통계적 안정성을 보장하기에 충분한지 검증 미흡 | p.9 |
| 교사 모델 1회만 학습 | Semi-weakly 접근에서 교사 사전학습을 1회만 수행 — 반복 불가 | p.9 |
| Wilcoxon Rank-Sum 검정의 적용 | 검정의 구체적인 p-value 수치가 보고되지 않고 별표(*)만 표시 | p.11 |

### ⚠️ 비교 불가능한 수치

| 비교 쌍 | 이유 |
|--------|------|
| 이 연구 vs. Ström et al.(2019) ($\kappa=0.83$) | 훈련 데이터 규모가 6682 WSI vs. 641 TMA로 비교 불가 |
| 이 연구 vs. Bulten et al.(2020) 내부 테스트 ($\kappa=0.91$) | 내부 vs. 외부 테스트셋 차이 존재 |
| 이 연구 vs. Nagpal et al.(2019) (ACC=0.94) | 평가 지표가 다름(ACC vs. κ) |
| TCGA-PRAD WSI-level 결과 | 테스트셋이 46 WSIs로 매우 소규모 — 통계적 신뢰성 낮음 |

---

## 6. 논문이 답하지 않는 질문

| # | 미답 질문 | 중요도 |
|---|---------|------|
| 1 | 전립선 외 다른 암종(폐암, 유방암 등)에서의 적용 가능성은? | 매우 높음 |
| 2 | K=10,000 이상의 의사 레이블을 사용하면 어떻게 되는가? | 높음 |
| 3 | 교사 모델의 품질이 학생 성능에 미치는 정량적 영향은? | 높음 |
| 4 | 다른 아키텍처(ViT, Transformer 등) 조합 시 성능은? | 높음 |
| 5 | 색상 정규화(stain normalization)와의 병용 효과는? | 높음 |
| 6 | 의사 레이블의 노이즈율을 정량적으로 측정하였는가? | 중간 |
| 7 | 다수결 투표보다 우수한 집계 방법은 무엇인가? | 중간 |
| 8 | 완전히 새로운(미지의) 기관 데이터에도 일반화되는가? | 매우 높음 |
| 9 | 각 Gleason Pattern별 오분류 패턴의 임상적 의미는? | 중간 |
| 10 | 레이블 효율성: 몇 개의 강한 주석이 최소 필요한가? | 높음 |

---

## 7. 중요 그림 5개 해석

### 📊 Fig.1 (p.2) — 데이터 이질성 시각화

**내용:** TCGA-PRAD(위, ~100,000² 픽셀)와 TMAZ(아래, ~3,100² 픽셀)의 예시 이미지

**해석:** 두 데이터셋의 크기 차이(약 1,000배)와 함께 염색 색조, 조직 밀도, 이미지 선명도가 육안으로도 뚜렷이 다름을 보여준다. 이는 단순 데이터 증강으로는 해결할 수 없는 수준의 도메인 이동을 시각적으로 정당화하며, 논문의 문제 설정의 타당성을 직접적으로 지지한다.

---

### 📊 Fig.2 (p.7) — Teacher/Student 학습 파이프라인 전체 구조

**내용:** 교사 학습(상단)과 학생 학습(하단)의 전체 흐름, 두 접근법(실선/점선)과 세 변형(색상별)을 한눈에 표현

**해석:** 이 그림은 논문의 핵심 아이디어를 집약한다. 교사 모델이 강한 주석(solid line) 또는 약한 주석+강한 주석(dotted line)으로 학습되는 두 경로가 명확히 구분된다. 학생의 세 변형은 서로 다른 데이터 혼합 전략을 나타내며, 베이스라인(회색 점선)과의 비교 구조가 명확하다. 이 구조의 핵심은 각 모델이 서로 다른 역할(의사 레이블 생성 vs. 효율적 추론)을 담당한다는 점이다.

---

### 📊 Fig.4 (p.9) — Semi-supervised 학습 결과 (K값별 성능 곡선)

**내용:** 의사 레이블 수(K: 0~10,000)에 따른 세 Variant의 κ 변화 (A: TMAZ 패치, B: TMAZ 코어, C: TCGA WSI)

**해석:**
- **Sub-fig A,B:** Variant II(초록)와 III(자홍)이 베이스라인(회색 점선)을 안정적으로 상회. Variant I(파란)은 하회 → 의사 레이블만으로는 TMAZ에 일반화 불가
- **Sub-fig C:** 모든 Variant가 베이스라인($\kappa≈0.23$)을 크게 상회하나 신뢰구간이 큼 → TCGA에서의 불안정성 확인
- **핵심 관찰:** 성능이 단조 증가하지 않고 K=7,000~10,000 부근에서 피크 후 하락 → 의사 레이블 노이즈 누적 효과

---

### 📊 Fig.6 (p.12) — 혼동 행렬 (최고 성능 모델)

**내용:** TMAZ Gleason grading(A), TMAZ Gleason scoring(B), TCGA-PRAD Gleason scoring(C)에서 최고 성능 모델의 원시 및 정규화 혼동 행렬

**해석:**
- **Fig.6A (Gleason grading, TMAZ, κ=0.6613):** GP4-GP5 경계 혼동이 가장 큼(GP4→GP5 오분류: 0.145). Benign-GP3 경계도 혼동(0.151). 이는 임상적으로 중요한 경계인 GP3/GP4 구분 어려움을 시사
- **Fig.6B (Gleason scoring, TMAZ, κ=0.8124):** GS9-10 범주에서 대각선 비율 0.737로 낮음. GS9를 GS8로 혼동(0.440)이 가장 빈번
- **Fig.6C (Gleason scoring, TCGA-PRAD, κ=0.6015):** GS7(4+3)의 정분류율이 0.364로 가장 낮음 → 이질적 외부 데이터에서 중간 등급 구분이 가장 어려움

---

### 📊 Fig.7 (p.14) — 의사 레이블 패치 시각화

**내용:** 교사 모델이 생성한 각 클래스(Benign, GP3, GP4, GP5)별 상위(확률 높음) 및 하위(확률 낮음) 랭크 패치

**해석:**
- 상위 랭크 패치(P≈0.999)들은 해당 Gleason pattern의 전형적인 조직 형태학(morphology)을 보임 → 교사 모델의 강한 주석 학습이 효과적으로 이루어졌음을 입증
- 하위 랭크 패치(P≈10⁻¹²~10⁻¹⁴)는 클래스 형태학과 불일치 → 이러한 저확률 패치를 제외함으로써 노이즈 감소 전략의 합리성을 시각적으로 지지
- GP4 하위 패치들이 실제로 다른 패턴처럼 보이는 점은 교사 모델의 불확실성이 조직 형태학적 경계와 일치함을 시사

---

## 8. 결론 분석

### 8-1. 모델의 일반화 성능 향상 가능성

#### 저자가 제시한 시사점 (p.15, Section 5)
> "The teacher/student paradigm allows to face the inter-dataset heterogeneity and to limit the overfitting during the training of the models."

> "The models trained with the approach generalize on both datasets used in this paper, the TMAZ and the TCGA-PRAD."

#### 저자의 후속 연구 계획 (p.15)
1. 다른 조직 유형(different types of tissues)으로 패러다임 확장
2. K값을 10,000 이상으로 확장하여 대규모 비주석 데이터 활용
3. 코드 및 사전학습 모델 GitHub 공개 (https://github.com/ilmaro8/Semi_Supervised_Learning)

#### 일반화 성능 향상 가능성 분석

**현재 일반화의 제한:**

| 측면 | 현 상태 | 개선 가능성 |
|-----|--------|-----------|
| 데이터 도메인 | 전립선만 | 범종양(pan-cancer) 확장 가능 |
| 기관 수 | TCGA: 19개 기관 | 더 많은 기관 데이터 포함 시 향상 |
| 입력 해상도 | 40x 단일 배율 | 다배율(multi-scale) 학습 도입 시 개선 |
| 집계 방법 | 단순 다수결 | MIL, Attention-based 방법 도입 가능 |
| 노이즈 제거 | 확률 임계값 기반 | 신뢰도 기반 curriculum learning 적용 가능 |

**일반화 향상을 위한 핵심 방향:**

1. **도메인 불변 특성 학습:** 스테인 정규화(stain normalization) 또는 도메인 적대적 훈련(domain adversarial training)과의 결합
2. **메타러닝 통합:** MAML 등 few-shot 학습 방법론과의 결합으로 새로운 기관에 빠르게 적응
3. **자기지도학습(SSL) 사전학습:** SimCLR, DINO 등을 활용한 병리 이미지 특화 사전학습으로 교사 모델의 초기 표현력 강화

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 주의사항:** 아래 분석은 제 학습 데이터(2021년 초까지)에 기반하며, 일부 세부 사항은 정확하지 않을 수 있습니다. 실제 연구 진행 시 해당 논문 원문을 직접 확인하시기 바랍니다.

#### 2020년 이후 주요 관련 연구 동향

| 연구 방향 | 대표 접근법 | 본 논문과의 관계 |
|---------|-----------|--------------|
| **자기지도 병리 이미지 학습** | MoCo, SimCLR 병리 특화 버전 (예: HIPT, 2022 추정) | 교사 모델의 사전학습 방식을 크게 개선 가능 |
| **Attention 기반 MIL** | CLAM, TransMIL (2021) | 다수결 투표 대신 attention map 기반 집계로 WSI 성능 향상 가능 |
| **Vision Transformer (ViT) 병리 적용** | 패치 기반 ViT의 병리 이미지 분류 | 본 논문의 CNN 기반 접근을 transformer로 대체 가능 |
| **Federated Learning** | 다기관 연합 학습 | 이질적 데이터 일반화를 프라이버시 보호와 함께 해결 |
| **Foundation Model** | 대규모 병리 이미지 foundation model (UNI, Conch 등, 2023-2024 추정) | 본 논문 교사 모델보다 훨씬 강력한 사전학습 활용 가능 |

> **용어 설명:**
> - **MIL (Multiple Instance Learning):** 개별 패치(instance)에 레이블이 없고 전체 이미지(bag)에만 레이블이 있는 설정에서 학습하는 방법론
> - **Foundation Model:** 대규모 데이터로 사전학습된 범용 모델. 다양한 downstream task에 few-shot으로 적용 가능
> - **DINO/SimCLR:** 레이블 없이 자기지도(self-supervised) 방식으로 강력한 표현을 학습하는 대조 학습 방법론

#### 본 논문이 앞으로의 연구에 미치는 영향

1. **방법론적 기여:**
   - 이질적 병리 데이터셋 간 teacher-student 전이의 실증적 프레임워크 제시
   - Blue-Ratio 기반 패치 선택 + 확률 기반 의사 레이블 선택의 파이프라인은 후속 연구에서 자주 참조됨

2. **데이터 공개의 기여:**
   - TMAZ + TCGA-PRAD 공개 데이터셋 조합의 벤치마크 확립

3. **한계가 후속 연구 동기 부여:**
   - 다수결 투표의 한계 → attention 기반 집계 연구 활성화
   - 의사 레이블 노이즈 → curriculum learning, confidence-based filtering 연구

#### 앞으로의 연구 시 고려할 점

| 고려사항 | 구체적 방향 |
|--------|-----------|
| **더 강력한 교사 모델** | Instagram 사전학습 대신 병리 특화 foundation model 사용 |
| **노이즈 레이블 처리** | Co-training, mixup 등 노이즈 강건 학습 기법 통합 |
| **집계 방법 개선** | 다수결 → Transformer 기반 attention aggregation |
| **다중 배율(multi-scale)** | 40x 단일 배율 → 10x/20x/40x 계층적 특성 통합 |
| **색상 불변성** | 스테인 정규화 또는 도메인 적대적 훈련 추가 |
| **엄밀한 통계 검증** | p-value 명시, 효과 크기(effect size) 보고, 다중 비교 보정 |
| **임상 실용성 평가** | κ 지표 외 실제 임상 의사결정 영향력 평가 |
| **데이터 확장성** | PANDA Challenge 데이터(11,000 WSI)와의 결합 실험 |

---

## 참고자료 목록

**논문 원문:**
- Marini, N., Otálora, S., Müller, H., Atzori, M. (2021). Semi-supervised training of deep convolutional neural networks with heterogeneous data and few local annotations: An experiment on prostate histopathology image classification. *Medical Image Analysis*, 73, 102165. https://doi.org/10.1016/j.media.2021.102165

**논문 내 인용 주요 참고문헌:**
- Arvaniti, E., et al. (2018). Automated gleason grading of prostate cancer tissue microarrays via deep learning. *Scientific Reports*, 8.
- Yalniz, I.Z., et al. Billion-scale semi-supervised learning for image classification. *arXiv:1905.00546*
- Bulten, W., et al. (2020). Automated deep-learning system for gleason grading. *Lancet Oncology*
- Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method. *ICML Workshop*
- Huang, G., et al. (2016). Densely connected convolutional networks. *arXiv:1608.06993*
- Xie, S., et al. (2017). Aggregated residual transformations (ResNeXt). *CVPR*
- Chang, H., et al. (2012). Nuclear segmentation via MRGC. *ISBI*
- Wilcoxon, F. (1992). Individual comparisons by ranking methods. *Breakthroughs in Statistics*
- Shaw, S., et al. (2020). Teacher-student chain for semi-supervised histology classification. *ICLR Workshop*
- Tolkach, Y., et al. (2020). High-accuracy prostate cancer pathology using deep learning. *Nature Machine Intelligence*
