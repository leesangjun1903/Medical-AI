# Effective Active Learning in Digital Pathology: A Case Study in Tumor Infiltrating Lymphocytes

---

## 1. Executive Summary (10문장 이내)

1. 디지털 병리학에서 딥러닝 모델은 뛰어난 성능을 보이지만, 전문 병리학자의 대규모 어노테이션(annotation)이 필수적이라는 심각한 병목 문제가 존재한다.
2. 본 논문은 능동 학습(Active Learning, AL)을 활용하여 종양 침윤 림프구(Tumor Infiltrating Lymphocytes, TIL) 분류에 필요한 어노테이션 비용을 최소화하는 방법을 제안한다.
3. 기존 AL 방법들(BALD, CoreSet 등)을 디지털 병리에 단순 적용할 경우, 선택된 샘플들이 유사한 형태학적 특성을 가져 무작위 샘플링 대비 소폭의 성능 향상에 그친다는 문제점을 확인하였다.
4. 이를 해결하기 위해 저자들은 **다양성 인식 데이터 획득(Diversity-Aware Data Acquisition, DADA)** 이라는 새로운 획득 함수를 제안한다.
5. DADA는 CNN의 딥 피처(deep feature)를 기반으로 패치를 클러스터링한 후, 각 클러스터에서 불확실성이 높은 샘플을 선택하여 다양성과 불확실성을 동시에 고려한다.
6. 또한, 동적 서브풀링(dynamic subpooling) 기법을 도입하여 대규모 패치 풀에서의 AL 수행 시간을 약 2.12배 단축하였다.
7. MNIST 및 TIL 데이터셋 실험에서 제안 방법(ENS DDBALD(2k))은 기존 최고 성능 방법(ENS BALD(100k)) 대비 약 1.8배 적은 패치(2,700개 vs 4,900개)로 동등한 최대 AUC에 도달하였다.
8. 무작위 샘플링이 동등한 AUC에 도달하기 위해서는 약 60,000개의 패치가 필요했음을 고려하면, DADA의 효율성은 매우 두드러진다.
9. 실험 결과, DADA는 병리 이미지 분석뿐만 아니라 MNIST와 같은 일반적인 딥러닝 태스크에서도 유효함을 보인다.
10. 본 연구는 CNN 기반 조직병리 이미지 분석에서 어노테이션 비용을 획기적으로 절감할 수 있는 실용적 프레임워크를 제시한다.

---

### 1-1. 연구의 목적과 필요성

**목적:** 전문 병리학자의 어노테이션 수요를 최소화하면서 디지털 병리학의 딥러닝 모델 성능을 효율적으로 달성하는 AL 전략 개발.

**필요성:**
- 전체 슬라이드 이미지(Whole Slide Image, WSI) 하나에는 수만 개의 이미지 패치가 포함되어 있어, 전문가의 전수 어노테이션은 현실적으로 불가능하다 (p.1).
- 기존 크라우드소싱, 합성 데이터 생성 접근법은 품질 관리, 도메인 특수성 등의 한계를 가진다 (p.1–2).
- 기존 AL 방법의 단순 적용은 유사한 형태의 패치를 반복 선택하는 문제로 인해 무작위 대비 미미한 성능 향상만을 보인다 (p.2, [18,19]).

> **💡 용어 설명**
> - **어노테이션(Annotation):** 학습 데이터에 정답 레이블(TIL 양성/음성 등)을 전문가가 직접 표시하는 작업.
> - **WSI(Whole Slide Image):** 조직 슬라이드 전체를 고해상도로 스캔한 디지털 병리 이미지.
> - **TIL(Tumor Infiltrating Lymphocytes):** 종양 조직에 침윤한 림프구. 밀도가 높을수록 환자 예후와 양의 상관관계가 있음.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | 기존 AL(BALD, CoreSet)을 디지털 병리에 직접 적용하면 무작위 대비 성능 향상이 미미함 | ENS BALD, CoreSet 모두 약 2,100패치 이후에야 무작위 대비 높은 AUC 달성 | Fig. 4, Section 3.2.1 |
| 2 | 불확실성 기반 AL은 유사한 형태학적 특성의 패치를 중복 선택하는 문제 발생 | Fig. 2(a): Ensemble+BALD 선택 패치들이 시각적으로 유사함 | Fig. 2(a), p.3 |
| 3 | DADA는 클러스터 기반 다양성 확보로 AL 수렴 속도를 향상시킴 | ENS DDBALD(2k)가 1,700패치에서 AUC 0.78 달성 vs ENS BALD(100k) 최대 0.73 | Fig. 5, Section 3.2.2 |
| 4 | 동적 서브풀링은 AL 실행 시간을 단축하면서 AUC도 개선함 | ENS DDBALD(2k)가 ENS DDBALD(100k) 대비 2.12배 빠름 | Fig. 10, Section 3.2.4 |
| 5 | DADA는 병리 외 일반 분야(MNIST)에서도 유효함 | ENS DDVR이 59,500 이미지 없이 680개로 0.98 정확도 달성 | Section 3.1, Table 3 |
| 6 | 앙상블(Ensemble) 방식이 MC-Dropout 방식보다 TIL 분류에서 우수하고 빠름 | ENS BALD가 MC BALD 대비 약 1.92배 빠름 | Fig. 10, Section 3.2.4 |

---

## 2-1. 상세 분석

### 해결하고자 하는 문제

디지털 병리학에서 딥러닝 모델 학습 시 대규모 전문가 어노테이션이 필요한 문제. 특히:
1. **수량 문제:** 단일 WSI에 수만 개의 패치 존재 → 전수 어노테이션 불가능.
2. **질적 문제 (다양성 부재):** 기존 불확실성 기반 AL이 유사한 시각적 특성을 가진 패치를 반복 선택 → 훈련 데이터의 다양성 저하.
3. **계산 비용 문제:** 대규모 풀에서 매 AL 반복마다 전체 풀에 대한 불확실성 계산 → 높은 계산 비용.

---

### 제안하는 방법 (수식 포함)

#### ① 풀 기반 능동 학습 (Pool-Based Active Learning)

반복 $i$에서 모델 $M$은 레이블된 집합 $L^i$로 학습되고, 획득 함수 $a(U^i, M)$를 통해 $s$개의 샘플 집합 $A^i$를 선택:

$$L^{i+1} = A^i \cup L^i, \quad U^{i+1} = U^i - A^i$$

> **💡 용어 설명**
> - $U^i$: $i$번째 반복에서의 비레이블 데이터 풀 (unlabeled pool)
> - $L^i$: $i$번째 반복에서의 레이블된 학습 데이터셋
> - $A^i$: $i$번째 반복에서 선택된 샘플 집합
> - 획득 함수(acquisition function): 어떤 샘플을 어노테이션할지 결정하는 함수

---

#### ② MC-Dropout(Bayesian) 불확실성 추정 (Eq. 1)

드롭아웃을 학습 및 테스트 시 적용하여 클래스 $c$에 대한 예측 확률 근사:

$$p(y = c \mid x, \mathcal{D}_{train}) = \int p(y = c \mid x, \omega) p(\omega \mid \mathcal{D}_{train}) d\omega$$

$$\approx \int p(y = c \mid x, \omega) q^*_\theta(\omega) d\omega \approx \frac{1}{F} \sum_{f=1}^{F} p(y = c \mid x, \hat{\omega}_f)$$

- $\omega$: 모델 가중치(weights)
- $\hat{\omega}\_f \sim q^*_\theta(\omega)$: 드롭아웃 분포에서 샘플링된 가중치
- $q^*_\theta(\omega)$: 드롭아웃 분포 (variational 근사)
- $F$: 전방 패스(forward pass) 횟수
- $p(y = c \mid x, \hat{\omega}_f)$: $f$번째 패스에서의 소프트맥스(softmax) 출력

> **💡 용어 설명**
> - **MC-Dropout:** 테스트 시에도 드롭아웃을 켜두어 여러 번 예측을 수행하고, 그 분산으로 불확실성을 추정하는 베이지안 근사 기법.
> - **Softmax:** 다중 클래스 분류에서 각 클래스에 대한 확률을 출력하는 활성화 함수.

---

#### ③ BALD (Bayesian Active Learning by Disagreement)

예측과 모델 사후 분포 사이의 상호 정보를 최대화하는 샘플 선택:

$$\mathcal{I}[y, \omega \mid x, \mathcal{D}_{train}] = \mathbb{H}[y \mid x, \mathcal{D}_{train}] - \mathbb{E}_{p(\omega \mid \mathcal{D}_{train})}[\mathbb{H}[y \mid x, \omega]]$$

최대 엔트로피:

$$\mathbb{H}[y \mid x, \mathcal{D}_{train}] = -\sum_c p(y = c \mid x, \mathcal{D}_{train}) \log p(y = c \mid x, \mathcal{D}_{train})$$

> **💡 용어 설명**
> - **상호 정보(Mutual Information):** 모델 예측과 가중치 사후 분포 간의 정보 공유량. 이 값이 클수록 모델이 해당 샘플에 대해 "확신하면서도 의견이 다른" 상태, 즉 정보 가치가 높음.
> - **엔트로피(Entropy):** 예측의 불확실성 척도. 높을수록 더 불확실.

---

#### ④ Variation Ratios (VR)

가장 빈번한 예측 클래스 $c^*$에 대한 불확실성:

$$VR[x] = 1 - p(y = c^* \mid x, \mathcal{D}_{train})$$

$$c^* = \underset{c=1,\ldots,C}{\arg\max} \; p(y = c \mid x, \mathcal{D}_{train})$$

---

#### ⑤ CoreSet (기하학적 접근)

선택되지 않은 점 $u$와 선택된 집합 $L^i$ 내 최근접 이웃 $l$ 간 최대 거리를 최소화하는 샘플 $p$ 선택:

$$p = \underset{u \in U^i}{\arg\max} \; \underset{l \in L^i}{\min} \; \text{dist}(u, l)$$

> **💡 용어 설명**
> - **CoreSet:** 집합 커버 문제(set cover problem)에 기반한 기하학적 AL 방법. 전체 데이터셋을 가장 잘 대표하는 샘플을 선택.

---

#### ⑥ DADA의 클러스터별 기여 비율 (PR)

클러스터 $i$의 평균 불확실성을 $Z(i)$라 할 때, 클러스터 $i$의 기여 비율:

$$PR(i) = \frac{Z(i)}{\sum_K Z(i)}$$

클러스터 $i$에서 선택할 패치 수:

$$\text{acqSize}_i = PR(i) \times Q$$

- $Q$: 한 획득 단계에서 선택할 총 패치 수
- $K$: 클러스터 수

---

#### ⑦ 동적 서브풀링 (Subpooling)

재생성 주기 $\alpha$마다 원본 풀 $U^i$에서 서브풀 $S^i_U$ 재샘플링:

$$S^{\alpha}_U \subset U^i$$

각 획득 단계 후 서브풀 업데이트:

$$S^{i+1}_U = S^i_U - A^i, \quad U^{i+1} = U^i - A^i$$

---

### 모델 구조

| 구성 요소 | MNIST | TIL |
|-----------|-------|-----|
| CNN 모델 | Keras+2 (4 conv layers + ReLU) | Inception V4 |
| 딥 피처 차원 | 7,744 | 1,536 (Bayesian) / 4,608 (Ensemble) |
| PCA 출력 차원 | - | 50 |
| 클러스터 수 (K) | - | 20 |
| 옵티마이저 | Adadelta (lr=0.0005) | Adam (lr=0.00005) |
| 배치 크기 | 128 | 62 |
| 서브풀 재생성 주기 (α) | - | 2 |
| 훈련 에포크 | 50 | 50 |

> **💡 용어 설명**
> - **Inception V4:** Google이 개발한 고성능 CNN 아키텍처. 다양한 크기의 필터를 병렬로 적용하는 Inception 모듈로 구성.
> - **PCA(Principal Component Analysis):** 고차원 데이터를 저차원으로 선형 변환하여 주요 변동 방향을 보존하는 차원 축소 기법.
> - **Mini-batch K-Means:** 대규모 데이터에서 효율적인 클러스터링을 위해 전체 데이터 대신 미니배치를 사용하는 K-평균 알고리즘의 변형.

---

### 성능 향상 및 한계

**성능 향상:**
- ENS DDBALD(2k)는 2,700패치로 최대 AUC(0.78) 달성 vs. ENS BALD(100k)는 4,900패치 필요 (약 **1.8배 효율**) (Section 3.2.3, Fig. 6)
- 무작위 샘플링은 60,000패치에서도 최대 AUC 0.76으로 DADA 미만 (Fig. S5)
- 실행 시간: ENS DDBALD(2k)는 ENS BALD(100k) 대비 1.21배, ENS DDBALD(100k) 대비 2.12배 빠름 (Fig. 10)
- MNIST에서 ENS DDVR은 680개 이미지로 0.98 정확도 달성 (전체 59,500개 사용 시 0.99) (Section 3.1)

**한계:**
- 평가에 사용된 CNN 모델이 두 가지(TIL용 InceptionV4, MNIST용 Keras+2)로 제한됨 (Discussion, p.8)
- TIL 데이터셋이 62개의 WSI(10가지 암 유형)로 한정되어 더 넓은 일반화 검증 부재 (Section 3, p.5)
- AL 실행 시간이 현대 GPU(NVIDIA V100)에서도 40시간으로 여전히 상당함 (Section 3.2.4)
- 클래스 불균형(TIL 양성 16%) 문제에 대한 별도의 처리 전략 미제시 (Section 3, p.5)
- 사전 훈련된 가중치(pre-trained weights) 미사용으로 인한 잠재적 성능 손실 (Section 3, p.5)

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| 기존 AL은 유사 패치를 반복 선택 | p.2 (Section 2.3), Fig. 2(a) |
| DADA 알고리즘 설계 | p.3–4, Algorithm 1 |
| DADA가 AUC 개선 | p.6, Fig. 5, Section 3.2.2 |
| ENS DDBALD(2k) vs ENS BALD(100k) CI 비교 | p.6, Fig. 6 |
| 서브풀링의 시간 단축 효과 | p.8, Fig. 10, Section 3.2.4 |
| DADA 선택 패치의 공간적 다양성 | p.7, Fig. 8 |
| 불확실성 분포 비교 | p.7, Fig. 7 |
| MNIST에서의 DADA 효과 | p.5, Fig. 3, Table 3 |
| VR 함수의 태스크 의존성 | p.6, Fig. 4, Section 3.2.1 |
| AUC 플래토(plateau) 원인 분석 | p.6–7, Fig. S5 |

---

## 4. 저자 보고 결과 vs. 독자 해석

### 저자가 직접 보고한 결과

**연구 주제:**
- TIL 분류에서의 효과적 AL 전략 개발 (Abstract, p.1)

**방법:**
- DADA: 딥 피처 기반 클러스터링 + 클러스터 내 불확실성 기반 샘플 선택 (Section 2.3)
- 동적 서브풀링: 매 $\alpha$스텝마다 서브풀 재생성 (Section 2.4)

**결과:**
- "our method (ENS DDBALD(2k)), the state-of-the-art (ENS BALD(100k)), and random sampling reached near the same maximum AUC value by acquiring 2700, 4900, and 60,000 tissue patches" (p.2)
- "ENS DDBALD(2k) is 1.21× and 2.12× faster than ENS BALD(100k) and ENS DDBALD(100k), respectively" (p.8)
- "ENS DDBALD(2k) achieved an average AUC of 0.75 over time compared to 0.70 by ENS BALD and 0.71 by CoreSet" (p.6)

---

### 검토자(독자)의 해석

1. **DADA의 핵심 성공 요인:** 단순히 불확실성이 높은 샘플만 선택하는 것이 아닌, **불확실성 + 다양성의 균형**을 구현한 것. 이는 기존 AL이 디지털 병리에서 실패했던 근본 원인(형태학적 편향)을 직접 해결한다.

2. **서브풀링의 이중 효과:** 서브풀링이 단순 시간 단축이 아닌 **AUC 향상**도 가져왔다는 점(ENS DDBALD(2k) > ENS DDBALD(100k))은 흥미롭다. 이는 소규모 서브풀이 오히려 다양한 영역을 순차적으로 탐색하게 만드는 정규화 효과를 가질 수 있음을 시사한다.

3. **VR 함수의 태스크 의존성 문제:** MNIST에서는 VR이 최선이었지만 TIL에서는 BALD가 우수했다. 이는 획득 함수 선택이 도메인에 따라 달라야 함을 의미하며, 실제 임상 적용 시 함수 선택을 위한 검증 데이터 필요성을 시사한다.

4. **클래스 불균형 처리의 부재:** TIL 양성 패치가 16%에 불과한 불균형 데이터셋에서, DADA가 다양성을 위해 불확실성이 낮은 클러스터에서도 샘플링함으로써 음성 패치 비율을 높이는 효과(Fig. 9)는 불균형 문제 완화에 기여했을 가능성이 있다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| ⚠️ 항목 | 문제점 |
|---------|--------|
| **Fig. 6 신뢰구간 실험이 5회 반복만 수행** | n=5로 통계적 신뢰도가 낮음. 충분한 통계 검증(t-test, Wilcoxon 등) 미제시 |
| **TIL 데이터셋이 62개 WSI, 10개 암 유형** | 각 암 유형별 샘플 수가 매우 적을 가능성. 암 유형별 성능 분해 미제시 |
| **MNIST와 TIL 비교** | 완전히 다른 도메인으로 방법론의 일반화 주장 근거로 사용하기에는 부족 |
| **AUC 최대값이 0.78로 제한적** | 임상 적용 기준에서 낮은 수준이나, 이에 대한 임상적 해석 미제시 |
| **실행 시간 40시간 제한** | 일부 방법이 시간 제한으로 완전히 수렴하지 못했을 가능성 (MC는 3,100패치만 획득) |
| **사전 훈련 가중치 미사용** | 실용적 시나리오(전이학습 사용)와 다른 환경에서의 실험 |
| **CoreSet 비교 조건** | CoreSet은 greedy 버전만 사용 (optimal CoreSet은 NP-hard로 미사용) |

---

## 6. 논문이 답하지 않는 질문

1. **DADA의 클러스터 수(K=20) 선택 근거는?** K에 따른 성능 민감도 분석이 부록(Section S2)으로만 제시됨.
2. **다양한 암 유형별로 DADA 성능이 일정한가?** 10개 암 유형을 통합하여 평가하였으나 유형별 분석 미제시.
3. **AL이 진행됨에 따라 어노테이션의 난이도가 변화하는가?** 병리학자의 실제 어노테이션 부담 변화 분석 부재.
4. **전이학습(Transfer Learning) 적용 시 DADA의 효과는?** 모든 실험이 랜덤 초기화에서 시작.
5. **다른 병리학적 분류 태스크(예: 종양 분류, 세포 분할)에서의 성능은?** TIL 분류에만 집중.
6. **실제 임상 환경에서 병리학자의 AL 참여 워크플로우 통합 방법은?** 시스템 통합 관점 미제시.
7. **DADA의 PCA 대신 다른 차원 축소 기법(UMAP, t-SNE) 적용 시 성능은?** 언급만 되고 실험 미수행.
8. **클래스 불균형(16% 양성)에 대한 명시적 처리 없이 신뢰할 수 있는가?** 별도 불균형 처리 전략 미제시.

---

## 7. 가장 중요한 그림 5개 해석

### Fig. 1 — TIL 분석 워크플로우 (p.3)

```
[WSI] → [패치 추출] → [전문가 어노테이션] → [CNN 훈련] → [반복 검토]
                                                              ↓
[TIL 맵] ← [역치 기반 검토] ← [정확도 기반 그룹화] ← [훈련된 CNN]
```

**해석:** 상단은 AL 기반 모델 훈련, 하단은 훈련된 모델을 새로운 WSI에 적용하는 파이프라인. 이 그림은 AL이 어노테이션 루프에 어떻게 통합되는지를 보여주며, 병리학자의 검토가 반복적으로 발생하는 지점을 명시한다. AL은 상단 훈련 루프의 효율화에 핵심 역할을 한다.

---

### Fig. 2 — DADA 유무에 따른 선택 패치 시각적 비교 (p.4)

- **(a) DADA 없음 (ENS+BALD):** 6개 패치가 모두 유사한 조직 구조(핑크색 종양 조직)를 가짐 → 형태학적 다양성 낮음.
- **(b) DADA 있음:** 6개 패치가 각기 다른 조직 특성(림프구 밀집 영역, 섬유질 조직, 근육 조직 등)을 포함 → 다양성 확보.

**해석:** 이 그림은 논문의 핵심 동기를 가장 직관적으로 전달한다. 기존 방법의 병목이 "유사 패치의 반복 선택"임을 시각적으로 증명하며, DADA의 클러스터링 전략이 이를 실질적으로 해결함을 보여준다. 임상적 관점에서 다양한 조직 유형을 학습 데이터에 포함시키는 것이 모델 일반화에 필수적임을 시사한다.

---

### Fig. 5 — DADA 및 서브풀 크기에 따른 AUC 비교 (p.6)

**해석:** 
- ENS DDBALD(100k): 서브풀 없이 DADA만 적용 → ENS BALD(100k), CoreSet(100k) 대비 일관된 AUC 우위.
- ENS DDBALD(2k): 가장 작은 서브풀 → **모든 조건 중 최고 성능**. 이는 직관에 반하는 결과로, 소규모 서브풀이 오히려 탐색 다양성을 증가시키는 효과를 가짐.
- 서브풀 크기가 클수록 성능이 낮아지는 경향: 2k > 10k > 20k > 100k (DADA 조건).
- 이 결과는 서브풀링이 단순한 계산 최적화가 아닌, **샘플 선택 전략 자체를 개선**하는 역할을 함을 시사한다.

---

### Fig. 6 — ENS DDBALD(2k) vs ENS BALD vs CoreSet 신뢰구간 비교 (p.6)

**해석:**
- ENS DDBALD(2k)는 더 빠른 AUC 상승 곡선과 **좁은 신뢰구간** → 안정적이고 재현 가능한 학습.
- ENS BALD(100k)와 CoreSet(100k)은 더 넓은 신뢰구간 → 실행마다 성능 변동이 큼.
- 약 4,000패치에서 AUC 플래토 발생 → 데이터 자체의 한계 또는 충분한 정보 획득 가능성.

> ⚠️ **통계적 주의:** 5회 반복(n=5)으로 산출한 신뢰구간은 통계적 신뢰도가 낮을 수 있음. 공식 통계 검정(p-value 등) 미제시.

---

### Fig. 10 — AL 단계별 실행 시간 비교 (p.8)

**해석:**
- 각 막대의 하단(획득 시간)과 상단(훈련 시간)으로 분리하여 병목 지점 명시.
- MC-Dropout(베이지안)은 20회 전방 패스 → 앙상블의 3회 모델 평가보다 획득 시간이 압도적으로 많음.
- ENS DDBALD(100k)는 PCA + 클러스터링 오버헤드로 ENS BALD(100k)보다 1.75배 느림 → **DADA 자체는 계산 비용 증가.**
- ENS DDBALD(2k)는 서브풀링으로 이 오버헤드를 상쇄하고 ENS BALD(100k)보다 **1.21배 빠름**.
- CoreSet이 가장 빠르지만 AUC는 최저 → 속도-성능 트레이드오프 명확.

---

## 8. 결론: 시사점, 후속 연구 계획 및 추가 방향

### 저자들이 제시한 시사점

1. 불확실성과 다양성을 동시에 고려하는 AL 전략이 디지털 병리학에서 효과적임.
2. 동적 서브풀링이 AL의 계산 효율과 모델 성능을 동시에 개선.
3. 제안 방법이 병리 외 일반 도메인(MNIST)에서도 적용 가능.

### 저자들이 제시한 후속 연구 계획 (p.8, Discussion)

- 추가 조직병리 이미지 분석 응용으로 실험 확장.
- 그래픽 UI 및 이미지 뷰어가 포함된 AL 툴킷 개발.
- 다양한 CNN 모델(InceptionV4 외)로 평가 확장.
- AL 실행 시간 추가 최적화 연구.

---

### 8-1. 모델의 일반화 성능 향상 가능성

**현재 일반화 한계:**

| 한계 | 구체적 문제 |
|------|-------------|
| 단일 태스크 검증 | TIL 분류 한 가지만 평가 |
| 제한된 암 유형 | 10개 암 유형, 62개 WSI |
| 사전 학습 미사용 | 실제 임상 환경과 상이 |
| 단일 염색법 | H&E 염색만 평가 (IHC, IF 등 미포함) |

**일반화 성능 향상을 위한 제언:**

1. **전이학습 통합 (Transfer Learning):**
   - ImageNet 또는 병리 특화 사전 학습 모델(PathologyFoundation 모델 등)과 DADA 결합.
   - 적은 어노테이션으로 빠른 수렴 가능성.
   - 수식적으로: 사전 학습 피처 $f_\theta(x)$를 그대로 DADA의 딥 피처로 활용하여 PCA 입력으로 사용.

2. **멀티 도메인 AL:**
   - 여러 암 유형, 여러 센터의 WSI를 동시에 고려하는 도메인 적응 AL.
   - 도메인 간 피처 분포 차이를 클러스터링에서 명시적으로 고려.

3. **자기 지도 학습(Self-Supervised Learning)과 결합:**
   - 레이블 없이 딥 피처 사전 학습 후 DADA 적용.
   - 초기 AL 단계에서 더 풍부한 피처 공간 제공.

4. **연합 학습(Federated Learning)과 AL 통합:**
   - 여러 병원의 데이터를 직접 공유하지 않고 AL 수행.
   - 프라이버시 보호와 데이터 다양성 동시 확보.

5. **메타 학습(Meta-Learning) 기반 획득 함수:**
   - 새로운 도메인에 빠르게 적응하는 AL 전략 학습.
   - DADA의 클러스터 수(K)와 서브풀 크기(α)를 태스크별로 자동 최적화.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 고지:** 아래는 본 검토자가 알고 있는 2020년 이후 관련 연구 동향에 기반하며, 논문 원문에는 포함되지 않은 내용입니다. 일부 세부 수치는 확인이 필요할 수 있습니다.

| 연구 방향 | 대표 연구 | 본 논문과의 비교 |
|-----------|-----------|-----------------|
| **Foundation Model 기반 병리 분석** | UNI (Chen et al., 2024), CONCH (Lu et al., 2024) | 대규모 사전 학습으로 어노테이션 필요성 자체를 줄임. DADA는 이와 상보적 관계. |
| **약한 지도 학습(Weakly Supervised)** | CLAM (Lu et al., 2021), TransMIL | 슬라이드 수준 레이블만 사용. DADA의 패치 수준 AL과 접근법이 다름. |
| **자기 지도 + AL 결합** | LADA (Kim et al., 2023 계열 연구) | 사전 학습 없이 시작하는 DADA의 한계를 보완 가능. |
| **변환기(Transformer) 기반 병리 모델** | ViT-based pathology models (2021~) | DADA의 딥 피처가 ViT의 패치 임베딩으로 대체될 경우 성능 향상 가능성. |
| **생성 모델 기반 데이터 증강 + AL** | Diffusion-based augmentation (2022~) | 합성 데이터와 AL의 하이브리드 접근. |

**본 논문이 앞으로의 연구에 미치는 영향:**

1. **AL + 다양성 원칙의 확립:** DADA에서 제안된 "불확실성 + 다양성"의 이중 기준은 이후 많은 병리 AL 연구의 설계 원칙으로 자리잡을 가능성이 높음.
2. **서브풀링 아이디어의 확장:** 대규모 Foundation Model 기반 분석에서 계산 효율화를 위한 서브풀링 전략이 더욱 중요해질 것.
3. **평가 프레임워크:** TIL 데이터셋을 AL 벤치마크로 활용하는 선례 제공.

**앞으로 연구 시 고려할 점:**

1. **Foundation Model과의 통합:** 최신 병리 특화 대규모 모델(UNI, CONCH 등)의 피처를 DADA의 딥 피처로 활용할 경우, 초기 AL 단계에서 클러스터링 품질이 크게 향상될 것으로 예상.

2. **AUC 한계치(0.78) 돌파:** 더 큰 데이터셋, 더 나은 어노테이션 품질, 사전 학습 모델 활용이 필요.

3. **공정한 비교 기준 설정:** AL 방법 간 비교 시 동일한 어노테이션 예산(annotation budget), 동일한 초기 훈련 세트, 동일한 평가 지표를 표준화한 벤치마크 필요.

4. **임상 검증:** 실제 병리학자 참여 하에 AL 워크플로우의 임상적 효용성 및 사용성 평가 필요.

5. **설명 가능성(Explainability):** AL이 선택한 패치의 근거를 병리학자가 이해할 수 있도록 시각화 및 설명 도구 통합.

---

## 참고 자료

**논문 원문:**
- Meirelles, A.L.S., Kurc, T., Saltz, J., Teodoro, G. (2022). "Effective active learning in digital pathology: A case study in tumor infiltrating lymphocytes." *Computer Methods and Programs in Biomedicine*, 220, 106828. DOI: 10.1016/j.cmpb.2022.106828

**논문 내 주요 참고문헌:**
- [15] Gal, Y., Islam, R., Ghahramani, Z. (2017). "Deep Bayesian active learning with image data." *ICML 2017*.
- [16] Sener, O., Savarese, S. (2018). "Active learning for convolutional neural networks: a core-set approach." *ICLR 2018*.
- [17] Beluch, W.H., et al. (2018). "The power of ensembles for active learning in image classification." *CVPR 2018*.
- [18] Raczkowski, L., et al. (2019). "ARA: accurate, reliable and active histopathological image classification framework with Bayesian deep learning." *Scientific Reports*.
- [19] Carse, J., McKenna, S. (2019). "Active learning for patch-based digital pathology using CNNs to reduce annotation costs." *European Congress on Digital Pathology*.
- [21] Saltz, J., et al. (2018). "Spatial organization and molecular correlation of tumor-infiltrating lymphocytes using deep learning on pathology images." *Cell Reports*.
- [41] Sculley, D. (2010). "Web-scale k-means clustering." *WWW 2010*.
- [43] Szegedy, C., et al. (2017). "Inception-v4, inception-resnet." *AAAI 2017*.
