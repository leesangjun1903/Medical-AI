# Terabyte-scale Deep Multiple Instance Learning for Classification and Localization in Pathology

> **참고 자료**: Campanella, G., Silva, V. W. K., & Fuchs, T. J. (2018). *Terabyte-scale Deep Multiple Instance Learning for Classification and Localization in Pathology*. arXiv:1805.06983v2.

---

## 1. Executive Summary (10문장 이내)

본 논문은 디지털 병리학에서 대규모 레이블 데이터 없이 전립선암을 자동 진단하는 딥러닝 프레임워크를 제안한다.  
핵심 아이디어는 **Multiple Instance Learning(MIL)** 패러다임으로, 슬라이드 전체 진단 레이블만으로 CNN을 학습시킨다.  
데이터셋은 12,160개의 Whole Slide Image(WSI)로, 이전 연구보다 약 100배 큰 규모이며 ImageNet의 25배에 해당하는 픽셀 수를 보유한다.  
WSI는 타일로 분할되어 각 타일이 하나의 인스턴스(instance)로 처리되고, 가장 높은 양성 확률을 가진 타일이 학습 신호로 활용된다.  
VGG11-BN과 ResNet34 모델이 각각 AUC 0.977, 0.976을 달성했고, 멀티스케일 앙상블 시 AUC 0.979에 도달했다.  
데이터셋 크기가 MIL 성능의 결정적 요인임을 실험적으로 입증했으며, 약 6,000개 이상의 WSI가 있어야 안정적인 일반화가 가능함을 보였다.  
특징 임베딩 시각화(PCA)를 통해 모델이 악성/양성 조직을 의미론적으로 구분하는 표현을 학습함을 확인했다.  
픽셀 단위 어노테이션 없이도 종양 위치 특정(localization)이 가능하다는 점에서 임상 배포 가능성을 열었다.  
본 연구는 병리학 AI의 실제 임상 적용을 위한 방법론적 토대를 마련한 선구적 연구다.

---

### 1-1. 연구의 목적과 필요성

| 구분 | 내용 |
|------|------|
| **문제 상황** | 기존 병리학 AI는 수백 장 수준의 소규모 데이터셋에 의존 → 임상 적용 불가 |
| **기술적 장벽** | 픽셀 단위 어노테이션은 비용과 시간이 과도하게 소요됨 |
| **임상적 필요성** | 전립선암은 관찰자 간 진단 일치율이 낮고 미세 병변 탐지가 어려움 (p.2) |
| **연구 목적** | 슬라이드 수준의 레이블만으로 임상 수준의 진단 모델 훈련 가능성 증명 |
| **기여** | 12,160 WSI 규모 데이터셋 구축 + 약지도(weakly supervised) MIL 프레임워크 제안 |

> 💡 **WSI(Whole Slide Image)**: 병리 슬라이드를 디지털 스캐너로 전체 스캔한 수 기가픽셀 이미지. 하나의 이미지가 수십~수백 GB에 달하기도 함.

> 💡 **약지도학습(Weakly Supervised Learning)**: 픽셀/영역별 세밀한 레이블 없이 슬라이드 전체의 진단명처럼 '거친' 레이블만 사용하는 학습 방식.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거/실험 | 위치 |
|---|-----------|-----------|------|
| 1 | MIL 프레임워크로 슬라이드 수준 레이블만으로 고성능 모델 훈련 가능 | AUC 0.977 (VGG11-BN, test set) | p.7, Fig. 6a |
| 2 | **데이터셋 크기**가 MIL 성능의 결정적 요인 | 100~8,000 슬라이드 실험, 오류율 단조 감소 | p.6, Fig. 4 |
| 3 | 높은 배율(20x)이 낮은 배율(5x, 10x)보다 우수한 성능 | 배율별 비교 실험 | p.9, Fig. 6b, Fig. F.1 |
| 4 | 멀티스케일 앙상블이 단일 배율보다 우수 | Max-pooling 앙상블: AUC 0.979 | p.9, Fig. 6b |
| 5 | 데이터 증강(회전/뒤집기)은 성능 향상에 기여하지 않음 | ResNet34 증강 vs 비증강 비교 | p.9, Fig. E.1 |
| 6 | 모델이 의미론적 특징을 학습함 (악성/양성 조직 구분) | PCA 특징 임베딩 시각화 | p.7, Fig. 5 |
| 7 | 양성 클래스에 더 높은 가중치($w_1=0.9$) 부여 시 최적 | 가중치 0.5~0.99 비교 실험 | p.6, Fig. B.1 |

---

### 2-1. 해결 문제 / 제안 방법 / 모델 구조 / 성능 / 한계 상세 설명

#### 🔴 해결하고자 하는 문제

- **픽셀 단위 어노테이션의 비효율성**: 전문가가 WSI에서 종양 영역을 직접 그리는 작업은 수십~수백 시간 소요
- **소규모 데이터의 일반화 한계**: CAMELYON16 최대 400 WSI → 실제 임상 분포를 대표하지 못함
- **전립선 바늘 생검의 복잡성**: 병변이 슬라이드 표면의 1% 미만에 불과할 수 있음 (p.2, Fig. 1)

#### 🟢 제안하는 방법 (수식 포함)

**① MIL 형식화 (p.3~4)**

각 슬라이드 $s_i$를 **백(bag)**으로 모델링:

$$S = \{s_i : i = 1, 2, \ldots, n\}$$

$$B_{s_i} = \{b_{i,1},\ b_{i,2},\ \ldots,\ b_{i,m}\}$$

- $S$: 전체 슬라이드 풀
- $B_{s_i}$: 슬라이드 $s_i$의 백 (m개의 타일 인스턴스 포함)
- $b_{i,j}$: 슬라이드 $i$의 $j$번째 타일

> 💡 **백(Bag)**: MIL에서 여러 인스턴스(타일)의 집합. 슬라이드 1장 = 1개의 백. 백의 레이블만 알고, 개별 인스턴스의 레이블은 모름.

**② 최고 순위 인스턴스 선택 (p.4)**

모델 $f_\theta$가 각 타일에 양성 확률을 부여하고, 슬라이드 내 가장 높은 확률의 타일 인덱스를 선택:

$$k_i = \underset{j}{\arg\max}(\overrightarrow{o_i})$$

- $f_\theta$: 파라미터 $\theta$를 가진 CNN (예: VGG11-BN, ResNet34)
- $\overrightarrow{o_i}$: 슬라이드 $s_i$ 내 모든 타일의 양성 확률 벡터
- $k_i$: 슬라이드 $i$에서 가장 높은 양성 확률을 가진 타일의 인덱스
- $b_{i,k}$: 최고 순위 타일 (학습에 사용되는 인스턴스)

**③ 가중 크로스 엔트로피 손실 (Equation 1, p.5)**

$$l = -w_1[y_i \log(\tilde{y}_i)] - w_0[(1 - y_i)\log(1 - \tilde{y}_i)]$$

- $l$: 손실(loss) 값
- $y_i$: 슬라이드 $s_i$의 실제 레이블 (양성=1, 음성=0)
- $\tilde{y}\_i = f_\theta(b_{i,k})$: 최고 순위 타일에 대한 모델의 양성 예측 확률
- $w_1$: 양성 클래스 가중치 (실험 결과 0.9가 최적)
- $w_0 = 1 - w_1$: 음성 클래스 가중치
- **클래스 불균형** (양성 19.9% vs 음성 80.1%)을 보정하기 위해 가중치 적용

> 💡 **크로스 엔트로피 손실(Cross-Entropy Loss)**: 모델 예측 확률과 실제 레이블 간의 차이를 정량화하는 손실 함수. 이진 분류에서 자주 사용됨.

> 💡 **클래스 불균형(Class Imbalance)**: 데이터에서 한 클래스(음성)가 다른 클래스(양성)보다 훨씬 많은 상황. 보정하지 않으면 모델이 다수 클래스만 예측하는 편향이 생김.

**④ 4단계 학습 파이프라인 (p.4, Fig. 3)**

```
각 에폭마다:
1단계: 슬라이드 타일링 (배경 제거 포함)
  ↓
2단계: 전체 데이터 추론 (inference pass)
  ↓
3단계: 슬라이드 내 타일 순위 매기기 (ranking by positive probability)
  ↓
4단계: 최고 순위 타일로 모델 학습 (top-1 ranked instance)
```

#### 🔵 모델 구조

| 구성 요소 | 상세 |
|-----------|------|
| **백본(Backbone)** | AlexNet, VGG11-BN, ResNet18, ResNet34 (ImageNet 사전학습) |
| **입력** | WSI에서 추출한 타일 (5x, 10x, 20x 배율) |
| **출력** | 타일별 양성/음성 확률 |
| **최적화** | Adam optimizer, 학습률 0.0001 |
| **미니배치 크기** | AlexNet: 512, ResNet: 256, VGG: 128 |
| **하드웨어** | 7× NVIDIA DGX-1 (각 8× V100 Volta GPU) |
| **앙상블** | 다중 배율 모델의 max-pooling |

> 💡 **사전학습(Pre-training)**: ImageNet 데이터셋으로 미리 학습된 모델의 가중치를 초기값으로 사용하는 기법. 의료 이미지처럼 레이블이 적은 분야에서 효과적.

> 💡 **ResNet(Residual Network)**: 잔차 연결(skip connection)을 도입하여 매우 깊은 네트워크 학습을 가능케 한 CNN 구조. ResNet34는 34개 레이어를 가짐.

#### 🟡 성능 향상

| 실험 조건 | AUC | Balanced Error | FNR |
|-----------|-----|---------------|-----|
| AlexNet 20x | 0.925 | - | - |
| ResNet18 20x | 0.955 | - | - |
| ResNet34 20x | 0.976 | ~7% | - |
| VGG11-BN 20x | **0.977** | ~7% | 9.4% |
| Ensemble (max) | **0.979** | 5.8% | **4.8%** |

> 💡 **AUC(Area Under the ROC Curve)**: ROC 곡선 아래 면적. 1에 가까울수록 분류 성능이 우수함. 0.5는 랜덤 분류기 수준.

> 💡 **Balanced Error**: 양성/음성 클래스의 오류율을 균등하게 평균한 지표. 클래스 불균형 상황에서 더 신뢰할 수 있는 성능 지표.

> 💡 **FNR(False Negative Rate, 위음성률)**: 실제 양성인데 음성으로 잘못 예측한 비율. 암 진단에서 특히 중요한 지표 (놓친 암 환자 비율).

#### 🔴 한계

1. **단일 기관 데이터**: MSKCC(Memorial Sloan Kettering Cancer Center) 데이터만 사용 → 외부 기관 일반화 검증 없음
2. **단일 암 유형**: 전립선 바늘 생검에만 적용, 타 암종으로의 전이 가능성 미검증
3. **Top-1 instance 방식의 한계**: 슬라이드당 1개 타일만 학습에 사용 → 정보 손실 가능성
4. **Gleason 점수 미예측**: 암 유무만 판별하며, 임상적으로 중요한 등급(grade) 정보 미제공
5. **데이터 증강 실험 규모**: ResNet34 1개 모델로만 테스트 (소규모 실험)
6. **설명 가능성(XAI) 부족**: PCA 시각화 외 세밀한 모델 해석 부재

---

## 3. 각 주장에 페이지/Figure 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| 12,160 WSI 데이터셋 구축 | p.1 (Abstract), p.3 (Section 3) |
| MIL 파이프라인 제안 | p.3~5 (Section 4), **Fig. 3** |
| 데이터셋 크기 = 성능 결정 요인 | p.6 (Section 5), **Fig. 4**, **Fig. C.1** |
| 최적 양성 가중치 $w_1=0.9$ | p.6, **Fig. B.1** |
| VGG11-BN AUC 0.977 달성 | p.7, **Fig. 6a** |
| 앙상블 AUC 0.979 (최고) | p.9, **Fig. 6b** |
| 20x > 10x > 5x 성능 순서 | p.9, **Fig. F.1**, **Fig. 6b** |
| 증강 효과 없음 | p.9, **Fig. E.1** |
| PCA 의미론적 특징 학습 확인 | p.7~8, **Fig. 5** |
| 오류 분석 (FP 55개, FN 33개) | p.7 (Error Analysis) |

---

## 4. 저자 보고 결과 vs. 내 해석 분리

### 4-1. 저자가 직접 보고한 결과

- **AUC**: VGG11-BN 0.977, ResNet34 0.976, 앙상블(max) 0.979 (p.7, p.9)
- **오류율**: FP 55개(3.7%), FN 33개(9.4%) — VGG11-BN 기준 (p.7)
- **데이터셋**: 총 12,160 슬라이드, 양성 2,424 / 음성 9,736 (p.3)
- **최적 가중치**: $w_1 = 0.9$ (p.6)
- **증강 효과**: 없음 — 저자 결론 (p.9)
- **특징 임베딩**: 악성/양성 조직이 PCA 공간에서 명확히 분리됨 (p.7)

### 4-2. 내 해석 및 평가

> ⚠️ 아래는 논문 내용에 근거한 **분석자의 해석**으로, 저자의 직접 주장이 아닙니다.

- **데이터 스케일 효과의 포화 여부**: Fig. 4에서 8,000개 이상에서도 성능 향상이 지속되는 추세가 보임. 저자는 포화 여부를 명시하지 않았으나, 더 많은 데이터가 여전히 유익할 가능성이 높음.
- **증강 실험의 한계**: 오직 ResNet34 단일 모델, 소규모 실험 → 결론의 통계적 강건성 낮음. 더 정교한 증강(stain normalization, color jitter 등)은 미탐색.
- **Top-1 MIL의 잠재적 비효율**: 슬라이드당 수백~수천 타일 중 1개만 학습에 사용 — 정보 활용 효율이 낮을 수 있으며, attention-based MIL 등 더 정교한 집계 방식이 성능을 향상시킬 수 있음.
- **외부 검증 부재**: 단일 기관 데이터로 훈련/검증/테스트 모두 수행 → 진정한 임상 일반화 성능은 미지수.
- **오류 분석의 임상적 의의**: FN 중 6개는 전문 병리학자도 암을 발견하지 못함 → 일부 FN은 데이터 레이블 자체의 노이즈일 가능성 존재 (레이블 불확실성 문제).

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 문제점 |
|------|--------|
| ⚠️ **증강 실험** | ResNet34 단 1개 아키텍처, 소수 실험 횟수 → 일반화 불가 (Fig. E.1) |
| ⚠️ **모델 비교 실험** | 각 5회 반복 — 통계적 유의성 검정(p-value, CI) 미제시 |
| ⚠️ **AUC 소수점 이하 차이** | VGG11-BN 0.977 vs ResNet34 0.976 — 오차 범위 내 차이일 수 있음 |
| ⚠️ **앙상블 AUC 0.9793** | 단일 테스트셋 평가 — 신뢰구간 미제시 |
| 🚫 **타 연구와 비교 불가** | 동일 전립선 바늘 생검 데이터로 비교한 연구 없음 → 절대 수치 비교 불가 |
| 🚫 **CAMELYON16과 비교 불가** | 완전히 다른 암종 + 다른 레이블 수준 + 다른 데이터셋 → 직접 수치 비교 부적절 |
| ⚠️ **데이터셋 크기 실험** | 검증 세트 2,000개가 훈련 세트와 동일 분포 — 진정한 외부 검증 아님 |
| ⚠️ **클래스 불균형** | 양성 19.9%로 Balanced Error를 주요 지표로 쓰나, 임상 기준 민감도/특이도 임계값 미제시 |

---

## 6. 논문이 답하지 않는 질문

1. **다른 기관(외부) 데이터에서의 성능은?** — 단일 기관(MSKCC) 내부 데이터만 사용
2. **Gleason 점수 등 암 등급 예측이 가능한가?** — 이진 분류(암 유무)에만 집중
3. **Top-1 MIL 방식이 최적인가?** — Top-k, attention-based 집계 방식과 비교 없음
4. **다른 암종(유방암, 폐암 등)에도 동일 접근이 유효한가?** — 전립선암만 실험
5. **임상 배포 시 병리학자와의 협업 프로토콜은?** — 결론에서 필요성만 언급
6. **모델의 불확실성 정량화(uncertainty quantification)는 가능한가?** — 미탐색
7. **염색 방법(H&E 외) 또는 다중 염색 슬라이드에도 적용 가능한가?** — 미언급
8. **훈련/추론에 소요되는 총 계산 비용(시간, 비용)은?** — 하드웨어 스펙만 언급
9. **레이블 노이즈(오진 포함)가 성능에 미치는 영향은?** — FN 6개 분석에서 시사만 됨
10. **데이터 증강의 종류(stain normalization, color jitter 등)에 따른 효과 차이는?** — 회전/뒤집기만 실험

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.3) — 전립선암 진단의 어려움

**해석**: 63,744 × 28,649 픽셀 크기의 슬라이드에서 암 진단의 근거가 되는 종양 선(gland)이 약 300픽셀 영역에 불과한 6개에 그침. 이는 전체 조직 면적의 약 1% 미만. 이 그림은 왜 MIL이 필요한지를 직관적으로 설명: 미세한 병변 위치를 알 수 없어도(레이블 없음), 슬라이드 전체 진단만으로 학습해야 하는 현실적 제약을 시각화함.

---

### Figure 3 (p.5) — MIL 파이프라인 개요

**해석**: 제안된 방법의 전체 흐름을 보여주는 핵심 그림.
1. 슬라이드(백) → 타일(인스턴스) 분해
2. 현재 모델로 모든 타일에 대해 추론
3. 종양 확률 기준 순위 정렬
4. **최고 순위 타일 1개**만 크로스엔트로피 손실로 역전파

이 설계는 계산 효율성과 MIL 가정(양성 백에는 적어도 1개의 양성 인스턴스 존재)을 동시에 만족시키는 실용적 접근법.

---

### Figure 4 (p.6) — 데이터셋 크기와 MIL 성능의 관계

**해석**: x축 로그 스케일에서 훈련 WSI 수 증가에 따라 최소 밸런스 검증 오류가 단조 감소. 약 100~500 슬라이드(이전 연구 규모)에서는 오류율 0.4~0.5 수준으로 실용 불가. 2,000개 이상에서 가파른 개선, 8,000개 수준에서 약 0.17까지 감소. **이 그림의 핵심 메시지**: "기존 소규모 데이터로는 MIL 기반 WSI 분류가 근본적으로 불가능하다." 향후 연구에서 데이터 수집 규모 기준을 최소 수천 장으로 설정해야 함을 시사.

---

### Figure 5 (p.8) — PCA 특징 임베딩 시각화

**해석**: ResNet34 분류 레이어 직전의 특징 벡터를 PCA로 2D 투영. 주요 관찰:
- **PC1축**: 스트로마(결합 조직) 타일이 왼쪽에 밀집 → 배경 역할
- **PC2축**: 위→아래 방향으로 양성 선(악성)~음성 선(양성) 순서로 분리
- **비회색 점(top-ranked 타일)**: 양성 슬라이드의 top-ranked 타일은 악성 선 영역에, 음성 슬라이드의 top-ranked 타일은 가장자리/잉크 영역에 위치
- **의의**: 픽셀 레이블 없이 슬라이드 레이블만으로 훈련했음에도, 모델이 병리학적으로 의미 있는 특징 공간을 자발적으로 학습했음을 증명.

> 💡 **PCA(Principal Component Analysis)**: 고차원 데이터를 분산이 큰 방향 순서로 저차원으로 압축하는 선형 차원 축소 기법. 시각화에 자주 사용됨.

---

### Figure 6 (p.9) — 테스트 세트 ROC 곡선

**해석**:
- **(a) 모델 비교**: VGG11-BN > ResNet34 > ResNet18 > AlexNet 순서. 상위 두 모델의 AUC 차이(0.977 vs 0.976)는 실질적으로 동등. AlexNet과의 격차(0.977 vs 0.925)는 모델 용량의 중요성을 시사.
- **(b) 배율 비교 + 앙상블**: 단일 배율에서 20x > 10x > 5x. 멀티스케일 앙상블(max-pooling)이 모든 단일 모델을 상회 → 다양한 해상도의 보완적 정보가 존재함. 임상적으로 고배율(세포 형태)과 저배율(구조적 패턴) 정보를 동시에 활용하는 것이 병리학자의 진단 방식과 일치.

> 💡 **ROC 곡선**: 분류 임계값을 변화시킬 때 민감도(TPR)와 1-특이도(FPR)의 관계를 나타낸 곡선. 왼쪽 상단에 가까울수록 성능이 좋음.

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 시사점 (p.9~10)**:
- 슬라이드 수준 레이블만으로 임상 수준의 진단 모델 훈련 가능함을 증명
- 대규모 데이터셋이 MIL 성능의 전제 조건임을 실증적으로 확인
- 병리학 디지털화 흐름과 결합 시 의사결정 지원 시스템으로 실제 임상 배포 가능성 제시

**저자가 직접 언급한 후속 연구 방향**: 논문 내 명시적 후속 연구 계획 없음 (저자 결론부에서 임상 배포 가능성만 언급)

---

### 8-1. 모델 일반화 성능 향상 가능성

이 논문에서 **일반화 성능**과 관련된 핵심 발견:

1. **데이터 규모가 일반화의 핵심 변수**: Fig. 4에서 8,000개 WSI에서도 성능이 포화되지 않음 → 더 많은 다기관 데이터 수집이 일반화 성능 향상의 가장 직접적 방법
2. **단일 기관 한계**: MSKCC 데이터만 사용 → 다른 병원의 염색 방식, 스캐너, 절차 차이에 대한 강건성 미검증
3. **증강 효과 없음**: 기본 회전/뒤집기 증강이 효과 없었지만, **stain normalization**, **Macenko/Vahadane 방법** 등 병리학 특화 증강이 일반화에 기여할 가능성은 미탐색
4. **Top-1 MIL의 일반화 제약**: 슬라이드당 1개 타일만 사용 → 다양한 병변 패턴을 학습하지 못할 수 있음

**일반화 향상을 위해 필요한 방향 (분석자 제안)**:

| 전략 | 기대 효과 |
|------|-----------|
| 다기관(multi-site) 데이터 수집 | 스캐너/염색 변동성에 대한 강건성 |
| 도메인 적응(domain adaptation) | 기관 간 특징 분포 차이 극복 |
| Stain normalization 적용 | 염색 편향 제거로 일반화 향상 |
| Attention-based MIL | 더 많은 인스턴스를 학습에 활용 |
| 자기지도학습(Self-supervised pre-training) | 병리학 도메인 특화 표현 학습 |

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 AI 훈련 데이터 기반 지식으로, 개별 세부 수치의 정확성은 원문 확인을 권장합니다.

#### 주요 후속 연구 흐름

| 연구 | 연도 | 핵심 기여 | 본 논문과의 관계 |
|------|------|-----------|----------------|
| Campanella et al. (Nature Medicine, 2019) | 2019 | 본 논문의 확장판; 15,187 WSI, 전립선/피부/자궁경부암 포함, AUC 0.98 | 본 논문의 직접 후속 |
| Lu et al. (Nature Biomedical Engineering, 2021) — CLAM | 2021 | Attention-based MIL + 클러스터링 기반 감독 학습 | Top-1 MIL의 한계 극복 시도 |
| Chen et al. (CVPR, 2022) — HIPT | 2022 | 계층적 이미지 피라미드 트랜스포머, ViT 기반 WSI 분류 | 트랜스포머 구조로 패러다임 전환 |
| Shao et al. (MICCAI, 2021) — TransMIL | 2021 | Transformer + MIL 결합, 공간적 맥락 모델링 | CNN에서 Transformer로의 진화 |
| Li et al. (Nature Communications, 2021) | 2021 | Graph Neural Network 기반 WSI 분류 | 타일 간 공간 관계 모델링 |
| Kather et al. (Nature Cancer, 2020) | 2020 | 범암종 게놈 특성 예측 (대규모 TCGA 데이터) | 응용 범위 확장 |

> **참고 문헌**:
> - Campanella et al. "Clinical-grade computational pathology using weakly supervised deep learning on whole slide images." *Nature Medicine* 25, 1301–1309 (2019).
> - Lu et al. "Data-efficient and weakly supervised computational pathology on whole-slide images." *Nature Biomedical Engineering* 5, 555–570 (2021).
> - Chen et al. "Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning." *CVPR* (2022).

#### 본 논문이 미친 영향

1. **데이터 규모 기준 설정**: 이 논문 이후 computational pathology 연구에서 "수천 장 이상의 WSI"가 사실상의 최소 기준으로 자리잡음
2. **약지도 MIL의 주류화**: 픽셀 어노테이션 없는 접근법이 연구 커뮤니티에서 주류 패러다임으로 정착
3. **실제 임상 데이터 활용 선례**: LIS(실험실 정보 시스템) 레이블을 학습 신호로 활용하는 접근법의 선례가 됨
4. **산업계 영향**: 저자 소속 Paige.AI가 FDA 승인을 받는 계기 중 하나

#### 향후 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|-----------|-------------|
| **Attention 기반 MIL 채택** | Top-1 방식의 정보 손실 → ABMIL(Attention-Based MIL)이 현재 표준 |
| **Foundation model 활용** | UNI, CONCH, PLIP 등 병리학 특화 비전-언어 모델로 표현 학습 고도화 |
| **다기관 검증 필수화** | 단일 기관 결과는 임상 신뢰성 불충분 → 외부 코호트 검증 필수 |
| **설명 가능성(XAI)** | Grad-CAM, SHAP 등으로 의사결정 근거 시각화 — 규제 기관 요구 사항 |
| **불확실성 정량화** | Bayesian deep learning, conformal prediction 등으로 예측 신뢰도 제공 |
| **레이블 품질 관리** | LIS 기반 레이블의 노이즈 문제 → 능동 학습(active learning)으로 핵심 슬라이드 검증 |
| **공정성(fairness) 검증** | 인종, 연령, 기관별 성능 편차 분석 필요 |
| **Gleason 등급 예측** | 이진 분류를 넘어 예후 예측으로 임상 가치 확장 |
| **계산 효율성** | 테라바이트 스케일 추론을 위한 경량화 모델/스트리밍 추론 연구 |
| **멀티모달 통합** | 게놈, 임상 데이터와 WSI를 결합한 통합 예측 모델 |

---

## 📚 참고 자료 목록

1. **본 논문**: Campanella, G., Silva, V. W. K., & Fuchs, T. J. (2018). "Terabyte-scale Deep Multiple Instance Learning for Classification and Localization in Pathology." *arXiv:1805.06983v2.*
2. Campanella et al. (2019). "Clinical-grade computational pathology using weakly supervised deep learning on whole slide images." *Nature Medicine*, 25, 1301–1309.
3. Lu, M. Y. et al. (2021). "Data-efficient and weakly supervised computational pathology on whole-slide images." *Nature Biomedical Engineering*, 5, 555–570.
4. Chen, R. J. et al. (2022). "Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning." *CVPR 2022*.
5. Dietterich, T. G., Lathrop, R. H., & Lozano-Pérez, T. (1997). "Solving the multiple instance problem with axis-parallel rectangles." *Artificial Intelligence*, 89(1), 31–71.
6. Ehteshami Bejnordi, B. et al. (2017). "Diagnostic assessment of deep learning algorithms for detection of lymph node metastases." *JAMA*, 318(22), 2199.
7. Goode, A. et al. (2013). "OpenSlide: A vendor-neutral software foundation for digital pathology." *Journal of Pathology Informatics*, 4(1), 27.
8. Siegel, R. L., Miller, K. D., & Jemal, A. (2016). "Cancer statistics, 2016." *CA: A Cancer Journal for Clinicians*, 66(1), 7–30.
9. Kather, J. N. et al. (2020). "Pan-cancer image-based detection of clinically actionable genetic alterations." *Nature Cancer*, 1, 789–799.
