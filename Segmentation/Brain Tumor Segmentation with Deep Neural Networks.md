# Brain Tumor Segmentation with Deep Neural Networks

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Havaei et al. (2016)은 MRI 영상에서 뇌종양(특히 교모세포종, glioblastoma)을 **완전 자동**으로 세분화(segmentation)하기 위해 CNN 기반 딥러닝 방법론을 제안하며, 당시 최신 기법 대비 **정확도와 속도를 동시에 향상**시킬 수 있음을 주장합니다.

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| **①** 완전 자동 세분화 | BRATS 2013 스코어보드 2위 달성 |
| **②** 처리 속도 | 25초~3분 (기존 대비 30배 이상 빠름) |
| **③** Two-Pathway CNN | 로컬/글로벌 문맥 동시 학습 + 2단계 학습 절차 |
| **④** Cascaded Architecture | CRF 대체 가능한 효율적 구조화 출력 모델링 |

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 및 한계

### 2-1. 해결하고자 하는 문제

뇌종양 세분화는 다음과 같은 이유로 매우 어려운 문제입니다:

- 종양의 위치, 크기, 형태가 환자마다 상이
- MRI 기기 종류 및 프로토콜에 따라 복셀(voxel) 강도 값이 비표준화
- 종양 경계가 불분명하고 정상 조직과 구분 어려움
- **레이블 불균형**: 건강한 복셀이 전체의 98%를 차지

기존 수작업 피처(hand-designed feature) 기반 방법론의 한계를 극복하고, 데이터로부터 직접 피처를 학습하는 딥러닝 방식을 적용합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 합성곱 레이어 연산

**1단계: 커널 합성곱 (Convolution)**

$$\mathbf{O}_s = b_s + \sum_{r} \mathbf{W}_{sr} * \mathbf{X}_r $$

여기서:
- $\mathbf{X}_r$: $r$번째 입력 채널 (MRI 모달리티)
- $\mathbf{W}_{sr}$: 해당 채널의 서브 커널
- $*$: 합성곱 연산
- $b_s$: 편향(bias)

**2단계: Maxout 비선형 활성화 함수**

```math
Z_{s,i,j} = \max\left\{O_{s,i,j},\ O_{s+1,i,j},\ \ldots,\ O_{s+K-1,i,j}\right\}
```

Maxout은 적응형 볼록(convex) 활성화 함수로, 유용한 피처 학습에 효과적입니다.

**3단계: Max Pooling**

$$H_{s,i,j} = \max_{p}\ Z_{s,\,i+p,\,j+p} $$

여기서 $p$는 풀링 윈도우 크기. 출력 크기는 $D = \frac{Q - p}{S} + 1$ ($S$: stride)

---

#### (B) Softmax 출력 및 손실 함수

출력 레이어에서 Softmax를 이용한 다중 클래스 분류:

$$\text{softmax}(\mathbf{a}) = \frac{\exp(\mathbf{a})}{Z}, \quad Z = \sum_c \exp(a_c) $$

각 위치 $(i,j)$에서의 레이블 확률:

$$p(\mathbf{Y}|\mathbf{X}) = \prod_{ij} p(Y_{ij}|\mathbf{X}) $$

학습 목표(손실 함수): 음의 로그 확률 최소화 + 정규화

$$\mathcal{L} = -\log p(\mathbf{Y}|\mathbf{X}) + \lambda_1\|\mathbf{W}\|_1 + \lambda_2\|\mathbf{W}\|^2 $$

여기서 $\lambda_1$, $\lambda_2$는 L1, L2 정규화 계수.

---

#### (C) 모멘텀 기반 SGD 최적화

$$\mathbf{V}_{i+1} = \mu \cdot \mathbf{V}_i - \alpha \cdot \nabla\mathbf{W}_i $$

$$\mathbf{W}_{i+1} = \mathbf{W}_i + \mathbf{V}_{i+1} $$

- 초기 모멘텀: $\mu = 0.5$, 최종: $\mu = 0.9$
- 초기 학습률: $\alpha = 0.005$, 에포크마다 $10^{-1}$ 비율로 감소

---

#### (D) 평가 지표

$$\text{Dice}(P, T) = \frac{|P_1 \wedge T_1|}{(|P_1| + |T_1|)/2} $$

$$\text{Sensitivity}(P, T) = \frac{|P_1 \wedge T_1|}{|T_1|} $$

$$\text{Specificity}(P, T) = \frac{|P_0 \wedge T_0|}{|T_0|} $$

---

### 2-3. 모델 구조

#### (A) TwoPathCNN (두 경로 아키텍처)

```
입력 (4×33×33: T1, T1C, T2, FLAIR)
        ├── [Local Path]  Conv 7×7 → Maxout → Pooling 4×4 → Conv 3×3 → Maxout → Pooling 2×2
        │                (세부 로컬 피처: 엣지 검출기 역할)
        └── [Global Path] Conv 13×13 → Maxout
                         (글로벌 문맥 피처: 위치 정보 반영)
                    ↓
             Feature Concatenation (224×21×21)
                    ↓
             Conv 21×21 + Softmax
                    ↓
             출력 (5×1×1: 5개 클래스)
파라미터 수: 651,488
```

- **Local Path**: 7×7 수용야(receptive field), 세밀한 종양 경계 학습
- **Global Path**: 13×13 수용야, 종양의 위치적 맥락 파악
- 최종 레이어를 **Fully Convolutional** 구조로 구현 → 추론 시 45배 속도 향상

---

#### (B) 2단계 학습 절차 (Two-Phase Training)

**레이블 불균형 문제:**

| 레이블 | 비율 |
|--------|------|
| 정상(label 0) | 98% |
| 괴사(label 1) | 0.18% |
| 부종(label 2) | 1.1% |
| 비강화 종양(label 3) | 0.12% |
| 강화 종양(label 4) | 0.38% |

- **1단계**: 모든 레이블을 균등 확률로 샘플링 → 다양한 클래스 피처 학습
- **2단계**: 출력 레이어만 재훈련, 실제 데이터 분포 반영 → 확률 보정 (다른 레이어 고정)

---

#### (C) 캐스케이드 아키텍처 (Cascaded Architectures)

CRF의 계산 비용 문제를 해결하는 효율적인 대안:

| 모델 | 설명 | 파라미터 수 |
|------|------|------------|
| **InputCascadeCNN** | 1번째 CNN 출력을 2번째 CNN 입력에 연결 | 802,368 |
| **LocalCascadeCNN** | 1번째 CNN 출력을 2번째 CNN 로컬 경로 첫 번째 은닉층에 연결 | 654,368 |
| **MFCascadeCNN** | 1번째 CNN 출력을 2번째 CNN 출력 직전에 연결 (Mean-Field 추론과 유사) | 662,513 |

---

### 2-4. 성능 결과

**BRATS 2013 테스트 세트 성능 (Dice Score):**

| 방법 | Complete | Core | Enhancing | 처리 시간 | 순위 |
|------|----------|------|-----------|-----------|------|
| **InputCascadeCNN\*** | **0.88** | **0.79** | **0.73** | 3분 | **2위** |
| Tustison (1위) | 0.87 | 0.78 | 0.74 | 100분 | 1위 |
| MFCascadeCNN\* | 0.86 | 0.77 | 0.73 | 1.5분 | 4-a |
| TwoPathCNN\* | 0.85 | 0.78 | 0.73 | 25초 | 4위 |

**처리 속도 비교:**
- InputCascadeCNN\*: Tustison 대비 **30배 이상** 빠름
- TwoPathCNN\*: Tustison 대비 **200배 이상** 빠름

---

### 2-5. 한계

논문에서 명시하거나 유추 가능한 한계:

1. **2D 슬라이스 처리**: MRI 볼륨의 z축 해상도가 비등방적(anisotropic)이어서 3D 정보를 충분히 활용하지 못함
2. **소규모 데이터셋**: BRATS 2013 훈련 데이터가 30개 피험자에 불과
3. **도메인 이동(Domain Shift) 취약성**: 다른 병원, 기기, 프로토콜에 대한 일반화 성능이 제한적
4. **2D 처리의 일관성 문제**: 축면(axial) 슬라이스별 처리로 3D 공간적 일관성 미흡
5. **데이터 증강 효과 미미**: 이미지 플리핑(flipping) 증강이 성능 향상에 기여하지 못함
6. **Enhancing 영역 정확도 저하**: 경계가 불분명하고 annotator 간 편차가 큼

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 논문 내 일반화 관련 기법

#### (A) 정규화 전략

논문은 다음 세 가지 정규화를 조합하여 과적합을 방지:

$$\mathcal{L}_{\text{reg}} = -\log p(\mathbf{Y}|\mathbf{X}) + \lambda_1\|\mathbf{W}\|_1 + \lambda_2\|\mathbf{W}\|^2 $$

- **L1 정규화**: 가중치 희소성(sparsity) 유도
- **L2 정규화**: 가중치 크기 억제
- **Dropout**: 은닉 유닛을 확률 0.5로 마스킹 → 공동 적응(co-adaptation) 방지
- **Early Stopping**: 검증 세트 성능이 향상되지 않으면 훈련 중단

#### (B) 뇌 MRI 도메인 특성 활용

논문은 다음과 같이 언급합니다:

> *"MRI images of brains are very similar from one patient to another. Since the variety of those images is much lower than those in real-image datasets such as CIFAR and ImageNet, a fewer number of training samples is thus needed."*

이는 의료 영상 도메인의 특수성(환자 간 유사성)이 일반화를 돕는다고 주장하지만, 동시에 **병원 간, 기기 간 편차**가 존재하는 현실을 간과할 수 있는 한계를 내포합니다.

#### (C) 전처리를 통한 도메인 정규화

```
① 상위/하위 1% 강도값 제거
② T1, T1C에 N4ITK 편향 보정 적용
③ 채널별 평균 빼기 + 표준편차 나누기
```

이 전처리는 MRI 기기 간 강도 차이를 부분적으로 완화하지만, 완전한 해결책은 아닙니다.

#### (D) BRATS 2015 실험에서의 일반화 시도

BRATS 2015 (274개 피험자) 실험에서 강도 분포의 이질성 문제를 해결하기 위해:

$$\text{앙상블} = \text{Voted Average of 7-fold Cross-Validation Models}$$

- 7-fold 교차 검증 + 모델 앙상블을 통해 데이터 다양성에 대한 일반화 향상
- 결과: Complete tumor, Core tumor 범주에서 1~2위 달성

### 3-2. 일반화 성능의 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|-----------|
| 소규모 데이터 (30개) | 데이터 증강(회전, 탄성 변형), 전이 학습 |
| 도메인 이동 취약성 | 도메인 적응(Domain Adaptation), 연합 학습(Federated Learning) |
| 2D 처리의 3D 일관성 부재 | 3D Conv, Volumetric 아키텍처 도입 |
| 레이블 노이즈 | Semi-supervised, 약지도 학습(Weakly-supervised) |

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4-1. 연구에 미치는 영향

#### (A) 의료 영상 분야 딥러닝 방법론의 표준화

이 논문은 다음을 제시하며 이후 연구의 기반이 됨:
- **멀티스케일 컨텍스트 학습** → nnU-Net, TransBTS 등으로 계승
- **2단계 학습 (클래스 불균형 대응)** → 이후 Focal Loss, OHEM 등의 개념과 연결
- **Cascaded 구조** → 이후 DeepMedic, 3D U-Net 등 계층적 아키텍처 설계에 영향

#### (B) 완전 합성곱 네트워크(FCN)의 의료 영상 적용 촉진

최종 레이어를 완전 합성곱으로 구현하여 패치 단위 예측 대비 40~45배 속도 향상을 달성함으로써, 임상 적용 가능성을 높이는 실용적 기여를 함.

#### (C) 공개 벤치마크(BRATS)의 표준화 기여

논문이 BRATS 2013을 엄격하게 활용하고 정량적 비교를 체계화함으로써, 이후 모든 뇌종양 세분화 연구의 표준 평가 프레임워크 확립에 기여.

### 4-2. 앞으로 연구 시 고려할 점

#### ① 3D 완전 합성곱 아키텍처로의 전환

본 논문이 2D 처리를 채택한 이유(MRI의 z축 비등방성)는 이후 연구에서 3D-UNet이나 nnU-Net이 해결. 향후 연구에서는 **3D volumetric 처리의 계산 효율성**을 핵심 고려 사항으로 삼아야 함.

#### ② 도메인 일반화 (Domain Generalization)

단일 데이터셋(BRATS 2013, 30개 피험자)에 특화된 모델이 다른 의료 기관/MRI 기기에서 얼마나 작동하는지에 대한 검증이 필요. **연합 학습(Federated Learning)** 이나 **Test-Time Adaptation**을 고려해야 함.

#### ③ 설명 가능성(Explainability)

논문 자체에서도 언급:
> *"visualizing the learned mid/high level features of a CNN is still very much an open research problem"*

임상 적용을 위해서는 **Grad-CAM, SHAP** 등 XAI 기법의 통합이 필수적.

#### ④ 레이블 효율성

30개 피험자 데이터로 학습하는 한계를 극복하기 위해 **Semi-supervised**, **Self-supervised**, **Few-shot learning** 접근법을 고려해야 함.

#### ⑤ 클래스 불균형의 고도화된 처리

논문의 2단계 학습보다 발전된 방법:
- Focal Loss: $\mathcal{L}_{\text{focal}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$
- Dice Loss와 Cross-Entropy의 결합
- Oversampling/Undersampling 전략

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 주요 기법 | Dice (Complete) | 주요 개선점 |
|------|-----------|-----------------|-------------|
| **Havaei et al. (2016)** | TwoPathCNN, Cascade | 0.88 | 멀티스케일 CNN, 2단계 학습 |
| **nnU-Net (Isensee et al., 2021)** | 자동 설정 3D U-Net | ~0.91+ | 자동 하이퍼파라미터, 3D 처리 |
| **TransBTS (Wang et al., 2021)** | CNN + Transformer | ~0.90 | 글로벌 자기 어텐션 통합 |
| **Swin UNETR (Hatamizadeh et al., 2022)** | Swin Transformer + U-Net | ~0.92 | 순수 Transformer 기반 인코더 |
| **SegResNet (Myronenko, 2019)** | ResNet + VAE 정규화 | ~0.91 | 비지도 보조 손실 활용 |

> ⚠️ 위 Dice 수치는 BRATS 데이터셋 버전(2017/2018/2020)에 따라 다르며, 논문별 정확한 수치는 해당 논문을 직접 확인하시기 바랍니다.

### 발전 트렌드 분석

```
Havaei et al. (2016)
    → 3D 처리 (DeepMedic, 3D U-Net)
        → 인코더-디코더 + 스킵 연결 (U-Net 계열)
            → 어텐션 메커니즘 (Attention U-Net)
                → Vision Transformer 통합 (TransBTS, Swin UNETR)
                    → 기반 모델 (Foundation Model, SAM for Medical Imaging)
```

**핵심 발전 방향:**
1. **2D → 3D**: 볼류메트릭 정보 완전 활용
2. **CNN → Transformer**: 장거리 의존성(long-range dependency) 모델링
3. **단일 도메인 → 다중 도메인**: 도메인 일반화
4. **지도 학습 → 자기지도/반지도**: 레이블 효율성 향상

---

## 참고 자료

**주요 참고 논문 (본 논문 및 인용 문헌):**
1. Havaei, M., et al. "Brain Tumor Segmentation with Deep Neural Networks." *Medical Image Analysis*, 2016. (arXiv:1505.03540v3)
2. Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods*, 2021.
3. Wang, W., et al. "TransBTS: Multimodal Brain Tumor Segmentation Using Transformer." *MICCAI*, 2021.
4. Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." *MICCAI BrainLesion Workshop*, 2022.
5. Goodfellow, I.J., et al. "Maxout Networks." *ICML*, 2013.
6. Srivastava, N., et al. "Dropout: A simple way to prevent neural networks from overfitting." *JMLR*, 2014.
7. Menze, B., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." *IEEE Trans. Medical Imaging*, 2015.

> **⚠️ 정확도 주의**: 2020년 이후 최신 연구의 Dice Score 수치는 데이터셋 버전(BRATS 2017/2018/2020/2021)과 평가 방식에 따라 상이합니다. 정확한 수치 비교를 위해서는 각 논문의 원문과 동일한 벤치마크 조건을 반드시 확인하시기 바랍니다.
