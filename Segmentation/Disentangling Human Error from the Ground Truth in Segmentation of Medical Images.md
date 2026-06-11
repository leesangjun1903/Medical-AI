# Disentangling Human Error from the Ground Truth in Segmentation of Medical Images

**저자:** Le Zhang, Ryutaro Tanno, Mou-Cheng Xu, Chen Jin, Joseph Jacob, Olga Ciccarelli, Frederik Barkhof, Daniel C. Alexander
**발표:** NeurIPS 2020 (arXiv: 2007.15963)

---

## 1. 핵심 주장 및 주요 기여 요약

최근 지도 학습 기반 세그멘테이션 방법이 활발히 사용되고 있으나, 알고리즘의 예측 성능은 레이블의 품질에 크게 의존하며, 특히 의료 영상 분야에서는 annotation 비용과 관찰자 간 변동성(inter-observer variability)이 모두 높다. 일반적인 레이블 수집 과정에서 서로 다른 전문가들이 자신의 편향과 역량 수준의 영향 하에 "참(true)" 세그멘테이션 레이블을 추정하며, 이러한 노이즈 레이블을 맹목적으로 ground truth로 취급하면 자동 세그멘테이션 알고리즘의 성능이 제한된다.

본 논문은 순수하게 노이즈가 포함된 관찰값만으로부터 개별 어노테이터의 신뢰도와 참(true) 세그멘테이션 레이블 분포를 **두 개의 결합된 CNN(coupled CNNs)**을 통해 동시에 학습하는 방법을 제시한다.

**주요 기여:**

1. 하나의 CNN은 전문가 합의 레이블 확률을 추정하고, 다른 CNN은 개별 어노테이터의 특성(예: 과잉 세그멘테이션 경향, 클래스 간 혼동 등)을 픽셀별 혼동 행렬(pixel-wise Confusion Matrices, CMs) 형태로 이미지별로 추정하는 새로운 딥러닝 아키텍처를 제안하였다.
2. 두 요소의 분리(disentangling)는 추정된 어노테이터가 최대한 신뢰할 수 없도록(maximally unreliable) 유도하면서도 노이즈가 있는 훈련 데이터에 대한 높은 충실도를 달성하도록 하여 이루어진다.
3. 기존 STAPLE 및 그 변형과 달리, 딥 신경망을 통해 입력 이미지에서 어노테이터 행동 및 전문가 합의 레이블로의 복잡한 매핑을 모델링하고 분리한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

이 논문의 핵심 목표는 **미지의 참(true) 세그멘테이션 맵을 추정**하는 것이다. 의료 영상 세그멘테이션 작업에서의 높은 관찰자 간 변동성 문제를 해결하기 위해, 다수의 노이즈 어노테이션으로 훈련되는 지도 분리(disentangling) 모델을 제안한다.

기존 방법의 한계:
- **Majority Voting / STAPLE** 등의 label fusion 방법은 입력 이미지의 특성을 고려하지 못함
- CNN의 파라미터는 서로 다른 이미지 샘플 간에 최적화되는 "전역 변수"로 작동하여, 유사한 이미지 샘플 간의 상관관계를 기반으로 어노테이터의 실수와 참 레이블을 강건하게 분리할 수 있다. 이미지당 사용 가능한 어노테이션 수가 적은 경우(예: 이미지당 단일 어노테이션)에도 가능하다.

### 2.2 제안하는 방법 (수식 포함)

#### 확률적 노이즈 모델

저자는 확률적 모델(probabilistic model)을 사용하여 노이즈가 있는 의료 어노테이션의 원리를 기술한다.

입력 이미지 $\mathbf{x}$에 대해, 각 픽셀 $i$에서의 참(true) 세그멘테이션 레이블 확률을 $\mathbf{p}(\mathbf{x})$로 표기하고, $r$번째 어노테이터의 관측 레이블을 $\hat{y}_i^{(r)}$로 나타내면:

$$p(\hat{y}_i^{(r)} = c \mid \mathbf{x}) = \sum_{l=1}^{C} A_{lc}^{(r)}(\mathbf{x}, i) \cdot p(y_i = l \mid \mathbf{x})$$

여기서:
- $C$: 세그멘테이션 클래스 수
- $A_{lc}^{(r)}(\mathbf{x}, i)$: $r$번째 어노테이터가 픽셀 $i$에서 참 레이블이 $l$일 때 $c$로 어노테이션할 확률 (픽셀별 혼동 행렬의 원소)
- $p(y_i = l \mid \mathbf{x})$: 참 세그멘테이션 레이블 분포

이를 행렬 형태로 표현하면:

$$\hat{\mathbf{p}}^{(r)}(\mathbf{x}) = \hat{\mathbf{A}}_\phi^{(r)}(\mathbf{x}) \cdot \hat{\mathbf{p}}_\theta(\mathbf{x})$$

여기서 $\hat{\mathbf{A}}\_\phi^{(r)}(\mathbf{x})$는 어노테이터 네트워크(파라미터 $\phi$ )가 추정한 혼동 행렬이고, $\hat{\mathbf{p}}_\theta(\mathbf{x})$는 세그멘테이션 네트워크(파라미터 $\theta$ )가 추정한 참 레이블 분포이다.

#### 손실 함수

기본 손실은 각 어노테이터의 관측 레이블에 대한 **크로스 엔트로피**이다:

$$\mathcal{L}_{CE}(\theta, \phi) = -\sum_{n=1}^{N} \sum_{r=1}^{R} \sum_{i} \log \hat{p}_{\theta,\phi}^{(r)}(\hat{y}_{n,i}^{(r)} \mid \mathbf{x}_n)$$

그러나 이 손실 함수만으로는 어노테이션 노이즈와 참 레이블 분포를 분리할 수 없다. $\hat{\mathbf{A}}\_\phi^{(r)}(\mathbf{x})$와 세그멘테이션 모델 $\hat{\mathbf{p}}_\theta(\mathbf{x})$의 여러 조합이 참 어노테이터 분포를 완벽히 일치시킬 수 있기 때문이다(예: CM 행의 순열).

이 문제를 해결하기 위해, 분류 과제에서 유사한 문제를 다룬 Tanno et al.에서 영감을 받아, 추정된 혼동 행렬의 **trace**를 손실 함수에 정규화 항으로 추가한다.

최종 결합 손실 함수:

$$\mathcal{L}(\theta, \phi) = \mathcal{L}_{CE}(\theta, \phi) + \lambda \sum_{n=1}^{N} \sum_{r=1}^{R} \sum_{i} \text{Tr}\left(\hat{\mathbf{A}}_\phi^{(r)}(\mathbf{x}_n, i)\right)$$

여기서 $\lambda > 0$는 정규화 강도를 조절하는 하이퍼파라미터이다.

#### Trace 정규화의 정당성

Trace 정규화의 근거: Tanno et al.은 어노테이터의 평균 CM이 **대각 우세(diagonally dominant)**이고 크로스 엔트로피 항이 0이면, 추정된 CM의 trace를 최소화하면 참 CM을 유일하게 복구할 수 있음을 보였다. 다만 이는 데이터 전체에 대한 평균 CM에 관한 결과였으며, 본 논문에서는 이미지별(sample-specific) 환경에서 유사하지만 약간 약한 결과를 보여주었다.

직관적으로, trace를 최소화하는 것은 $\hat{\mathbf{A}}\_\phi^{(r)}$의 대각 원소를 최소화하는 것, 즉 어노테이터가 "최대한 신뢰할 수 없는" 것처럼 추정하게 유도한다. 이렇게 하면, 세그멘테이션 네트워크 $\hat{\mathbf{p}}_\theta$가 참 레이블 분포의 대부분을 담당하게 되어 두 요소의 분리가 달성된다:

$$\min_{\theta, \phi} \mathcal{L}(\theta, \phi) = \min_{\theta, \phi} \left[ \mathcal{L}_{CE}(\theta, \phi) + \lambda \sum_{n,r,i} \text{Tr}\left(\hat{\mathbf{A}}_\phi^{(r)}(\mathbf{x}_n, i)\right) \right]$$

### 2.3 모델 구조

제안된 아키텍처는 두 개의 결합된 CNN으로 구성되며, 하나는 참 세그멘테이션 확률을 추정하고 다른 하나는 개별 어노테이터의 특성을 픽셀별 혼동 행렬(CM)을 이미지별로 추정하여 모델링한다.

구체적으로:

| 구성 요소 | 역할 | 출력 |
|---|---|---|
| **Segmentation Network** ($\theta$) | 참(true) 세그멘테이션 레이블 분포 추정 | $\hat{\mathbf{p}}_\theta(\mathbf{x}) \in \mathbb{R}^{H \times W \times C}$ |
| **Annotator Network** ($\phi$) | 각 어노테이터 $r$의 픽셀별 혼동 행렬 추정 | $\hat{\mathbf{A}}_\phi^{(r)}(\mathbf{x}) \in \mathbb{R}^{H \times W \times C \times C}$ |

세그멘테이션 아키텍처는 U-Net 기반 세그멘테이션 네트워크와 혼동 행렬 네트워크를 결합하여 다수 어노테이터의 노이즈 레이블을 처리한다. 다수의 인코더-디코더 블록과 스킵 연결을 사용하여 입력의 세밀한 디테일과 대략적 특징을 모두 포착하며, 혼동 행렬 레이어가 어노테이션의 노이즈를 모델링한다. 이는 어노테이터 레이블 노이즈가 우려되는 세그멘테이션 작업을 위해 설계된 유연하고 높은 맞춤형 아키텍처이다.

```
Input Image (x)
     │
     ├─────────────────────┐
     ▼                     ▼
┌─────────────┐   ┌──────────────────┐
│ Segmentation │   │ Annotator Network │
│ Network (θ)  │   │      (φ)          │
│  (U-Net)     │   │  (U-Net variant)  │
└──────┬──────┘   └────────┬─────────┘
       │                    │
       ▼                    ▼
  p̂_θ(x)              Â_φ^(r)(x) for r=1,...,R
  [H×W×C]             [H×W×C×C] per annotator
       │                    │
       └────────┬───────────┘
                ▼
         p̂^(r)(x) = Â_φ^(r)(x) · p̂_θ(x)
         (Predicted noisy annotation)
```

### 2.4 성능 향상

이 방법은 합성 어노테이션과 다수의 전문가에 의한 실제 어노테이션을 포함하는 네 개의 데이터셋에서 검증되었다. 다섯 가지 label fusion 기준선과 비교하여 모든 기준에서 더 나은 성능을 보였으며, 시각적으로도 더 나은 결과를 얻었다.

**실험 데이터셋:**
1. **MNIST 기반 Toy Dataset**: 알고리즘 속성 연구
2. **ISBI 2015 MS Lesion Segmentation**: 다발성 경화증 병변 세그멘테이션
3. **LIDC-IDRI**: 폐 결절 CT 세그멘테이션
4. **BraTS**: 뇌종양 세그멘테이션 (합성 노이즈)

실험 결과는 어노테이터 실수의 복잡한 공간적 특성을 포착하는 강한 능력도 보여주었다.

관련 후속 연구(LF-Net)에서는 다양한 이미징 모달리티의 5개 데이터셋에서 최신 방법 대비 세그멘테이션 정확도를 향상시켰으며, MS 병변 세그멘테이션에서 DSC 7% 개선을 달성하였다.

### 2.5 한계

논문 및 리뷰에서 언급된 한계들:

1. **계산 비용**: 두 개의 CNN을 동시에 훈련해야 하므로 메모리 및 계산 비용이 증가
2. **어노테이터 수 의존성**: 어노테이터 수가 극히 적을 때(1명) 분리 성능이 저하될 수 있음
3. **대각 우세 가정**: Trace 정규화의 이론적 보장은 어노테이터의 CM이 대각 우세라는 가정에 의존
4. 이후 연구에서 지적된 바와 같이, 이러한 방법들은 여전히 다수의 네트워크를 구성해야 하며 제한된 응용 시나리오에 초점을 맞추어 실용적 유연성이 부족하고, 거친 경계(coarse boundary) 레이블 문제를 명시적으로 고려하지 않아 최적이 아닌 결과를 낳을 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

CNN의 파라미터가 서로 다른 이미지 샘플 간에 최적화되는 "전역 변수"로 작동하기 때문에, 유사한 이미지 샘플 간의 상관관계를 기반으로 어노테이터의 실수와 참 레이블을 강건하게 분리할 수 있다. 이미지당 사용 가능한 어노테이션 수가 적을 때에도 가능하다.

이는 기존 STAPLE 기반 방법에 비해 핵심적인 **일반화 우위**이다:

1. **이미지 조건부(image-conditional) 모델링**: 혼동 행렬이 이미지별로 추정되므로, 어노테이터의 오류 패턴이 이미지 콘텐츠에 따라 달라지는 현실을 반영
2. **공유 파라미터를 통한 전이 학습**: 네트워크 파라미터가 전체 데이터셋에 걸쳐 공유되므로, 한 이미지에서 학습된 어노테이터 패턴이 다른 이미지에 전이
3. **단일 어노테이션에서도 동작**: 이미지당 하나의 어노테이션만 있어도, 다른 이미지에서 학습된 어노테이터 모델을 활용 가능

### 3.2 일반화 성능 향상을 위한 핵심 수식적 통찰

Trace 정규화가 일반화에 미치는 영향을 분석하면:

$$\text{Tr}(\hat{\mathbf{A}}_\phi^{(r)}) = \sum_{c=1}^{C} \hat{A}_{cc}^{(r)}$$

이 값을 최소화하면 대각 원소(정답을 맞추는 확률)가 줄어들어, 어노테이터 모델이 "최대한 나쁜" 방향으로 유도된다. 크로스 엔트로피 손실과 결합하면:

- $\mathcal{L}_{CE}$: 전체 시스템이 관측 데이터에 충실하도록 강제
- $\lambda \cdot \text{Tr}$: 어노테이터 네트워크가 가능한 한 많은 오류를 설명하도록 강제

이 두 힘의 균형이 세그멘테이션 네트워크를 **노이즈에 강건한 참 분포**를 학습하도록 유도하여, 미지의 테스트 데이터에 대한 일반화 성능을 향상시킨다.

### 3.3 도메인/데이터셋 간 일반화

- 다양한 의료 영상 모달리티(MRI, CT)에 걸쳐 일관된 성능 향상
- 합성 노이즈와 실제 어노테이터 노이즈 모두에서 효과 검증
- 한 구성 요소는 실제 세그멘테이션 맵을 학습하고, 다른 구성 요소는 이를 기반으로 노이즈 어노테이션을 재구성하기 위해 어노테이터의 행동을 모델링한다. 이 분리 구조가 도메인 전이 시에도 세그멘테이션 네트워크의 깨끗한 표현 학습을 보장한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

1. **노이즈 레이블 학습 패러다임 확립**: 이 논문은 의료 영상 세그멘테이션에서 "어노테이터 모델링 + 참 레이블 분리"라는 패러다임을 확립하여 수많은 후속 연구의 기반이 됨
2. **혼동 행렬 기반 어노테이터 모델링의 표준화**: 픽셀별 혼동 행렬을 사용한 공간적 어노테이터 모델링이 이후 연구의 핵심 구성 요소로 자리잡음
3. **단일 어노테이션 환경으로의 확장**: 이미지당 하나의 어노테이션만으로도 동작 가능하다는 점은 실제 임상 환경에서의 적용 가능성을 크게 높임
4. **품질 인식 학습(quality-aware learning)**: 어노테이터의 신뢰도를 명시적으로 모델링하는 접근은 능동 학습(active learning), 커리큘럼 학습(curriculum learning) 등과의 결합 가능성을 열어줌

### 4.2 향후 연구 시 고려할 점

1. **확장성(Scalability)**: 어노테이터 수가 매우 많아질 경우의 파라미터 효율성
2. **3D 볼류메트릭 세그멘테이션**: 2D 슬라이스 기반 방법의 3D 확장 시 공간적 일관성 보장
3. **어노테이터 비정상성**: 의도적으로 잘못된 어노테이션(adversarial annotators)에 대한 강건성
4. **비대칭 노이즈 패턴**: 특정 클래스에 편향된 노이즈 처리
5. **Foundation 모델과의 결합**: SAM 등 대규모 사전학습 모델과 결합 시 노이즈 처리 전략

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | Zhang et al. 대비 차별점 |
|---|---|---|---|
| **Adaptive Early-Learning Correction (CVPR 2022)** | 2022 | 조기 학습 교정을 통한 노이즈 어노테이션 세그멘테이션 | 학습 초기의 깨끗한 패턴을 활용하여 적응적으로 레이블 교정; 별도의 어노테이터 모델 불필요 |
| **Joint Class-Affinity Loss Correction (MICCAI 2022)** | 2022 | 클래스 친화도 손실 교정을 통한 강건한 의료 영상 세그멘테이션 | 클래스 간 관계를 활용한 손실 교정으로 전역적 노이즈 처리 |
| **Superpixel-guided Iterative Learning (2021)** | 2021 | 슈퍼픽셀 기반 반복 학습을 통한 노이즈 어노테이션에서의 의료 영상 세그멘테이션 | 슈퍼픽셀 단위의 노이즈 식별로 공간적 일관성 활용 |
| **SEAMAL (IEEE TMI 2024)** | 2024 | 노이즈 레이블 하의 강건한 세그멘테이션은 의료 영상에서 중요한 문제이며, SEAMAL은 동시적 에지 정렬 및 메모리 보조 학습 프레임워크를 통해 노이즈 레이블에 강건한 세그멘테이션을 수행한다. | 단일 네트워크 구조로 경계 레이블 문제를 명시적으로 처리 |
| **MetaDCSeg (2025~2026)** | 2025-2026 | 의료 영상 노이즈 지도학습에서 카테고리 수준 모델링과 픽셀 수준 디노이징 전략으로 분류되며, 픽셀 수준 디노이징은 신뢰도 기반 메커니즘을 설계하여 잠재적으로 깨끗한 픽셀을 식별·활용한다. | 픽셀별 메타 학습으로 공간적으로 변화하는 노이즈 분포를 포착 |
| **ProSona (2025)** | 2025 | 어노테이션 스타일의 연속 잠재 공간을 학습하여 자연어 프롬프트를 통한 제어 가능한 개인화를 지원하며, Probabilistic U-Net 백본이 다양한 전문가 가설을 포착하고, 프롬프트 기반 투영 메커니즘이 개인화된 세그멘테이션을 생성한다. | 자연어 프롬프트 기반 개인화 세그멘테이션으로 확장 |
| **Learning Confident Classifiers (2025)** | 2025 | 입력 이미지를 세그멘테이션과 어노테이션 두 개의 별도 네트워크에 공급하며, U-Net 기반 세그멘테이션 네트워크와 혼동 행렬 네트워크를 결합한다. | Zhang et al.의 프레임워크를 기반으로 새로운 정규화 기법 적용 |
| **Teacher-guided Early-Learning (2024)** | 2024 | Mean Teacher 아키텍처 기반으로 고품질 및 저품질 레이블의 혼합을 사용하는 노이즈 레이블 학습 방법을 제안 | 자가 앙상블(self-ensemble) 기반 접근으로 별도 어노테이터 모델 불필요 |

### 최신 연구 동향 분석

최근 의료 영상 노이즈 세그멘테이션 접근법은 카테고리 수준 모델링과 픽셀 수준 디노이징으로 분류되며, 카테고리 수준 방법은 노이즈와 깨끗한 레이블 간의 확률적 관계를 추정하지만 종종 전역 가정에 의존하여 의료 영상에 내재한 복잡하고 공간적으로 변화하는 노이즈 분포를 포착하지 못한다. Zhang et al.의 접근법은 이 두 카테고리를 결합한 선구적 연구로, 이후의 방법들에 직접적 영향을 미쳤다.

---

## 참고 자료

1. **Zhang, L., Tanno, R., Xu, M.C., Jin, C., Jacob, J., Ciccarelli, O., Barkhof, F., & Alexander, D.C.** (2020). "Disentangling Human Error from the Ground Truth in Segmentation of Medical Images." *NeurIPS 2020*. [arXiv:2007.15963](https://arxiv.org/abs/2007.15963)
2. **Tanno, R., Saeedi, A., Sankaranarayanan, S., Alexander, D.C., & Silberman, N.** (2019). "Learning From Noisy Labels By Regularized Estimation Of Annotator Confusion." *CVPR 2019*. [arXiv:1902.03680](https://arxiv.org/abs/1902.03680)
3. **NeurIPS 2020 Review**: [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2020/file/b5d17ed2b502da15aa727af0d51508d6-Review.html)
4. **Zhang, L., Tanno, R., et al.** (2023). "Learning from multiple annotators for medical image segmentation." *Pattern Recognition* / PMC. [PMC10533416](https://pmc.ncbi.nlm.nih.gov/articles/PMC10533416/)
5. **Ye, S., et al.** (2024). "Learning a Single Network for Robust Medical Image Segmentation with Noisy Labels (SEAMAL)." *IEEE TMI*, 43(9):3188-3199.
6. **Liu, S., et al.** (2022). "Adaptive Early-Learning Correction for Segmentation From Noisy Annotations." *CVPR 2022*.
7. **Not All Pixels Are Equal: Pixel-wise Meta-Learning (MetaDCSeg)** (2026). [arXiv:2511.18894](https://arxiv.org/html/2511.18894)
8. **Adaptive Label Correction for Robust Medical Image Segmentation with Noisy Labels** (2025). [arXiv:2503.12218](https://arxiv.org/html/2503.12218v2)
9. **Learning Confident Classifiers in the Presence of Label Noise** (2025). [arXiv:2301.00524](https://arxiv.org/html/2301.00524v3)

---

> **참고**: 본 분석의 수식은 원 논문(arXiv:2007.15963)과 선행 연구(arXiv:1902.03680)의 수학적 프레임워크를 기반으로 정리한 것입니다. 일부 세부적인 수식 표현은 논문 본문의 정확한 표기와 미세하게 다를 수 있으므로, 정확한 수식 확인을 위해서는 원 논문의 PDF를 직접 참조하시기 바랍니다.
