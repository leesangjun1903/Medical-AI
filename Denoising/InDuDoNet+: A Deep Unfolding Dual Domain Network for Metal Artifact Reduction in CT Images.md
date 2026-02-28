# InDuDoNet+: A Deep Unfolding Dual Domain Network for Metal Artifact Reduction in CT Images

## 종합 분석 보고서

---

## 1. 핵심 주장 및 주요 기여 요약

**InDuDoNet+**는 CT 영상에서 금속 임플란트로 인한 아티팩트(Metal Artifact)를 제거하기 위해, CT 물리적 영상 기하학(imaging geometry)을 심층 네트워크에 체계적으로 내재화한 **해석 가능한(interpretable) 딥 언폴딩(deep unfolding) 이중 도메인 네트워크**이다.

### 주요 기여:

1. **공간-라돈 도메인 결합 복원 모델** 수립 및 근위 경사(proximal gradient) 기반의 단순 연산자만으로 구성된 최적화 알고리즘 설계
2. **해석 가능한 네트워크 구조**: 반복 알고리즘의 각 단계를 네트워크 모듈로 언폴딩하여 각 모듈의 물리적 의미가 명확
3. **지식 기반 Prior-net** 설계를 통한 일반화 성능 향상: CT 값의 조직별 차이에 대한 사전 지식을 네트워크에 통합
4. **경량화된 네트워크**: WNet을 통해 파라미터 수를 대폭 축소(약 178만 개)하면서도 SOTA 대비 우수한 성능 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

CT 촬영 시 환자 체내의 **금속 임플란트**는 투영 데이터(sinogram)에 결손을 유발하고, 재구성된 CT 영상에 줄무늬(streaky) 및 음영(shading) 아티팩트를 발생시켜 임상 진단을 저해한다.

기존 딥러닝 기반 MAR 방법의 **두 가지 주요 한계**:

- **(1) CT 물리적 영상 기하학 제약의 불충분한 반영**: 대부분의 이중 도메인 방법이 시노그램과 CT 영상을 별도의 U-Net으로 개별 처리하며, 물리적 열화 과정을 완전히 모델링하지 않음
- **(2) 해석 가능성 부족**: 기성(off-the-shelf) 네트워크 블록에 의존하여 각 모듈의 MAR 작업에서의 역할을 평가하기 어려움

### 2.2 제안하는 방법 (수식 포함)

#### (A) 이중 도메인 모델 정식화

관측된 금속 영향 시노그램 $Y \in \mathbb{R}^{N_b \times N_p}$에 대해, 기존의 단일 도메인 복원 문제:

$$\min_{X} \|(1 - Tr) \odot (\mathcal{P}X - Y)\|_F^2 + \lambda g(X) $$

를 공간-라돈 도메인 결합 정규화로 확장:

$$\min_{S, X} \|\mathcal{P}X - S\|_F^2 + \alpha \|(1 - Tr) \odot (S - Y)\|_F^2 + \lambda_1 g_1(S) + \lambda_2 g_2(X) $$

여기서:
- $X \in \mathbb{R}^{H \times W}$: 금속이 없는 깨끗한 CT 영상 (공간 도메인)
- $S \in \mathbb{R}^{N_b \times N_p}$: 깨끗한 시노그램 (라돈 도메인)
- $\mathcal{P}$: 순방향 투영(forward projection) 연산자
- $Tr \in \mathbb{R}^{N_b \times N_p}$: 이진 금속 트레이스(metal trace)
- $\odot$: 요소별 곱셈
- $g_1(\cdot)$, $g_2(\cdot)$: 정규화 함수

#### (B) 정규화된 시노그램 복원

더 균질한(homogeneous) 영역에서 시노그램 보완이 용이하다는 관찰에 기반하여, 시노그램을 정규화:

$$S = \widetilde{Y} \odot \widetilde{S} $$

여기서 $\widetilde{Y} = \mathcal{P}\widetilde{X}$는 사전 영상(prior image)의 순방향 투영으로 구한 정규화 계수, $\widetilde{S}$는 정규화된 시노그램이다.

이를 Eq. (2)에 대입하면 **최종 이중 도메인 복원 문제**:

$$\min_{\widetilde{S}, X} \|\mathcal{P}X - \widetilde{Y} \odot \widetilde{S}\|_F^2 + \alpha \|(1 - Tr) \odot (\widetilde{Y} \odot \widetilde{S} - Y)\|_F^2 + \lambda_1 g_1(\widetilde{S}) + \lambda_2 g_2(X) $$

#### (C) 최적화 알고리즘

**근위 경사 기법(proximal gradient technique)**을 적용하여 $\widetilde{S}$와 $X$를 교대로 업데이트:

**$\widetilde{S}$ 업데이트:**

$$\widetilde{S}_n = \text{prox}_{\lambda_1 \eta_1}\left(\widetilde{S}_{n-1} - \eta_1 \left(\widetilde{Y} \odot \left(\widetilde{Y} \odot \widetilde{S}_{n-1} - \mathcal{P}X_{n-1}\right) + \alpha(1 - Tr) \odot \widetilde{Y} \odot \left(\widetilde{Y} \odot \widetilde{S}_{n-1} - Y\right)\right)\right) $$

**$X$ 업데이트:**

$$X_n = \text{prox}_{\lambda_2 \eta_2}\left(X_{n-1} - \eta_2 \mathcal{P}^T\left(\mathcal{P}X_{n-1} - \widetilde{Y} \odot \widetilde{S}_n\right)\right) $$

핵심적으로, 이 알고리즘은 **행렬 역연산 없이 점별 곱셈(point-wise multiplication)** 등 단순 연산만으로 구성되어, 네트워크로의 언폴딩이 용이하다.

#### (D) 학습 손실 함수

$$\mathcal{L} = \sum_{n=0}^{N} \beta_n \|X_n - X_{gt}\|_F^2 \odot (1 - M) + \gamma \left(\sum_{n=1}^{N} \beta_n \|\widetilde{Y} \odot \widetilde{S}_n - Y_{gt}\|_F^2\right) $$

여기서 $\beta_N = 1$, $\beta_n = 0.1$ ($n \neq N$), $\gamma = 0.1$로 설정하여 최종 스테이지 출력에 주도적 역할을 부여하고, 중간 스테이지도 감독한다.

### 2.3 모델 구조

InDuDoNet+의 전체 구조는 세 가지 핵심 모듈로 구성된다:

| 모듈 | 역할 | 구조적 특징 |
|------|------|-----------|
| **Prior-net** | 사전 영상 $\widetilde{X}$ 추정 → 정규화 계수 $\widetilde{Y} = \mathcal{P}\widetilde{X}$ 산출 | 사전 지식 기반 클러스터링 + 얕은 WNet (Conv 3층) |
| **$\widetilde{S}$-net** (N 스테이지) | 정규화된 시노그램 $\widetilde{S}$ 복원 | Eq. (8)의 언폴딩, proxNet (ResNet 4블록) |
| **X-net** (N 스테이지) | 아티팩트 제거된 CT 영상 $X$ 복원 | Eq. (10)의 언폴딩, proxNet (ResNet 4블록) |

**Prior-net 상세 (지식 기반 설계)**:

1. $X_{LI}$ (선형 보간 보정 영상)에 k-means 클러스터링 적용 → 공기(-1000 HU), 연조직(0 HU), 뼈(원래 값) 분류
2. 조잡한 사전 분할 영상 $\widetilde{X}_c$ 생성
3. $X_{ma}$ (금속 영향 영상)와 $\widetilde{X}_c$를 입력으로 하는 **WNet** (3층 Conv-BN-ReLU)이 픽셀 단위 가중치 행렬 생성
4. $\widetilde{X} = \widetilde{X}\_c \odot \text{WNet}(X_{ma})$로 정밀 사전 영상 획득

**채널별 연결/분리(Channel-wise Concatenation/Detachment)**: 단일 채널 입력의 정보 전파 한계를 극복하기 위해, 보조 변수 $Q^s_{n-1} \in \mathbb{R}^{N_b \times N_p \times N_s}$를 도입하여 채널 차원으로 연결 후 proxNet에 입력한다.

### 2.4 성능 향상

#### 합성 DeepLesion 데이터셋:

| 방법 | 평균 PSNR (dB) | 평균 SSIM | RMSE (HU) | 파라미터 수 |
|------|:---:|:---:|:---:|:---:|
| DuDoNet++ | 39.69 | 0.9886 | 27.78 | 25,983,627 |
| InDuDoNet | 41.48 | **0.9904** | 18.20 | 5,174,936 |
| **InDuDoNet+** | **41.50** | 0.9891 | **16.93** | **1,782,007** |

- RMSE 기준 DuDoNet++ 대비 **약 39% 감소** (27.78 → 16.93 HU)
- 파라미터 수가 DuDoNet++의 **약 1/14.6**로 대폭 경량화
- 추론 시간: 0.3782초로 DuDoNet++(0.8062초) 대비 **약 53% 단축**

#### 일반화 성능 (합성 Dental, 크로스 바디 사이트):

| 방법 | Dental 전체 평균 PSNR | SSIM |
|------|:---:|:---:|
| DuDoNet++ | 38.51 | 0.9882 |
| InDuDoNet | 41.95 | 0.9757 |
| **InDuDoNet+** | **42.68** | **0.9910** |

복부/흉부 CT로 학습 후 치과 CT에 적용했을 때, InDuDoNet+가 가장 우수한 일반화 성능을 보였다.

### 2.5 한계

1. **금속 분할 마스크의 수동 설정**: 임상 데이터에서 2500 HU 임계값 기반 분할에 의존하며, 부정확한 임계값은 성능 저하를 초래
2. **대형 금속 임플란트 제한**: 큰 금속일수록 영상 손상이 심해 복원의 여지가 제한적이며, DuDoNet++ 대비 약간의 우위만 확보
3. **합성 데이터-실제 데이터 간 도메인 갭**: 학습은 합성 데이터에서 이루어지며, 실제 임상 데이터의 ground truth가 부재하여 정량적 검증 제한
4. **지도 학습 의존**: 현재 프레임워크는 완전 지도 학습 방식이며, 반지도/비지도 학습으로의 확장은 미래 과제로 남아 있음

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

InDuDoNet+의 일반화 성능 향상은 **세 가지 핵심 메커니즘**에 기인한다:

### 3.1 물리적 영상 기하학 제약의 내재화

CT 영상 형성의 물리적 과정($\mathcal{P}$: 순방향 투영, $\mathcal{P}^T$: 역투영)이 네트워크 전체에 걸쳐 명시적으로 반영된다. 이는 **도메인 불변(domain-invariant)** 특성을 가지므로, 학습 데이터와 다른 신체 부위(예: 복부 → 치과)에서도 일관된 물리적 모델이 적용되어 일반화에 유리하다.

실험적 근거: Prior-net 없이도(InDuDoNet w/o Prior-net, 1,743,734 파라미터) DuDoNet++(25,983,627 파라미터)와 유사한 평균 PSNR/SSIM을 달성(40.18 vs 39.69 dB). 이는 물리적 기하학 제약의 내재화만으로도 상당한 MAR 성능을 확보할 수 있음을 시사한다.

### 3.2 지식 기반 Prior-net에 의한 과적합 완화

기존 InDuDoNet의 Prior-net은 **U-shape 블랙박스 구조**였으나, InDuDoNet+는:

1. **조직별 CT 값 분포에 대한 사전 지식**(공기: -1000 HU, 연조직: 0 HU, 뼈: 실제 값)을 Thresholding 기반 클러스터링으로 반영 → 조잡한 사전 분할 영상 $\widetilde{X}_c$
2. **얕은 WNet** (3층 CNN)으로 $\widetilde{X}_c$를 미세 조정

이 설계는 **모델 기반(model-driven)과 데이터 기반(data-driven)의 결합**으로, 사전 지식이 초기 추정의 안정성을 보장하고, 얕은 네트워크가 도메인별 미세 조정을 수행한다.

### 3.3 파라미터 축소에 의한 과적합 방지

| 방법 | 파라미터 수 | Dental PSNR (일반화) |
|------|:---:|:---:|
| DuDoNet++ | 25,983,627 | 38.51 |
| InDuDoNet | 5,174,936 | 41.95 |
| **InDuDoNet+** | **1,782,007** | **42.68** |

InDuDoNet+ → InDuDoNet 대비 파라미터가 약 **65.6% 감소**하면서 Dental(크로스 바디 사이트) 일반화 PSNR이 **0.73 dB 향상**되었다. 이는 적은 파라미터가 과적합을 완화하여 일반화 성능을 향상시킨다는 통계 학습 이론적 관점과 일치한다.

### 3.4 해석 가능성과 일반화의 연결

각 네트워크 모듈의 물리적 의미가 명확하므로:
- X-net의 각 스테이지: 이전 추정 CT 영상 $X_{n-1}$의 시노그램 $\mathcal{P}X_{n-1}$과 현재 추정 시노그램 $\widetilde{Y} \odot \widetilde{S}_n$의 잔차를 역투영하여 CT 영상 업데이트
- 이 과정이 물리적으로 타당한 방향으로 네트워크를 제약하므로, **다양한 하이퍼파라미터 설정에서도 안정적 성능** 달성 (Table 5: 모든 Loss 변형에서 40.34~41.50 dB)

### 3.5 일반화 성능 추가 향상 가능성

- **자동 금속 위치 검출 알고리즘** 통합: 수동 임계값 의존성 제거 가능
- **반지도/비지도 학습 확장**: 실제 임상 데이터의 paired ground truth 부재 문제 해결
- **다양한 CT 기하학(cone-beam 등)으로의 확장**: 현재 fan-beam에 특화
- **사전 지식의 다양화**: 조직별 CT 값 외에 해부학적 아틀라스(atlas) 등 추가 사전 지식 통합

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **Deep Unfolding 패러다임의 의료 영상 적용 확대**: InDuDoNet+는 deep unfolding 방법론이 MAR에서도 물리적 해석 가능성과 우수한 성능을 동시에 달성할 수 있음을 실증하였으며, 이는 sparse-view CT 복원, limited-angle CT, PET/MRI 영상 복원 등 다른 의료 영상 역문제(inverse problem)에도 적용 가능성을 시사한다.

2. **도메인 지식과 딥러닝의 체계적 융합 방법론**: Prior-net의 "사전 지식 기반 조잡 추정 + 얕은 CNN 미세 조정" 패러다임은 의료 영상 분야에서 도메인 전문 지식을 네트워크에 통합하는 새로운 방법론적 템플릿을 제공한다.

3. **경량 네트워크 설계의 중요성 재확인**: 178만 파라미터로 2,600만 파라미터급 모델을 능가하는 결과는, 무분별한 모델 크기 증대보다 물리적 모델에 기반한 구조적 설계가 더 효과적임을 보여준다.

4. **일반화 성능 평가 프로토콜 확립**: 크로스 바디 사이트(DeepLesion → Dental/SpineWeb) 평가 방식은 MAR 연구의 일반화 벤치마크로 활용될 수 있다.

### 4.2 향후 연구 시 고려할 점

1. **자동 금속 분할**: 현재 수동 임계값(2500 HU) 기반 분할의 한계를 극복하기 위해, 세그멘테이션 네트워크를 MAR 프레임워크에 통합하는 end-to-end 학습이 필요
2. **3D 볼륨 처리**: 현재 2D 슬라이스 단위 처리의 한계를 극복하기 위한 3D 확장
3. **반지도/비지도 학습**: 임상 데이터의 paired ground truth 부재 문제 해결을 위한 학습 패러다임 전환
4. **Cone-beam CT로의 확장**: 현재 fan-beam 기하학에 특화된 구조의 일반화
5. **다중 금속/다양한 금속 재질 대응**: 실제 임상에서의 다양한 금속 유형에 대한 로버스트니스 확보
6. **계산 효율성 추가 개선**: 10 스테이지 반복에 따른 순방향/역투영 연산의 계산 비용 절감

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근 방식 | InDuDoNet+와의 차이점 |
|------|------|----------|-------------------|
| **DuDoNet++** (Lyu et al., 2020) | 2020 | 이중 도메인, 금속 마스크 투영 인코딩 | 물리적 모델 미내재화, 휴리스틱 U-Net 기반, 약 26M 파라미터 |
| **DSCMAR** (Yu et al., 2020) | 2020 | 이중 도메인, 시노그램 완성 + FBP | 최종 CT 영상이 FBP로 고정되어 유연성 제한 |
| **ADN** (Liao et al., 2019b) | 2020 (TMI) | 비지도 학습, 아티팩트 분리 | 비지도이나 물리적 모델 미반영 |
| **DICDNet** (Wang et al., 2021a) | 2021 (TMI) | 해석 가능한 합성곱 사전 네트워크, 이미지 도메인 | 이미지 도메인에 한정, 시노그램 도메인 미활용 |
| **InDuDoNet** (Wang et al., 2021b) | 2021 (MICCAI) | Deep unfolding 이중 도메인 | Prior-net이 U-shape 블랙박스, 약 517만 파라미터 |
| **DuDoDR-Net** (Zhou et al., 2022) | 2022 (MedIA) | 이중 도메인 반복 네트워크, sparse view + MAR 동시 처리 | 다중 작업 동시 처리이나 물리적 모델 통합 정도가 상이 |
| **ACDNet** (Wang et al., 2022a) | 2022 | 적응형 합성곱 사전 네트워크 | MAR 특화 사전 학습, 이미지 도메인 중심 |
| **OSCNet** (Wang et al., 2022b) | 2022 (MICCAI) | 방향 공유 합성곱 표현 | 금속 아티팩트의 방향적 특성 활용, 단일 도메인 |

### 핵심 비교 관점:

**물리적 모델 통합 수준**: InDuDoNet+는 CT 순방향 투영 $\mathcal{P}$와 역투영 $\mathcal{P}^T$를 매 스테이지마다 명시적으로 사용하여, 시노그램-영상 간 상호 학습을 물리적으로 제약한다. DuDoNet++, DSCMAR 등은 FP/FBP 층을 도메인 변환에만 사용하고, 각 도메인의 복원은 독립적인 U-Net에 위임한다.

**해석 가능성**: ISTA-Net (Zhang & Ghanem, 2018), ADMM-Net (Yang et al., 2017) 등의 deep unfolding 전통에 따라, InDuDoNet+는 각 네트워크 모듈이 최적화 알고리즘의 특정 단계에 대응한다. 이는 DuDoNet++, DSCMAR의 휴리스틱 설계와 근본적으로 다르다.

**일반화 성능**: 크로스 바디 사이트 실험에서 InDuDoNet+의 우위가 가장 두드러지며, 이는 물리적 모델 제약과 경량화된 Prior-net의 시너지 효과로 분석된다.

---

## 참고자료

1. **Wang, H., Li, Y., Zhang, H., Meng, D., & Zheng, Y.** (2022). "InDuDoNet+: A Deep Unfolding Dual Domain Network for Metal Artifact Reduction in CT Images." *arXiv:2112.12660v2*, submitted to *Medical Image Analysis*. — 본 논문 원문
2. **Wang, H., Li, Y., Zhang, H., Chen, J., Ma, K., Meng, D., & Zheng, Y.** (2021). "InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction." *MICCAI 2021*, pp. 107–118. — 이전 버전 (학회 발표)
3. **Lyu, Y., Lin, W.A., Liao, H., Lu, J., & Zhou, S.K.** (2020). "Encoding Metal Mask Projection for Metal Artifact Reduction in Computed Tomography." *MICCAI 2020*, pp. 147–157. — DuDoNet++ 비교 방법
4. **Yu, L., Zhang, Z., Li, X., & Xing, L.** (2020). "Deep Sinogram Completion with Image Prior for Metal Artifact Reduction in CT Images." *IEEE Transactions on Medical Imaging*, 40, 228–238. — DSCMAR 비교 방법
5. **Lin, W.A., et al.** (2019). "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction." *CVPR 2019*, pp. 10512–10521.
6. **Zhang, J. & Ghanem, B.** (2018). "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing." *CVPR 2018*, pp. 1828–1837. — Deep unfolding 방법론적 기반
7. **Meyer, E., et al.** (2010). "Normalized Metal Artifact Reduction (NMAR) in Computed Tomography." *Medical Physics*, 37, 5482–5493. — 정규화 기법의 원류
8. **Zhou, B., et al.** (2022). "DuDoDR-Net: Dual-Domain Data Consistent Recurrent Network." *Medical Image Analysis*, 75, 102289.
9. **Wang, H., et al.** (2021). "DICDNet: Deep Interpretable Convolutional Dictionary Network for Metal Artifact Reduction in CT Images." *IEEE Transactions on Medical Imaging*, 41, 869–880.
10. **Beck, A. & Teboulle, M.** (2009). "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems." *SIAM Journal on Imaging Sciences*, 2, 183–202. — 근위 경사 기법의 이론적 기반
11. GitHub 리포지토리: https://github.com/hongwang01/InDuDoNet_plus
