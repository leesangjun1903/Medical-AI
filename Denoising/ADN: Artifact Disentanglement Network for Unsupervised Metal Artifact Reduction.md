# ADN: Artifact Disentanglement Network for Unsupervised Metal Artifact Reduction

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
기존의 CT 금속 아티팩트 제거(Metal Artifact Reduction, MAR) 딥러닝 방법들은 합성(synthesized) 데이터로 학습하는 **지도학습(supervised)** 방식에 의존하며, 이로 인해 실제 임상 데이터에 적용 시 **일반화 성능(generalization)이 크게 저하**된다. ADN은 **쌍을 이루는(paired) 데이터 없이** 아티팩트 제거를 학습하는 **최초의 비지도학습(unsupervised)** 접근법을 제안하며, 잠재 공간(latent space)에서 아티팩트와 콘텐츠를 분리(disentangle)함으로써 임상 데이터에서 우수한 일반화 성능을 달성한다.

### 주요 기여
1. **최초의 비지도학습 MAR 프레임워크**: 쌍을 이루는 학습 데이터 없이 금속 아티팩트를 제거하는 최초의 딥러닝 방법론 제안
2. **아티팩트 분리(Disentanglement) 개념**: 아티팩트와 해부학적 콘텐츠를 잠재 공간에서 분리하여, 아티팩트 제거·전이·자기복원 등 다양한 이미지 변환을 지원
3. **특화된 손실 함수 설계**: Adversarial loss, reconstruction loss, artifact consistency loss, self-reduction loss 등 4가지 손실 함수를 결합하여 쌍 데이터의 필요성을 제거
4. **임상 데이터에서의 우수한 일반화 성능**: 합성 데이터에서는 지도학습 방법과 유사한 성능을, 임상 데이터에서는 지도학습 방법을 유의미하게 능가하는 일반화 성능을 입증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

CT 영상에서 금속 임플란트(치과 충전재, 고관절 보철물 등)는 X-선을 비균일하게 감쇠시켜 **streaking 및 shading 아티팩트**를 유발한다. 기존 DNN 기반 MAR 방법들은 다음과 같은 근본적 한계를 가진다:

- **쌍 데이터 획득의 비현실성**: 동일 환자의 금속 아티팩트가 있는/없는 CT 이미지 쌍을 임상에서 얻는 것이 불가능
- **합성 데이터의 한계**: 빔 경화(beam hardening), 산란(scatter), 부분 체적 효과(partial volume effect) 등 복잡한 물리적 메커니즘을 합성 데이터가 완전히 재현하지 못함
- **도메인 이동(domain shift)**: 합성 데이터로 학습한 모델이 실제 임상 데이터에 적용 시 성능 저하

### 2.2 제안하는 방법

#### 2.2.1 문제 재정의: 아티팩트 분리(Artifact Disentanglement)

아티팩트가 있는 영상 도메인을 $\mathcal{I}^a$, 아티팩트가 없는 영상 도메인을 $\mathcal{I}$로 정의한다. 기존의 MAR 모델 $f: \mathcal{I}^a \to \mathcal{I}$를 학습하기 위해, 아티팩트가 있는 영상 $x^a$를 **콘텐츠 공간(content space)** $\mathcal{C}$와 **아티팩트 공간(artifact space)** $\mathcal{A}$로 분리한다.

#### 2.2.2 인코더와 디코더 (수식 포함)

**인코딩 단계**: 비쌍(unpaired) 입력 $x^a \in \mathcal{I}^a$와 $y \in \mathcal{I}$에 대해:

$$c_x = E^c_{\mathcal{I}^a}(x^a), \quad a = E^a_{\mathcal{I}^a}(x^a), \quad c_y = E_{\mathcal{I}}(y) $$

여기서 $E^c_{\mathcal{I}^a}$는 콘텐츠 인코더, $E^a_{\mathcal{I}^a}$는 아티팩트 인코더, $E_{\mathcal{I}}$는 아티팩트-프리 영상 인코더이다.

**디코딩 단계 — 아티팩트 합성**:

$$\hat{x}^a = G_{\mathcal{I}^a}(c_x, a), \quad \hat{y}^a = G_{\mathcal{I}^a}(c_y, a) $$

**디코딩 단계 — 아티팩트 제거**:

$$\hat{x} = G_{\mathcal{I}}(c_x), \quad \hat{y} = G_{\mathcal{I}}(c_y) $$

**자기 복원(Self-Reduction)**:

$$\tilde{y} = G_{\mathcal{I}}(E^c_{\mathcal{I}^a}(\hat{y}^a)) $$

최종 MAR 모델은 다음과 같이 구성된다:

$$f = G_{\mathcal{I}} \circ E^c_{\mathcal{I}^a}$$

#### 2.2.3 손실 함수

전체 목적 함수는 다음과 같다:

$$\mathcal{L} = \lambda_{\text{adv}}(\mathcal{L}^{\mathcal{I}}_{\text{adv}} + \mathcal{L}^{\mathcal{I}^a}_{\text{adv}}) + \lambda_{\text{art}}\mathcal{L}_{\text{art}} + \lambda_{\text{rec}}\mathcal{L}_{\text{rec}} + \lambda_{\text{self}}\mathcal{L}_{\text{self}}$$

**(1) Adversarial Loss** — 생성된 이미지의 현실성을 보장:

$$\mathcal{L}^{\mathcal{I}}_{\text{adv}} = \mathbb{E}_{\mathcal{I}}[\log D_{\mathcal{I}}(y)] + \mathbb{E}_{\mathcal{I}^a}[1 - \log D_{\mathcal{I}}(\hat{x})]$$

$$\mathcal{L}^{\mathcal{I}^a}_{\text{adv}} = \mathbb{E}_{\mathcal{I}^a}[\log D_{\mathcal{I}^a}(x^a)] + \mathbb{E}_{\mathcal{I},\mathcal{I}^a}[1 - \log D_{\mathcal{I}^a}(\hat{y}^a)] $$

**(2) Reconstruction Loss** — 인코딩-디코딩 과정에서 정보 보존:

$$\mathcal{L}_{\text{rec}} = \mathbb{E}_{\mathcal{I},\mathcal{I}^a}[\|\hat{x}^a - x^a\|_1 + \|\hat{y} - y\|_1]$$

**(3) Artifact Consistency Loss** — 해부학적 정확성 보장 (핵심 기여):

$$\mathcal{L}_{\text{art}} = \mathbb{E}_{\mathcal{I},\mathcal{I}^a}[\|(x^a - \hat{x}) - (\hat{y}^a - y)\|_1] $$

이 손실 함수는 $x^a$와 $\hat{x}$의 차이(아티팩트 제거량)와 $\hat{y}^a$와 $y$의 차이(아티팩트 합성량)가 동일한 아티팩트 코드 $a$에 의해 일관되어야 한다는 가정에 기반한다. $x^a$와 $\hat{x}$를 직접 비교하면 아티팩트를 보존하게 되는 문제를 피한다.

**(4) Self-Reduction Loss** — 합성된 아티팩트 제거를 통한 자기 감독:

$$\mathcal{L}_{\text{self}} = \mathbb{E}_{\mathcal{I},\mathcal{I}^a}[\|\tilde{y} - y\|_1] $$

### 2.3 모델 구조

ADN은 다음 구성 요소로 이루어진다:

| 구성 요소 | 역할 | 구조 |
|---------|------|------|
| $E_{\mathcal{I}} / E^c_{\mathcal{I}^a}$ | 콘텐츠 인코딩 | Downsampling blocks + Residual blocks |
| $E^a_{\mathcal{I}^a}$ | 아티팩트 인코딩 | Downsampling blocks (다중 스케일 피라미드 출력) |
| $G_{\mathcal{I}}$ | 아티팩트-프리 영상 생성 | Residual blocks + Upsampling blocks + Final block |
| $G_{\mathcal{I}^a}$ | 아티팩트 영상 생성 (APD) | Residual + Merge + Upsampling blocks |
| $D_{\mathcal{I}} / D_{\mathcal{I}^a}$ | 판별기 | PatchGAN 스타일 |

**핵심 구조적 기여 — Artifact Pyramid Decoding (APD)**:
- Feature Pyramid Network (FPN)에서 영감을 받아, $E^a_{\mathcal{I}^a}$가 다중 스케일 feature map을 출력
- $G_{\mathcal{I}^a}$가 디코딩 시 각 스케일에서 아티팩트 코드를 merge block ($1 \times 1$ convolution)으로 결합
- 고해상도 아티팩트 디테일을 효과적으로 복원

**설계 세부사항**:
- Strided convolution으로 다운샘플링 (max pooling 대신 — 생성 모델에 유리)
- Nearest neighbor interpolation + convolution으로 업샘플링 (checkerboard 효과 방지)
- Reflection padding 사용 (경계 아티팩트 최소화)
- Instance normalization + ReLU 활성화 함수

### 2.4 성능 향상

#### 합성 데이터(SYN) 정량적 결과

| 방법 | 유형 | PSNR | SSIM |
|-----|------|------|------|
| LI | 전통적 | 32.0 | 91.0 |
| NMAR | 전통적 | 32.1 | 91.2 |
| CNNMAR | 지도학습 | 32.5 | 91.4 |
| UNet | 지도학습 | **34.8** | 93.1 |
| cGANMAR | 지도학습 | 34.1 | **93.4** |
| CycleGAN | 비지도 | 30.8 | 72.9 |
| DIP | 비지도 | 26.4 | 75.9 |
| MUNIT | 비지도 | 14.9 | 7.5 |
| DRIT | 비지도 | 25.6 | 79.7 |
| **ADN (Ours)** | **비지도** | **33.6** | **92.4** |

- 기존 비지도학습 방법 대비 **PSNR +8.0 이상, SSIM +12.7 이상** 향상
- 지도학습 방법(UNet 34.8, cGANMAR 34.1)과 **유사한 수준** 달성

#### Ablation Study 결과

| 모델 | 손실 함수 | PSNR | SSIM |
|------|---------|------|------|
| M1 | $\mathcal{L}_{\text{adv}}$ only | 21.7 | 61.5 |
| M2 | M1 + $\mathcal{L}_{\text{rec}}$ | 26.3 | 82.1 |
| M3 | M2 + $\mathcal{L}_{\text{art}}$ | 32.8 | 91.6 |
| M4 (ADN) | M3 + $\mathcal{L}_{\text{self}}$ | **33.6** | **92.4** |

$\mathcal{L}\_{\text{art}}$의 추가가 가장 큰 성능 향상(+6.5 PSNR, +9.5 SSIM)을 가져왔으며, $\mathcal{L}_{\text{self}}$가 추가적인 개선(+0.8 PSNR, +0.8 SSIM)을 제공한다.

### 2.5 한계

1. **합성 데이터에서 지도학습 대비 약간의 성능 격차**: PSNR 기준 UNet(34.8) 대비 ADN(33.6)으로 약 1.2dB 차이
2. **정량적 평가의 제한**: 임상 데이터에 대한 ground truth가 없어 정량적 비교가 불가하며, 정성적 평가에 의존
3. **아티팩트 유형에 대한 가정**: 아티팩트와 콘텐츠가 잠재 공간에서 분리 가능하다는 가정이 모든 유형의 아티팩트에 성립하지 않을 수 있음
4. **아티팩트와 유사한 해부학적 구조의 오분리**: 금속 아티팩트와 유사한 외관의 병변이나 해부학적 구조가 아티팩트로 잘못 전이될 가능성
5. **단일 슬라이스 2D 처리**: 3D 볼륨 정보를 활용하지 못하여 슬라이스 간 일관성이 보장되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 ADN의 일반화 성능이 우수한 근본적 이유

ADN의 일반화 성능 우수성은 다음과 같은 구조적·방법론적 요인에 기인한다:

**(1) 합성 데이터 의존성 제거**

지도학습 방법은 $f: x^a_{\text{synth}} \to x$로 학습되어, 학습 시 접하는 아티팩트 분포 $P_{\text{synth}}(x^a)$와 테스트 시 실제 아티팩트 분포 $P_{\text{clinical}}(x^a)$ 간의 **도메인 이동(domain shift)**에 취약하다. ADN은 실제 임상 데이터에서 직접 아티팩트를 학습하므로 이 문제를 회피한다.

**(2) 귀납적 편향(Inductive Bias)의 활용**

ADN은 두 가지 수준의 귀납적 편향을 도입한다:
- **1차 귀납적 편향**: CT 영상을 아티팩트 있는 그룹과 없는 그룹으로 분류하여, 모델이 두 그룹 비교를 통해 자연스럽게 아티팩트 분리를 학습
- **2차 귀납적 편향**: 아티팩트 분리 가정이 잠재 공간에서의 조작을 안내하여, 모델 출력 간의 자기 감독(self-supervision) 신호를 생성 — 이것이 $\mathcal{L}\_{\text{art}}$와 $\mathcal{L}_{\text{self}}$의 기반

**(3) 교차 모달리티(Cross-Modality) 일반화**

CL2 실험에서 ADN은 cone-beam CT (CBCT) 영상에 대해서도 훈련 및 적용이 가능했다. 지도학습 방법은 CT 합성 데이터로만 학습되어 CBCT에 적용 시 완전히 실패한 반면, ADN은 아티팩트가 있는 CBCT 영상과 아티팩트가 없는 CT 영상을 직접 사용하여 학습했다.

**(4) 아티팩트 성질에 대한 불가지론적(Agnostic) 설계**

ADN의 문제 정의에서 아티팩트의 물리적 성질(빔 경화, 산란 등)에 대한 어떠한 가정도 하지 않는다. 이로 인해:
- 노이즈 제거 (CL1의 약간 노이즈한 입력)
- Streaking 아티팩트 제거 (CL2)
- 다양한 유형의 아티팩트 동시 처리

등이 가능하다.

### 3.2 일반화 성능 향상을 위한 향후 방향

**(1) 3D 볼륨 정보 활용**

현재 ADN은 2D 슬라이스 단위로 처리한다. 3D convolution이나 슬라이스 간 attention 메커니즘을 도입하면 슬라이스 간 일관성과 3D 아티팩트 패턴 포착이 가능하다.

**(2) 다중 도메인 학습(Multi-Domain Learning)**

여러 CT 장비, 스캔 프로토콜, 신체 부위에서 동시에 학습하면 아티팩트의 다양한 패턴을 포착하여 일반화 성능이 더욱 향상될 수 있다.

**(3) 물리 기반 사전 지식 통합**

아티팩트의 물리적 생성 메커니즘에 대한 약한 사전 지식을 네트워크 구조나 손실 함수에 반영하면, 순수 데이터 기반 접근보다 더 견고한 분리가 가능하다.

**(4) 사전 학습(Pre-training) 및 자기 지도 학습(Self-Supervised Learning)**

대규모 의료 영상 데이터셋에서 사전 학습된 표현을 활용하면, 소규모 임상 데이터에서도 효과적인 아티팩트 분리가 가능하다.

**(5) 적응적 아티팩트 코드 설계**

현재의 고정 차원 아티팩트 코드 대신, 아티팩트의 복잡도에 따라 적응적으로 코드 차원을 조절하는 메커니즘이 다양한 아티팩트에 대한 일반화를 향상시킬 수 있다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**(1) 비지도학습 MAR의 패러다임 전환**

ADN은 MAR 분야에서 "합성 데이터 기반 지도학습"이라는 기존 패러다임에 대한 근본적 대안을 제시했다. 이후 연구들이 비지도 또는 반지도(semi-supervised) 접근법을 적극적으로 탐구하는 계기가 되었다.

**(2) 분리 표현 학습(Disentangled Representation Learning)의 의료 영상 적용 확대**

잠재 공간에서 의미 있는 요소를 분리하는 아이디어가 MAR을 넘어 다양한 의료 영상 문제(디노이징, 디블러링, 모달리티 변환 등)에 확장 가능함을 보여주었다.

**(3) 임상 적용 가능성의 실증**

합성 데이터가 아닌 실제 임상 데이터에서 직접 학습하는 것이 더 실용적이고 견고한 접근임을 실험적으로 입증하여, 의료 AI의 임상 전환(clinical translation)에 대한 새로운 관점을 제시했다.

**(4) 아티팩트 합성의 부가적 활용**

ADN의 아티팩트 전이 기능은 데이터 증강(data augmentation)에 활용 가능하며, 이는 아티팩트가 있는 환경에서도 견고한 하류 작업(downstream task) 모델 학습에 기여할 수 있다.

### 4.2 향후 연구 시 고려할 점

1. **정량적 평가의 어려움**: 임상 데이터에 ground truth가 없으므로, 임상 전문가 평가, 하류 작업(segmentation, detection) 성능 비교 등 간접 평가 지표의 개발이 필요
2. **의료 안전성 검증**: 아티팩트 제거 과정에서 진단적으로 중요한 정보(병변, 미세 구조)가 손실될 위험성에 대한 체계적 검증
3. **계산 효율성**: 인코더-디코더 다중 경로 구조로 인한 학습/추론 비용 고려
4. **아티팩트 분리의 완전성**: 복잡한 아티팩트 패턴(예: 다중 금속 임플란트, 큰 금속체)에서의 분리 완전성 검증
5. **규제 요건**: 의료기기 소프트웨어로의 인허가를 위해 비지도학습 모델의 설명 가능성(explainability)과 재현성 확보

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구

| 연구 | 연도 | 핵심 접근법 | ADN과의 차이점 |
|------|------|-----------|-------------|
| **DuDoNet++** (Lyu et al.) | 2020 | 이중 도메인(sinogram+image) 비지도 MAR | 프로젝션 도메인 정보도 활용; ADN은 이미지 도메인만 사용 |
| **InDuDoNet** (Wang et al., TMI 2022) | 2022 | 반복적 이중 도메인 네트워크 | 물리적 모델 기반 해석 가능한 구조; ADN 대비 물리 사전 지식 통합 |
| **Score-MAR** (Song et al.) | 2022 | Score-based diffusion model 활용 MAR | 확산 모델 기반 생성으로 더 높은 품질; 하지만 추론 속도 느림 |
| **ACDNet** (Liang et al., MedIA 2021) | 2021 | Adaptive artifact disentanglement | ADN의 아이디어를 확장하여 적응적 분리 메커니즘 도입 |
| **UDAMAR** (Unsupervised Domain Adaptation) | 2021+ | 도메인 적응 기반 MAR | 합성→임상 도메인 적응; ADN은 도메인 적응 없이 직접 학습 |
| **CycleMedGAN** 변형들 | 2020+ | Cycle-consistency 기반 의료 영상 변환 | ADN의 artifact consistency loss가 CycleGAN의 cycle loss보다 MAR에 특화 |
| **MIST-net** (Peng et al., IEEE TMI 2022) | 2022 | Multi-scale implicit sinogram-to-image transformation | Sinogram 도메인 활용 가능 시 우수; ADN은 sinogram 불필요 |
| **Diffusion-based MAR** (다수 연구, 2023+) | 2023+ | DDPM/Score 기반 아티팩트 제거 | 생성 품질 우수하나 계산 비용 높고, 여전히 지도학습 의존 경향 |

### 5.2 비교 분석의 핵심 관찰

**(1) 이중 도메인 접근법의 부상**

DuDoNet++ 계열 연구들은 sinogram과 image 도메인을 모두 활용하여 ADN 대비 합성 데이터에서 더 높은 정량적 성능을 달성한다. 그러나 **sinogram 데이터의 가용성**이 전제되어야 하므로 임상 적용성에 제약이 있다. ADN의 이미지 도메인만 사용하는 접근은 이 측면에서 여전히 실용적 장점을 가진다.

**(2) 확산 모델(Diffusion Model)의 등장**

2022-2023년 이후 확산 모델 기반 MAR 연구가 급증했다. 이들은 생성 품질 면에서 GAN 기반 ADN을 능가할 수 있으나:
- 추론 시간이 수십~수백 배 느림
- 대부분 여전히 지도학습에 의존
- ADN의 비지도 아티팩트 분리 개념을 확산 모델에 결합하는 연구는 아직 초기 단계

**(3) ADN 프레임워크의 지속적 영향력**

ADN의 핵심 아이디어(아티팩트-콘텐츠 분리, artifact consistency loss)는 이후 다양한 연구에서 변형·확장되어 활용되고 있으며, 비지도 의료 영상 처리의 **기준선(baseline)**으로 빈번히 인용된다.

**(4) 물리 정보 통합의 트렌드**

최근 연구들은 순수 데이터 기반 접근에서 벗어나 CT 물리 모델(Beer-Lambert 법칙, Radon 변환 등)을 네트워크에 통합하는 경향이 있다. ADN은 물리 모델을 사용하지 않아 일반성은 높으나, 물리 정보를 통합한 후속 연구들이 특정 시나리오에서 더 높은 성능을 보인다.

### 5.3 연구 방향 제안

1. **ADN + Diffusion Model**: 비지도 아티팩트 분리 프레임워크에 확산 모델의 생성 능력을 결합
2. **ADN + Physics-informed Learning**: 아티팩트 분리에 약한 물리적 제약을 추가하여 분리의 정확성 향상
3. **ADN + Foundation Model**: 대규모 의료 영상 기반 모델(MedSAM 등)의 표현을 활용한 일반화 성능 극대화
4. **3D ADN**: 볼륨 데이터 처리를 통한 슬라이스 간 일관성 확보

---

## 참고 자료 및 출처

1. **Liao, H., Lin, W.-A., Zhou, S. K., & Luo, J.** "ADN: Artifact Disentanglement Network for Unsupervised Metal Artifact Reduction." *arXiv:1908.01104v4*, 2019. (본 논문 원문)
2. **Liao, H., Lin, W.-A., Yuan, J., Zhou, S. K., & Luo, J.** "Artifact Disentanglement Network for Unsupervised Metal Artifact Reduction." *MICCAI 2019*. (예비 버전)
3. **Lin, W.-A., Liao, H., et al.** "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction." *CVPR 2019*.
4. **Zhang, Y. & Yu, H.** "Convolutional Neural Network Based Metal Artifact Reduction in X-ray Computed Tomography." *IEEE Trans. Med. Imaging*, 37(6), 2018.
5. **Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A.** "Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks." *CVPR 2017*.
6. **Huang, X., Liu, M., Belongie, S. J., & Kautz, J.** "Multimodal Unsupervised Image-to-Image Translation." *ECCV 2018*.
7. **Wang, J., Zhao, Y., Noble, J. H., & Dawant, B. M.** "Conditional Generative Adversarial Networks for Metal Artifact Reduction in CT Images of the Ear." *MICCAI 2018*.
8. **Meyer, E., et al.** "Normalized Metal Artifact Reduction (NMAR) in Computed Tomography." *Medical Physics*, 2010.
9. **Locatello, F., et al.** "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations." *arXiv:1811.12359*, 2018.
10. **Gjesteby, L., et al.** "Metal Artifact Reduction in CT: Where Are We After Four Decades?" *IEEE Access*, 4, 2016.
11. **Wang, T., et al.** "InDuDoNet+: A Deep Unfolding Dual Domain Network for Metal Artifact Reduction." *IEEE TMI*, 2022. (2020년 이후 관련 연구)
12. **Peng, C., et al.** "An Interpretable Model-driven Deep Network for CBCT Reconstruction (MIST-net)." *IEEE TMI*, 2022. (2020년 이후 관련 연구)
13. **ADN GitHub Repository**: https://github.com/liaohaofu/adn

> **참고**: 2020년 이후 최신 연구의 비교 분석 부분은 해당 분야의 일반적인 연구 동향에 기반한 것이며, 특정 논문의 정확한 수치적 비교는 각 원논문을 직접 참조하시기를 권장합니다. 개별 후속 연구의 정확한 실험 결과는 원 논문을 통해 확인해 주십시오.
