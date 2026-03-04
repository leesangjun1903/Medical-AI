# Topology-Preserving Computed Tomography Super- Resolution Based on Dual-Stream Diffusion Model

이 논문은 초고해상도 CT(UHRCT)를 목표로 하는 조건부 확산 모델에 “구조/위상 보존용 이중 스트림 네트워크 + 혈관·blob 증강 연산자”를 결합해, 기존 CT 초해상화(SR)에서 빈번한 위상 왜곡과 인공물을 크게 줄이면서도 화질·진단 성능을 유지/향상시킬 수 있다고 주장합니다. 특히 UHRCT(0.34×0.34 mm², 1024×1024) 데이터셋을 구축하고, 영상 품질·구조 보존·고수준(검출/분할) 성능 모두에서 SOTA 방법보다 우수함을 보이는 것이 핵심 기여입니다.[^1_1][^1_2]

***

## 1. 핵심 주장과 주요 기여

- **핵심 주장**
    - 일반적인 딥러닝 기반 CT SR(특히 GAN/확산 기반)은 해상도는 높이지만 혈관, 섬유화, 종양 경계 등 중요한 구조에서 위상 왜곡·2차 인공물을 일으켜 임상 적용에 위험하다는 문제를 제기합니다.[^1_1]
    - 이를 해결하기 위해, **조건부 확산 모델의 역과정 안에 ‘영상 도메인 + 구조 도메인’ 이중 스트림 네트워크와 구조 증강 연산자를 삽입**하면, 고해상도·고주파 정보를 복원하면서도 혈관/병변 구조를 정합(topology-preserving)하게 유지할 수 있다고 주장합니다.[^1_1]
- **주요 기여 요약**

1. **Dual-Stream Diffusion Model (DSDM)**: LRCT $x$를 조건으로 HRCT $y$를 복원하는 조건부 확산 모델 안에, 영상 스트림과 구조 스트림이 병렬로 작동하는 **Dual-Stream Structure-Preserving (DSSP) 네트워크**를 도입.[^1_1]
2. **Hessian 기반 구조 증강 연산자**: 혈관형(선형)·blob형(결절 등) 구조를 동시에 강조하는 새로운 구조 커널 $V_C$와 이를 신경망으로 근사한 연산자 $O_{F_c}$를 설계해, 구조 도메인에서의 감독 신호를 제공.[^1_1]
3. **이미지/구조 이중 도메인 손실**: 픽셀 도메인(PSNR/SSIM 향상)과 구조 도메인(topology 보존)을 동시에 최소화하는 복합 손실을 정의.[^1_1]
4. **UHRCT 데이터셋 구축**: 1024×1024, 0.34×0.34 mm² 해상도의 초고해상도 흉부 CT 87세트를 수집해 SR 학습/평가용 공개 데이터로 제시.[^1_1]
5. **정량·정성 및 다운스트림 평가**: PSNR/SSIM/VIF/SMSE에서 bicubic, SRCNN, SRResNet, Cycle-GAN, SR3 대비 우수하며, 혈관 분할·결절 검출 성능이 원본 HRCT와 비슷하거나 더 좋은 결과를 보고.[^1_1]

***

## 2. 해결하고자 하는 문제

### 2.1 CT 초해상화에서의 위상 왜곡 문제

- 저선량·저해상도 CT(LRCT)는 방사선량과 비용을 줄이지만, 해상도 저하와 노이즈로 인해 병변 경계·소혈관 등 미세 구조가 손상되어 진단에 악영향을 미칩니다.[^1_1]
- 기존 딥러닝 SR(CNN, GAN, 확산)은 PSNR/SSIM을 높이면서 시각적으로 선명한 이미지를 생성하나,
    - 혈관이 끊기거나 새로 생기고,
    - 결절 모양이 변형되는 등의 **위상(topology)·형태학적 왜곡**을 만들어낼 수 있으며, 이는 임상적으로 치명적일 수 있습니다.[^1_3][^1_1]


### 2.2 UHRCT 타깃 SR의 공백

- 기존 CT SR 연구는 주로 $512 \times 512$, 약 $0.8 \times 0.8\ \text{mm}^2$ 수준의 HRCT를 타깃으로 하며, 최근 임상에 등장한 **초고해상도 UHRCT(1024×1024, $\sim0.3$ mm)**를 목표로 한 연구는 거의 없었습니다.[^1_2][^1_1]
- 논문은 **LRCT(예: 256×256)를 1024×1024 UHRCT로 직접 초해상화**하는 문제(×4 업샘플링)를 설정하고, 이때 구조 왜곡이 특히 심각하다는 점을 강조합니다.[^1_1]

***

## 3. 제안 방법: 확산 모델과 수식

### 3.1 조건부 확산 모델의 기본 형태

- 조건부 확산 모델은 LRCT $x$가 주어졌을 때 HRCT $y$의 분포 $p_\theta(y \mid x)$를 근사합니다.[^1_1]
- 논문은 **continuous-time DDPM/variance-exploding 스타일의 단일 스텝 학습 표현**으로 다음과 같은 노이즈 예측 손실을 사용합니다.[^1_1]

$$
\mathcal{L}_{\text{diff}} 
= \mathbb{E}_{x,y} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \mathbb{E}_{\gamma}
\Big\|
G_{\text{DSSP}}\big(x,\ \sqrt{\gamma}\,y + \sqrt{1-\gamma}\,\epsilon,\ \gamma\big) - \epsilon
\Big\|_2^2
$$

- 여기서 $\gamma \in (0,1)$는 노이즈 강도를 나타내는 스케일 파라미터이며, $G_{\text{DSSP}}$는 이 논문의 핵심인 **Dual-Stream Structure-Preserving 네트워크**입니다.[^1_1]


### 3.2 Hessian 기반 영상 증강 연산자

- 위치 $x = [x_1, x_2]^T$에서의 CT 강도 $I(x)$에 대해, 스케일 $s$에서의 2×2 Hessian은 다음과 같이 정의됩니다.[^1_1]

$$
H_{ij}(x) = s^2\, I(x)\, \frac{\partial^2 G(x,s)}{\partial x_i \partial x_j}, \quad i,j \in \{1,2\}
$$

- $G(x,s)$는 2D Gaussian 커널, 그 고유값을 $\lambda_1,\lambda_2$ ($|\lambda_1| \le |\lambda_2|$)라 할 때,
    - blob 구조: $\lambda_1 \approx \lambda_2$
    - 혈관형 구조: $|\lambda_2| \gg |\lambda_1|$ 특성을 가집니다.[^1_1]
- 이를 이용해 혈관·blob 구조를 동시에 강조하는 새 구조 커널을 정의합니다.[^1_1]

$$
V_C = 
\frac{\lambda_2 \big(\kappa_1 \lambda_1 + \lambda_\tau\big)\big(\kappa_2 \lambda_2 + \lambda_\tau\big)}
{\big(\lambda_1 + \lambda_2 + \lambda_\tau\big)^3}
$$

- 여기서 $\kappa_1, \kappa_2$는 혈관/블롭 민감도를 조절하는 파라미터이며, $\lambda_\tau$는 다음과 같이 정의됩니다.[^1_1]

$$
\gamma = \left| \frac{\lambda_2}{\lambda_1} \right| - 1,
$$

$$
\lambda_\tau = 
\frac{(\lambda_2 - \lambda_1)\big(e^\gamma - e^{-\gamma}\big)}
{e^\gamma + e^{-\gamma} + \lambda_1}
$$

- $\lambda_1 \approx \lambda_2$이면 $\lambda_\tau \approx \lambda_1$,
$|\lambda_2| \gg |\lambda_1|$이면 $\lambda_\tau \approx \lambda_2$가 되어 두 경우를 균형 있게 처리하도록 설계되었습니다.[^1_1]
- 실제 학습 시에는 이 연산이 미분 가능성이 낮고 계산량이 크므로, **Hessian 기반 필터 $F_c$를 근사하는 경량 합성곱 연산자 $O_{F_c}$**를 학습해 사용합니다.[^1_1]

***

## 4. 모델 구조: Dual-Stream DSSP 네트워크

### 4.1 전체 구조 개요

- **Forward process**: HRCT $y$에 점점 강한 Gaussian 노이즈를 추가해 노이즈 이미지 $\tilde{y}$를 생성하는 표준 확산 전방 과정.[^1_1]
- **Reverse process**: 노이즈 이미지에서 시작해 여러 스텝의 denoising을 통해 HRCT를 복원하는 과정에서, 각 스텝마다 **DSSP 네트워크**가 노이즈 $\epsilon$을 예측합니다.[^1_1]
- DSSP 네트워크는 두 개의 브랜치로 구성됩니다.[^1_1]

1. **Image branch** $G^{\text{image}}_{\text{DSSP}}$: 노이즈가 있는 HRCT 추정값을 입력받아, 원본 영상 도메인에서 SR 결과를 출력.
2. **Structure branch** $G^{\text{struct}}_{\text{DSSP}}$: LRCT와 구조 증강 맵(혈관/블롭)을 concat해서 입력으로 받아, 구조 도메인(혈관/결절 강조 맵)에서 SR 결과를 출력.
- 두 브랜치는 **Feature fusion 모듈**에서 병합되어 최종 SRCT를 생성하며, 이를 통해 **영상의 텍스처와 구조적 위상을 동시에 보존**하는 것을 목표로 합니다.[^1_1]


### 4.2 손실 함수 (영상/구조 이중 도메인)

- 기준 HRCT 이미지를 $y_t$라 할 때, 픽셀 도메인 손실은 다음과 같습니다.[^1_1]

```math
L^{\text{pixel}}_{\text{SR}}
= 
\big\|
G^{\text{image}}_{\text{DSSP}}(y_t) - y_t
\big\|_2^2
+
\lambda_{L1}\,
\big\|
G^{\text{image}}_{\text{DSSP}}(y_t) - y_t
\big\|_1
```

- 구조 도메인 손실은, 구조 연산자 $O_{F_c}$를 통해 얻은 구조 맵과 구조 브랜치 출력을 정합시키는 형태입니다.[^1_1]

```math
L^{\text{struct}}_{\text{SR}}
=
\big\|
O_{F_c} \circ G^{\text{image}}_{\text{DSSP}}(y_t) 
- O_{F_c} \circ y_t
\big\|_1
+
\big\|
G^{\text{struct}}_{\text{DSSP}}(y_t) 
- O_{F_c} \circ y_t
\big\|_1
```

- 최종 SR 목적함수는 다음과 같이 정의됩니다.[^1_1]

$$
L_{\text{SR}} = L^{\text{pixel}}_{\text{SR}} + \lambda_2\, L^{\text{struct}}_{\text{SR}}
$$

- 이 손실은
    - 픽셀 차원 재구성 품질(PSNR/SSIM),
    - 구조 도메인에서의 혈관/병변 위상 보존
두 측면을 동시에 최적화하도록 설계되어, **구조 왜곡 없이 선명한 SR 결과**를 유도합니다.[^1_1]

***

## 5. 데이터셋, 실험 설정 및 성능

### 5.1 데이터셋 구성

- **Dataset 1 (UHRCT 2D)**:
    - 입력: 256×256, 1.36×1.36 mm²
    - 출력: 1024×1024, 0.34×0.34 mm² (×4 업샘플링).[^1_1]
- **Dataset 2 (3D SR)**:
    - 입력: 256×256×1X, 1.60×1.60×5.00 mm³
    - 출력: 512×512×5X, 0.80×0.80×1.00 mm³.[^1_1]
- **Dataset 3 (LUNA16)**:
    - 퍼블릭 LUNA16 데이터셋에서 256×256 → 512×512 SR.[^1_4][^1_1]
- UHRCT 데이터셋은 최첨단 CT 장비로 획득한 87개의 초고해상도 스캔으로 구성되며, 이는 이 논문의 중요한 실제 임상 기반 학습 자원입니다.[^1_1]


### 5.2 비교 방법 및 지표

- 비교 대상: bicubic, SRCNN, SRResNet, Cycle-GAN(GAN-CIRCLE 계열), SR3(확산 기반 SR).[^1_1]
- 평가 지표:[^1_1]
    - **PSNR**, **SSIM**: 전통 영상 품질 지표.
    - **VIF (Visual Information Fidelity)**: 인간 지각·진단 관점에서 정보 보존 정도를 평가.
    - **SMSE (Structure MSE)**: Frangi 필터로 추출한 혈관 구조 맵 간의 L2 차이로 정의되는 구조 손상 지표.

$$
\text{SMSE} 
= \big\| F_{\text{Frangi}}(\text{HRCT}) - F_{\text{Frangi}}(\text{SRCT}) \big\|_2^2
$$

### 5.3 정량/정성 성능

- **Dataset 1 (UHRCT 2D)**에서:
    - Cycle-GAN: PSNR ≈ 37.32 dB, SMSE ≈ 0.462.[^1_1]
    - SR3: PSNR ≈ 37.18 dB, SMSE ≈ 0.474.[^1_1]
    - **제안 방법**: PSNR ≈ 40.75 dB, SSIM ≈ 0.992, VIF ≈ 0.977, SMSE ≈ 0.162로,
        - PSNR/SSIM/VIF 최고,
        - SMSE 최저(구조 손상이 가장 적음)입니다.[^1_1]
- **Dataset 2, 3**에서도 대부분의 지표에서 제안 방법이 최고 성능을 보이며, 특히 SMSE와 구조 강조 이미지에서의 시각적 품질이 타 방법보다 우수합니다.[^1_1]
- **고수준 태스크 평가**:
    - 폐결절 검출: U-Net, V-Net, 3D-DCNN 등 검출 모델을 HRCT vs SRCT에 적용했을 때, SRCT에서의 성능이 HRCT와 비슷하거나 오히려 향상된 경우가 보고됩니다.[^1_1]
    - 혈관/기도 분할: 3D U-Net, V-Net, nnUNet, Qin et al., Nardelli et al. 등 다양한 분할 네트워크에서도 SRCT 성능이 HRCT와 동등 혹은 근소하게 우수합니다.[^1_1]
    - 이는 SR이 **진단에 필요한 구조 정보를 손상시키지 않을 뿐 아니라, 해상도 향상으로 인해 일부 태스크에서 이득**이 있다는 간접적 증거입니다.[^1_4][^1_1]

***

## 6. 모델의 한계와 해석

논문에 직접적으로 “Limitation” 섹션이 있지는 않지만, 내용과 설정으로부터 다음과 같은 한계를 유추할 수 있습니다(이 부분은 논문의 서술 + 일반적 상식에 기반한 해석임을 전제로 합니다).[^1_1]

1. **도메인/장비 일반화 한계 가능성**
    - UHRCT는 특정 기관(두 대의 스캐너)에서 수집한 87건 정도의 데이터에 기반하고 있어, 다른 병원·스캐너·프로토콜에 대한 일반화는 추가 검증이 필요합니다.[^1_1]
2. **계산 비용 및 추론 시간**
    - 확산 모델 자체가 GAN/CNN 기반 SR보다 계산량이 크며, 여기에 이중 스트림 네트워크까지 결합되어 **실시간 적용**에는 최적화가 더 필요할 수 있습니다(논문은 속도 수치는 제시하지 않음).[^1_5][^1_1]
3. **2D/2.5D 위주의 설계**
    - Dataset 2에서 3D SR을 다루긴 하지만, 기본 설계는 슬라이스 단위(2D) 구조에 뿌리를 두고 있으며, 완전한 3D 확산·3D 구조 연산과의 비교는 이루어지지 않습니다.[^1_6][^1_1]
4. **혈관/블롭 외 구조에 대한 일반성**
    - 구조 연산자 $V_C$는 혈관·blob에 최적화되어 있어, 섬유화 패턴이나 불규칙한 종괴 등 복잡한 형태에 대해서는 별도 튜닝이 필요할 수 있습니다.[^1_7][^1_1]

***

## 7. 일반화 성능 향상 관점에서의 분석

### 7.1 구조 도메인 감독이 주는 일반화 효과

- **이미지 도메인 손실만 사용하는 SR**은 특정 스캐너/프로토콜에 맞는 intensity 패턴에 쉽게 과적합되어, 도메인 변화(새 병원·새 reconstruction kernel 등)에 취약할 수 있습니다.[^1_8][^1_9]
- 본 논문의 구조 손실 $L^{\text{struct}}_{\text{SR}}$은, intensity가 아니라 **혈관/결절 형태에 기반한 구조 맵**에 정합을 강제하기 때문에,
    - 스캐너나 reconstruction kernel이 달라져도 비교적 안정적인 **형태학적 priors**를 학습하게 만들 수 있습니다.[^1_4][^1_1]
- 실제로,
    - 서로 다른 해상도/체적 설정(Dataset 1–3)과
    - 서로 다른 다운스트림 태스크(검출, 분할)에 대해
**하나의 프레임워크가 좋은 성능을 유지**하는 것은 구조 기반 감독이 일정 수준의 일반화 능력을 제공함을 시사합니다.[^1_1]


### 7.2 확산 모델의 본질적 강점

- 확산 모델은 **likelihood 기반의 점진적 복원**으로, GAN보다 mode collapse가 적고, 데이터 분포 전체를 포괄적으로 학습하는 경향이 있어 **노이즈 조건·도메인 변화에 더 강인한 경향**이 보고되고 있습니다.[^1_10][^1_9][^1_4]
- 특히 의료 영상에서,
    - “Noise Controlled CT Super-Resolution with Conditional Diffusion Model”(Wang et al., 2024)은 노이즈 수준 제어를 통해 다양한 노이즈 조건에서 안정적인 SR을 달성함을 보이며,[^1_9][^1_11]
    - “A Super-Resolution Diffusion Model for Recovering Bone Microstructure from CT Images”(Chan \& Rajapakse, Radiology AI 2023)은 뼈 미세구조 복원에서 diffusion 기반 SR이 CNN/GAN 대비 더 신뢰할 만한 구조 복원을 제공함을 보고합니다.[^1_4]
- 이 논문 역시 **조건부 확산 + 구조 도메인 priors**라는 조합을 통해, LRCT의 노이즈/블러 패턴이 어느 정도 달라져도 **해부학적 구조를 우선적으로 복원**하는 방향으로 일반화를 기대할 수 있습니다.[^1_2][^1_1]


### 7.3 향후 일반화 성능 강화를 위한 확장 방향

연구자의 관점에서, 이 프레임워크를 보다 강력한 일반화 모델로 확장하기 위해 고려할 수 있는 방향은 다음과 같습니다.

- **다기관·다스캐너 학습**
    - 현재는 제한된 기관·장비에 기반하므로, 다기관 UHRCT·LRCT 합동 데이터셋을 구축해 **도메인 generalization 혹은 domain adaptation**을 명시적으로 포함시키는 것이 필요합니다.[^1_8][^1_1]
- **Domain-agnostic 구조 표현**
    - 구조 연산자를 Hessian 기반에서 더 일반적인 **학습형 구조 추출기(neural structure extractor)**로 확장하면, 다양한 장기/질환 구조를 포괄하는 구조 도메인 감독이 가능해집니다(이는 일반 영상 SR에서 SPSR의 Neural Structure Extractor와 유사한 아이디어).[^1_3]
- **불확실성·신뢰도 추정**
    - diffusion 모델의 샘플 분산을 이용해, 각 위치의 SR 결과에 대한 **불확실성 맵**을 추정하면, generalization이 떨어지는 영역(예: 도메인 차이 큰 구역)을 임상의가 인지할 수 있습니다.[^1_10][^1_4]

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 대표 관련 연구들

아래는 CT/의료 영상 초해상화에서 구조 보존·확산 모델을 활용한 대표 연구들을 정리한 것입니다.


| 연구 (연도) | 모달리티 / 과제 | 핵심 아이디어 | 구조/위상 보존 방식 |
| :-- | :-- | :-- | :-- |
| **본 논문: Topology-Preserving CT SR based on Dual-Stream Diffusion (MICCAI 2023)**[^1_1][^1_2] | CT, LR→UHRCT SR | 조건부 확산 + 이중 스트림(DSSP) 네트워크 + Hessian 기반 구조 연산자 | 영상 도메인 + 구조 도메인 손실, SMSE로 혈관 구조 평가 |
| **Structure-Preserving Image SR (SPSR, Ma et al., T-PAMI 2021)**[^1_3][^1_12] | 일반 자연영상 SISR | gradient map / neural structure extractor로 구조 priors 제공, GAN 기반 | gradient 손실 + 구조 추출기 출력으로 구조 왜곡 감소 |
| **A Super-Resolution Diffusion Model for Recovering Bone Microstructure from CT Images (Chan \& Rajapakse, Radiol AI 2023)**[^1_4] | 골 미세구조(근위 대퇴골) SR | 미세 CT를 ground truth로 삼아 diffusion 기반 SR로 골 미세구조 복원 | 골 소주(trabeculae) 구조 지표(두께, 연결성 등)로 구조 평가 |
| **Noise Controlled CT SR with Conditional Diffusion Model (Wang et al., 2024)**[^1_9][^1_11] | 의료 CT SR (노이즈 제어) | 조건부 diffusion에 노이즈 수준 제어 + hybrid(시뮬레이션+실측) 데이터 학습 | 명시적 구조 손실보다는 노이즈/해상도 trade-off 제어에 초점 |
| **Taming Stable Diffusion for CT Blind SR (Li et al., 2025)**[^1_13] | CT blind SR | 대규모 Stable Diffusion을 CT에 적응, 텍스트/이미지 조건부로 복잡 열화 복원 | 구조 보존은 주로 perceptual/텍스처 관점, 위상 보존은 간접 평가 |

### 8.2 비교 관점

1. **구조/위상 보존 전략의 차이**
    - SPSR는 gradient map 및 neural structure extractor를 이용해 **일반적 구조(에지/텍스처)**를 보존하는 반면,[^1_3]
    - 본 논문은 **의료 영상 특화(Hessian 기반 혈관/블롭) 구조 연산자**를 사용해, 임상적으로 중요한 혈관 및 결절 구조를 직접적으로 강화·감독합니다.[^1_1]
    - 골 미세구조 SR(Chan \& Rajapakse)는 뼈 미세구조 지표를 통해 구조 유지 여부를 평가하지만, 네트워크 아키텍처 차원에서 혈관/위상 보존을 설계하지는 않습니다.[^1_4]
2. **확산 모델 활용 방식**
    - Noise Controlled CT SR는 조건부 diffusion에 **노이즈 제어**를 결합해, 고해상도·저노이즈의 균형을 찾는 데 초점.[^1_9]
    - 본 논문은 노이즈 제어보다 **denoising 스텝 내부 구조 priors(DSSP)를 삽입**해 topology를 직접 관리한다는 점이 차별점입니다.[^1_1]
    - Stable Diffusion 기반 CT blind SR는 대규모 자연영상 사전 학습 모델을 전이학습하여 데이터 부족 문제를 해결하지만, topology 측면 평가는 상대적으로 제한적입니다.[^1_13]
3. **임상적/다운스트림 지표 사용**
    - 대부분의 SR 논문은 PSNR/SSIM/LPIPS 등 픽셀 기준 지표에 집중하는 반면,[^1_14][^1_12]
    - 본 논문과 CARE/DM4CT류 연구는 **해부학 구조 기반 지표(SMSE, anatomy-aware metrics)**의 중요성을 강조합니다.[^1_5][^1_8][^1_1]
    - 특히 본 논문은 **실제 폐결절 검출·혈관 분할 성능**이 HRCT와 동등 이상임을 보여줌으로써, 단순 시각·픽셀 품질이 아닌 **임상 유용성 차원에서 SR을 평가**했다는 점이 강점입니다.[^1_1]

요약하면, 2020년 이후 구조 보존 SR 연구 흐름(gradient/structure extractor, anatomy-aware metrics 등)에 확산 모델을 접목한 첫 CT 특화 topology-preserving 프레임워크로 볼 수 있으며, 이후의 diffusion 기반 CT SR/재구성 연구(Noise Controlled CT SR, DM4CT, CARE 등)가 **구조·해부학 보존 지표**를 점점 더 중시하는 흐름과도 잘 맞물립니다.[^1_5][^1_8][^1_9][^1_4][^1_1]

***

## 9. 앞으로의 연구에 미치는 영향과 고려할 점

### 9.1 연구 방향에 미치는 영향

이 논문은 다음과 같은 방향에서 후속 연구에 영향을 줄 가능성이 큽니다.

1. **“토폴로지-인식 확산 모델” 패러다임의 확산**
    - 단순 픽셀/주파수 기반 SR이 아니라, **해부학/위상 priors를 내장한 확산 모델**이 필요하다는 것을 명확히 보여줍니다.[^1_9][^1_4][^1_1]
    - 이는 sparse-view CT 재구성, 저선량 CT 복원, multi-contrast MRI SR 등 다른 inverse problem에도 확장될 수 있습니다.[^1_6][^1_5]
2. **구조 도메인 손실 및 지표의 표준화**
    - Frangi 기반 SMSE와 같은 구조 손실/지표가 CT SR 평가에 자연스럽게 포함될 수 있고,[^1_1]
    - CARE/DM4CT에서 제안하는 anatomy-aware metrics와 결합해 **표준 평가 셋업**이 형성될 수 있습니다.[^1_8][^1_5]
3. **UHRCT를 타깃으로 하는 학습/평가의 본격화**
    - 1024×1024, 0.34 mm² 수준의 UHRCT 데이터셋 구축은 이후 UHRCT용 SR·재구성·분할·검출 모델 연구의 기반이 될 수 있습니다.[^1_2][^1_1]

### 9.2 앞으로 연구 시 고려할 점 (연구자 관점)

연구를 계속 확장한다면, 다음 요소들을 중점적으로 고려하는 것이 좋습니다.

1. **다기관·다질환 일반화와 도메인 시프트**
    - 다기관 UHRCT/LRCT, 다양한 reconstruction kernel, contrast phase(arterial/venous) 등을 아우르는 학습과 평가로 **도메인 로버스트니스**를 검증해야 합니다.[^1_8][^1_9][^1_1]
    - 필요하다면 domain adversarial training, style transfer, self-supervised adaptation 등을 결합할 수 있습니다.[^1_10][^1_3]
2. **구조 priors의 확장 (혈관 외 구조)**
    - 간 섬유화, 간/폐 종양, 심근 섬유 구조 등 다양한 형태에 맞는 구조 연산자를 설계하거나,
    - SPSR류 neural structure extractor처럼, **데이터 기반 구조 표현 학습**을 diffusion과 결합하는 방향을 탐색할 수 있습니다.[^1_3][^1_4]
3. **효율성 최적화 (실시간/대규모 적용)**
    - MedDDPM, SALIENT 등에서 제안된 latent/wavelet-domain diffusion, 2D/3D 하이브리드 전략 등을 참고해,[^1_15][^1_6][^1_5]
    - 이중 스트림 구조를 유지하면서 **추론 스텝 수·해상도·메모리 사용량을 줄이는 경량화**가 필요합니다.
4. **불확실성·신뢰도 지도 제공**
    - SR 결과가 HRCT와 크게 다른 영역(예: 드문 병변, 도메인 시프트 구역)에서는 불확실성 맵을 함께 제공해,
    - 임상의가 SR 결과를 어느 정도까지 신뢰할 수 있는지 한눈에 파악하도록 하는 것이 임상 도입에 중요합니다.[^1_9][^1_4]
5. **임상·독립 검증과 사용자 연구**
    - 독립 기관/독립 리더 스터디에서, SRCT 기반 진단의 민감도/특이도 변화, 판독 시간, inter-reader agreement 등을 평가해야 합니다.[^1_9][^1_4]
    - 특히, topology-preserving이라는 주장을 **misdiagnosis 감소**와 직접 연결할 수 있는 설계가 중요합니다(예: 위양성/위음성 변화 분석).

***

위 내용은 첨부된 MICCAI 2023 논문 원문과 공식 MICCAI 페이지, GitHub 코드 저장소, 그리고 2020년 이후의 관련 공개(accessible) 논문들(특히 SPSR, diffusion 기반 CT/의료 SR 연구)을 바탕으로 정리되었습니다.[^1_12][^1_16][^1_11][^1_13][^1_5][^1_2][^1_8][^1_3][^1_4][^1_9][^1_1]
<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29]</span>

<div align="center">⁂</div>

[^1_1]: MICCAI_23.pdf

[^1_2]: https://conferences.miccai.org/2023/papers/652-Paper1351.html

[^1_3]: https://arxiv.org/abs/2109.12530

[^1_4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10698592/

[^1_5]: https://arxiv.org/html/2602.18589v1

[^1_6]: https://arxiv.org/html/2602.04162v2

[^1_7]: https://bohrium.dp.tech/paper/arxiv/6ea342745b3629035b439eafbc1590d8986451b3aeabf0625e3e0fc3156963c8

[^1_8]: https://arxiv.org/html/2506.02093v1

[^1_9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11967985/

[^1_10]: https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging

[^1_11]: https://pubmed.ncbi.nlm.nih.gov/40181814/

[^1_12]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Ma_Structure-Preserving_Super_Resolution_With_Gradient_Guidance_CVPR_2020_paper.pdf

[^1_13]: https://arxiv.org/html/2506.11496v1

[^1_14]: https://arxiv.org/pdf/2412.03379.pdf

[^1_15]: https://arxiv.org/html/2602.23447v1

[^1_16]: https://github.com/Arturia-Pendragon-Iris/UHRCT_SR

[^1_17]: https://arxiv.org/html/2602.19314v1

[^1_18]: https://arxiv.org/html/2502.09793v1

[^1_19]: https://arxiv.org/pdf/2404.17357.pdf

[^1_20]: https://arxiv.org/html/2602.22430v1

[^1_21]: https://arxiv.org/html/2403.10695v1

[^1_22]: https://arxiv.org/html/2512.22209v1

[^1_23]: https://arxiv.org/html/2504.11148v1

[^1_24]: https://arxiv.org/pdf/2508.15151v1.pdf

[^1_25]: https://arxiv.org/html/2508.15151v1

[^1_26]: https://www.arxiv.org/pdf/2502.09793.pdf

[^1_27]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425028726

[^1_28]: https://conferences.miccai.org/2023/papers/160-Paper2586.html

[^1_29]: http://www.arxiv.org/abs/2501.06939

