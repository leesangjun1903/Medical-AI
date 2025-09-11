# Learning Fourier-Constrained Diffusion Bridges for MRI Reconstruction 

## 핵심 주장과 주요 기여

**핵심 주장:**
기존 diffusion 모델들이 Gaussian noise에서 fully-sampled 데이터로의 task-agnostic 변환을 학습하는 반면, MRI reconstruction에 최적화된 **Fourier-Constrained Diffusion Bridge (FDB)**는 moderately undersampled 데이터에서 fully-sampled 데이터로의 직접적인 변환을 학습하여 성능을 크게 향상시킬 수 있다는 것입니다.[1]

**주요 기여:**
1. **최초의 MRI reconstruction용 diffusion bridge**: MRI reconstruction을 위한 최초의 diffusion bridge 모델 제안[1]
2. **stochastic frequency-removal 기반 generalized diffusion process**: 랜덤한 공간 주파수 제거를 통한 새로운 diffusion 프로세스[1]
3. **enhanced sampling algorithm**: soft dealiasing을 위한 learned correction term을 포함한 개선된 샘플링 알고리즘[1]

### Diffusion Model : Task-agnostic
Diffusion 모델들이 task-agnostic하다는 설명은, 일반적으로 특정 작업(task)에 종속되지 않고 여러 데이터 생성 문제에 적용 가능한 특성을 의미합니다.  
Diffusion 모델은 데이터에 점진적으로 노이즈를 더하는 순전파(forward process)와 그 반대를 통해 노이즈로부터 원래 데이터를 복원하는 역전파(reverse process)를 학습하는 생성 모델로, 이 과정은 특정 task에 국한되지 않고 다양한 데이터 분포 학습에 범용적으로 사용됩니다.

즉, Diffusion 모델은 이미지, 텍스트, 그래프 등 다양한 형태의 데이터를 생성할 수 있어 task에 특화된 설계 없이도 작동 가능한 구조를 갖추고 있다는 점에서 "task-agnostic"이라 할 수 있습니다.  
이는 모델이 특정 분류, 회귀 등 고정된 목적이 아닌, 임의 데이터 생성이라는 보다 넓은 범위의 문제를 대상으로 설계되기 때문입니다.

undersampled(부분적으로 축소된) 데이터를 입력으로 받아 fully-sampled(완전한 샘플) 데이터를 출력하도록 모델을 학습시키며, 이 과정은 학습 데이터 내에서 undersampled-fully sampled 쌍을 활용하는 지도학습 문제로 정의됩니다.  
이를 통해 모델은 undersampled 데이터의 정보 손실을 보완하고 완전한 데이터로 변환하는 패턴을 학습합니다.

### Diffusion bridge model
Diffusion bridge 모델은 확산 과정(diffusion process)을 시작점과 끝점을 고정한 조건부 확산(bridge)으로 다루는 확률 모형입니다.  
즉, 미리 정해진 두 시점 사이에서 확산이 특정 값을 통과하도록 제한하여 그 경로를 모델링합니다.

쉽게 말해, 특정 시점 시작 위치에서 출발해 특정 시점 끝 위치에 도달하는 확산 경로를 시뮬레이션하거나 생성하는 방법입니다.  
이때 경로는 확률적인 노이즈를 포함한 연속적인 움직임이며, 시작과 끝점이 고정되어 있어 일반 확산 과정과 다릅니다.

주로 이 모델은 확률적 경로 생성, 시간에 따른 경로 변화 예측, 이미지 또는 데이터 사이 연속 보간 등에서 활용합니다.  
예를 들어, 이미지 변환 모델에서는 두 이미지 분포 간의 연속적인 전환을 구현하는 데 사용됩니다.

## 해결하고자 하는 문제

### 기존 방법들의 한계
1. **Task-specific priors**: 특정 imaging operator에 종속되어 domain shift에 취약[1]
2. **Common diffusion priors**: asymptotic normality assumption으로 인한 성능 손실 - target transformation(undersampled → fully-sampled)과 learned transformation(Gaussian noise → fully-sampled) 간의 불일치[1]
3. **기존 diffusion bridges**: deterministic degradation에 기반하여 MRI의 stochastic undersampling pattern에 적합하지 않음[1]

## 제안하는 FDB 방법론

### 1. 수학적 모델링

**Diffusion Process:**
FDB는 fully-sampled data $$X_0$$에서 moderately undersampled data $$X_{T_f}$$로의 매핑을 수행합니다.

**Degradation Operator:**
대각선 frequency-removal 매트릭스 $$\Lambda_t$$를 사용:

$$
X_t = \left(\prod_{\tau=1}^{t} \Lambda_\tau\right) X_0 = \bar{\Lambda}_t X_0
$$

**Spatio-temporal Point Process:**

$$
S_t = \{(k_1, ..., k_n) : k_i \sim U[1, N_K], k_i \notin \bigcup_{\tau=1}^{t-1} S_\tau, r(k_i) > \bar{r}_t\}
$$

**Radius Threshold Scheduling:**

$$
\bar{r}_t = r_{\max} - r_{\max}(1 - \sqrt{R'})(t/T_f)
$$

### 2. Training Objective

**Generalized Diffusion Process:**

$$
x_t = \alpha_t C_t x_0 + \sigma_t z
$$

여기서 $$\alpha_t = 1, \sigma_t = 0$$ (noise-free process)

**Training Loss:**

$$
L_{FDB-ub} = \mathbb{E}_{t,x(0,t)}[\|G_\theta(x_t, t) - x_0\|^2]
$$

### 3. Enhanced Sampling Algorithm

**Corrected Sampling:**

$$
x_{t-1} = x_t + (C_{t-1} - C_t)\tilde{x}_0 + w_t C_t(\tilde{x}_0 - x_t)
$$

**Learned Correction Weight:**

$$
w_t = \frac{\mathbb{E}_{t,X(t-1,t)}[\|X_{t-1}\|^2 - \|X_t\|^2]}{\mathbb{E}_{t,X(0,t)}[\|X_0\|^2 - \|X_t\|^2]}
$$

## 모델 구조

FDB는 UNet 기반 architecture를 사용하며, 다음과 같은 특징을 가집니다:[1]
- **Sinusoidal time encoding**: 다양한 time step 범위로의 extrapolation 가능
- **Multi-channel input/output**: real과 imaginary component를 별도 채널로 처리
- **Coil compression**: multi-coil 데이터를 5개 virtual coil로 압축하여 90% 이상의 variance 보존

## 성능 향상

### Within-Domain Performance
- **IXI dataset**: LORAKS 대비 평균 7.5dB PSNR, 24.4% SSIM 향상[1]
- **fastMRI dataset**: task-specific prior 대비 평균 4.3dB PSNR, 12.0% SSIM 향상[1]
- **전체 비교**: 기존 diffusion prior 대비 평균 4.8dB PSNR, 7.6% SSIM 향상[1]

### Cross-Domain Performance
1. **Sampling density shift** (2D → 1D): 평균 5.6dB PSNR, 30.1% SSIM 향상[1]
2. **Dataset shift** (fastMRI → IXI): 평균 4.8dB PSNR, 34.8% SSIM 향상[1]

## 일반화 성능 향상

### 핵심 일반화 메커니즘
1. **Task-relevant degradation**: MRI physics에 맞는 frequency removal을 통해 target transformation과의 alignment 향상[1]
2. **Finite start-point**: asymptotic start-point 대신 moderate degradation level 사용으로 domain shift에 더 robust[1]
3. **Stochastic degradation**: deterministic degradation과 달리 다양한 undersampling pattern에 적응 가능[1]

### Ablation Study 결과
- **R' = 2 (lowest degradation)**: 가장 높은 성능, 낮은 R' 값이 일반적으로 더 나은 성능 제공[1]
- **Correction term**: 없을 경우 약 4-5dB PSNR 감소[1]
- **Learned weight scheduling**: linear scheduling 대비 significant 성능 향상[1]

## 한계점

1. **Inference time**: 여전히 multiple diffusion step이 필요하여 one-step GAN 대비 느림[1]
2. **Fully-sampled data dependency**: training에 fully-sampled acquisition 필요[1]
3. **Architecture limitation**: UNet 기반으로 long-range context capture에 한계[1]
4. **Contrast-specific optimization 부재**: 여러 contrast를 pooling하여 학습하지만 contrast-specific modulation 없음[1]

## 향후 연구에 미치는 영향

### 긍정적 영향
1. **Diffusion bridge paradigm**: MRI reconstruction에서 diffusion bridge 접근법의 효과성 입증으로 관련 연구 활성화 예상
2. **Physics-informed generative modeling**: 의료 영상의 물리적 특성을 고려한 generative model 설계의 중요성 강조
3. **Cross-domain robustness**: domain shift 문제 해결을 위한 새로운 접근법 제시

### 향후 고려사항

1. **Architecture 개선**: Transformer-based backbone 적용을 통한 long-range dependency 모델링[1]
2. **Acceleration 전략**: 
   - Adversarial learning 통합으로 large step size 가능
   - Distillation 기법을 통한 inference 가속화[1]
3. **Self-supervised learning**: Fully-sampled data 의존성 해결을 위한 self-supervised 또는 cycle-consistent learning 도입[1]
4. **Multi-contrast optimization**: Contrast-specific feature modulation 또는 joint reconstruction 접근법[1]
5. **Subject-specific adaptation**: 개별 피험자에 대한 prior adaptation 방법론 개발[1]

**결론적으로**, 이 논문은 MRI reconstruction 분야에서 physics-informed diffusion bridge의 가능성을 보여주며, 향후 의료 영상 재구성 연구에서 task-specific degradation modeling의 중요성을 강조하는 landmark work로 평가됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e26b58e5-dbfb-453d-9c15-676ab1ef25e9/2308.01096v3.pdf)
