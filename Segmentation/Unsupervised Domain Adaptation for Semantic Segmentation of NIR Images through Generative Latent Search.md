# Unsupervised Domain Adaptation for Semantic Segmentation of NIR Images through Generative Latent Search

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **NIR(근적외선) 이미지에 대한 피부 분할(skin segmentation)** 문제를 해결하기 위해, **타겟 데이터(NIR 이미지)에 전혀 접근하지 않고도** 소스 도메인(가시광선 Red 채널)에서 훈련된 분할 네트워크를 타겟 도메인에 적용할 수 있는 새로운 **Target-Independent UDA(비지도 도메인 적응)** 방법을 제안합니다.

핵심 아이디어: 타겟 이미지가 주어지면, **소스 도메인에서 해당 이미지와 가장 유사한 이미지('nearest-clone')를 VAE의 잠재 공간 최적화를 통해 탐색**하고, 이를 소스 도메인에서 훈련된 분할 네트워크의 프록시(proxy)로 사용합니다.

### 주요 기여

| 번호 | 기여 내용 |
|------|-----------|
| 1 | NIR 피부 분할을 UDA 문제로 재정의 (Red채널 → NIR 도메인) |
| 2 | Target-independent 'nearest-clone' 탐색 방법 제안 |
| 3 | 'nearest-clone' 존재성의 이론적 증명 (Lemma 1) |
| 4 | VAE 잠재 공간 최적화를 통한 joint sampling-optimization 알고리즘 개발 |
| 5 | 두 개의 NIR 피부 분할 데이터셋 신규 공개 및 SOTA 달성 |
| 6 | Synthia→Cityscapes 표준 UDA 벤치마크에서도 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 배경:**
- NIR 카메라는 야간 감시, 원격 심박수 측정 등 중요 응용에서 필수적이나, 대규모 NIR 레이블 데이터가 존재하지 않음
- 기존 딥러닝 기반 분할 방법은 대량의 레이블 데이터 요구
- 기존 UDA 방법들도 타겟 도메인의 비레이블 데이터에 대한 접근을 가정

**핵심 도전:**
> NIR 이미지의 레이블 데이터는 물론, **비레이블 타겟 데이터조차 없는 상황**에서 소스(Red 채널) → 타겟(NIR) 도메인 적응을 수행해야 함

---

### 2.2 제안 방법 및 수식

#### Step 1: 'nearest-clone' 존재성 증명 (Lemma 1)

소스 분포 $\mathcal{P}_s(\mathbf{x})$와 타겟 분포 $\mathcal{P}_t(\mathbf{x})$가 정의된 분리 가능한 거리 공간(separable metric space) $\{\mathcal{X}, \mathcal{D}\}$에서:

$$\mathbf{Prob}\Big(\mathbb{B}_r(\tilde{\mathbf{x}})\Big) \triangleq \int_{\mathbb{B}_r(\tilde{\mathbf{x}})} \mathcal{P}_s(\mathbf{x})\, d\mathbf{x} > 0, \quad \forall r > 0 \tag{1}$$

임의의 $\delta > 0$에 대해, $\mathcal{S}\_n$의 어떤 점도 $\mathbb{B}_\delta(\tilde{\mathbf{x}})$ 안에 없을 확률:

$$\mathbf{Prob}\left[\min_{i=1,2,..,n} \mathcal{D}\{\mathbf{x}_i, \tilde{\mathbf{x}}\} \geq \delta\right] = \left[1 - \mathbf{Prob}\Big(\mathbb{B}_\delta(\tilde{\mathbf{x}})\Big)\right]^n \tag{2}$$

따라서 'nearest-clone' $\tilde{\mathbf{x}}\_S$가 $\mathbb{B}_\delta(\tilde{\mathbf{x}})$ 내에 있을 확률:

$$\mathbf{Prob}\left[\tilde{\mathbf{x}}_S \in \mathbb{B}_\delta(\tilde{\mathbf{x}})\right] = 1 - \left[1 - \mathbf{Prob}\Big(\mathbb{B}_\delta(\tilde{\mathbf{x}})\Big)\right]^n \xrightarrow{n \to \infty} 1 \tag{3, 4}$$

즉, $n \to \infty$일 때 확률 1로 'nearest-clone'이 존재함.

---

#### Step 2: VAE를 통한 소스 분포 모델링

VAE는 변분 추론(variational inference)에 기반한 생성 모델로, 데이터 로그우도를 다음과 같이 분해합니다:

$$\ln \mathcal{P}_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi) + \mathbb{D}_{KL}\left[\mathcal{Q}_\phi(\mathbf{z}|\mathbf{x})\,||\,\mathcal{P}_\theta(\mathbf{z}|\mathbf{x})\right] \tag{5}$$

여기서 ELBO(Evidence Lower Bound):

$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{\mathcal{Q}_\phi(\mathbf{z}|\mathbf{x})}\left[\ln\left(\mathcal{P}_\theta(\mathbf{x}|\mathbf{z})\right)\right] - \mathbb{D}_{KL}\left[\mathcal{Q}_\phi(\mathbf{z}|\mathbf{x})\,||\,\mathcal{P}_\theta(\mathbf{z})\right] \tag{6}$$

VAE 학습 후, 디코더 $h_\theta$를 소스 분포 샘플러로 사용:

$$\forall \mathbf{z} \sim \mathcal{N}(0, I),\quad \hat{\mathbf{x}} = h_\theta(\mathbf{z}) \sim \mathcal{P}_s(\hat{\mathbf{x}}) \tag{7}$$

VAE 전체 손실함수:

$$\mathcal{L}_{total} = \mathcal{L}_{vae} + \beta \mathcal{L}_p \tag{Supplementary Eq. 3}$$

---

#### Step 3: Latent Search (LS) - 'nearest-clone' 탐색

타겟 샘플 $\tilde{\mathbf{x}} \sim \mathcal{P}_t(\mathbf{x})$에 대해 'nearest-clone' $\tilde{\mathbf{x}}_S$를 찾는 최적화 문제:

$$\tilde{\mathbf{z}}_S = \arg\min_{\mathbf{z}}\; \mathcal{D}\!\left(\tilde{\mathbf{x}},\, h_\theta(\mathbf{z})\right) \tag{9}$$

$$\tilde{\mathbf{x}}_S = h_\theta(\tilde{\mathbf{z}}_S) \tag{10}$$

거리 메트릭 $\mathcal{D}$로 **SSIM 기반 손실함수** 사용:

$$\mathcal{L}_{ssim}(\mathbf{x}, \hat{\mathbf{x}}) = 1 - \text{SSIM}(\mathbf{x}, \hat{\mathbf{x}}) \tag{Supplementary Eq. 8}$$

SSIM의 정의 (세 가지 유사도 측면):

$$l(\mathbf{x}, \hat{\mathbf{x}}) = \frac{2\mu_\mathbf{x}\mu_{\hat{\mathbf{x}}} + C_1}{\mu_\mathbf{x}^2 + \mu_{\hat{\mathbf{x}}}^2 + C_1} \tag{Supp. Eq. 4}$$

$$c(\mathbf{x}, \hat{\mathbf{x}}) = \frac{2\sigma_\mathbf{x}\sigma_{\hat{\mathbf{x}}} + C_2}{\sigma_\mathbf{x}^2 + \sigma_{\hat{\mathbf{x}}}^2 + C_2} \tag{Supp. Eq. 5}$$

$$s(\mathbf{x}, \hat{\mathbf{x}}) = \frac{\sigma_{\mathbf{x}\hat{\mathbf{x}}} + C_3}{\sigma_\mathbf{x}\sigma_{\hat{\mathbf{x}}} + C_3} \tag{Supp. Eq. 6}$$

$$\text{SSIM}(\mathbf{x}, \hat{\mathbf{x}}) = l(\mathbf{x}, \hat{\mathbf{x}})^\alpha \cdot c(\mathbf{x}, \hat{\mathbf{x}})^\beta \cdot s(\mathbf{x}, \hat{\mathbf{x}})^\gamma \tag{Supp. Eq. 7}$$

잠재 벡터 업데이트 (경사하강법):

$$\mathbf{z} \leftarrow \mathbf{z} + \eta \nabla_{\mathbf{z}} \mathcal{L}_{ssim}$$

---

### 2.3 모델 구조

전체 시스템은 **GLSS(Generative Latent Search for Segmentation)**로 명명되며 두 단계로 구성됩니다.

#### 훈련 단계 (VAE)

```
소스 데이터 (Red 채널 이미지)
         ↓
   인코더 gφ → [μz, σz] → z ~ N(μz, σz²)
         ↓
   디코더 hθ → 재구성 이미지 x̂
         + Edge 정보 (Sobel 필터) skip connection
         ↓
   손실: Lr(MSE) + DKL(KL발산) + Lp(Perceptual)
```

- **인코더** $g_\phi$: 6개의 Conv 레이어 (128→64→32→16→8→4→latent 64차원)
- **디코더** $h_\theta$: 대칭적 Deconv 레이어 + Edge skip connection (Sobel 필터)
- **Perceptual Model** $P_\psi$: DeepLabv3+ 또는 UNet의 중간 레이어 특징 활용
- 잠재 공간 prior: $\mathcal{P}_\theta(\mathbf{z}) = \mathcal{N}(0, I)$

#### 추론 단계 (Latent Search)

```
타겟 이미지 x̃T 입력
         ↓
z ~ N(0,1) 초기화
         ↓ (반복 최적화, ~90-100 iterations)
Lssim(x̃T, hθ*(z)) 최소화 → z 업데이트
         ↓ (수렴)
z̃S → hθ*(z̃S) = x̃S ('nearest-clone')
         ↓
분할 네트워크 Sψ → 예측 마스크
```

#### 분할 네트워크 $S_\psi$

- **UNet** (EfficientNet 백본): $\mathcal{L}\_s = \mathcal{L}\_{dice} + \mathcal{L}_{bce}$, RMSProp (lr=0.001)
- **DeepLabv3+** (Xception 백본): $\mathcal{L}\_s = \mathcal{L}_{focal}$ ( $\gamma=2.0, \alpha=0.75$ ), RMSProp (lr=0.01)

---

### 2.4 성능 향상

#### NIR 피부 분할 (Table 2)

| 방법 | SNV IoU (UNet) | SNV IoU (DLv3+) | Hand Gesture IoU (UNet) | Hand Gesture IoU (DLv3+) |
|------|:-:|:-:|:-:|:-:|
| Source Only | 0.295 | 0.215 | 0.601 | 0.505 |
| AdaptSegNet | 0.315 | 0.230 | 0.641 | 0.542 |
| ADVENT | 0.341 | 0.332 | 0.612 | 0.508 |
| CLAN | 0.248 | 0.225 | 0.625 | 0.513 |
| BDL | 0.320 | 0.301 | 0.647 | 0.536 |
| DISE | 0.341 | 0.339 | 0.672 | 0.563 |
| DADA | 0.332 | 0.314 | 0.643 | 0.559 |
| **GLSS (제안)** | **0.406** | **0.385** | **0.736** | **0.698** |

#### Synthia→Cityscapes (Table 3)

| 방법 | mIoU (13cls) | mIoU* (16cls) |
|------|:-:|:-:|
| AdaptSegNet | 46.7 | - |
| ADVENT | 48.0 | 41.2 |
| BDL | 51.4 | - |
| DISE | 48.8 | 41.5 |
| DADA | 49.8 | 42.6 |
| **GLSS** | **52.3** | **44.5** |

**성능 향상 포인트:**
- SNV에서 기존 최고(DISE/ADVENT 0.341) 대비 **+19% IoU 향상**
- Hand Gesture에서 기존 최고(DISE 0.672) 대비 **+9.5% IoU 향상**
- 타겟 데이터 없이도 타겟 데이터를 사용하는 SOTA 방법들을 능가

---

### 2.5 한계점

1. **추론 시간 비용**: 샘플당 Latent Search 시간이 SNV는 약 450ms, Hand Gesture는 약 120ms 소요 (실시간 적용에 제약)

2. **VAE의 표현 한계**: VAE는 GAN 대비 생성 품질(FID: VAE=44, BigGAN=29.7)이 낮으며, 블러(blur)가 발생할 수 있음 (Edge 연결 및 Perceptual Loss로 부분 완화)

3. **도메인 갭의 한계**: 소스와 타겟 간 도메인 갭이 매우 클 경우, 의미론적으로 정확한 'nearest-clone' 탐색이 어려울 수 있음

4. **다중 클래스 분할 복잡성**: 이진 분할(피부/비피부)에서 상대적으로 강점을 보이나, 다중 클래스의 경우 잠재 공간 최적화의 복잡성 증가

5. **하이퍼파라미터 민감도**: Latent Search 반복 횟수 $K$, 학습률, 잠재 공간 차원 등에 따른 성능 변동 가능성

6. **소스 데이터 품질 의존성**: 소스 분포의 품질과 다양성이 VAE 표현력에 직접적 영향을 미침

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Target-Independence가 일반화에 미치는 효과

GLSS의 가장 큰 일반화 강점은 **타겟 도메인에 독립적**이라는 점입니다. 논문의 Supplementary Table 1은 이를 명확히 보여줍니다:

| 타겟 데이터 비율 | AdaptSegNet | BDL | CLAN | ADVENT | DADA | **GLSS** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 60% | 0.23 | 0.30 | 0.22 | 0.33 | 0.31 | **0.37** |
| 40% | 0.22 | 0.26 | 0.22 | 0.29 | 0.28 | **0.37** |
| 20% | 0.21 | 0.22 | 0.21 | 0.24 | 0.23 | **0.38** |

**관찰**: 기존 SOTA 방법들은 타겟 데이터 비율이 줄어들수록 성능이 저하되는 반면, GLSS는 타겟 데이터 양과 무관하게 **일관된 성능**을 유지합니다.

### 3.2 일반화 성능 향상 메커니즘

#### (a) 이론적 근거
Lemma 1이 보장하는 'nearest-clone'의 존재성은 **소스 분포가 충분히 다양하면** 임의의 타겟 샘플에 대해 근접한 소스 샘플이 항상 존재함을 의미합니다. 이는 GLSS가 새로운 타겟 도메인에도 적용 가능한 이론적 토대를 제공합니다.

#### (b) 구조 보존 (SSIM 기반 탐색)
SSIM 손실은 픽셀 레벨 정보가 아닌 **밝기(luminance), 대비(contrast), 구조(structure)를 종합적으로** 고려하므로, 다양한 도메인 변화(조명, 배경, 피부색)에서도 의미론적 구조를 보존합니다:

$$\mathcal{L}_{ssim} = 1 - l(\mathbf{x}, \hat{\mathbf{x}})^\alpha \cdot c(\mathbf{x}, \hat{\mathbf{x}})^\beta \cdot s(\mathbf{x}, \hat{\mathbf{x}})^\gamma$$

**MSE/MAE 대비 SSIM 우수성**: 모든 반복 횟수에서 SSIM이 일관되게 더 높은 IoU를 달성 (Fig. 5 참조)

#### (c) Edge 정보의 도메인 불변성
NIR과 Red 채널 이미지 모두에서 **엣지(윤곽선) 정보는 상당히 유사**합니다. Sobel 필터로 추출한 엣지를 디코더에 skip connection으로 제공함으로써, VAE가 도메인에 무관한 구조 정보를 활용하여 더 선명한 'nearest-clone'을 생성할 수 있습니다.

#### (d) Perceptual Loss의 의미론적 일반화
$$\mathcal{L}_p = \sum_{i=1}^{B}\left\|P_\psi(\mathbf{x}_i) - P_\psi\left(h_\theta(\mathbf{z}_i)\right)\right\|_2^2$$

DeepLabv3+나 UNet의 중간 레이어 특징을 활용하는 Perceptual Loss는 픽셀 수준이 아닌 **의미론적 유사성**을 강제하여, VAE가 도메인에 관계없이 유의미한 패턴을 학습하도록 합니다.

#### (e) 다른 도메인으로의 확장 가능성
- **Synthia→Cityscapes**: 합성→실제 도메인 변환에서도 SOTA 달성 (mIoU 52.3)
- **WBC 분류** (관련 논문 [37] 참조): 백혈구 분류에도 동일한 GLSS 프레임워크 적용
- 이는 GLSS가 특정 도메인 쌍에 국한되지 않는 **범용 UDA 프레임워크**임을 시사

#### (f) VAE vs GAN: 잠재 탐색 가능성
GLSS는 **생성(generation)과 추론(inference) 모두**가 가능한 모델이 필요합니다. BigGAN(FID=29.7)이 더 높은 생성 품질을 보여도, 잠재 공간에서의 직접 최적화가 불가능해 IoU가 현저히 낮습니다:

| | SNV IoU | Hand Gesture IoU |
|--|:-:|:-:|
| GLSS + BigGAN | 0.09 | 0.21 |
| GLSS + VAE | 0.38 | 0.69 |

이는 **생성 품질보다 잠재 공간의 구조화(structured latent space)가** 도메인 적응 일반화에 더 중요함을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

#### (a) Target-Free UDA 패러다임 개척
기존 UDA 연구의 암묵적 가정("비레이블 타겟 데이터는 항상 접근 가능")을 근본적으로 재검토하게 만듭니다. **타겟 데이터 수집이 불가능한 실제 시나리오**에서의 적용 가능성을 열어줍니다.

#### (b) 잠재 공간 탐색을 통한 도메인 브리징
VAE 잠재 공간을 통한 최적화 방법은 이후 연구들에서:
- **Diffusion Model 기반 잠재 탐색**: DDPM/DDIM 등 최신 생성 모델과의 결합 가능성
- **Contrastive Learning과의 통합**: 잠재 공간의 의미론적 구조 강화
- **Few-shot/Zero-shot 적응**: 극소량의 소스 데이터만으로의 확장

#### (c) NIR 이미지 연구 촉진
두 개의 공개 NIR 데이터셋(SNV, Hand Gesture) 공개는 NIR 도메인 연구의 벤치마크로 활용 가능합니다.

#### (d) 의료 이미징 등 레이블 희소 도메인 적용
MRI→CT, 히스토패솔로지(H&E→IHC) 등 의료 영상 도메인에서도 타겟 레이블 획득이 극히 어려운 문제를 해결하는 데 GLSS 프레임워크가 활용될 수 있습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래의 비교 분석은 제공된 논문(arXiv:2006.08696v2)을 기준으로 하되, 2020년 이후 발표된 관련 연구들의 일반적 동향을 설명합니다. 각 논문의 정확한 수치는 해당 논문 원문을 통해 반드시 확인하시기 바랍니다.

#### GLSS 이후 UDA 연구 동향 비교

| 연구 방향 | 대표적 접근 | GLSS와의 차별점 |
|-----------|------------|----------------|
| **Diffusion 기반 DA** | 이미지 생성에 DDPM 활용 | 더 높은 생성 품질, 그러나 추론 비용 더 큰 |
| **Transformer 기반 UDA** | DAFormer (Hoyer et al., 2022) | 어텐션 메커니즘으로 장거리 의존성 포착 |
| **Prompt 기반 DA** | 대규모 언어-비전 모델 활용 | 사전 훈련 지식 활용, 단 타겟 데이터 여전히 필요 |
| **Test-Time Adaptation** | TTT, TENT 등 | 테스트 시 적응, 단 온라인 학습 필요 |
| **Source-Free DA** | SFDA 계열 방법들 | 소스 데이터 없이 적응, 단 타겟 데이터는 필요 |

**GLSS의 독자적 위치**: SFDA(Source-Free Domain Adaptation)와 달리 GLSS는 **소스 모델만 있으면 타겟 데이터 없이 추론 시 적응**이 가능한 점에서 더욱 극단적인 시나리오를 다룹니다.

---

### 4.3 앞으로 연구 시 고려할 점

#### (a) 더 강력한 생성 모델 탐색
```
VAE → VQ-VAE → Latent Diffusion Model → ?
```
- **Latent Diffusion Models(LDM)**: 더 풍부한 잠재 공간 구조와 높은 생성 품질
- 단, **잠재 공간의 최적화 가능성(differentiability)** 보장 필요
- Score-based 모델의 역방향 과정을 활용한 잠재 탐색 방법 연구 필요

#### (b) 효율적인 Latent Search 알고리즘
- 현재 샘플당 120~450ms 소요 → **실시간 응용에 병목**
- 메타러닝(MAML 등)을 통해 빠른 잠재 탐색 초기화 학습
- 멀티해상도 탐색 전략(coarse-to-fine latent search)
- **Neural Architecture Search**를 통한 최적 VAE 구조 탐색

#### (c) 잠재 공간의 의미론적 구조화 강화
- 현재 VAE는 소스 도메인 데이터만으로 학습되어 잠재 공간이 단순 재구성에 최적화
- **Semantic VAE**: 클래스 레이블 정보를 잠재 공간에 명시적으로 인코딩
- **Disentangled Representation**: 도메인 변화와 의미론적 내용을 분리하여 표현

$$\mathbf{z} = \mathbf{z}_{content} \oplus \mathbf{z}_{style}$$

#### (d) 다양한 거리 메트릭 탐색
- SSIM이 GLSS에서 효과적이나, **학습 기반 거리 메트릭**(e.g., LPIPS, DISTS) 탐색
- 특히 세그멘테이션 태스크에 특화된 **태스크-어웨어 유사도 함수** 개발 필요

$$\mathcal{L}_{task-aware} = \lambda_1 \mathcal{L}_{ssim} + \lambda_2 \mathcal{L}_{perceptual} + \lambda_3 \mathcal{L}_{semantic}$$

#### (e) 다중 소스 도메인 확장
- 현재는 단일 소스 도메인(Red 채널)만 사용
- 여러 소스 도메인(RGB 3채널, 다양한 조명 조건)의 VAE 혼합 모델 학습을 통해 더 넓은 소스 분포 커버

#### (f) 대규모 사전학습 모델과의 통합
- **CLIP, DINO, SAM** 등 대규모 비전 모델의 특징을 perceptual 모델로 활용
- Foundation Model의 일반화 능력과 GLSS의 target-independent 특성 결합

#### (g) 이론적 보장 강화
- 현재의 존재성 증명(Lemma 1)은 $n \to \infty$ 극한에 의존
- **유한 샘플에서의 수렴률(convergence rate)** 분석 필요
- 도메인 갭의 크기와 필요한 소스 샘플 수 간의 관계 이론화

#### (h) 연속적 도메인 변화(Continual DA) 대응
- 실제 환경에서는 타겟 도메인이 시간에 따라 변화 (야간 → 주간, 계절 변화 등)
- GLSS 프레임워크를 **온라인/연속적 도메인 적응**으로 확장하는 연구 필요

---

## 참고 자료

**본 답변의 근거 자료:**

1. **주 논문**: Pandey, P., Tyagi, A. K., Ambekar, S., & Prathosh, A. P. (2020). "Unsupervised Domain Adaptation for Semantic Segmentation of NIR Images through Generative Latent Search." *arXiv preprint arXiv:2006.08696v2.*

2. **논문 내 참조 문헌** (논문 References 기준):
   - Kingma, D. P., & Welling, M. (2013). "Auto-encoding variational bayes." *arXiv:1312.6114* [VAE 원논문]
   - Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). "Image quality assessment: from error visibility to structural similarity." *IEEE Transactions on Image Processing, 13(4)* [SSIM]
   - Tsai, Y. H. et al. (2018). "Learning to adapt structured output space for semantic segmentation." *CVPR* [AdaptSegNet]
   - Vu, T. H. et al. (2019). "Advent: Adversarial entropy minimization for domain adaptation in semantic segmentation." *CVPR* [ADVENT]
   - Li, Y., Yuan, L., & Vasconcelos, N. (2019). "Bidirectional learning for domain adaptation of semantic segmentation." *arXiv:1904.10620* [BDL]
   - Chang, W. L. et al. (2019). "All about structure: Adapting structural information across domains." *CVPR* [DISE]
   - Vu, T. H. et al. (2019). "Dada: Depth-aware domain adaptation in semantic segmentation." *ICCV* [DADA]
   - Luo, Y. et al. (2019). "Taking a closer look at domain shift: Category-level adversaries." *CVPR* [CLAN]
   - Pandey, P. et al. (2020). "Target-independent domain adaptation for wbc classification using generative latent search." *arXiv:2005.05432* [관련 선행 연구]
   - Johnson, J., Alahi, A., & Fei-Fei, L. (2016). "Perceptual losses for real-time style transfer." *ECCV* [Perceptual Loss]
   - Cover, T., & Hart, P. (1967). "Nearest neighbor pattern classification." *IEEE Transactions on Information Theory* [Lemma 1 이론적 근거]

3. **코드 및 데이터셋**: https://github.com/ambekarsameer96/GLSS

> **정확도 관련 고지**: 2020년 이후 최신 연구(DAFormer, Diffusion 기반 DA 등)와의 정량적 비교 수치는 해당 논문들의 원문 미확인으로 구체적 수치를 제시하지 않았습니다. 위에 제시된 모든 수식과 실험 결과는 제공된 논문 PDF에서 직접 확인된 내용입니다.
