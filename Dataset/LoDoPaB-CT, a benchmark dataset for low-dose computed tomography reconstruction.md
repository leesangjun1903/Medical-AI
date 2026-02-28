# LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction

---

## 1. 핵심 주장과 주요 기여 요약

**핵심 주장:** 딥러닝 기반 CT 영상 재구성 방법들의 공정한 비교가 어려운 이유는 각 방법이 사용하는 데이터와 실험 설정에 크게 의존하기 때문이며, 이를 해결하기 위해 **표준화된 벤치마크 데이터셋**이 필요하다.

**주요 기여:**
1. **대규모 오픈 액세스 데이터셋 구축:** LIDC/IDRI 데이터베이스로부터 약 800명의 환자, 40,000장 이상의 CT 슬라이스와 시뮬레이션된 저선량 측정 데이터 쌍을 제공
2. **재현 가능한 시뮬레이션 파이프라인:** 2D 평행빔(parallel beam) 기하학 기반의 저광자수(low photon count) 측정 시뮬레이션 설정을 상세히 기술하고, 생성 스크립트를 공개
3. **표준화된 평가 체계:** Python 라이브러리(DIVaℓ), 온라인 재구성 챌린지, 사전 정의된 train/validation/test/challenge 분할을 제공
4. **확장성:** 전이 학습(transfer learning), 희소 각도(sparse-angle), 제한 각도(limited-angle) 재구성 시나리오에도 활용 가능

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

CT 영상 재구성은 X선 감쇠 측정값으로부터 내부 분포를 복원하는 **역문제(inverse problem)**이다. 저선량 CT에서는 방사선량을 줄이면서 발생하는 **높은 노이즈**와 **언더샘플링** 문제를 극복해야 한다. 딥러닝 기반 재구성 방법들이 크게 발전했으나, 각 연구가 서로 다른 데이터와 설정을 사용하기 때문에 **공정한 비교가 불가능**했다. 기존 공개 CT 데이터셋들은 투영 데이터를 포함하지 않거나, 환자 수가 부족하거나, 딥러닝 훈련에 부적합한 형태로 제공되었다.

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 CT 역문제 수학적 모델

2D 평행빔 기하학에서 CT 역문제는 다음과 같이 정의된다:

$$\mathcal{A}x + \varepsilon(\mathcal{A}x) = y^\delta $$

여기서:
- $\mathcal{A}$: 스캔 기하학에 의해 정의된 선형 레이 변환 (ray transform)
- $x$: X선 감쇠 계수의 미지 내부 분포 (영상)
- $\varepsilon$: 이상적 측정 $\mathcal{A}x$에 의존하는 노이즈 분포의 샘플
- $y^\delta$: 노이즈가 포함된 CT 측정값 (사이노그램)

#### 2.2.2 Radon 변환

평행빔 기하학에서 레이 변환 $\mathcal{A}$는 Radon 변환으로, X선 경로 $L_{s,\varphi}(t)$를 따라 영상 $x$를 적분한다:

```math
L_{s,\varphi}(t) := s\omega(\varphi) + t\omega^\perp(\varphi), \quad \omega(\varphi) := \begin{bmatrix} \cos(\varphi) \\ \sin(\varphi) \end{bmatrix}, \quad \omega^\perp(\varphi) := \begin{bmatrix} -\sin(\varphi) \\ \cos(\varphi) \end{bmatrix}
```

$$\mathcal{A}x(s,\;\varphi) := \int_{\mathbb{R}} x(L_{s,\varphi}(t))\,\mathrm{d}t $$

#### 2.2.3 Beer-Lambert 법칙과 노이즈 모델

투영과 검출기에서의 이상적 강도 측정 $I_1(s,\varphi)$의 관계는 Beer-Lambert 법칙에 의해 다음과 같이 표현된다:

$$\mathcal{A}(s,\;\varphi) = -\ln\left(\frac{I_1(s,\;\varphi)}{I_0}\right) = y(s,\;\varphi) $$

양자 노이즈를 포아송 분포로 모델링한다:

$$\widetilde{N}_1(s,\;\varphi) \sim \text{Pois}(N_0 \exp(-\mathcal{A}x(s,\;\varphi))), \quad \frac{\widetilde{I}_1(s,\;\varphi)}{I_0} = \frac{\widetilde{N}_1(s,\;\varphi)}{N_0} $$

여기서 포아송 분포는 다음과 같이 정의된다:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \mathbb{N}_0 $$

#### 2.2.4 이산화된 모델

실제 적용을 위해 순방향 연산자를 유한 차원 선형 맵 $A: \mathbb{R}^n \to \mathbb{R}^m$으로 이산화한다:

$$Ax + \varepsilon(Ax) = y^\delta, \quad \varepsilon(Ax) = -Ax - \ln(\widetilde{N}_1/N_0), \quad \widetilde{N}_1 \sim \text{Pois}(N_0 \exp(-Ax)) $$

#### 2.2.5 Hounsfield Unit (HU) 변환과 정규화

HU 값에서 선형 감쇠 계수 $\mu$로의 변환:

$$\text{HU} = 1000 \frac{\mu - \mu_{\text{water}}}{\mu_{\text{water}} - \mu_{\text{air}}} \quad \Leftrightarrow \quad \mu = \text{HU} \frac{\mu_{\text{water}} - \mu_{\text{air}}}{1000} + \mu_{\text{water}} $$

사용된 감쇠 계수:

$$\mu_{\text{water}} = 20/\text{m}, \quad \mu_{\text{air}} = 0.02/\text{m} $$

$[0, 1]$ 범위로의 정규화를 위한 최대값:

$$\mu_{\max} = 3071 \frac{\mu_{\text{water}} - \mu_{\text{air}}}{1000} + \mu_{\text{water}} = 81.35858/\text{m} $$

클리핑 함수:

$$\hat{\mu} = \text{clip}(\mu/\mu_{\max},\;[0,\;1]) = \begin{cases} 0 & ,\mu/\mu_{\max} \leq 0 \\ \mu/\mu_{\max} & ,0 < \mu/\mu_{\max} \leq 1 \\ 1 & ,1 < \mu/\mu_{\max} \end{cases} $$

#### 2.2.6 평가 지표

**PSNR (Peak Signal-to-Noise Ratio):**

$$\text{PSNR}(\bar{x},\;x) := 10\log_{10}\left(\frac{\max_x^2}{\text{MSE}(\bar{x},\;x)}\right), \quad \text{MSE}(\bar{x},\;x) := \frac{1}{n}\sum_{i=1}^{n}|\bar{x}_i - x_i|^2 $$

**SSIM (Structural Similarity):**

$$\text{SSIM}(\bar{x},\;x) := \frac{1}{M}\sum_{j=1}^{M}\frac{(2\bar{\mu}_j\mu_j + C_1)(2\Sigma_j + C_2)}{(\bar{\mu}_j^2 + \mu_j^2 + C_1)(\bar{\sigma}_j^2 + \sigma_j^2 + C_2)} $$

여기서 $C_1 = (K_1 L)^2$, $C_2 = (K_2 L)^2$이며, $K_1 = 0.01$, $K_2 = 0.03$, $L = \max(x) - \min(x)$.

### 2.3 모델 구조 및 데이터셋 설계

**시뮬레이션 파라미터:**
| 파라미터 | 값 |
|---|---|
| 영상 해상도 | $362 \times 362$ px ($26\text{cm} \times 26\text{cm}$) |
| 검출기 빈 수 | 513개 (등간격) |
| 투영 각도 수 | 1000개 (0 ~ $\pi$, 등간격) |
| 평균 광자 수 ($N_0$) | 4096/검출기 빈 |
| 시뮬레이션 해상도 (inverse crime 방지) | $1000 \times 1000$ px |

**데이터 분할:**
| 분할 | 환자 수 | 샘플 수 |
|---|---|---|
| 훈련 (Train) | 632 | 35,820 |
| 검증 (Validation) | 60 | 3,522 |
| 테스트 (Test) | 60 | 3,553 |
| 챌린지 (Challenge) | 60 | 3,678 |

각 분할은 **서로 다른 환자 집합**으로 구성되어, 학습된 재구성기가 미지의 환자에 적용되는 시나리오를 평가한다.

**기준(Baseline) 모델:**
- **FBP (Filtered Back-Projection):** Hann 필터, 주파수 스케일링 0.641
- **FBP + U-Net:** FBP 결과를 5-스케일 U-Net으로 후처리. MSE 손실, Adam 옵티마이저 ($\text{lr} = 10^{-3} \to 10^{-4}$, 코사인 어닐링), 최대 250 에폭, 배치 크기 32

### 2.4 성능 향상

| 방법 | 테스트 PSNR (dB) | 테스트 SSIM |
|---|---|---|
| FBP | $30.52 \pm 3.10$ | $0.7372 \pm 0.1467$ |
| FBP + U-Net | $35.84 \pm 4.59$ | $0.8443 \pm 0.1501$ |

FBP + U-Net이 FBP 대비 약 **5 dB의 PSNR 향상**과 SSIM에서도 유의미한 개선을 보여, 이 데이터셋이 딥러닝 방법의 학습에 효과적임을 입증하였다.

### 2.5 한계

1. **시뮬레이션의 한계:** Radon 변환 + 포아송 노이즈만 사용하며, 산란(scattering) 등 실제 물리 효과를 반영하지 않음
2. **기하학적 제한:** 평행빔(parallel beam) 2D 설정만 사용하며, 실제 임상에서 사용하는 헬리컬 팬빔/콘빔 기하학과 차이 존재
3. **2D 슬라이스 기반:** 3D 볼륨 재구성으로의 일반화가 직접적이지 않으며, 리비닝(rebinning) 효과 평가 필요
4. **크롭된 영상:** $362 \times 362$ px로 중앙 크롭되어 전체 피사체 측정에 대한 성능과 다를 수 있음
5. **Ground truth의 불완전성:** 정상 선량 재구성 결과를 ground truth로 사용하므로 자체적으로 노이즈와 아티팩트 포함 가능
6. **평가 지표의 한계:** PSNR/SSIM이 임상적 품질의 모든 측면을 포착하지 못함

---

## 3. 모델의 일반화 성능 향상 가능성

LoDoPaB-CT 데이터셋은 **일반화 성능**과 관련하여 여러 중요한 설계 결정과 확장 가능성을 포함한다:

### 3.1 데이터셋 수준의 일반화 설계

- **환자 기반 분할(Patient-wise Split):** 훈련/검증/테스트/챌린지 세트가 완전히 다른 환자 집합으로 구성되어, 학습된 모델이 **미지의 환자**에 대해 얼마나 잘 일반화되는지를 직접 평가할 수 있다.
- **이질적 데이터 출처:** LIDC/IDRI 데이터베이스는 7개 학술 기관과 8개 의료 영상 회사에서 수집된 이질적인(heterogeneous) 데이터로 구성되어, 다양한 스캐너 모델과 기술 파라미터(관전압 120–140 kV, 관전류 40–627 mA)를 포함한다. 이는 모델이 특정 스캐너에 과적합되는 것을 방지한다.
- **대규모 데이터:** 40,000장 이상의 샘플과 800명 이상의 환자 데이터는 딥러닝 모델의 성공적인 훈련에 필요한 충분한 양과 다양성을 제공한다.

### 3.2 역범죄(Inverse Crime) 방지

시뮬레이션과 재구성에 동일한 이산 모델을 사용하는 "역범죄"를 피하기 위해, 시뮬레이션 시 ground truth를 $362 \times 362$에서 $1000 \times 1000$으로 업스케일링한 후 투영을 생성한다. 이를 통해 특정 해상도의 이산화 문제 속성에서만 좋은 성능을 보이는 것이 아닌, **해석적 모델에 대한 진정한 역문제 해결 능력**을 평가할 수 있다.

### 3.3 전이 학습을 통한 일반화

LoDoPaB-CT는 전이 학습의 기반 데이터셋으로 활용 가능하다:
- **다양한 스캔 시나리오로의 전이:** 각도 수 변경(예: 1000→200 각도의 희소 각도 문제), 제한 각도 문제, 초해상도 실험 등으로 쉽게 변환 가능
- **2D→3D 전이:** 2D 데이터로 사전 학습 후 3D 데이터로 미세 조정하는 전략에 활용 가능 (Shan et al., 2018)
- **실제 CT 데이터 대비 이점:** ImageNet의 자연 영상보다 실제 흉부 CT 재구성 영상이 유사한 CT 재구성 작업에 더 유리할 수 있음 (Raghu et al., 2019)

### 3.4 일반화의 제약과 향후 과제

- **2D 평행빔 → 3D 임상 설정:** 리비닝 효과를 평가해야 하며, 3D 재구성에 직접 대응하는 알고리즘이 필요
- **시뮬레이션 → 실측 데이터:** 시뮬레이션이 실제 측정의 모든 측면을 포착하지 못하므로, 실제 데이터에 대한 추가 검증이 필수적
- **광자 수 변경:** $N_0 = 4096$은 특정 선량 수준을 나타내며, 다른 선량 조건에서는 재시뮬레이션이 필요

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

1. **표준화된 벤치마크 확립:** LoDoPaB-CT는 저선량 CT 재구성 분야에서 방법론 간 **공정한 비교를 위한 공통 기반**을 제공하여, 연구 결과의 재현성과 비교 가능성을 크게 향상시켰다.

2. **딥러닝 재구성 연구 가속화:** 대규모의 쌍(pair) 데이터 제공으로, 지도 학습(supervised learning) 기반 방법뿐 아니라 비지도/반지도 학습 방법의 개발도 촉진한다.

3. **챌린지 플랫폼:** Grand Challenge 웹사이트를 통한 온라인 평가 시스템은 연구 커뮤니티의 지속적인 참여와 최신 기술 비교를 가능하게 한다.

4. **전이 학습 연구 촉진:** CT 도메인 내 전이 학습의 기반 데이터셋으로서, 데이터가 부족한 특수 CT 응용(예: 특수 신체 부위, 희귀 질환) 연구에 기여한다.

### 4.2 향후 연구 시 고려할 점

1. **실측 데이터와의 간극(Domain Gap):** 시뮬레이션 데이터에서 훈련된 모델의 실제 CT 스캐너 데이터에 대한 적용 시, 도메인 적응(domain adaptation) 기법이 필요할 수 있다.

2. **3D 재구성으로의 확장:** 현재 2D 슬라이스 기반이므로, 실제 임상 적용을 위해서는 3D 볼륨 재구성 방법으로의 확장과 이에 따른 계산 비용 문제를 고려해야 한다.

3. **다양한 노이즈 모델:** 포아송 노이즈 외에 검출기 노이즈(가우시안), 산란 등을 포함하는 보다 현실적인 노이즈 모델의 적용이 필요하다.

4. **임상적 유효성 평가:** PSNR/SSIM 외에 진단 과업 기반(task-based) 평가, 방사선 전문의에 의한 주관적 평가 등 임상적 관점의 평가가 보완되어야 한다.

5. **팬빔/콘빔 기하학:** 임상 스캐너에서 실제 사용되는 기하학으로의 확장을 고려한 데이터셋 설계가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

LoDoPaB-CT가 2021년에 발표된 이후, 이 데이터셋을 활용하거나 저선량 CT 재구성 벤치마킹 분야에서 관련된 주요 최신 연구들을 비교 분석한다.

### 5.1 LoDoPaB-CT를 활용한 주요 연구

| 연구 | 연도 | 핵심 내용 | LoDoPaB-CT 대비 성과 |
|---|---|---|---|
| Baguer et al., "Computed tomography reconstruction using deep image prior and learned reconstruction methods" (Inverse Problems, 2020) | 2020 | Deep Image Prior(DIP)와 학습 기반 방법(Learned Primal-Dual 등)을 LoDoPaB-CT에서 비교. 200 각도 희소 설정도 평가 | DIP는 학습 데이터 없이도 합리적 결과 도출, 학습 기반 방법이 전반적으로 우수 |
| Genzel et al., "Solving inverse problems with deep neural networks — Robustness included?" (IEEE TPAMI, 2022) | 2022 | LoDoPaB-CT를 포함한 역문제에서 DL 기반 재구성기의 **적대적 강건성(adversarial robustness)**을 체계적으로 분석 | DL 재구성기가 작은 적대적 섭동에 취약할 수 있음을 보이며, 일반화/강건성 문제를 제기 |
| Leuschner et al., "Quantitative comparison of deep learning-based image reconstruction methods for low-dose and sparse-angle CT applications" (Journal of Imaging, 2021) | 2021 | LoDoPaB-CT에서 다양한 DL 재구성 방법(Learned Primal-Dual, iUNet, U-Net 후처리 등)의 정량적 비교 | Learned Primal-Dual 등 모델 기반 학습 방법이 순수 후처리 방법보다 우수 |

### 5.2 경쟁 벤치마크 데이터셋

| 데이터셋 | 연도 | 특징 | LoDoPaB-CT와의 차이 |
|---|---|---|---|
| LDCT-and-Projection-data (Mayo Clinic, McCollough et al.) | 2020 | 299명 환자의 실측 정상 선량 투영 데이터 (DICOM-CT-PD 형식), 팬빔 기하학 | **실측 데이터** 제공이나 저선량 시뮬레이션 설정이 표준화되어 있지 않음; 팬빔 기하학 |
| AAPM Low Dose CT Grand Challenge | 2016 | 30명 환자, 시뮬레이션 저선량 데이터 | 환자 수가 적어 대규모 DL 학습에 부적합 |
| Der Sarkissian et al. (42 walnuts, cone-beam) | 2019 | 42개 호두의 콘빔 CT 투영 데이터 | 의료 영상이 아니며 규모가 작음 |
| fastMRI (Knoll et al.) | 2020 | MRI k-space 데이터 1,600건 | CT가 아닌 MRI 모달리티; 유사한 벤치마크 철학 공유 |

### 5.3 최신 재구성 방법론의 발전 (2020년 이후)

| 방법론 카테고리 | 대표 연구 | 핵심 아이디어 | LoDoPaB-CT 적용 가능성 |
|---|---|---|---|
| **Score-based diffusion models** | Song et al., "Solving inverse problems in medical imaging with score-based generative models" (ICLR 2022) | 사전 학습된 score 함수를 이용한 조건부 샘플링으로 역문제 해결 | 비지도 학습 기반으로 LoDoPaB-CT의 ground truth 영상 분포 학습에 활용 가능 |
| **Equivariant neural networks** | Chen et al., "Equivariant imaging: Learning beyond the range space" (ICCV 2021) | 순방향 모델의 등변성(equivariance) 활용한 자기 지도 학습 | 투영 데이터만으로도 학습 가능, LoDoPaB-CT의 측정 데이터 분포 활용 |
| **Plug-and-Play (PnP) / RED** | Hurault et al., "Gradient step denoiser for convergent PnP" (ICLR 2022) | 사전 학습된 디노이저를 반복 최적화에 플러그인 | LoDoPaB-CT로 디노이저 학습 후 다양한 설정에 적용 가능 |
| **Neural Radiance Fields for CT** | Zha et al., "NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction" (MICCAI 2022) | NeRF 개념을 CT에 적용, 희소 뷰에서 암시적 표현 학습 | 3D 확장에 대한 시사점 제공 |
| **Transformer 기반 재구성** | Wang et al., "CTformer: Convolution-free Token2Token Dilated Vision Transformer for Low-dose CT Denoising" (PMB 2023) | 트랜스포머 아키텍처를 저선량 CT 디노이징에 적용 | LoDoPaB-CT에서 U-Net 기반 후처리와의 비교 가능 |

### 5.4 핵심 트렌드와 시사점

1. **비지도/자기 지도 학습의 부상:** 쌍(pair) 데이터 없이도 학습 가능한 방법들이 발전하고 있으나, LoDoPaB-CT 같은 표준 벤치마크에서의 정량적 비교가 여전히 중요하다.

2. **물리 기반 + 데이터 기반 하이브리드:** Learned Primal-Dual, unrolled optimization 등 물리 모델을 네트워크에 통합하는 방법이 일반화 성능에서 유리한 것으로 나타나고 있다.

3. **강건성과 안전성:** Genzel et al. (2022)의 연구가 보여준 바와 같이, DL 재구성기의 적대적 강건성 문제는 임상 적용에서 핵심 과제이며, LoDoPaB-CT가 이러한 연구의 테스트베드로 활용되고 있다.

4. **Diffusion 모델의 가능성:** Score-based generative model은 불확실성 정량화(uncertainty quantification)까지 가능하여 의료 영상 분야에서 주목받고 있다.

---

## 참고자료

1. Leuschner, J., Schmidt, M., Otero Baguer, D. & Maass, P. "LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction." *Scientific Data* 8, 109 (2021). https://doi.org/10.1038/s41597-021-00893-z

2. Baguer, D. O., Leuschner, J. & Schmidt, M. "Computed tomography reconstruction using deep image prior and learned reconstruction methods." *Inverse Problems* 36, 094004 (2020).

3. Genzel, M., Macdonald, J. & März, M. "Solving inverse problems with deep neural networks — Robustness included?" *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44(11), 7944-7960 (2022).

4. Song, Y., Shen, L., Xing, L. & Ermon, S. "Solving inverse problems in medical imaging with score-based generative models." *ICLR 2022*.

5. McCollough, C. et al. "Data from Low Dose CT Image and Projection Data." *The Cancer Imaging Archive* (2020).

6. Armato, S. G. III et al. "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)." *Med. Phys.* 38, 915–931 (2011).

7. Wang, Z. et al. "Image quality assessment: from error visibility to structural similarity." *IEEE Trans. Image Processing* 13, 600–612 (2004).

8. Ronneberger, O., Fischer, P. & Brox, T. "U-Net: Convolutional networks for biomedical image segmentation." *MICCAI 2015*.

9. Adler, J. & Öktem, O. "Learned primal-dual reconstruction." *IEEE Trans. Medical Imaging* 37, 1322–1332 (2018).

10. Chen, D., Tachella, J. & Davies, M. E. "Equivariant imaging: Learning beyond the range space." *ICCV 2021*.

11. Hurault, S., Leclaire, A. & Papadakis, N. "Gradient step denoiser for convergent Plug-and-Play." *ICLR 2022*.

12. Raghu, M., Zhang, C., Kleinberg, J. & Bengio, S. "Transfusion: Understanding transfer learning for medical imaging." *NeurIPS 2019*.

13. Shan, H. et al. "3-D convolutional encoder-decoder network for low-dose CT via transfer learning from a 2-D trained network." *IEEE Trans. Medical Imaging* 37, 1522–1534 (2018).

14. Wang, C. et al. "CTformer: Convolution-free Token2Token Dilated Vision Transformer for Low-dose CT Denoising." *Physics in Medicine and Biology* 68(6) (2023).

15. Zha, R. et al. "NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction." *MICCAI 2022*.

16. Leuschner, J. et al. "Quantitative comparison of deep learning-based image reconstruction methods for low-dose and sparse-angle CT applications." *Journal of Imaging* 7(3), 44 (2021).
