# 핵심 주장 및 주요 기여 요약

**“A Review on CT Image Noise and Its Denoising”** 논문은 CT(Computed Tomography) 영상에 내재된 다양한 잡음의 특성과 이를 억제하기 위한 후처리(denoising) 기법들을 체계적으로 정리·분류하고, 각 기법들의 전제, 장·단점 및 임상적 적용 가능성을 비교·분석한 종합적 리뷰이다.  
- CT 영상 잡음의 주요 원인을 통계적(quantal) 잡음, 전자(electronic) 잡음, 라운드오프(roundoff) 오류 등으로 정의하고, 이들 잡음이 영상 재구성 과정에서 어떻게 분포되는지 물리·확률 모델 차원에서 고찰.  
- 잡음 특성(분포, 스펙트럼)과 검출기·재구성 파라미터(튜브 전류, 필터 백프로젝션 vs. 반복 재구성 등)가 어떻게 영상 품질에 영향을 미치는지 정리.  
- 공간 영역 및 변환 영역 기반 필터링, 사전(딕셔너리) 학습, 비국소 평균(NLM), 총변동(TV) 및 이의 변형, 심층 학습(Convolutional Neural Network, Autoencoder) 등 주요 후처리 기법들의 수학적 배경, 알고리즘 구조, 성능 비교를 포괄적으로 기술.  
- 각 기법들의 잡음 저감-특징 보존(trade‐off) 특성과 계산 복잡도를 비교·분류하여, 연구자들이 목적에 맞는 적절한 방법을 선택할 수 있도록 가이드라인 제시.

***

# 1. 문제 정의

CT 영상은 환자 선량 절감을 위해 저선량(low-dose)으로 촬영될 경우 잡음이 급격히 증가하여 낮은 대조도 병변 검출에 장애가 된다.  
- 주요 잡음 원인  
  - 양자 통계적 잡음(quantal noise): 유한한 X선 광자 수 검출에서 기인  
  - 전자 잡음(electronic noise), 연산 라운드오프  
- 잡음 분포: 다중 검출기 MDCT는 가우시안 근사, 단일 광자 모델은 포아송 분포  
- 목표: “진료에 필요한 해부학적 구조(모서리·세부 조직)를 보존하면서 노이즈만 효과적으로 제거”  

# 2. 제안 방법 및 모델 구조

논문은 직접 새 모델을 제안하기보다, 대표 기법들의 수식적 기초와 구조를 다음과 같이 정리·분석한다.

## 2.1. 공간(domain) 필터링
- **선형 필터**: 평균, Wiener 필터  
  ‣ Wiener: $$\hat{f}(x) = \frac{\sigma_f^2}{\sigma_f^2 + \sigma_n^2} g(x) $$  
- **비선형 필터**: 미디언, Bilateral, Non-Local Means (NLM)  
  ‣ NLM:  

$$
    \hat{f}(i) = \frac{1}{Z(i)} \sum_{j} w(i,j)\,g(j), \quad
    w(i,j) = \exp\!\bigl(-\|P_i - P_j\|^2/\!h^2\bigr)
  $$  
  
  여기서 $$P_i$$는 픽셀 i 주변 패치, $$h$$는 제어 파라미터

## 2.2. 정규화 기반 기법
- **티호노프(Tikhonov) 정규화**  

$$
    \min_f \|A f - g\|_2^2 + \lambda\|f\|_2^2
  $$
  
  ‣ 경계 부드러움 과도→엣지 소실  
- **총변동(TV) 정규화**  

$$
    \min_f \|A f - g\|_2^2 + \alpha\int \|\nabla f\|\,dx
  $$
  
  ‣ 엣지 보존 우수하나 계단 효과(staircasing)  
- **분할 브레그만(Split Bregman)** 등 가속화 기법

## 2.3. 사전 학습(dictionary) 기법
- 각 패치를 희소 표현(sparse coding)  

$$
    \min_{\alpha} \|p - D\alpha\|_2^2 + \beta\|\alpha\|_1
  $$
  
  ‣ Dual‐energy CT 등에서 에지 보존 강화

## 2.4. 심층 학습 기반
- **Residual Encoder–Decoder CNN (RED‐CNN)**  
  ‣ 잔차 학습: $$\hat{y} = x + \mathcal{F}(x; \theta) $$  
- **Wavelet‐Domain CNN**  
  ‣ 다중 해상도 특징 학습으로 잡음-신호 분리  
- **Denoising Autoencoder**  
  ‣ 입력 $$x+ n$$ → 은닉 $$h=\sigma(Wx)$$ → 재구성

***

# 3. 성능 향상 및 한계

|기법                         |장점                                       |단점                                    |
|-----------------------------|------------------------------------------|----------------------------------------|
|F-TV(Second‐Order TV)        |계단 현상 완화, 균질 영역 잡음 제거 효과 탁월     |고역 대조 구조 소실 위험                  |
|AT‐PCA Patch 기반            |엣지 보존 우수, 잡음 억제 탁월                   |패치 탐색 및 PCA 반복 → 계산량 과다         |
|NLM                          |엣지 및 질감 보존                           |검색 범위·가중치 파라미터 민감, 계산비용 과다  |
|BM3D                         |최첨단 성능, AWGN 억제 우수                   |컬러 CT, 비정형 잡음 모델 시 제한적          |
|RED‐CNN                      |잔차 학습으로 고속, 병변 유지 및 잡음 제거 동시 달성 |학습 데이터셋 의존, 일반화 어려움           |

- **성능 개선**: Split‐Bregman, 딥러닝 잔차 학습, high‐order TV, adaptive threshold 기법이 종전 대비 PSNR·SSIM 1–3 dB, 0.02–0.05 향상 보고  
- **한계**  
  - 계산 복잡도 vs. 실시간 임상 적용  
  - 과적합 위험: low‐dose CT 전용 학습 시 다른 프로토콜 일반화 저하  
  - 잡음 모델 불일치: 실제 CT는 단일 AWGN 가정 미충족

***

# 4. 일반화 성능 향상 관점

- **도메인 적응(Domain Adaptation)**: 서로 다른 CT 제조사·파라미터별로 잡음 특성 차이→메타 학습, 제로샷 적응 필요  
- **강건성 강화**: 다양한 잡음 레벨·분포(포아송+가우시안 혼합)에 대응하는 하이브리드 학습  
- **경량화 모델 설계**: 모바일 실시간 적용을 위한 Knowledge Distillation, Pruning  
- **물리 모델 융합**: 재구성 원리를 반영한 Physics‐Informed 네트워크로 실제 잡음 분포 근사

***

# 5. 향후 연구 영향 및 고려 사항

- **임상 적용**: 계산 자원 제약 환경에서 알고리즘 속도-정확도 균형  
- **표준화 벤치마크**: 공개 저선량 CT 데이터셋과 공식 평가 지표(SSIM, GMSD, DIV 등) 마련  
- **통계 모델 통합**: 단일 잡음 모델을 넘어 CT 스캔 파이프라인 전체 노이즈 전파 모델링  
- **다중 모달 융합**: MRI·PET 등 다중 모달 영상의 잡음 특성을 결합한 초해상도 및 고감도 모형 개발  
- **윤리적 고려**: AI 기반 후처리가 진단에 미치는 영향 평가, FDA/CE 가이드라인 준수

이 리뷰는 CT 영상 잡음 완화 연구의 **종합 로드맵**을 제시하며, 후속 연구자들이 효율적·강건한 denoising 모델을 설계할 수 있는 이론적·실험적 토대를 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9dd222cd-ba4a-45d0-b11f-82ef60a5c649/1-s2.0-S1746809418300107-main.pdf

## 4. CT Image Denoising Methods

CT(Computed Tomography) 이미지의 잡음 제거(denoising) 기법은 크게 공간 영역(spatial domain) 처리와 변환 영역(transform domain) 처리로 나뉩니다. 각 기법은 “노이즈 억제”와 “의료 의미 있는 구조(엣지·세부 조직) 보존”이라는 두 가지 목표를 달성하기 위해 고안되었습니다.

### 4.1 공간 영역 필터링 (Spatial-Domain Filtering)

영상을 직접 픽셀 값으로 다루며 노이즈를 줄이는 방법입니다.

#### 4.1.1 선형 필터와 비선형 필터  
- **평균(Mean) 필터, 윈너(Wiener) 필터 (선형)**  
  - 픽셀을 주변 이웃의 가중 평균으로 대체  
  - 노이즈를 줄이지만 가장자리가 뭉개지는 블러(blur) 현상이 발생  
- **미디언(Median) 필터, 모폴로지 기반 (비선형)**  
  - 픽셀 값을 주변 픽셀 중 중간값으로 교체하여 소금·후추 잡음(salt-and-pepper noise)에 강함  
- **양방향 필터(Bilateral Filter)**  
  - 공간 거리와 강도 차이에 모두 가중치를 두어 엣지 보존  
  -

$$
      \hat f(i)
      = \frac{1}{W_i} \sum_j \exp\!\Bigl(-\tfrac{\|i-j\|^2}{2\sigma_s^2}
      -\tfrac{|f(i)-f(j)|^2}{2\sigma_r^2}\Bigr)\,f(j)
    $$

- **비국소 평균(Non-Local Means, NLM)**  
  - 픽셀 주변 패치(patch) 유사도를 계산하여 멀리 떨어진 유사 패치까지 평균에 참여  
  -

$$
      \hat f(i)
      = \sum_{j\in\Omega(i)}
      w(i,j)\,f(j),\quad
      w(i,j)\propto e^{-\|P_i-P_j\|^2/h^2}.
    $$  
  
  - 텍스처·엣지 보존에 탁월하지만 계산량이 매우 큼  

#### 4.1.2 티호노프 정규화, 비등방성 확산, 총변동(TV)  
- **티호노프(Tikhonov) 정규화**  

$$
    \min_f \|Af-g\|_2^2 + \lambda\|f\|_2^2
  $$  
  
  → 과도한 평활화로 엣지 손실  
- **비등방성 확산(Anisotropic Diffusion)**  
  - 확산 과정에서 엣지 위치에서는 확산 계수를 낮춰 엣지 보존  
- **총변동(Total Variation, TV)**  

$$
    \min_f \|Af-g\|_2^2 + \alpha\int\!\|\nabla f\|\,dx
  $$  
  
  → 엣지 보존 우수, 균질 영역에서 계단 현상(staircasing)  
- **분할 브레그만(Split Bregman) 기법**  
  - TV 최적화를 빠르게 수치해석  

#### 4.1.3 사전 학습(Dictionary Learning)  
- 영상의 작은 패치들을 사전(dictionary) 기저로 희소 표현(sparse coding)  

$$
    \min_{\alpha} \|p - D\alpha\|_2^2 + \beta\|\alpha\|_1
  $$  

- Dual-energy CT 영상 등에서 서로 다른 에너지 스펙트럼 이미지를 결합하여 노이즈 제거  

#### 4.1.4 양방향·비국소 TV(Non-Local TV)  
- TV에 비국소(non-local) 가중치를 결합하여 영역 간 유사도를 이용한 노이즈 억제  
- 소규모 조직과 텍스처를 잘 보존  

#### 4.1.5 딥러닝 기반  
- **잔차 인코더-디코더 CNN (RED-CNN)**  
  - 입력 영상 $$x$$에 대해 학습된 맵 $$\mathcal F(x)$$을 더함  
    $$\hat y = x + \mathcal F(x)$$  
- **Wavelet-Domain CNN**  
  - 다중 해상도(wavelet) 특징 학습  
- **Denoising Autoencoder**  
  - 잡음 영상을 입력으로 잠재 변수(encoded)로 압축 → 재구성 시 잡음 제거  
- 풍부한 학습 데이터로 복잡한 잡음 모델에도 적응 가능하나, 과적합·일반화 한계 존재  

***

### 4.2 변환 영역 필터링 (Transform-Domain Filtering)

영상을 주파수 영역이나 멀티해상도 영역으로 분해한 뒤 잡음 성분을 억제하고 역변환으로 복원하는 방법입니다.

#### 4.2.1 웨이블릿 기반 Denoising  
- 영상에 이산 웨이블릿 변환(DWT) 적용 → 저·고주파 성분 분리  
- 고주파(디테일) 계수에 임계값(threshold) 적용, 저주파 유지  
- **절단(shrinkage) 규칙**  
  - Hard threshold: $$\alpha\mapsto0$$ if $$|\alpha| < T $$  
  - Soft threshold: $$\alpha\mapsto\text{sign}(\alpha)\,(|\alpha|-T)_+ $$  
- **임계값 추정**  
  - VisuShrink(비적응, σ √(2 log N)), SureShrink(SURE 기반), BayesShrink(베이지안 리스크 최솟값)  

#### 4.2.2 스케일 간 및 스케일 내 의존성 활용  
- **Intra-scale dependency**: 같은 레벨 내 인접 계수 공통 구조 반영  
- **Inter-scale dependency**: 부모–자식 계수 간 상관성 모델링 (Hidden Markov Tree 등)  
- 통계적 모델(Gaussian Scale Mixture, GMM) 결합 시 잡음 모델링 정확도 향상  

#### 4.2.3 확장 변환 기법  
- DWT 한계(shift 민감성, 방향 선택성 부족) 극복용  
  - **Contourlet, Shearlet, Curvelet, Nonsubsampled Shearlet**  
  - 방사형·곡선형 특징을 잘 포착하여 선·곡선형 경계 보존  
- **Wavelet Packet, Tetrolet Transform** 등 다양한 기저로 잡음·신호 분리 효율 개선  

#### 4.2.4 BM3D (Block-Matching and 3D Filtering)  
- 노이즈 억제 최첨단 기법  
- 유사 패치를 블록 매칭해 3차원 배열로 정렬 → 협력적 필터링(3D 변환+역변환)  
- AWGN 억제 성능 우수  
- Curvelet/BM3D 하이브리드로 엣지 보존 추가 강화  

***

이상이 CT 이미지 잡음 제거를 위한 주요 기법들입니다. 각 방법은 잡음 저감과 구조 보존 간의 트레이드오프를 가지며, 실제 임상 적용 시 계산 효율·일반화 능력을 함께 고려해야 합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9dd222cd-ba4a-45d0-b11f-82ef60a5c649/1-s2.0-S1746809418300107-main.pdf
