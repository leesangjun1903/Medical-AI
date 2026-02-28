# Deep-Neural-Network Based Sinogram Synthesis for Sparse-View CT Image Reconstruction

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
본 논문은 **sparse-view CT에서 결손된 사이노그램(sinogram) 데이터를 심층 신경망(DNN)을 활용하여 합성(synthesis)**함으로써, 기존의 해석적(analytic) 보간법 및 반복적(iterative) 영상 재구성 방법보다 우수한 CT 영상 재구성 품질을 달성할 수 있음을 주장한다.

### 주요 기여
1. **Residual U-Net 기반 사이노그램 합성 프레임워크 제안**: 사이노그램 도메인에서 누락된 뷰 데이터를 DNN으로 합성하여 FBP(Filtered Backprojection)로 고품질 영상 재구성을 가능하게 함
2. **풀링 층의 학습 가능한 합성곱 층으로의 대체**: max-pooling 대신 stride-based convolution을 사용하여 다운샘플링 가중치도 학습 대상에 포함
3. **잔차 학습(residual learning) 통합**: 입력과 출력의 차이만 학습하여 수렴 속도와 효율성 향상
4. **포괄적 비교 실험**: 선형 보간, 방향성 보간, 20-layer 순차 CNN, POCS-TV 반복 재구성과의 정량적·정성적 비교 수행

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

X-ray CT의 임상 사용 증가로 방사선 피폭 위험이 대두되었으며, 저선량 CT 구현 방안 중 하나인 **sparse-view sampling**이 주목받고 있다. Sparse-view CT에서는 투영(projection) 뷰의 수를 줄여 방사선량을 감소시키지만, 이로 인해 다음과 같은 문제가 발생한다:

- **FBP 알고리즘 직접 적용 시**: 심각한 **streak artifacts** 발생
- **반복적 재구성 알고리즘(예: POCS-TV)**: 긴 계산 시간, 재구성 파라미터 의존적 **cartoon artifacts**, 미세 구조 손실 위험
- **기존 보간법(선형, 방향성 등)**: 복원 능력의 한계로 인해 잔여 아티팩트 존재

이러한 문제들은 본질적으로 **부족한 데이터로부터의 ill-posed 역문제(inverse problem)**에 해당한다.

### 2.2 제안하는 방법

#### (a) 전체 파이프라인

1. 원본 사이노그램에서 4배 서브샘플링하여 sparse-view 사이노그램 생성 (720 views → 180 views)
2. 선형 보간으로 초기 full-size 사이노그램 생성 (업샘플링)
3. **Residual U-Net**을 통해 보간된 사이노그램을 ground truth에 가깝게 합성
4. 합성된 사이노그램에 **FBP 알고리즘**을 적용하여 최종 영상 재구성

#### (b) 합성곱 연산

각 층에서 가중치 $W$, 편향 $b$, 입력 $x$에 대해 다음 연산을 수행한다:

$$W * x + b $$

#### (c) 비용 함수 (Cost Function)

네트워크 출력 패치 $x^k$와 ground truth 패치 $y^k$ 간의 유클리드 오차를 최소화한다:

$$\mathcal{L} = \frac{1}{2N} \sum_{k} \| x^k - y^k \|_2^2 $$

여기서:
- $x^k$: 네트워크 출력 패치 (벡터 형태)
- $y^k$: ground truth 패치 (벡터 형태)
- $N$: 한 iteration에 사용되는 배치 수
- $k$: 학습 데이터셋 내 패치 인덱스

#### (d) 최적화

**Adam optimizer**를 사용하며, 1차 모멘텀 $\beta_1 = 0.9$, 2차 모멘텀 $\beta_2 = 0.999$, 학습률 $\alpha = 0.0001$로 설정하였다. Adam의 파라미터 업데이트 규칙은 다음과 같다:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

여기서 $g_t$는 시간 $t$에서의 gradient, $\theta$는 네트워크 파라미터이다 (Kingma & Ba, 2014 [49]).

### 2.3 모델 구조: Residual U-Net

#### 핵심 구조적 특징

| 구성 요소 | 설명 |
|---------|------|
| **기본 아키텍처** | U-Net (Ronneberger et al., 2015) |
| **잔차 학습** | 입력 데이터와 마지막 합성곱 층 출력을 합산 (residual connection) |
| **다운샘플링** | 풀링 층 대신 **stride=2인 합성곱 층** 사용 (학습 가능한 다운샘플링) |
| **업샘플링** | Deconvolution (전치 합성곱) + ReLU + Batch Normalization |
| **Skip connections** | 인코더-디코더 간 대응 층을 연결하여 세부 정보 보존 |
| **합성곱 커널** | 3×3 크기의 합성곱 + 1×1 합성곱 |
| **활성화 함수** | ReLU (Rectified Linear Unit) |
| **정규화** | Batch Normalization |

#### 풀링 층 대체의 근거

사이노그램 도메인에서는 측정된 픽셀 값이 보간된 초기 추정값보다 중요하므로, max-pooling처럼 최대값에 가중치를 부여하는 것보다 **측정 픽셀 및 상관관계가 높은 픽셀에 더 높은 가중치**를 부여하는 학습 기반 다운샘플링이 적합하다.

#### 패치 기반 학습

- **패치 크기**: $50 \times 50$
- **패치 추출 stride**: 10 (중첩 영역을 통해 타일링 아티팩트 완화)
- **학습 데이터**: 2,142,667 패치
- **검증 데이터**: 918,285 패치
- **데이터 소스**: TCIA (The Cancer Imaging Archive)의 Lung CT, 7명 환자 634 슬라이스

#### 비교 대상 네트워크: 20-layer 순차 CNN

20개의 연속적 합성곱 층(3×3, ReLU)으로 구성된 단순 네트워크로, U-Net 구조의 이점을 입증하기 위한 비교 기준으로 사용되었다.

### 2.4 성능 향상

#### 사이노그램 도메인 평가 (8명 환자, 662 슬라이스 — 학습/검증 미참여)

| 메트릭 | 선형 보간 | 방향성 보간 | 20-Conv | **U-Net** |
|-------|---------|---------|---------|----------|
| NRMSE ($\times 10^{-3}$, Patient #4) | 2.34 | 1.79 | 0.74 | **0.66** |
| PSNR (Patient #4) | 52.60 | 54.95 | 62.59 | **63.59** |
| SSIM (Patient #4) | 0.947 | 0.956 | 0.983 | **0.984** |

#### 재구성 영상 도메인 평가

| 메트릭 | POCS-TV | 선형 보간 | 방향성 보간 | 20-Conv | **U-Net** |
|-------|---------|---------|---------|---------|----------|
| NRMSE ($\times 10^{-3}$, Patient #4) | 6.09 | 6.18 | 4.70 | 2.05 | **1.81** |
| PSNR (Patient #4) | 41.71 | 41.58 | 43.95 | 51.18 | **52.23** |
| SSIM (Patient #4) | 0.962 | 0.964 | 0.974 | 0.992 | **0.993** |

- **U-Net은 모든 환자, 모든 메트릭에서 최고 성능**을 달성
- POCS-TV 대비 PSNR 약 10 dB 이상 향상
- 선형 보간 대비 NRMSE 약 60~70% 감소
- U-Net이 20-Conv 대비에서도 일관되게 소폭 우위

#### 정성적 평가
- **FBP (sparse)**: 심각한 streak artifacts
- **POCS-TV**: cartoon artifacts, 미세 구조 손실
- **선형/방향성 보간**: 중간 수준의 streak artifacts 잔존
- **U-Net**: **streak artifacts 최소**, ground truth에 가장 근접

### 2.5 한계점

1. **학습 시간**: U-Net은 약 12일(GPU: GTX Titan X 12GB), 순차 CNN은 약 5일 소요
2. **추론 시간**: U-Net 약 50초, 순차 CNN 약 10초 (실시간 적용 어려움)
3. **2D fan-beam 기하에 국한**: cone-beam CT나 helical CT에 대한 확장 미검증
4. **고정 서브샘플링 비율**: 4배 서브샘플링(1/4 뷰)만 실험, 다른 서브샘플링 비율에 대한 성능 미확인
5. **시뮬레이션 기반 데이터**: 실제 sparse-view 스캔이 아닌 reprojection 기반 시뮬레이션 사이노그램 사용
6. **단일 해부학적 부위(폐 CT)로 학습**: 다른 해부학적 부위(복부, 두경부 등)에 대한 일반화 미검증
7. **POCS-TV 파라미터 최적화 부족**: 저자들도 인정하듯이 반복적 알고리즘의 파라미터 미세 조정에 따라 성능이 달라질 수 있음
8. **MSE 기반 손실 함수의 한계**: 지각적(perceptual) 품질이나 고주파 세부 구조 보존에 대한 고려 부족

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 논문에서 나타난 일반화 근거

논문은 **학습에 참여하지 않은 8명의 환자 데이터(662 슬라이스)**로 평가를 수행하였으며, 학습 오차와 검증 오차가 유사한 값을 보여(Fig. 5) **과적합(overfitting) 징후가 관찰되지 않았다**. 이는 다음 설계 요소에 기인한다:

- **패치 기반 학습**: 대량의 패치(약 214만 개)로 데이터 다양성 확보
- **Batch Normalization**: 내부 공변량 변화(internal covariate shift) 감소
- **잔차 학습**: 입력-출력 간 차이만 학습하여 학습 난이도 감소

### 3.2 일반화 성능의 한계 요인

| 한계 요인 | 상세 설명 |
|---------|---------|
| **해부학적 다양성 부족** | 폐 CT 데이터만 사용, 복부·두경부·근골격계 등 다른 부위에서의 성능 보장 불가 |
| **스캔 기하 제한** | fan-beam 기하만 고려, 실제 임상에서 주로 사용되는 cone-beam/helical 기하 미포함 |
| **서브샘플링 패턴 고정** | 등간격 4배 서브샘플링만 학습, 불규칙 샘플링이나 다른 배율에 대한 적응력 미검증 |
| **노이즈 미고려** | reprojection 기반 시뮬레이션은 실제 측정 노이즈를 반영하지 못함 |
| **환자 수 제한** | 학습 7명, 평가 8명으로 통계적 유의성 확보 어려움 |

### 3.3 일반화 성능 향상을 위한 방향

#### (a) 데이터 측면
- **다양한 해부학적 부위**(복부, 두경부, 심장 등)의 CT 데이터 통합 학습
- **Data augmentation**: 회전, 플립, 스케일링, 노이즈 주입 등
- **실제 sparse-view 스캔 데이터** 활용 또는 현실적 노이즈 모델 적용
- **다중 서브샘플링 비율**에 대한 학습 (예: 2배, 4배, 8배 등)

#### (b) 모델 측면
- **도메인 적응(domain adaptation)** 기법 도입으로 새로운 스캔 기하/부위에 대한 전이 학습
- **조건부 생성 모델**: 서브샘플링 비율을 조건 입력으로 하여 단일 네트워크로 다양한 비율 처리
- **자가 지도 학습(self-supervised learning)** 또는 **비지도 학습** 패러다임 도입으로 paired data 의존성 감소
- **물리 기반 제약조건** 통합: 데이터 일관성(data consistency) 층을 네트워크에 내장

#### (c) 학습 전략 측면
- **적대적 학습(adversarial training)** 도입으로 지각적 품질 향상 및 과도한 스무딩 방지
- **perceptual loss** 또는 **SSIM loss** 등 다중 손실 함수 조합
- **교차 검증(cross-validation)** 프로토콜 강화
- **불확실성 추정(uncertainty estimation)** 통합으로 신뢰도 기반 임상 의사결정 지원

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구적 영향

본 논문은 **사이노그램 도메인에서의 딥러닝 적용**이라는 패러다임을 CT 재구성 분야에 확립하는 데 기여한 초기 연구 중 하나이다. 이후 연구들에 미친 주요 영향은 다음과 같다:

1. **이중 도메인 접근법의 기반 마련**: 사이노그램 도메인과 영상 도메인을 모두 활용하는 연구의 토대
2. **딥러닝 기반 데이터 완성(data completion)** 개념의 CT 분야 도입
3. **전통적 재구성 알고리즘(FBP)과의 호환성 유지**: 딥러닝을 전처리로 활용하여 기존 임상 파이프라인과의 통합 용이성 시사
4. **반복적 재구성 대비 계산 효율성**: 학습 완료 후 빠른 추론 속도의 잠재력 제시

### 4.2 향후 연구 시 고려할 핵심 사항

| 고려 사항 | 상세 내용 |
|---------|---------|
| **데이터 일관성(Data Consistency)** | 네트워크 출력이 물리적 측정과 일관성을 유지하도록 명시적 제약 필요 |
| **임상적 검증** | NRMSE/PSNR/SSIM 외에 진단 정확도, ROC 분석 등 임상적 평가 필수 |
| **3D 확장** | 2D 패치 기반에서 3D 볼류메트릭 처리로의 확장 필요 |
| **실시간 처리** | 추론 속도 최적화를 통한 실시간 임상 적용 가능성 확보 |
| **안정성/강건성** | 적대적 사례(adversarial examples), 분포 밖(out-of-distribution) 데이터에 대한 강건성 검증 |
| **해석 가능성** | 네트워크가 학습한 특징의 물리적 의미 분석, 신뢰도 맵 제공 |
| **규제 및 인증** | 의료기기 인허가를 위한 표준화된 성능 검증 프로토콜 수립 |

### 4.3 저자들이 제시한 향후 연구 방향

- **Cone-beam CT** 및 **helical multiple fan-beam CT**로의 확장
- **불규칙 각도 샘플링** 대응
- **결손 검출기 채널 문제** 해결
- 학습 데이터 **중복성 감소**를 통한 학습 속도 향상

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 관련 연구 동향

| 연구 | 연도 | 핵심 접근 | 본 논문과의 차별점 |
|-----|------|---------|--------------|
| **Learned Primal-Dual (Adler & Öktem)** | 2018/후속 2020+ | 반복적 재구성 과정을 unrolling하여 학습 | 사이노그램과 영상 도메인을 번갈아 처리하는 end-to-end 학습 |
| **FBPConvNet (Jin et al.)** [35] 후속 연구들 | 2020+ | FBP 후 영상 도메인 후처리 | 본 논문은 사이노그램 도메인 전처리에 초점 |
| **DRONE (Wu et al., 2021)** | 2021 | 이중 도메인(dual-domain) 네트워크 | 사이노그램 + 영상 도메인 동시 최적화 |
| **ISTA-Net++ / ADMM-Net 계열** | 2020+ | Algorithm unrolling 기반 | 최적화 알고리즘의 반복 과정을 네트워크 층으로 전개 |
| **Score-based diffusion models for CT** | 2022+ | 확산 모델 기반 영상 재구성 | 확률적 생성 모델로 불확실성 추정 가능 |
| **Neural Attenuation Fields (NAF)** | 2022 | 암시적 신경 표현(implicit neural representation) | 단일 스캔에 대한 최적화, 학습 데이터 불필요 |

### 5.2 상세 비교

#### (a) Dual-Domain Networks (이중 도메인 네트워크)

본 논문이 사이노그램 도메인에만 집중한 반면, 2020년 이후의 연구들은 **사이노그램 도메인과 영상 도메인을 동시에 활용**하는 이중 도메인 접근법을 선호한다.

- **Lin et al. (2019, "DuDoNet") 및 후속 연구들**: 사이노그램 도메인에서 결손 데이터를 복원한 뒤, 영상 도메인에서 추가 개선을 수행하는 두 단계 네트워크. 본 논문의 접근법을 영상 도메인까지 확장한 형태로 볼 수 있다.
  - *출처: Lin et al., "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction," CVPR 2019*

#### (b) Algorithm Unrolling / Learned Iterative Methods

$$x^{(n+1)} = \text{CNN}_\theta\left(x^{(n)} - \alpha \nabla \frac{1}{2}\|Ax^{(n)} - y\|_2^2\right)$$

여기서 $A$는 시스템 매트릭스(forward projection operator), $y$는 측정된 사이노그램, $\text{CNN}_\theta$는 학습 가능한 정규화 모듈이다.

- **Learned Primal-Dual (Adler & Öktem, 2018) 후속 연구들 (2020+)**: primal 도메인(영상)과 dual 도메인(사이노그램)에서의 업데이트를 교대로 수행하는 구조. 물리적 forward/backward 모델을 네트워크에 통합하여 **데이터 일관성을 구조적으로 보장**.
  - *출처: Adler, J. and Öktem, O., "Learned Primal-Dual Reconstruction," IEEE TMI, 2018*

이러한 접근법은 본 논문의 한계인 **데이터 일관성 미보장 문제**를 구조적으로 해결한다.

#### (c) Generative Models (생성 모델)

- **Score-based Diffusion Models (Song et al., 2021; Chung et al., 2022)**: 확산 모델을 CT 재구성에 적용하여, 데이터 일관성 조건 하에서 사후 분포(posterior distribution)로부터 샘플링.

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

이 접근법은 **불확실성 추정**이 가능하며, MSE 기반 손실의 과도한 스무딩 문제를 완화한다.
  - *출처: Chung, H. et al., "Score-based diffusion models for accelerated MRI," Medical Image Analysis, 2022*
  - *출처: Song, Y. et al., "Solving Inverse Problems in Medical Imaging with Score-Based Generative Models," ICLR 2022*

#### (d) Implicit Neural Representations (암시적 신경 표현)

- **Neural Attenuation Fields (NAF, Zha et al., 2022)**: NeRF 스타일의 좌표 기반 네트워크로 감쇠 계수 필드를 직접 표현. **외부 학습 데이터 없이 단일 sparse-view 데이터만으로 재구성** 가능.
  - *출처: Zha, R. et al., "NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction," MICCAI 2022*

이 방법은 본 논문과 달리 **사전 학습 데이터가 불필요**하며, 일반화 문제를 근본적으로 우회한다.

#### (e) Transformer 기반 접근법

- **2021년 이후** Vision Transformer(ViT)와 Swin Transformer를 CT 재구성에 도입하는 연구들이 등장. 장거리 의존성(long-range dependency) 모델링 능력이 사이노그램의 전역적 일관성 유지에 유리하다.
  - *출처: Wang, C. et al., "CTformer: Convolution-free Token2Token Dilated Vision Transformer for Low-dose CT Denoising," PMB, 2023*

### 5.3 종합 비교 요약

| 특성 | 본 논문 (2017) | Dual-Domain (2019+) | Algorithm Unrolling (2020+) | Diffusion Models (2022+) | INR (2022+) |
|-----|------------|----------------|----------------------|---------------------|----------|
| **도메인** | 사이노그램 | 사이노그램 + 영상 | 사이노그램 + 영상 | 영상 (+ 사이노그램 제약) | 연속 공간 |
| **데이터 일관성** | 미보장 | 부분 보장 | 구조적 보장 | 조건부 보장 | 내재적 보장 |
| **학습 데이터 필요성** | 대량 필요 | 대량 필요 | 대량 필요 | 대량 필요 | 불필요 (test-time optimization) |
| **불확실성 추정** | 불가 | 불가 | 불가 | 가능 | 제한적 |
| **일반화** | 제한적 | 개선됨 | 개선됨 | 양호 | 우회 |
| **계산 비용** | 중간 | 높음 | 높음 | 매우 높음 | 높음 |

---

## 참고자료

1. **Lee, H., Lee, J., Kim, H., Cho, B., and Cho, S.** "Deep-neural-network based sinogram synthesis for sparse-view CT image reconstruction," *IEEE Transactions on Radiation and Plasma Medical Sciences*, 2019. (본 분석 대상 논문)
2. **Ronneberger, O., Fischer, P., and Brox, T.** "U-net: Convolutional networks for biomedical image segmentation," *MICCAI*, 2015. [논문 내 참고문헌 43]
3. **He, K., Zhang, X., Ren, S., and Sun, J.** "Deep residual learning for image recognition," *CVPR*, 2016. [논문 내 참고문헌 44]
4. **Kingma, D. and Ba, J.** "Adam: A method for stochastic optimization," *arXiv preprint arXiv:1412.6980*, 2014. [논문 내 참고문헌 49]
5. **Sidky, E. Y., Kao, C.-M., and Pan, X.** "Accurate image reconstruction from few-views and limited-angle data in divergent-beam CT," *Journal of X-ray Science and Technology*, vol. 14, pp. 119-139, 2006. [논문 내 참고문헌 51]
6. **Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P.** "Image quality assessment: from error visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, pp. 600-612, 2004. [논문 내 참고문헌 52]
7. **Adler, J. and Öktem, O.** "Learned Primal-Dual Reconstruction," *IEEE Transactions on Medical Imaging*, vol. 37, no. 6, pp. 1322-1332, 2018.
8. **Lin, W.-A. et al.** "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction," *CVPR*, 2019.
9. **Chung, H. et al.** "Score-based diffusion models for accelerated MRI," *Medical Image Analysis*, 2022.
10. **Song, Y. et al.** "Solving Inverse Problems in Medical Imaging with Score-Based Generative Models," *ICLR*, 2022.
11. **Zha, R. et al.** "NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction," *MICCAI*, 2022.
12. **Wang, C. et al.** "CTformer: Convolution-free Token2Token Dilated Vision Transformer for Low-dose CT Denoising," *Physics in Medicine & Biology*, 2023.
13. **Jin, K. H., McCann, M. T., Froustey, E., and Unser, M.** "Deep convolutional neural network for inverse problems in imaging," *IEEE Transactions on Image Processing*, vol. 26, pp. 4509-4522, 2017. [논문 내 참고문헌 35]
14. **Lee, H., Lee, J., and Cho, S.** "View-interpolation of sparsely sampled sinogram using convolutional neural network," *SPIE Medical Imaging*, 2017. [논문 내 참고문헌 48]
15. **The Cancer Imaging Archive (TCIA)**: https://www.cancerimagingarchive.net/ [논문 내 참고문헌 40]

---

> **주의사항**: 2020년 이후 최신 연구들의 비교 분석은 해당 분야의 일반적 동향에 기반하여 작성되었으며, 개별 연구의 정확한 실험 수치 비교는 각 논문의 원문을 직접 참조해야 합니다. 특히 diffusion model 기반 CT 재구성과 implicit neural representation 기반 접근법은 급격히 발전하는 분야이므로, 최신 survey 논문을 함께 참고하시기를 권장합니다.
