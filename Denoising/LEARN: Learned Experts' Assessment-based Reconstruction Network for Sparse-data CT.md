# LEARN: Learned Experts' Assessment-based Reconstruction Network for Sparse-data CT

## 종합 분석 보고서

---

## 1. 핵심 주장 및 주요 기여 요약

LEARN 논문은 **sparse-data CT(희소 데이터 CT) 재구성** 문제를 해결하기 위해, 기존의 압축 센싱(Compressive Sensing) 기반 반복 재구성(Iterative Reconstruction) 알고리즘을 **딥러닝 네트워크로 언폴딩(unfolding)**하는 접근법을 제안한다. 핵심 주장은 다음과 같다:

1. **Fields of Experts(FoE) 기반 반복 재구성 스킴을 고정된 횟수만큼 전개(unfold)하여** 데이터 기반 학습이 가능한 신경망(LEARN)을 구성한다.
2. **모든 정규화 항(regularization terms)과 균형 파라미터(balancing parameters)가 반복(iteration)마다 독립적으로 학습**되므로, 수작업 파라미터 튜닝 문제를 해소한다.
3. 단 **50개의 반복(layer)**으로 기존 반복 알고리즘의 수백~수천 회 반복을 대체하여 **계산 복잡도를 수 자릿수(orders of magnitude) 감소**시킨다.
4. Mayo Clinic Low-Dose CT Challenge Dataset에서 **아티팩트 감소, 특징 보존, 계산 속도** 면에서 기존 최첨단 기법들보다 우수한 성능을 입증한다.

**주요 기여:**
- 반복 재구성을 CNN으로 직접 변환하여 물리 모델(data fidelity)과 학습 기반 정규화를 통합
- 반복 인덱스별(iteration-dependent) 파라미터 학습이라는 새로운 패러다임 제시
- CT 재구성 분야에서 algorithm unrolling의 선구적 적용

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Sparse-data CT에서는 투영 데이터가 부족하여 선형 시스템이 **비결정(underdetermined)** 상태가 된다:

$$Ax = y$$

여기서 $x \in \mathbb{R}^J$는 감쇠 계수 벡터(이미지), $A \in \mathbb{R}^{I \times J}$는 시스템 행렬, $y \in \mathbb{R}^I$는 측정 데이터이다. $I \ll J$인 경우 해가 무한히 존재하며, FBP로는 심각한 아티팩트가 발생한다.

기존 CS 기반 반복 재구성의 **세 가지 주요 한계**:
- **(a) 계산 비용:** 수백~수천 회 반복에 따른 높은 계산 비용
- **(b) 범용 정규화의 부재:** 다양한 임상 과제에 보편적으로 적합한 정규화 항을 찾기 어려움
- **(c) 파라미터 설정의 어려움:** 데이터 충실도와 정규화 간 균형 파라미터를 적응적으로 설정하는 것이 미해결 문제

### 2.2 제안하는 방법 (수식 포함)

#### 정규화된 CT 재구성

일반적인 정규화 기반 목적함수:

$$x = \arg\min_x E(x) = \arg\min_x \frac{\lambda}{2}\|Ax - y\|_2^2 + R(x), \quad s.t. \; x_j \geq 0 \;\; \forall j$$

#### Fields of Experts(FoE) 정규화

FoE 모델을 정규화 항으로 채택:

$$R(x) = \sum_{k=1}^{K} \phi_k(G_k x)$$

여기서 $K$는 정규화기의 수, $G_k$는 $N_f$ 크기의 변환 행렬(합성곱 연산자), $\phi_k(\cdot)$는 포텐셜 함수이다. 이를 삽입하면:

$$x = \arg\min_x \frac{\lambda}{2}\|Ax - y\|_2^2 + \sum_{k=1}^{K} \phi_k(G_k x), \quad s.t. \; x \geq 0$$

#### 경사 하강법 기반 반복 스킴

경사 하강법을 적용하면:

$$x^{t+1} = x^t - \alpha \cdot \eta(x^t) = x^t - \alpha \frac{\partial E}{\partial x}$$

그래디언트:

$$\eta(x^t) = \lambda A^T(Ax^t - y) + \sum_{k=1}^{K} (G_k)^T \gamma_k(G_k x^t)$$

여기서 $\gamma(\cdot) = \phi'(\cdot)$이다.

#### 반복 의존적(iteration-dependent) 확장 — LEARN의 핵심

모든 파라미터를 반복 인덱스 $t$에 종속시켜:

$$x^{t+1} = x^t - \left(\lambda^t A^T(Ax^t - y) + \sum_{k=1}^{K} (G_k^t)^T \gamma_k^t(G_k^t x^t)\right)$$

$\lambda^t$, $G_k^t$, $\gamma_k^t$가 모두 반복마다 다른 값을 가지며, $\alpha$는 자유 스케일링 가능하므로 생략된다.

#### 3-Layer CNN 대체

각 반복에서 $\sum_{k=1}^{K}(G_k^t)^T\gamma_k^t(G_k^t x^t)$를 3층 CNN으로 대체:

$$M(x^{t-1}) = \mathbf{W}_3^{t-1} * \text{ReLU}(\mathbf{W}_2^{t-1} * \text{ReLU}(\mathbf{W}_1^{t-1} * x^{t-1} + \mathbf{b}_1^{t-1}) + \mathbf{b}_2^{t-1}) + \mathbf{b}_3^{t-1}$$

#### 학습 목적함수

훈련 데이터셋 $D = \{(y_s, x_s)\}_{s=1}^{N_D}$에 대해:

$$\min_\Theta L(D;\Theta) = \frac{1}{2N_D}\sum_{s=1}^{N_D}\|x_s^{N_t}(y_s, \Theta) - x_s\|_2^2$$

제약 조건:

$$x_s^{t+1} = x_s^t - \left(\lambda^t A^T(Ax_s^t - y_s) + M(x_s^t)\right), \quad t = 0, 1, 2, \ldots, N_t - 1$$

파라미터 집합: $\Theta^t = \{\lambda^t, \mathbf{W}_1^t, \mathbf{W}_2^t, \mathbf{W}_3^t, \mathbf{b}_1^t, \mathbf{b}_2^t, \mathbf{b}_3^t\}$

#### 역전파

데이터 충실도 항의 존재로 인해 일반 CNN과 다른 역전파가 필요:

$$\frac{\partial L}{\partial \Theta^t} = \frac{\partial x^{t+1}}{\partial \Theta^t} \cdot \frac{\partial x^{t+2}}{\partial x^{t+1}} \cdots \frac{\partial x^{N_t}}{\partial x^{N_t-1}} \cdot \frac{\partial L}{\partial x^{N_t}}$$

$$\frac{\partial L}{\partial x^{N_t}} = x^{N_t} - x$$

$$\frac{\partial x^{t+1}}{\partial \lambda^t} = -\left(A^T(Ax_s^t - y_s)\right)^T$$

$$\frac{\partial x^{t+2}}{\partial x^{t+1}} = I - \left(\lambda^{t+1}A^T A + \frac{\partial M(x^{t+1})}{\partial x^{t+1}}\right)$$

### 2.3 모델 구조

LEARN 네트워크의 전체 구조는 다음과 같다:

1. **입력:** 초기 이미지 $x^0$(FBP 결과 또는 영 이미지), 시스템 행렬 $A$, $A^T$, 측정 데이터 $y$
2. **반복 블록 ($N_t = 50$회 언폴딩):** 각 블록은 다음을 포함:
   - **데이터 충실도 항:** $\lambda^t A^T(Ax^t - y)$
   - **3-Layer CNN:** 학습된 정규화 (필터 수 $n_1 = n_2 = 48$, $n_3 = 1$, 커널 크기 $5 \times 5$)
   - **잔차 연결(shortcut connection):** $x^{t-1} \rightarrow x^t$
3. **출력:** 재구성 이미지 $x^{N_t}$
4. **총 레이어 수:** 약 150층 (반복당 3층 × 50회)
5. **활성화 함수:** ReLU
6. **손실 함수:** MSE
7. **최적화:** Adam optimizer (초기 학습률 $10^{-4}$)

각 반복 블록은 **잔차 네트워크(residual network)** 형태로, 구조적 세부사항 보존과 학습 가속을 동시에 달성한다.

### 2.4 성능 향상

**Mayo Clinic Low-Dose CT Challenge Dataset**에서 5가지 최첨단 방법과 비교:

| 방법 | 64 views PSNR | 128 views PSNR | 속도(초) |
|------|:---:|:---:|:---:|
| FBP | 25.18 | 28.46 | **0.10** |
| ASD-POCS | 34.16 | 36.12 | 15.92 |
| Dual-DL | 34.25 | 36.20 | 1654.25 |
| PWLS-TGV | 35.65 | 39.25 | 627.24 |
| Gamma-Reg | 36.01 | 38.79 | 50.23 |
| FBPConvNet | 35.36 | 38.59 | 1.08 |
| **LEARN** | **40.73** | **43.38** | 5.89 |

- 64 views에서 PSNR **약 5.2 dB 향상** (차순위 대비)
- 128 views에서 PSNR **약 3.1 dB 향상**
- ASD-POCS 대비 **3배**, Dual-DL 대비 **300배**, PWLS-TGV 대비 **100배** 빠른 속도
- 정성 평가에서 두 영상의학과 전문의 모두 LEARN이 참조 이미지와 통계적으로 유의한 차이가 없음을 확인 (Student's t-test, $p > 0.05$)

### 2.5 한계

1. **MSE 손실 함수의 한계:** 대비(contrast) 보존이 모든 상황에서 완벽하지 않음. Perceptual loss나 GAN 프레임워크가 더 적합할 수 있음.
2. **비볼록(non-convex) 최적화:** 에너지 함수의 전역 최적해 보장 불가. 초기값과 학습 경로에 의존.
3. **훈련 비용:** CPU에서 100개 이미지 학습에 약 49시간, 200개 이미지에 약 80시간 소요.
4. **시스템 행렬 학습 미포함:** $A$와 $A^T$는 고정된 상태로 사용. 시스템 행렬까지 학습하면 추가 성능 향상 가능성 존재.
5. **메모리 제약:** 150층 이상의 네트워크로 인해 GPU 메모리가 도전적.

---

## 3. 모델의 일반화 성능 향상 가능성

논문은 LEARN의 일반화 성능에 대해 다각도로 검증하고 논의한다:

### 3.1 초기값 강건성

FBP 초기값과 영(zero) 이미지로 각각 초기화한 실험에서, 시각적 차이가 거의 없으며 정량 지표 차이도 미미하였다 (예: 전체 교차 검증에서 FBP 초기화 PSNR 40.73 vs. Zero 초기화 PSNR 39.37). 이는 **데이터 충실도 항이 매 반복마다 강력한 제약으로 작용**하여, 초기값 의존성을 크게 완화함을 보여준다.

### 3.2 노이즈 수준에 대한 강건성

노이즈 없는 데이터로 학습한 후, 다양한 강도의 포아송 노이즈가 추가된 테스트 데이터에 적용한 결과, blank scan factor $b_0 > 1 \times 10^7$까지 성능이 안정적으로 유지되었다. 이는 **학습 데이터와 테스트 데이터 간 노이즈 불일치에 대해서도 일정 수준 강건함**을 의미한다.

### 3.3 학습/테스트 데이터 불일치에 대한 강건성

흉부(thoracic) 이미지만으로 학습한 네트워크를 복부(abdominal) 이미지 테스트에 적용하거나 그 반대의 경우에도, 다양한 부위를 포함한 훈련과 유사한 성능을 달성하였다:

| 설정 | RMSE | PSNR | SSIM |
|------|:---:|:---:|:---:|
| LEARN-All-T (혼합 훈련, 흉부 테스트) | 0.0140 | 37.20 | 0.9472 |
| LEARN-A-T (복부 훈련, 흉부 테스트) | 0.0139 | 37.11 | 0.9484 |
| LEARN-All-A (혼합 훈련, 복부 테스트) | 0.0100 | 40.08 | 0.9574 |
| LEARN-T-A (흉부 훈련, 복부 테스트) | 0.0109 | 39.32 | 0.9526 |

이 강건성의 원인으로 논문은 두 가지를 제시한다:
1. **투영 데이터의 직접 활용:** 매 반복마다 데이터 충실도 항이 재구성을 신뢰 가능한 범위 내에 유지
2. **잔차 학습의 효과:** 네트워크가 아티팩트 패턴에 집중하게 되며, 아티팩트는 신체 부위와 무관하게 유사한 특성을 가짐

### 3.4 적은 훈련 샘플로의 일반화

200개 훈련 샘플만으로 150층 네트워크를 학습하면서도 과적합(overfitting)이 발생하지 않았다. 이는:
- **저수준 특징(edge, shape, texture)만 학습**하면 되므로 고수준 패턴 인식에 비해 훨씬 적은 데이터로 충분
- **투영 데이터가 각 층에서 제약으로 작용**하여 DenseNet과 유사한 효과를 발휘

### 3.5 일반화 향상을 위한 미래 방향

- **시스템 행렬 $A$ 학습:** 다양한 CT 기하학 구성에 대한 적응성 향상
- **GAN 기반 손실 함수:** perceptual similarity 활용으로 임상적 대비 보존 개선
- **더 다양한 훈련 데이터셋:** 다양한 해부학적 부위, 병리, 스캔 프로토콜 포함
- **고급 최적화 기법 도입:** Quasi-Newton 방법 등으로 언폴딩 스킴 개선

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구적 영향

LEARN은 **algorithm unrolling** 패러다임을 CT 재구성에 본격 적용한 선구적 연구로서:

1. **물리 기반 딥러닝(Physics-informed Deep Learning)의 촉매:** 블랙박스 네트워크 대신 물리적·수학적 근거에서 출발하여 네트워크 구조를 설계하는 방법론을 확립
2. **반복 알고리즘과 딥러닝의 융합:** 데이터 충실도와 학습된 정규화를 동시에 활용하는 하이브리드 접근법의 기틀 마련
3. **임상 적용 가능성:** 속도와 품질의 균형을 달성하여, 실시간 CT 재구성에 대한 가능성 제시

### 4.2 앞으로 연구 시 고려할 점

1. **손실 함수 설계:** MSE 외에 perceptual loss, adversarial loss, SSIM loss 등 복합 손실 함수 적용
2. **수렴 보장과 이론적 분석:** 비볼록 최적화 환경에서의 수렴성에 대한 이론적 기반 확보 필요
3. **3D/4D 확장:** 현재 2D 슬라이스 기반 → 3D 볼륨 또는 4D(시간 축 포함) 재구성으로 확장
4. **다양한 영상 모달리티 적용:** MRI, PET, SPECT 등 다른 의료 영상 분야로의 확장
5. **임상 검증:** 대규모 다기관 임상 데이터에서의 일반화 성능 검증
6. **시스템 행렬 학습:** 스캐너 기하학까지 학습하여 완전한 end-to-end 학습 달성
7. **메모리 효율성:** 가역 네트워크(invertible network) 등을 활용한 메모리 최적화

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

LEARN 이후 algorithm unrolling 기반 CT/의료 영상 재구성 분야는 급격히 발전하였다. 주요 후속 연구들과의 비교:

### 5.1 ADMM-Net 계열의 발전

| 연구 | 주요 특징 | LEARN 대비 차별점 |
|------|---------|----------------|
| **ADMM-CSNet** (Yang et al., 2020, IEEE TPAMI) | ADMM 알고리즘을 언폴딩하여 CS-MRI 재구성 | ADMM 분할 구조 활용으로 부분 문제 분리; LEARN은 gradient descent 기반 |
| **ISTA-Net++** (Zhang & Ghanem, 2018/2020 확장) | ISTA 알고리즘 언폴딩, 학습 가능한 soft-thresholding | 수축 연산자 학습에 특화; LEARN은 FoE 기반 보다 일반적 정규화 학습 |

### 5.2 Learned Primal-Dual (LPD) 계열

| 연구 | 주요 특징 | LEARN 대비 차별점 |
|------|---------|----------------|
| **Learned Primal-Dual Reconstruction** (Adler & Öktem, 2018, IEEE TMI; 2020년대 후속 확장 다수) | Primal-dual 최적화를 언폴딩, primal과 dual 공간 모두에서 학습 | 이중 공간(dual space) 활용으로 더 풍부한 정보 처리; LEARN은 primal 공간만 사용 |

### 5.3 End-to-End Variational Networks

| 연구 | 주요 특징 | LEARN 대비 차별점 |
|------|---------|----------------|
| **iCTU-Net** (Li et al., 2020, Phys. Med. Biol.) | U-Net을 각 반복 블록의 정규화 모듈로 사용 | LEARN의 3-Layer CNN보다 표현력이 높은 U-Net 정규화기 |
| **LoDoPaB-CT Benchmark** (Leuschner et al., 2021, Scientific Data) | 대규모 벤치마크 데이터셋과 다양한 학습 기반 재구성 비교 | LEARN 계열을 포함한 체계적 비교 프레임워크 제공 |

### 5.4 Plug-and-Play (PnP) 및 Deep Equilibrium 접근법

| 연구 | 주요 특징 | LEARN 대비 차별점 |
|------|---------|----------------|
| **Deep Equilibrium Models (DEQ)** (Gilton et al., 2021, IEEE TCI) | 무한 깊이 네트워크의 고정점을 직접 계산 | LEARN의 고정 반복 수 제한 극복; 메모리 효율적 |
| **PnP-DL** (Ahmad et al., 2020, IEEE SPM) | 사전 학습된 denoiser를 반복 알고리즘에 플러그인 | LEARN과 달리 denoiser를 별도 학습; 유연하지만 end-to-end 최적화 아님 |

### 5.5 Transformer 기반 접근법

| 연구 | 주요 특징 | LEARN 대비 차별점 |
|------|---------|----------------|
| **Eformer** (Wang et al., 2023, Med. Image Anal.) | Transformer를 정규화 모듈에 도입하여 장거리 의존성 포착 | LEARN의 CNN 기반 정규화의 제한된 수용장(receptive field)을 극복 |
| **DuDoTrans** (Wang et al., 2022, MICCAI) | 이중 도메인(sinogram + image) Transformer | 데이터 도메인과 이미지 도메인 모두에서의 학습 |

### 5.6 자기지도/비지도 학습 기반 접근법

| 연구 | 주요 특징 | LEARN 대비 차별점 |
|------|---------|----------------|
| **Noise2Inverse** (Hendriksen et al., 2020, IEEE TCI) | 쌍 데이터 없이 투영 데이터 분할로 자기지도 학습 | LEARN은 지도 학습에 의존; 참조 데이터 없는 시나리오에서 유리 |
| **Deep Image Prior for CT** (Baguer et al., 2020, Inverse Problems) | 사전 학습 없이 네트워크 구조 자체를 prior로 활용 | 학습 데이터 불필요; 단일 영상 재구성에 특화 |

### 5.7 종합 비교 관점

| 측면 | LEARN (2018) | 최신 연구 (2020+) |
|------|:---:|:---:|
| 정규화 모듈 | 3-Layer CNN | U-Net, Transformer, DenseNet 등 |
| 학습 패러다임 | 지도 학습 (MSE) | 자기지도, 적대적 학습, perceptual loss |
| 최적화 기반 | Gradient Descent | ADMM, Primal-Dual, FISTA, DEQ |
| 도메인 | 이미지 도메인 | 이중 도메인 (sinogram + image) |
| 이론적 보장 | 제한적 | 수렴 보장 연구 증가 |
| 일반화 전략 | 데이터 충실도 제약 | 메타 학습, 도메인 적응 |
| 메모리 효율 | GPU 메모리 도전적 | 가역 네트워크, DEQ 등으로 개선 |

---

## 참고자료

1. **Hu Chen, Yi Zhang, et al.**, "LEARN: Learned Experts' Assessment-based Reconstruction Network for Sparse-data CT," *IEEE Transactions on Medical Imaging*, 37(6), 1333-1347, 2018. (본 논문)
2. **S. Roth and M. J. Black**, "Fields of Experts," *Int. J. Comput. Vis.*, 82(2), 205–229, 2009.
3. **Y. Chen and T. Pock**, "Trainable nonlinear reaction diffusion: a flexible framework for fast and effective image restoration," *IEEE TPAMI*, 39(6), 1256–1272, 2017.
4. **K. Gregor and Y. LeCun**, "Learning fast approximations of sparse coding," *Proc. ICML*, 2010, pp. 399-406.
5. **Y. Yang, J. Sun, H. Li, and Z. Xu**, "Deep ADMM-net for compressive sensing MRI," *Proc. NeurIPS*, 2016.
6. **K. Hammernik et al.**, "Learning a variational network for reconstruction of accelerated MRI data," *Magnetic Resonance in Medicine*, 79(6), 3055-3071, 2018.
7. **J. Adler and O. Öktem**, "Learned Primal-Dual Reconstruction," *IEEE TMI*, 37(6), 1322-1332, 2018.
8. **D. Gilton, G. Ongie, and R. Willett**, "Deep Equilibrium Architectures for Inverse Problems in Imaging," *IEEE TCI*, 7, 2021.
9. **R. Ahmad, C. A. Bouman, et al.**, "Plug-and-Play Methods for Magnetic Resonance Imaging," *IEEE Signal Processing Magazine*, 37(1), 105-116, 2020.
10. **A. A. Hendriksen et al.**, "Noise2Inverse: Self-Supervised Deep Convolutional Denoising for Tomography," *IEEE TCI*, 6, 1320-1335, 2020.
11. **D. O. Baguer, J. Leuschner, and M. Schmidt**, "Computed Tomography Reconstruction Using Deep Image Prior and Learned Reconstruction Methods," *Inverse Problems*, 36(9), 094004, 2020.
12. **J. Leuschner et al.**, "LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction," *Scientific Data*, 8, 109, 2021.
13. **K. H. Jin et al.**, "Deep convolutional neural network for inverse problems in imaging," *IEEE TIP*, 26(9), 4509–4522, 2017. (FBPConvNet)
14. **E. Y. Sidky and X. Pan**, "Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization," *Phys. Med. Biol.*, 53(17), 4777–4807, 2008. (ASD-POCS)
15. **Mayo Clinic Low-Dose CT Grand Challenge Dataset:** http://www.aapm.org/GrandChallenge/LowDoseCT/

---

**정확도 관련 참고사항:** 위 내용은 제공된 원 논문의 내용을 기반으로 작성되었습니다. 2020년 이후 최신 연구 비교 분석 부분은 해당 분야의 일반적으로 알려진 주요 연구들에 기반하였으며, 특정 정량 수치의 직접 비교는 실험 조건(데이터셋, 뷰 수, 전처리 등)이 다를 수 있으므로 해당 원논문을 직접 참조하시기 바랍니다.
