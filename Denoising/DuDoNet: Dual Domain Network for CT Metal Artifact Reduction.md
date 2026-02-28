# DuDoNet: Dual Domain Network for CT Metal Artifact Reduction

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
DuDoNet은 CT 영상에서 금속 임플란트로 인한 아티팩트(Metal Artifact)를 줄이기 위해 **시노그램(sinogram) 도메인과 이미지(CT image) 도메인을 동시에 학습하는 최초의 end-to-end 이중 도메인 네트워크**를 제안한다. 기존의 단일 도메인 접근법(시노그램 인페인팅 또는 이미지 향상)이 각각 이차 아티팩트(secondary artifact) 또는 비국소적(non-local) 금속 그림자 제거에 한계를 보이는 문제를 극복한다.

### 주요 기여 (4가지)
| 기여 | 설명 |
|------|------|
| **End-to-end 이중 도메인 학습** | 시노그램 복원과 CT 이미지 향상을 동시에 수행하는 최초의 end-to-end 네트워크 |
| **Mask Pyramid (MP) U-Net** | 작은 금속 임플란트의 마스크 정보가 다운샘플링으로 소실되는 문제를 해결하기 위한 다중 스케일 마스크 피라미드 구조 |
| **Radon Inversion Layer (RIL)** | FBP 알고리즘을 미분 가능한 네트워크 레이어로 구현하여 이미지 도메인에서 시노그램 도메인으로 그래디언트 역전파를 가능하게 함 |
| **Radon Consistency (RC) Loss** | 이미지 도메인에서 이차 아티팩트를 직접 페널라이징하고, RIL을 통해 시노그램 도메인의 일관성을 개선하는 손실 함수 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

CT 영상에서 금속 임플란트가 존재할 때 발생하는 **금속 아티팩트 저감(Metal Artifact Reduction, MAR)** 문제를 해결한다. 기존 방법의 두 가지 핵심 한계는:

1. **이미지 도메인 한계**: 금속 아티팩트는 **구조적이고 비국소적(structured and non-local)**이어서(streaking, shadowing 등), 단순한 이미지 도메인 향상으로는 효과적으로 제거할 수 없다.
2. **시노그램 도메인 한계**: 시노그램 인페인팅 기반 접근법은 보간된 데이터가 물리적 일관성을 위반하여 **심각한 이차 아티팩트(secondary artifact)**를 유발한다.

#### CT 이미징의 물리적 모델

다색(polychromatic) X-선 소스 하에서 에너지 분포 $\eta(E)$에 대한 CT 모델은:

$$Y = -\log \int \eta(E) \exp\{-\mathcal{P}X(E)\} \, dE $$

여기서 $\mathcal{P}$는 투영 생성 과정, $Y$는 시노그램, $X(E)$는 에너지 $E$에서의 감쇠 계수 분포이다.

금속이 없는 정상 조직에서는 $X(E) \approx X$ (에너지에 대해 거의 일정)이므로:

$$Y = \mathcal{P}X $$

그러나 금속 임플란트 $I_M(E)$가 존재하면 $X(E) = X + I_M(E)$이 되어:

$$Y = \mathcal{P}X - \log \int \eta(E) \exp\{-\mathcal{P}I_M(E)\} \, dE $$

재구성 알고리즘 $\mathcal{P}^\dagger$를 적용하면:

$$\mathcal{P}^\dagger Y = \hat{X} - \mathcal{P}^\dagger \log \int \eta(E) \exp\{-\mathcal{P}I_M(E)\} \, dE $$

Eq.(4)에서 $\hat{X}$ 뒤의 항이 바로 **금속 아티팩트**이다. 완벽한 MAR은 이 항을 억제하면서 $\hat{X}$는 영향받지 않게 해야 하나, 두 항 모두 metal trace 영역에 기여하므로 본질적으로 **비적정 문제(ill-posed problem)**이다.

---

### 2.2 제안하는 방법 및 수식

DuDoNet은 세 가지 주요 구성 요소로 이루어진다:

#### (a) Sinogram Enhancement Network (SE-Net)

금속으로 훼손된 시노그램 $Y$와 metal trace 마스크 $\mathcal{M}\_t \in \{0,1\}^{H_s \times W_s}$가 주어지면, 먼저 **선형 보간(Linear Interpolation, LI)**으로 초기 추정값 $Y_{LI}$를 생성한 후, Mask Pyramid U-Net 구조의 SE-Net $\mathcal{G}_s$가 metal trace 내부를 복원한다:

$$Y_{out} = \mathcal{M}_t \odot \mathcal{G}_s(Y_{LI}, \mathcal{M}_t) + (1 - \mathcal{M}_t) \odot Y_{LI} $$

SE-Net의 학습 손실:

$$\mathcal{L}_{\mathcal{G}_s} = \|Y_{out} - Y_{gt}\|_1 $$

여기서 $Y_{gt}$는 금속 아티팩트가 없는 ground truth 시노그램이다.

**Mask Pyramid의 핵심 아이디어**: 작은 금속 임플란트의 경우, 네트워크의 상위 레이어에서 다운샘플링으로 인해 metal trace 정보가 소실된다. MP 구조는 여러 스케일에서 마스크 정보를 유지하여 이 문제를 해결한다.

#### (b) Radon Inversion Layer (RIL)

RIL $f_R$은 FBP 알고리즘을 미분 가능한 네트워크 레이어로 구현한 것으로, 3개의 모듈로 구성된다:

**① Parallel-beam Conversion Module**: Fan-beam 시노그램 $Y_{fan}(\gamma, \beta)$을 parallel-beam $Y_{para}(t, \theta)$로 변환:

$$\begin{cases} t = D \sin \gamma, \\ \theta = \beta + \gamma. \end{cases} $$

**② Ram-Lak Filtering Module**: 주파수 도메인에서 Ram-Lak 필터링 적용:

$$Q(t, \theta) = \mathcal{F}_t^{-1} \{ |\omega| \cdot \mathcal{F}_t \{ Y_{para}(t, \theta) \} \} $$

여기서 $\mathcal{F}_t$와 $\mathcal{F}_t^{-1}$은 각각 DFT와 iDFT이다.

**③ Backprojection Module**: 필터링된 시노그램을 이미지 도메인으로 역투영:

$$X(u, v) = \int_0^{\pi} Q(u\cos\theta + v\sin\theta, \theta) \, d\theta $$

**RIL의 핵심 성질**: 역투영의 특성상, $f_R$의 입력 $Y_{out}$에 대한 미분은 투영 연산 $\mathcal{P}$와 동일하다. 즉, 이미지 도메인의 손실이 시노그램 도메인으로 직접 역전파된다.

> 기존 딥러닝 프레임워크에서 $\mathcal{P}$를 직접 역변환하려면 $O(H_s W_s H_c W_c)$의 시간/공간 복잡도가 필요하여 GPU 메모리 제약 상 비실용적이나, RIL은 FBP의 병렬 구조를 활용하여 이를 효율적으로 해결한다.

**Radon Consistency (RC) Loss**: RIL을 통해 재구성된 이미지에서 이차 아티팩트를 직접 페널라이징:

$$\mathcal{L}_{RC} = \|f_R(Y_{out}) - X_{gt}\|_1 $$

#### (c) Image Enhancement Network (IE-Net)

잔차 학습(residual learning)을 통한 최종 CT 이미지 향상:

$$X_{out} = X_{LI} + \mathcal{G}_i(\hat{X}, X_{LI}) $$

여기서 $X_{LI} = f_R(Y_{LI})$이고, IE-Net 손실:

$$\mathcal{L}_{\mathcal{G}_i} = \|X_{out} - X_{gt}\|_1 $$

#### 전체 목적 함수

$$\mathcal{L} = \mathcal{L}_{\mathcal{G}_s} + \mathcal{L}_{RC} + \mathcal{L}_{\mathcal{G}_i} $$

---

### 2.3 모델 구조

```
입력: 훼손된 시노그램 Y, Metal trace 마스크 M_t
         │
    ┌────┴────┐
    │  LI 보간  │ ──→ Y_LI ──→ RIL ──→ X_LI
    └─────────┘                          │
         │                               │
    ┌────┴────┐                     ┌────┴────┐
    │  SE-Net  │ ──→ Y_out          │         │
    │ (MP U-Net)│                    │  IE-Net  │
    └─────────┘                    │ (U-Net)  │
         │                         └────┬────┘
    ┌────┴────┐                         │
    │   RIL   │ ──→ X̂ ─────────────→  X_out
    └─────────┘
```

- **SE-Net**: Mask Pyramid U-Net — 다중 스케일에서 마스크 정보 융합
- **RIL**: Parallel-beam 변환 → Ram-Lak 필터링 → 역투영 (미분 가능)
- **IE-Net**: U-Net 기반 잔차 학습, $\hat{X}$와 $X_{LI}$를 입력으로 받아 최종 향상된 CT 출력

---

### 2.4 성능 향상

#### 정량적 결과 (Table 2 기준, 평균 PSNR/SSIM)

| 방법 | PSNR (dB) | SSIM |
|------|-----------|------|
| LI | 25.47 | 0.8917 |
| NMAR | 27.51 | 0.9001 |
| cGAN-CT | 28.07 | 0.8733 |
| RDN-CT | 31.74 | 0.9156 |
| CNNMAR | 29.52 | 0.9243 |
| **DuDoNet** | **33.51** | **0.9379** |

- DuDoNet은 모든 금속 크기 범주에서 PSNR과 SSIM 모두 일관되게 최고 성능 달성
- RDN-CT 대비 약 **1.77 dB PSNR** 향상, CNNMAR 대비 약 **3.99 dB** 향상
- 실행 시간: DuDoNet (0.1335s) vs RDN-CT (0.5150s) → **약 4배 빠름**

#### Ablation Study 주요 결과
- **RC Loss 효과**: RC Loss 사용 시 전체 금속 크기에서 최소 **0.3 dB** 이상 향상 (E→G)
- **Mask Pyramid 효과**: 작은 금속에서 약 **0.2 dB** 향상 (F→G), 큰 금속에서는 유사 성능
- **이중 도메인 효과**: 단일 이미지 도메인(IE-Net, 31.45 dB) 대비 이중 도메인(33.51 dB)이 약 **2 dB** 우수

---

### 2.5 한계점

논문에서 명시적으로 언급하거나 분석을 통해 도출할 수 있는 한계점은 다음과 같다:

1. **시뮬레이션 데이터 기반 학습 및 평가**: DeepLesion 데이터셋의 실제 환자 CT에 인공적으로 금속 아티팩트를 시뮬레이션하여 학습/평가함. 실제 임상 데이터에 대한 평가는 보충 자료(supplementary material)에만 제시되어 있으며, **sim-to-real 갭(시뮬레이션과 실제 간 차이)**이 존재할 수 있다.

2. **고정된 CT 기하학 가정**: Fan-beam 기하학과 특정 소스-회전 중심 거리( $D = 39.7$ cm), 고정된 프로젝션 뷰 수(320 views) 등 특정 CT 시스템 파라미터에 맞춰 설계되어, **다양한 CT 기하학에 대한 일반화가 보장되지 않음**.

3. **금속 종류/재질의 다양성 미반영**: 다양한 금속 재질(티타늄, 스테인리스 스틸, 코발트-크롬 등)에 따른 감쇠 특성 차이가 충분히 고려되지 않음.

4. **3D 확장 미비**: 2D 슬라이스 기반 처리로, 3D CT 볼륨에서의 슬라이스 간 일관성은 고려하지 않음.

5. **손실 함수의 단순성**: $L_1$ 손실만 사용하며, 지각적 손실(perceptual loss)이나 적대적 손실(adversarial loss)은 사용하지 않아 미세한 텍스처 복원에 한계가 있을 수 있음.

6. **각 손실 항의 가중치 튜닝 부재**: Eq.(14)에서 $\mathcal{L}\_{\mathcal{G}\_s}$, $\mathcal{L}\_{RC}$, $\mathcal{L}_{\mathcal{G}_i}$를 동일 가중치(1:1:1)로 사용하며, 최적 가중치 탐색이 이루어지지 않음.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 모델의 일반화 관련 특성

**긍정적 측면:**
- **물리 기반 레이어(RIL)의 도메인 지식 내장**: RIL은 FBP라는 물리적으로 정확한 재구성 알고리즘을 레이어로 구현함으로써, 순수 데이터 기반 접근보다 물리적 일관성이 보장되어 일반화에 유리하다.
- **이중 도메인 학습**: 시노그램과 이미지 도메인을 동시에 학습함으로써, 각 도메인의 보완적 정보를 활용하여 단일 도메인 방법보다 더 강건한 특징 학습이 가능하다.
- **대규모 데이터셋**: 320명 환자의 4,000장 이미지 × 90개 금속 형상 = 360,000개 학습 조합으로, 다양한 해부학적 구조와 금속 형상에 대한 학습이 이루어짐.

**개선이 필요한 측면:**

### 3.2 일반화 성능 향상을 위한 구체적 방향

#### (1) 다양한 CT 기하학으로의 확장
현재 모델은 특정 fan-beam 기하학에 고정되어 있다. 일반화를 위해:
- RIL을 cone-beam, helical CT 등 다양한 기하학을 지원하도록 확장
- CT 기하학 파라미터를 네트워크 입력으로 조건화(conditioning)하는 방법 고려

#### (2) 도메인 적응(Domain Adaptation) 기법 적용
시뮬레이션-실제 간 갭을 줄이기 위해:
- 비지도 도메인 적응(Unsupervised Domain Adaptation) 또는 반지도 학습(Semi-supervised Learning) 적용
- 실제 임상 데이터의 쌍을 이루지 않은(unpaired) 데이터를 활용한 학습 전략

#### (3) 데이터 증강 및 다양성 확대
- 다양한 금속 재질, 크기, 위치, 개수에 대한 증강
- 다양한 X-선 스펙트럼 조건, 노이즈 수준에 대한 학습
- 다양한 해부학적 부위(두경부, 척추, 골반, 사지 등)에 대한 학습 데이터 확보

#### (4) 자기 지도 학습(Self-supervised Learning) 도입
- Ground truth가 없는 실제 임상 데이터에서도 학습 가능한 프레임워크 개발
- Noise2Noise [15] 스타일의 접근법을 MAR에 적용

#### (5) 3D 볼류메트릭 확장
- 인접 슬라이스 간 정보를 활용하여 3D 일관성을 유지하는 확장
- 메모리 효율적 3D RIL 설계

#### (6) 테스트 시 적응(Test-time Adaptation)
- 테스트 시 입력 데이터의 특성에 맞게 모델을 적응시키는 기법 적용
- 물리적 일관성 제약을 활용한 테스트 시 미세 조정

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구적 영향

#### (1) 이중 도메인 학습 패러다임 확립
DuDoNet은 **"시노그램 + 이미지"라는 이중 도메인 학습 프레임워크**를 MAR 분야에 최초로 확립하여, 이후 수많은 후속 연구의 기반이 되었다. 이 패러다임은 CT MAR뿐 아니라 MRI 아티팩트 보정, PET 재구성 등 다른 의료 영상 복원 작업으로도 확장 가능하다.

#### (2) 미분 가능한 물리 기반 레이어의 선구적 도입
RIL의 개념은 **물리적 이미지 형성 과정을 딥러닝 파이프라인에 통합**하는 "physics-informed deep learning"의 중요한 사례로, CT 재구성, sparse-view CT, limited-angle CT 등 다양한 영상 복원 문제에 영감을 제공하였다.

#### (3) 의료 AI의 실용성 향상
MAR의 성능 향상은 금속 임플란트를 가진 환자의 **정확한 진단과 치료 계획 수립**에 직접적으로 기여하며, 방사선 치료 계획(radiation therapy planning) 등 임상 응용에서의 중요성이 크다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|----------|----------|
| **실제 임상 데이터 검증** | 시뮬레이션 기반 평가의 한계를 극복하기 위해, 실제 금속 임플란트가 있는 환자 CT와 임플란트 제거 후 CT 쌍을 활용한 검증 필요 |
| **3D 볼류메트릭 처리** | 2D 슬라이스 단위 처리의 한계를 극복하기 위한 3D 또는 2.5D 접근 |
| **다중 금속 처리** | 여러 종류/재질의 금속이 동시에 존재하는 복잡한 시나리오 |
| **적대적 학습(GAN) 통합** | 세밀한 텍스처 복원을 위한 적대적 손실 함수 도입 |
| **비지도/자기지도 학습** | 쌍을 이루는 데이터 없이도 학습 가능한 프레임워크 |
| **임상 평가 지표** | PSNR/SSIM 외에 임상적으로 의미 있는 평가 지표(예: 진단 정확도, 방사선 치료 계획 정확도) 사용 |
| **계산 효율성** | 실시간 응용을 위한 추론 속도 최적화 |
| **다양한 CT 시스템 호환성** | 다양한 제조사/모델의 CT 시스템에 대한 범용성 확보 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

DuDoNet 이후, 이중 도메인 학습 프레임워크를 발전시킨 주요 후속 연구들을 비교 분석한다.

### 5.1 주요 후속 연구

#### (1) DuDoNet++ (Liao et al., 2020)
- **출처**: Liao, H., Lin, W.-A., et al. "ADN: Artifact Disentanglement Network for Data-Driven Metal Artifact Reduction." *IEEE Transactions on Medical Imaging*, 39(3), 634–643, 2020.
- DuDoNet의 저자 그룹이 제안한 후속 연구로, **아티팩트 분리(disentanglement)** 전략을 도입하여 비지도(unpaired) 학습이 가능하도록 확장
- 실제 임상 데이터에 대한 적용 가능성 향상

#### (2) InDuDoNet (Wang et al., 2021)
- **출처**: Wang, H., Li, Y., et al. "InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction." *MICCAI 2021*, Lecture Notes in Computer Science, vol 12906, pp. 107–118, 2021.
- DuDoNet의 이중 도메인 프레임워크를 **해석 가능한(interpretable)** 반복 최적화 알고리즘의 언롤링(unrolling)으로 재구성
- 반복적(iterative) 이중 도메인 정제를 통해 성능 향상
- 모델의 각 단계가 물리적으로 해석 가능

#### (3) InDuDoNet+ (Wang et al., 2023)
- **출처**: Wang, H., Li, Y., et al. "InDuDoNet+: A Deep Unfolding Dual Domain Network for Metal Artifact Reduction in CT Images." *Medical Image Analysis*, 85, 102729, 2023.
- InDuDoNet의 확장판으로, **더 깊은 언폴딩(deep unfolding)** 구조와 향상된 시노그램 정제 모듈을 도입
- 임상 데이터에 대한 일반화 성능 개선에 중점

#### (4) DuDoTrans (Wang et al., 2022)
- **출처**: Wang, C., et al. "DuDoTrans: Dual-Domain Transformer Provides More Attention for Sinogram Restoration in Sparse-View CT Reconstruction." *arXiv preprint arXiv:2111.10790*, 2022.
- DuDoNet의 이중 도메인 개념을 **Transformer 아키텍처**로 확장
- Sparse-view CT 재구성에 적용
- 장거리 의존성(long-range dependency) 모델링 능력 향상

#### (5) ACDNet (Adaptive Convolutional Dictionary Network)
- **출처**: Wang, H., et al. "Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction." *IJCAI 2022*.
- 적응적 합성곱 사전(adaptive convolutional dictionary)을 이중 도메인 프레임워크에 통합
- 다양한 금속 아티팩트 패턴에 대한 적응력 향상

#### (6) Score-MAR (Song et al., 2023)
- **출처**: Song, T.A., et al. "Solving Inverse Problems with Score-Based Generative Priors learned from Noisy Data." *arXiv*, 2023; 및 관련 Score-based diffusion 모델을 MAR에 적용한 연구들.
- **확산 모델(Diffusion Model)** 기반의 MAR 접근
- 비지도 학습이 가능하며, 물리적 일관성을 유지하면서 고품질 복원 달성
- DuDoNet의 지도 학습 한계를 극복하는 대안적 접근

#### (7) DuDoDp-MAR (Dual-Domain Diffusion Prior, 2023–2024)
- 이중 도메인에서 확산 모델을 사전 분포(prior)로 사용하는 연구들이 등장
- DuDoNet의 이중 도메인 철학과 생성 모델의 강력한 사전 분포 모델링 능력을 결합

### 5.2 비교 분석 표

| 특성 | DuDoNet (2019) | InDuDoNet (2021) | InDuDoNet+ (2023) | Diffusion-based MAR (2023+) |
|------|:-:|:-:|:-:|:-:|
| **이중 도메인** | ✅ | ✅ | ✅ | ✅ (일부) |
| **End-to-end** | ✅ | ✅ | ✅ | ✅ |
| **해석 가능성** | ❌ | ✅ (Unrolling) | ✅ (Deep unfolding) | △ |
| **반복적 정제** | ❌ (1회) | ✅ (다단계) | ✅ (다단계) | ✅ (반복 샘플링) |
| **비지도 학습 가능** | ❌ | ❌ | △ | ✅ |
| **물리적 일관성 보장** | ✅ (RIL + RC Loss) | ✅ | ✅ | ✅ (Data consistency) |
| **Transformer 사용** | ❌ | ❌ | ❌ | △ |
| **일반화 성능** | 보통 | 향상 | 더 향상 | 높음 (비지도) |
| **계산 효율성** | 높음 | 보통 | 보통 | 낮음 (느림) |

### 5.3 발전 추세 요약

DuDoNet 이후 MAR 분야의 주요 발전 방향:

1. **해석 가능한 딥 언폴딩**: 최적화 알고리즘의 반복을 네트워크 단계로 펼치는 접근이 주류화
2. **Transformer 아키텍처 도입**: 장거리 의존성 모델링으로 비국소적 아티팩트 처리 능력 향상
3. **생성 모델(Diffusion, Score-based) 활용**: 비지도 학습 가능성과 강력한 사전 분포 모델링
4. **임상 데이터 직접 활용**: 시뮬레이션 의존도를 줄이고 실제 데이터에서의 학습/평가 강화
5. **3D 처리**: 슬라이스 간 일관성을 고려한 3D MAR 연구 증가

---

## 참고 자료 및 출처

1. **원 논문**: Lin, W.-A., Liao, H., et al. "DuDoNet: Dual Domain Network for CT Metal Artifact Reduction." *CVPR 2019*, pp. 10512–10521.
2. **Liao, H., Lin, W.-A., et al.** "ADN: Artifact Disentanglement Network for Data-Driven Metal Artifact Reduction." *IEEE Transactions on Medical Imaging*, 39(3), 634–643, 2020.
3. **Wang, H., Li, Y., et al.** "InDuDoNet: An Interpretable Dual Domain Network for CT Metal Artifact Reduction." *MICCAI 2021*, LNCS vol 12906, pp. 107–118.
4. **Wang, H., Li, Y., et al.** "InDuDoNet+: A Deep Unfolding Dual Domain Network for Metal Artifact Reduction in CT Images." *Medical Image Analysis*, 85, 102729, 2023.
5. **Wang, C., et al.** "DuDoTrans: Dual-Domain Transformer Provides More Attention for Sinogram Restoration in Sparse-View CT Reconstruction." *arXiv preprint arXiv:2111.10790*, 2022.
6. **Zhang, Y. and Yu, H.** "Convolutional Neural Network based Metal Artifact Reduction in X-ray Computed Tomography." *IEEE Transactions on Medical Imaging*, 2018. (논문 내 참고문헌 [33])
7. **Meyer, E., et al.** "Normalized Metal Artifact Reduction (NMAR) in Computed Tomography." *Medical Physics*, 37(10), 5482–5493, 2010. (논문 내 참고문헌 [18])
8. **Kak, A.C. and Slaney, M.** *Principles of Computerized Tomographic Imaging.* SIAM, 2001. (논문 내 참고문헌 [11])
9. **Kalender, W.A., et al.** "Reduction of CT Artifacts Caused by Metallic Implants." *Radiology*, 164(2), 576–577, 1987. (논문 내 참고문헌 [12])
10. **Wang, H., et al.** "Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction." *IJCAI 2022*.

> **주의**: 2020년 이후 후속 연구의 정확한 성능 수치 비교는 각 연구에서 사용하는 데이터셋과 실험 설정이 다를 수 있으므로, 직접적인 수치 비교보다는 방법론적 발전 방향에 초점을 맞추어 분석하였습니다. 구체적 수치 비교가 필요한 경우 각 논문의 원문을 직접 참조하시기 바랍니다.
