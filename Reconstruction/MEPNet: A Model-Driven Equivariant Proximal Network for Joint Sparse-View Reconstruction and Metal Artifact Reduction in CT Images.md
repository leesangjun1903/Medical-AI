# MEPNet: A Model-Driven Equivariant Proximal Network for Joint Sparse-View Reconstruction and Metal Artifact Reduction in CT Images

## 종합 분석 보고서

---

## 1. 핵심 주장 및 주요 기여 요약

MEPNet은 **Sparse-View CT 재구성(Sparse-View Reconstruction)**과 **금속 아티팩트 감소(Metal Artifact Reduction, MAR)**를 동시에 수행하는 **모델 기반(model-driven) 등변(equivariant) 근위 네트워크(proximal network)**이다.

### 핵심 주장
1. 기존 방법들은 물리적 CT 영상 기하학 제약(physical imaging geometry constraint)을 듀얼 도메인 학습에 충분히 반영하지 못하며, CT 스캐닝에 내재된 중요한 사전 지식(prior knowledge)을 깊이 탐구하지 않았다.
2. **회전 등변성(rotation equivariance)**은 CT 스캐닝의 본질적 특성(같은 장기가 다른 각도에서 촬영됨)을 반영하는 핵심 사전 정보이며, 이를 네트워크에 인코딩하면 **더 적은 파라미터로 더 높은 재구성 성능과 일반화 능력**을 달성할 수 있다.

### 주요 기여
- **듀얼 도메인 재구성 모델**: SVMAR 과제에 특화된 최적화 모델을 제안하고 근위 경사법(proximal gradient technique)으로 풀이
- **모델 기반 언롤링 네트워크**: 최적화 알고리즘을 직접 전개(unrolling)하여 명확한 작동 메커니즘을 가진 네트워크 구축
- **회전 등변 CNN 기반 근위 연산자**: 표준 CNN 대신 $p8$ 그룹 등변 CNN을 사용하여 CT 스캐닝 고유의 회전 대칭성을 인코딩
- **일반화 성능 향상**: 파라미터 수 감소(5,095,703 → 4,723,309)와 동시에 cross-domain 테스트에서 우수한 성능 달성

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

Sparse-view CT는 방사선량 감소와 데이터 획득 속도 향상을 위해 채택되지만, 불충분한 프로젝션 데이터로 인해 재구성 CT 영상에 심각한 아티팩트가 발생한다. 환자가 **금속 임플란트**(고관절 보형물, 척추 임플란트 등)를 가진 경우, **빔 경화(beam hardening)**와 **광자 결핍(photon starvation)** 때문에 아티팩트가 더욱 악화된다.

**SVMAR(Sparse-View Metal Artifact Reduction)** 과제의 기존 한계:
1. 대부분의 방법이 물리적 영상 기하학 제약을 듀얼 도메인 학습에 완전히 반영하지 못함
2. CT 스캐닝 절차에 내재된 중요한 사전 정보(예: 회전 대칭성)가 충분히 활용되지 않음

### 2.2 제안하는 방법 (수식 포함)

#### (A) 듀얼 도메인 재구성 모델

sparse-view metal-affected sinogram $Y_{svma} \in \mathbb{R}^{N_b \times N_p}$가 주어졌을 때, 깨끗한 CT 영상 $X \in \mathbb{R}^{H \times W}$를 복원하기 위한 기본 최적화 모델:

$$\min_{X} \|(1 - Tr \cup D) \odot (\mathcal{P}X - Y_{svma})\|_F^2 + \mu R(X) $$

여기서:
- $D \in \mathbb{R}^{N_b \times N_p}$: 이진 스파스 다운샘플링 행렬 (누락 영역 = 1)
- $Tr \in \mathbb{R}^{N_b \times N_p}$: 이진 금속 트레이스 (금속 영향 영역 = 1)
- $\mathcal{P}$: 순방향 프로젝션 연산자
- $R(\cdot)$: 정규화 함수
- $\odot$: 요소별 곱셈

시노그램과 CT 영상을 **동시 재구성**하기 위해 듀얼 정규화 항 $R_1(\cdot)$, $R_2(\cdot)$를 도입하고, 시노그램 $S$를 $\bar{Y} \odot \bar{S}$로 재작성하여 안정적 학습을 도모:

$$\min_{\bar{S}, X} \|\mathcal{P}X - \bar{Y} \odot \bar{S}\|_F^2 + \lambda \|(1 - Tr \cup D) \odot (\bar{Y} \odot \bar{S} - Y_{svma})\|_F^2 + \mu_1 R_1(\bar{S}) + \mu_2 R_2(X) $$

여기서 $\bar{Y}$는 사전 영상(prior image)의 순방향 프로젝션으로 구현된 정규화 계수, $\bar{S}$는 정규화된 시노그램이다.

#### (B) 반복 최적화 알고리즘

근위 경사법(proximal gradient technique)을 사용하여 $\bar{S}$와 $X$를 교대로 갱신:

$$\bar{S}_k = \text{prox}_{\mu_1 \eta_1}\left(\bar{S}_{k-1} - \eta_1 \left(\bar{Y} \odot (\bar{Y} \odot \bar{S}_{k-1} - \mathcal{P}X_{k-1}) + \lambda(1 - Tr \cup D) \odot \bar{Y} \odot (\bar{Y} \odot \bar{S}_{k-1} - Y_{svma})\right)\right)$$

$$X_k = \text{prox}_{\mu_2 \eta_2}\left(X_{k-1} - \eta_2 \mathcal{P}^T(\mathcal{P}X_{k-1} - \bar{Y} \odot \bar{S}_k)\right) $$

각 변수의 반복 규칙은 두 단계로 구성:
1. **명시적 경사 단계(explicit gradient step)**: 데이터 일관성 보장
2. **암시적 근위 계산(implicit proximal computation)**: 사전 $R_i(\cdot)$ 적용

#### (C) 언롤링 네트워크

반복 규칙 (5)를 $K$번 전개하여 신경망 구축. 반복 $k$에서:

$$\bar{S}_k = \text{proxNet}_{\theta_{\bar{s}}^{(k)}}\left(\bar{S}_{k-1} - \eta_1(\cdots)\right)$$

$$X_k = \text{proxNet}_{\theta_x^{(k)}}\left(X_{k-1} - \eta_2 \mathcal{P}^T(\mathcal{P}X_{k-1} - \bar{Y} \odot \bar{S}_k)\right) $$

#### (D) 회전 등변 근위 연산자

CT 스캐닝에서 같은 장기가 다른 각도로 촬영되므로, 재구성 과제는 회전에 대해 등변(equivariant)이다. 등변성의 수학적 정의:

$$\Phi(T_g(f)) = T_g'(\Phi(f)), \quad \forall g \in G $$

회전 그룹 $G$에 대해 등변인 컨볼루션 필터 $\psi$를 찾기 위해:

$$[T_\theta[f]] \star \psi = T_\theta[f \star \psi] = f \star \pi_\theta[\psi], \quad \forall \theta \in G $$

**푸리에 급수 전개 기반 필터 파라미터화**:

$$\psi(x) = \sum_{m=0}^{p-1} \sum_{n=0}^{p-1} a_{mn}\varphi_{mn}^c(x) + b_{mn}\varphi_{mn}^s(x) $$

여기서:
- $x = [x_i, x_j]^T$: 2D 공간 좌표
- $a_{mn}$, $b_{mn}$: 학습 가능한 전개 계수
- $\varphi_{mn}^c(x)$, $\varphi_{mn}^s(x)$: 2D 고정 기저 함수
- $p = 5$ (실험에서)

회전 연산자 $\pi_\theta$는 좌표 변환으로 구현:

```math
\pi_\theta[\psi](x) = \psi(U_\theta^{-1} x), \quad \text{where } U_\theta = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}, \quad \forall \theta \in G
```

$p8$ 그룹을 사용하여, **8개의 서로 다른 회전 방향에 대한 필터가 하나의 전개 계수 집합을 공유**하므로 네트워크 파라미터를 크게 줄인다.

### 2.3 모델 구조

MEPNet의 전체 구조 (Fig. 2 참조):

1. **Prior-net**: 사전 영상으로부터 정규화 계수 $\bar{Y}$를 학습 (InDuDoNet [24]의 설계 활용)
2. **$K$개 스테이지의 반복 구조** (실험에서 $K=10$):
   - **시노그램 도메인**: $\text{proxNet}\_{\theta_{\bar{s}}^{(k)}}$ — 표준 CNN 기반 (4개의 [Conv+BN+ReLU+Conv+BN+Skip Connection] 잔차 블록)
   - **이미지 도메인**: $\text{proxNet}_{\theta_x^{(k)}}$ — **회전 등변 CNN 기반** ($p8$ 그룹)
3. **엔드투엔드 학습**: 전개 계수 $\{\theta\_{\bar{s}}^{(k)}\}\_{k=1}^K$ , $\theta_{prior}$ , $\eta_1$, $\eta_2$, $\lambda$ 모두 학습 데이터로부터 유연하게 학습

### 2.4 성능 향상

#### 정량적 결과 (Table 1)

| Method | DeepLesion-test (×4) | Pancreas-test (×4) | CLINIC-test (×4) |
|--------|-------|-------|-------|
| Input | 13.63 / 0.3953 | 13.29 / 0.3978 | 14.99 / 0.4604 |
| FBPConvNet | 27.38 / 0.8851 | 25.44 / 0.8731 | 29.62 / 0.8766 |
| DuDoNet | 36.83 / 0.9634 | 35.14 / 0.9527 | 37.34 / 0.9493 |
| InDuDoNet | 40.24 / 0.9793 | 38.17 / 0.9734 | 39.67 / 0.9621 |
| **MEPNet** | **41.43 / 0.9889** | **40.69 / 0.9872** | **41.58 / 0.9820** |

#### 주요 성능 개선점:
- InDuDoNet 대비 PSNR **1~2.5 dB 향상** (×4 기준)
- 파라미터 수 **7.3% 감소** (5,095,703 → 4,723,309)
- 큰 금속에서 특히 큰 성능 향상 (Large Metal: 33.78 → 37.51 dB)
- 회전 구조 왜곡 완화 효과 확인

### 2.5 한계

1. **시뮬레이션-임상 간 도메인 갭**: 학습 데이터가 공통 프로토콜로 합성되어, 실제 임상 시나리오와 차이가 존재
2. **실제 임상 데이터 부재**: sparse-view metal-inserted 스캐닝 구성에서 획득된 실제 임상 데이터로의 검증이 이루어지지 않음
3. **$p8$ 그룹으로 제한된 이산화**: 연속적 회전 등변성이 아닌 8개의 이산 회전 방향만 고려

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 회전 등변성을 통한 일반화 메커니즘

MEPNet의 일반화 성능 향상은 **그룹 등변 CNN의 이론적 기반**에 근거한다. Cohen & Welling (2016) [3]이 보인 핵심 원리는 다음과 같다:

> 네트워크에 더 많은 대칭성(symmetry)을 인코딩할수록, **가중치 공유(weight sharing)**가 증가하여 파라미터가 줄어들고, 이는 **데이터 효율성(data efficiency)**과 **일반화 능력(generalization capability)**의 향상으로 이어진다.

구체적으로:
- **표준 CNN**: 평행 이동 등변성만 보유 → 한정된 가중치 공유
- **MEPNet의 회전 등변 CNN**: 평행 이동 + 회전 등변성 보유 → $p8$ 그룹의 8개 회전 방향에 대해 동일한 전개 계수를 공유

$$\text{파라미터 감소: } 5,095,703 \xrightarrow{\text{rotation equivariance}} 4,723,309 \quad (\approx 7.3\% \downarrow)$$

### 3.2 Cross-domain 실험 결과

DeepLesion으로 훈련하고 **다른 신체 부위 데이터셋**에서 테스트:

| Dataset | InDuDoNet (PSNR/SSIM) | MEPNet (PSNR/SSIM) | 향상폭 |
|---------|---------|---------|------|
| Pancreas-test (×4) | 38.17 / 0.9734 | **40.69 / 0.9872** | +2.52 dB |
| CLINIC-test (×4) | 39.67 / 0.9621 | **41.58 / 0.9820** | +1.91 dB |

이 결과는 회전 등변성이 **도메인 간 이전(transfer)** 시에도 강건한 사전 정보로 작용함을 입증한다. 특히:
- Pancreas-test에서 +2.52 dB의 대폭 향상은 **복부 장기의 회전 대칭 구조**가 잘 활용됨을 시사
- 뼈 구조의 회전 구조 왜곡이 완화됨 (Fig. 5, green box)

### 3.3 일반화 성능 향상의 이론적 근거

CT 스캐닝의 물리적 특성에 기반한 일반화 향상 논리:

1. **물리적 근거**: CT 스캐닝에서 X선 소스가 환자 주위를 회전하므로, 같은 장기가 다양한 각도에서 촬영됨. 이는 재구성 과제가 본질적으로 **회전에 대해 등변**임을 의미
2. **정규화 효과**: 회전 등변성은 네트워크의 솔루션 공간을 물리적으로 의미있는 방향으로 제약하여, **과적합(overfitting) 감소**
3. **표현 효율성**: 동일한 표현 용량을 더 적은 파라미터로 달성 → **bias-variance tradeoff**에서 유리

### 3.4 향후 일반화 성능 추가 향상 가능성

1. **연속 회전 등변성**: 현재 $p8$ (8방향)에서 더 세밀한 이산화 또는 연속 그룹( $SO(2)$ )으로 확장 가능
2. **스케일 등변성 추가**: Gunel et al. (2022) [7]의 scale-equivariant 접근과 결합하여 다중 해상도 일반화
3. **3D 등변성**: 실제 CT는 3D 데이터이므로 $SE(3)$ 그룹 등변성으로 확장 가능
4. **자기 지도 학습과의 결합**: Equivariant Imaging (Chen et al., 2021) [2]의 아이디어와 결합하여 레이블 없는 데이터에서의 일반화 가능

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구 영향

#### (1) CT 재구성 분야에서의 등변 네트워크 패러다임 제시
MEPNet은 **SVMAR 과제에서 회전 등변성을 최초로 연구**하였으며, 이는 의료 영상 재구성 분야에서 물리적 대칭성을 체계적으로 활용하는 새로운 방향을 열었다. 이후 연구에서 다양한 대칭 그룹(회전, 스케일, 반사 등)을 CT 재구성에 적용하는 연구가 확산될 것으로 예상된다.

#### (2) 모델 기반(model-driven) 딥러닝의 확장
최적화 모델에서 출발하여 네트워크를 구성하는 **algorithm unrolling** 접근의 가치를 재확인하였다. 특히 근위 연산자를 등변 CNN으로 대체하는 아이디어는 **ADMM-Net**, **KXNet** 등 다른 모델 기반 네트워크에도 적용 가능하다.

#### (3) 파라미터 효율성과 성능의 동시 달성
더 적은 파라미터로 더 높은 성능을 달성한 결과는, **임상 배포(clinical deployment)** 관점에서 중요한 의미를 가진다. 제한된 연산 자원에서의 실시간 CT 재구성 가능성을 높인다.

### 4.2 향후 연구 시 고려할 점

#### (1) 실제 임상 데이터 검증
- 논문의 가장 큰 한계인 **시뮬레이션-임상 도메인 갭** 해결이 필수적
- Sparse-view metal-inserted 스캐닝 구성의 실제 임상 데이터 수집 및 검증 필요
- **도메인 적응(domain adaptation)** 기법과의 결합 고려

#### (2) 확장된 등변 그룹의 탐색
- $p8$ → $p16$ 또는 연속 그룹 $SO(2)$로의 확장 시 성능-효율 트레이드오프 분석
- 3D CT 볼륨에 대한 $SE(3)$ 등변성 적용
- 스케일 등변성과 회전 등변성의 동시 인코딩

#### (3) 다양한 CT 재구성 시나리오 확장
- **Limited-angle CT** 재구성
- **Low-dose CT** 노이즈 제거
- **Cone-beam CT** 재구성
- 논문에서도 이러한 확장 가능성을 언급

#### (4) 비지도/자기 지도 학습과의 결합
- 쌍 데이터(paired data) 의존성 감소를 위한 **unsupervised MAR** (예: ADN [11]) 접근과의 통합
- Equivariant Imaging [2]의 range space 이론과 결합

#### (5) 계산 효율성 최적화
- 등변 컨볼루션의 실시간 추론 속도 개선
- 모바일/엣지 기기 배포를 위한 경량화

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 대상 연구 개요

| 연구 | 연도 | 접근법 | 도메인 | 핵심 특징 |
|------|------|--------|--------|----------|
| DuDoDR-Net [36] | 2022 | 듀얼 도메인 순환 네트워크 | 듀얼 | GRU 기반, 동시 SV+MAR |
| InDuDoNet [24] | 2021 | 해석 가능한 듀얼 도메인 네트워크 | 듀얼 | Algorithm unrolling, 해석 가능성 |
| InDuDoNet+ [25] | 2022 | InDuDoNet 확장 | 듀얼 | 깊은 언폴딩, 향상된 Prior |
| DICDNet [22] | 2021 | 해석 가능한 컨볼루션 사전 네트워크 | 이미지 | 컨볼루션 사전 학습 |
| DuDoTrans [21] | 2021 | 듀얼 도메인 트랜스포머 | 듀얼 | Transformer 기반 시노그램 복원 |
| NeRP [18] | 2022 | 암시적 신경 표현 | 이미지 | Prior embedding |
| Scale-Equivariant [7] | 2022 | 스케일 등변 언롤링 | MRI | 스케일 등변성 (MRI 적용) |
| Equivariant Imaging [2] | 2021 | 등변 영상 | 범용 | Range space 이론 |
| **MEPNet** | **2023** | **등변 근위 네트워크** | **듀얼** | **회전 등변성 + Algorithm unrolling** |

### 5.2 상세 비교

#### (A) DuDoDR-Net (Zhou et al., Medical Image Analysis, 2022) vs. MEPNet

DuDoDR-Net은 SVMAR 과제를 동시에 다루는 최초의 방법 중 하나로, U-Net과 GRU를 사용한 듀얼 도메인 순환 네트워크를 제안하였다. 그러나:
- **DuDoDR-Net**: 경험적으로 설계된 off-the-shelf 모듈 사용, 물리적 제약 불충분
- **MEPNet**: 최적화 모델에서 직접 도출된 네트워크, 물리적 기하학 제약 내장, 회전 등변성 활용

#### (B) InDuDoNet / InDuDoNet+ (Wang et al., MICCAI 2021 / MedIA 2022) vs. MEPNet

MEPNet은 InDuDoNet을 직접적으로 확장한 연구이다:
- **InDuDoNet**: 표준 CNN 기반 근위 연산자 → 평행 이동 등변성만 보유
- **MEPNet**: 회전 등변 CNN 기반 근위 연산자 → **회전 + 평행 이동 등변성** 보유
- InDuDoNet는 MEPNet에서 **회전 등변성을 제거한 ablation 모델**로 간주 가능

정량적 비교 (×4, DeepLesion-test):
$$\text{InDuDoNet: } 40.24 \text{ dB} \xrightarrow{+1.19 \text{ dB}} \text{MEPNet: } 41.43 \text{ dB}$$

#### (C) Scale-Equivariant Unrolled Networks (Gunel et al., MICCAI 2022) vs. MEPNet

이 연구는 **MRI 재구성**에서 스케일 등변성을 언롤링 네트워크에 적용하였다:
- **공통점**: 등변성을 algorithm unrolling 프레임워크에 통합, 데이터 효율성 및 일반화 향상
- **차이점**: Gunel et al.은 스케일 등변성 + MRI, MEPNet은 회전 등변성 + CT
- **시사점**: 두 접근을 결합하여 **다중 그룹 등변 언롤링 네트워크** 구축 가능성

#### (D) Equivariant Imaging (Chen et al., ICCV 2021) vs. MEPNet

Equivariant Imaging은 역문제에서 **등변성을 비지도 학습에 활용**하는 이론적 프레임워크를 제시하였다:
- **공통점**: CT 재구성에서의 등변성 활용
- **차이점**: Chen et al.은 등변성을 **학습 프레임워크**(비지도)에 활용, MEPNet은 **네트워크 구조**(지도 학습)에 활용
- **시사점**: 두 접근의 결합으로 **비지도 SVMAR** 가능성

#### (E) Equivariant Neural Networks for Inverse Problems (Celledoni et al., 2021) vs. MEPNet

이 연구는 역문제에서의 등변 신경망에 대한 이론적 분석을 제공하였다:
- **공통점**: 역문제에서의 등변성 활용
- **차이점**: Celledoni et al.은 일반적 역문제 프레임워크, MEPNet은 SVMAR에 특화

### 5.3 비교 요약표

| 비교 항목 | DuDoDR-Net | InDuDoNet | DuDoTrans | MEPNet |
|----------|-----------|-----------|-----------|--------|
| 모델 기반 여부 | ✗ (경험적) | ✓ | ✗ | ✓ |
| 듀얼 도메인 | ✓ | ✓ | ✓ | ✓ |
| 등변성 활용 | ✗ | ✗ | ✗ | ✓ (회전) |
| 물리적 제약 내장 | 부분적 | ✓ | 부분적 | ✓ |
| 해석 가능성 | 낮음 | 높음 | 중간 | 높음 |
| 파라미터 효율성 | 중간 | 중간 | 높음 | **높음** |
| 일반화 능력 | 중간 | 중간 | 중간 | **높음** |

---

## 참고자료

1. **Wang, H., Zhou, M., Wei, D., Li, Y., & Zheng, Y.** "MEPNet: A Model-Driven Equivariant Proximal Network for Joint Sparse-View Reconstruction and Metal Artifact Reduction in CT Images." *MICCAI 2023* (본 논문)
2. **Cohen, T., & Welling, M.** "Group equivariant convolutional networks." *ICML 2016*, pp. 2990–2999.
3. **Chen, D., Tachella, J., & Davies, M.E.** "Equivariant Imaging: Learning beyond the range space." *ICCV 2021*, pp. 4379–4388.
4. **Wang, H., Li, Y., Zhang, H., et al.** "InDuDoNet: An interpretable dual domain network for CT metal artifact reduction." *MICCAI 2021*, pp. 107–118.
5. **Wang, H., Li, Y., Zhang, H., Meng, D., & Zheng, Y.** "InDuDoNet+: A deep unfolding dual domain network for metal artifact reduction in CT images." *Medical Image Analysis*, p. 102729 (2022).
6. **Zhou, B., Chen, X., Zhou, S.K., Duncan, J.S., & Liu, C.** "DuDoDR-Net: Dual-domain data consistent recurrent network for simultaneous sparse view and metal artifact reduction in computed tomography." *Medical Image Analysis*, **75**, 102289 (2022).
7. **Gunel, B., Sahiner, A., et al.** "Scale-Equivariant unrolled neural networks for data-efficient accelerated MRI reconstruction." *MICCAI 2022*, pp. 737–747.
8. **Celledoni, E., Ehrhardt, M.J., et al.** "Equivariant neural networks for inverse problems." *Inverse Problems*, **37**(8), 085006 (2021).
9. **Xie, Q., Zhao, Q., Xu, Z., & Meng, D.** "Fourier series expansion based filter parametrization for equivariant convolutions." *IEEE TPAMI* (2022).
10. **Lin, W.A., Liao, H., et al.** "DuDoNet: Dual domain network for CT metal artifact reduction." *CVPR 2019*, pp. 10512–10521.
11. **Parikh, N., & Boyd, S.** "Proximal algorithms." *Foundations and Trends in Optimization*, **1**(3), 127–239 (2014).
12. **Jin, K.H., McCann, M.T., Froustey, E., & Unser, M.** "Deep convolutional neural network for inverse problems in imaging (FBPConvNet)." *IEEE TIP*, **26**(9), 4509–4522 (2017).
13. **Wang, C., Shang, K., et al.** "DuDoTrans: Dual-domain transformer provides more attention for sinogram restoration in sparse-view CT reconstruction." *arXiv:2111.10790* (2021).
14. **Weiler, M., Hamprecht, F.A., & Storath, M.** "Learning steerable filters for rotation equivariant CNNs." *CVPR 2018*, pp. 849–858.
