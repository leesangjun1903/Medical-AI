# Learned Primal-Dual Reconstruction

---

## 1. 핵심 주장과 주요 기여 요약

**Learned Primal-Dual Reconstruction** (Adler & Öktem, 2018)은 토모그래피 영상 재구성을 위한 딥러닝 기반 프레임워크로, 고전적인 **Primal-Dual Hybrid Gradient (PDHG)** 최적화 알고리즘을 신경망으로 **unrolling**하되, proximal 연산자를 **합성곱 신경망(CNN)**으로 대체한 방법이다.

### 주요 기여:
1. **모델 기반 + 데이터 기반 접근의 통합**: 순방향 연산자(forward operator)를 신경망 내부에 명시적으로 포함시켜, 원시 측정 데이터(raw data)로부터 직접 재구성을 수행함. FBP 등의 초기 재구성에 의존하지 않음.
2. **Primal-Dual 구조의 학습**: 재구성 공간(primal)과 데이터 공간(dual) 모두에서 CNN을 운용하며, 순방향 연산자와 그 수반(adjoint)으로 연결.
3. **우수한 성능**: 저선량 CT 재구성에서 FBP, TV 정규화, 학습 기반 후처리(U-Net) 대비 PSNR 및 SSIM 모두 유의미하게 향상.
4. **임상 적용 가능한 속도**: 단 10회의 forward-backprojection 연산만으로 재구성 완료.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

역문제(inverse problem)에서 신호 $f_{\text{true}} \in X$를 간접 관측 데이터 $g \in Y$로부터 재구성하는 것이 목표이다:

$$g = \mathcal{T}(f_{\text{true}}) + \delta g $$

여기서 $\mathcal{T}: X \to Y$는 순방향 연산자, $\delta g$는 노이즈이다.

전통적 변분 정규화(variational regularization) 접근은 다음을 최소화한다:

$$\min_{f \in X} \left[ \mathcal{L}\big(\mathcal{T}(f), g\big) + \lambda \mathcal{S}(f) \right] \quad \text{for a fixed } \lambda \geq 0 $$

여기서 $\mathcal{L}$은 음의 데이터 로그-우도, $\mathcal{S}$는 정규화 함수이다. 이 최적화는 ill-posed 문제이며, 기존 방법들은 (1) 순방향 모델을 활용하지만 사전 정보(prior) 설계가 수동적이거나, (2) 딥러닝으로 후처리하지만 순방향 모델 정보를 충분히 활용하지 못하는 한계가 있다.

**핵심 질문**: 순방향 모델의 지식을 신경망 설계에 어떻게 통합할 것인가?

### 2.2 제안하는 방법

#### 2.2.1 PDHG 알고리즘에서의 출발

Chambolle-Pock 알고리즘(PDHG)은 다음 구조의 최적화 문제를 풀기 위한 것이다:

$$\min_{f \in X} \left[ \mathcal{F}\big(\mathcal{K}(f)\big) + \mathcal{G}(f) \right] $$

PDHG의 반복 구조는:

$$h_{i+1} \leftarrow \text{prox}_{\sigma \mathcal{F}^*}\big(h_i + \sigma \mathcal{K}(\bar{f}_i)\big)$$

$$f_{i+1} \leftarrow \text{prox}_{\tau \mathcal{G}}\big(f_i - \tau [\partial \mathcal{K}(f_i)]^*(h_{i+1})\big)$$

$$\bar{f}_{i+1} \leftarrow f_{i+1} + \gamma(f_{i+1} - f_i)$$

여기서 proximal 연산자는:

$$\text{prox}_{\tau \mathcal{G}}(f) = \arg\min_{f' \in X} \left[ \mathcal{G}(f') + \frac{1}{2\tau}\|f' - f\|_X^2 \right] $$

#### 2.2.2 Learned Primal-Dual 알고리즘 (Algorithm 3)

PDHG의 proximal 연산자를 학습 가능한 CNN으로 대체하고, 다음과 같은 수정을 가한다:

- **메모리 확장**: primal 공간을 $f = [f^{(1)}, f^{(2)}, \ldots, f^{(N_{\text{primal}})}] \in X^{N_{\text{primal}}}$, dual 공간을 $U^{N_{\text{dual}}}$로 확장.
- **자유로운 결합 학습**: 네트워크가 연산자 평가 결과와 이전 업데이트를 어떻게 결합할지 자유롭게 학습.
- **반복별 별도 파라미터**: 각 반복마다 다른 학습된 proximal 연산자 사용.

**Algorithm 3 (Learned Primal-Dual)**:

```math
\text{Initialize } f_0 \in X^{N_{\text{primal}}}, \quad h_0 \in U^{N_{\text{dual}}}
```

$$\text{for } i = 1, \ldots, I \text{ do:}$$

$$h_i \leftarrow \Gamma_{\theta_i^d}\big(h_{i-1},\, \mathcal{K}(f_{i-1}^{(2)}),\, g\big) $$

$$f_i \leftarrow \Lambda_{\theta_i^p}\big(f_{i-1},\, [\partial \mathcal{K}(f_{i-1}^{(1)})]^*(h_i^{(1)})\big) $$

$$\text{return } f_I^{(1)}$$

여기서 $\Gamma_{\theta_i^d}$와 $\Lambda_{\theta_i^p}$는 학습된 CNN 기반 proximal 연산자이다.

#### 2.2.3 학습된 Proximal 연산자의 구조

각 학습된 proximal 연산자는 **residual network** 형태로 설계된다:

$$\text{Id} + \mathcal{W}_{w_3, b_3} \circ \mathcal{A}_{c_2} \circ \mathcal{W}_{w_2, b_2} \circ \mathcal{A}_{c_1} \circ \mathcal{W}_{w_1, b_1}$$

여기서:
- $\mathcal{W}_{w_j, b_j}: X^n \to X^m$는 아핀 합성곱 연산자:

$$\left[\mathcal{W}_{w_j, b_j}([f^{(1)}, \ldots, f^{(n)}])\right]^{(k)} = b_j^{(k)} + \sum_{l=1}^{n} w_j^{(l,k)} * f^{(l)}$$

- $\mathcal{A}_{c_j}$는 PReLU 비선형 활성화 함수:

$$\mathcal{A}_{c_j}(x) = \begin{cases} x & \text{if } x \geq 0 \\ -c_j x & \text{else} \end{cases}$$

#### 2.2.4 손실 함수

학습은 경험적 손실(empirical loss)을 최소화하여 수행:

$$\widehat{L}(\theta) := \frac{1}{N}\sum_{i=1}^{N} \left\| \mathcal{T}_\theta^\dagger(g_i) - f_i \right\|_X^2 $$

### 2.3 모델 구조

| 구성 요소 | 세부 사항 |
|---------|---------|
| 반복 횟수 ($I$) | 10 (forward + backprojection 각 10회) |
| Primal/Dual 채널 수 | $N_{\text{primal}} = N_{\text{dual}} = 5$ |
| Primal CNN 채널 | $6 \to 32 \to 32 \to 5$ |
| Dual CNN 채널 | $7 \to 32 \to 32 \to 5$ (데이터 $g$ 추가 입력) |
| 합성곱 커널 크기 | $3 \times 3$ |
| 총 합성곱 층 수 | 60 (10 iterations × 2 networks × 3 layers) |
| 총 파라미터 수 | 약 $2.4 \times 10^5$ |
| 초기화 | Zero-initialization ($f_0 = 0, h_0 = 0$) |
| 비선형 함수 | PReLU |
| 옵티마이저 | ADAM (cosine annealing 학습률) |

초기 학습률:

$$\eta_t = \frac{\eta_0}{2}\left(1 + \cos\left(\pi \frac{t}{t_{\max}}\right)\right), \quad \eta_0 = 10^{-3}$$

### 2.4 성능 향상

#### Ellipse Phantom (Table I)

| Method | PSNR (dB) | SSIM | Runtime (ms) |
|--------|-----------|------|-------------|
| FBP | 19.75 | 0.597 | 4 |
| TV | 28.06 | 0.929 | 5,166 |
| FBP + U-Net | 29.20 | 0.944 | 9 |
| Learned Gradient | 32.29 | 0.981 | 56 |
| **Learned Primal-Dual** | **38.28** | **0.989** | 49 |

- 모든 비교 방법 대비 **>6 dB PSNR 향상**

#### Human Phantom (Table II)

| Method | PSNR (dB) | SSIM | Runtime (ms) |
|--------|-----------|------|-------------|
| FBP | 33.65 | 0.830 | 423 |
| TV | 37.48 | 0.946 | 64,371 |
| FBP + U-Net | 41.92 | 0.941 | 463 |
| **Learned Primal-Dual (linear)** | **44.11** | **0.969** | 620 |

- TV 대비 **6.6 dB**, U-Net 후처리 대비 **2.2 dB** 향상
- SSIM에서도 유의미한 개선 (0.946 → 0.969)
- TV 대비 **100배 이상 빠른 속도**

### 2.5 한계

1. **과도한 평활화(Over-smoothing)**: MSE 손실함수($L^2$ norm)의 특성으로 인해 재구성 영상이 시각적으로 과도하게 매끄러워 보임. 저자들은 perceptual loss 등 대안적 손실함수를 제안함.
2. **비선형 순방향 모델의 이점 미미**: 비선형(pre-log) vs 선형(post-log) 순방향 모델 간 성능 차이가 크지 않음.
3. **네트워크 구조의 미세 조정 부족**: 더 나은 proximal 네트워크 구조가 성능을 향상시킬 가능성.
4. **일반화 성능에 대한 명시적 검증 부족**: 학습 데이터 분포와 크게 다른 새로운 데이터에 대한 체계적 일반화 실험이 부재.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 논문에서의 일반화 관련 논의

논문에서는 일반화를 다음과 같은 맥락에서 다루고 있다:

- **경험적 손실 vs 실제 손실**: 식 (4)의 기대값 기반 손실함수와 식 (5)의 경험적 손실 간의 관계를 명시하며, "경험적 손실을 최소화하면서 동시에 새로운 데이터에 일반화하기를 원한다"고 언급함.
  
$$L(\theta) := \mathbb{E}_{(f,g)\sim\mu}\left[\left\|\mathcal{T}_\theta^\dagger(g) - f\right\|_X^2\right] $$

- **순방향 모델의 내재**: 순수 데이터 기반 방법과 달리 물리적 순방향 모델 $\mathcal{T}$를 네트워크 내부에 포함시킴으로써, 네트워크가 물리적으로 일관된(consistent) 재구성을 학습하도록 유도. 이는 일반화를 돕는 **강력한 귀납적 편향(inductive bias)**으로 작용함.

- **적은 파라미터 수**: 약 $2.4 \times 10^5$개의 파라미터만 사용하면서도 60개의 합성곱 층을 가지는 깊은 네트워크. U-Net의 $10^7$개 파라미터와 비교하면 약 40배 적은 파라미터로 더 나은 성능을 달성하여, 과적합 위험이 상대적으로 낮음.

- **PDHG와의 이론적 연결**: Learned Primal-Dual는 PDHG, 경사하강법, L-BFGS 등 다양한 최적화 알고리즘을 **특수 경우로 포함**하며, 신경망의 **보편 근사 정리(universal approximation theorem)**에 의해 이 관계가 보장됨.

### 3.2 일반화 향상을 위한 잠재적 방향

1. **Data Augmentation**: 논문에서는 데이터 증강을 사용하지 않았다고 명시. 회전, 이동, 노이즈 변형 등의 증강이 일반화를 개선할 수 있음.

2. **정규화 기법 부재**: Dropout, batch normalization, weight decay 등을 사용하지 않았으며, 이를 적용하면 일반화 성능이 향상될 가능성이 있음.

3. **다양한 손실함수**: MSE 대신 perceptual loss, adversarial loss 등을 사용하면 시각적 품질과 일반화 모두 개선될 수 있음.

4. **물리적 일관성의 강화**: 순방향 연산자의 명시적 포함이 이미 일반화에 유리하지만, **data consistency layer**를 추가하면 물리적 일관성을 더욱 강화할 수 있음.

5. **Cross-domain 전이**: 논문에서는 ellipse phantom으로 학습 후 Shepp-Logan phantom에서 평가하여 어느 정도의 일반화를 보였으나, 다양한 해부학적 구조와 스캔 프로토콜에 대한 체계적인 일반화 연구가 필요.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

1. **Algorithm Unrolling 패러다임의 확립**: 이 논문은 고전적 최적화 알고리즘을 unrolling하여 딥러닝과 결합하는 방법론의 선구적 사례 중 하나로, 이후 수많은 unrolled optimization 연구의 기반이 됨.

2. **Physics-Informed Deep Learning의 촉진**: 순방향 모델을 네트워크 아키텍처에 직접 포함시키는 접근은 이후 Physics-Informed Neural Networks (PINNs) 및 model-based deep learning의 발전에 영향을 줌.

3. **End-to-End 학습의 가능성 입증**: 원시 측정 데이터에서 직접 최종 재구성까지의 전체 파이프라인을 end-to-end로 학습 가능함을 실증.

4. **다양한 영상 모달리티로의 확장**: CT뿐 아니라 MRI, PET, 초음파 등 다른 영상 기법으로의 적용 가능성을 열어줌.

### 4.2 향후 연구 시 고려할 점

1. **수렴 보장 및 이론적 분석**: Unrolled 네트워크의 수렴 보장, 안정성(stability), 재구성의 정확도에 대한 이론적 분석이 필요.

2. **적대적 강건성(Adversarial Robustness)**: 딥러닝 기반 재구성이 적대적 perturbation이나 분포 이동(distribution shift)에 취약할 수 있으며, 이에 대한 강건성 연구가 중요.

3. **3D 및 대규모 데이터로의 확장**: 논문은 2D 슬라이스 기반이며, 3D 볼류메트릭 재구성으로의 확장 시 메모리 및 계산 비용 문제 해결이 필요.

4. **해석 가능성(Interpretability)**: 학습된 proximal 연산자가 실제로 무엇을 수행하는지에 대한 해석이 임상 적용에 중요.

5. **임상 검증**: 실제 환자 데이터에서의 진단 정확도, 병변 검출 성능 등에 대한 임상적 검증이 필수.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구

| 연구 | 연도 | 핵심 기여 | Learned Primal-Dual과의 관계 |
|------|------|---------|---------------------------|
| **End-to-End Variational Networks (E2E-VarNet)** (Sriram et al.) | 2020 | MRI를 위한 end-to-end variational network; sensitivity map 추정과 재구성을 통합 학습 | Learned Primal-Dual의 MRI 확장 버전; cascade 구조에 data consistency 강화 |
| **Deep Equilibrium Models (DEQ)** (Gilton et al.) | 2021 | 무한 깊이 unrolled 네트워크의 고정점(fixed point)을 implicit하게 학습 | 고정 반복 수 대신 수렴점을 직접 학습하여 메모리 효율성 및 일반화 향상 |
| **Learned Primal-Dual with convergence guarantees** (Mukherjee et al.) | 2021 | Learned Primal-Dual에 수렴 보장 조건을 추가 | 원래 Learned Primal-Dual의 이론적 한계를 보완 |
| **AUTOMAP & successors** (Zhu et al.) | 2018/2020+ | 완전 학습 기반 영상 재구성; fully connected + CNN | 순방향 모델을 사용하지 않는 순수 데이터 기반 접근 |
| **Plug-and-Play (PnP) ADMM / RED** (Ahmad et al., Cohen et al.) | 2020+ | 사전 학습된 denoiser를 proximal 연산자 대신 사용 | Learned Primal-Dual와 유사한 철학이지만 denoiser는 별도 학습 |
| **Model-Based Deep Learning (MoDL)** (Aggarwal et al.) | 2019/2020 | Weight sharing을 통해 파라미터 수를 줄이고 data consistency를 명시적으로 부과 | Learned Primal-Dual에서 반복별 파라미터를 공유하는 변형 |
| **Equivariant Neural Networks for Inverse Problems** (Chen et al.) | 2021+ | 대칭성(equivariance)을 활용하여 일반화 성능 향상 | 물리적 대칭성을 아키텍처에 내재하여 일반화 강화 |
| **Score-based/Diffusion Models for Inverse Problems** (Song et al., Chung et al.) | 2021-2023 | 확산 모델(diffusion model)을 사전 분포(prior)로 사용하여 역문제 해결 | Learned Primal-Dual와 달리 생성 모델을 활용; 더 나은 perceptual quality |

### 5.2 상세 비교

#### 5.2.1 E2E-VarNet (Sriram et al., 2020)

fastMRI 챌린지에서 우수한 성능을 보인 방법으로, Learned Primal-Dual와 유사하게 unrolled optimization 구조를 사용하지만, **sensitivity map 추정 모듈**을 통합하고 **data consistency layer**를 명시적으로 포함한다. 이는 물리적 일관성을 더 강하게 부과하여 일반화에 유리하다.

$$f_{i+1} = f_i - \eta_i \cdot \text{CNN}_i\Big(\mathcal{A}^H(\mathcal{A}f_i - g)\Big)$$

여기서 $\mathcal{A}$는 multi-coil forward operator이다.

> 참고: Sriram, A., et al., "End-to-End Variational Networks for Accelerated MRI Reconstruction," *MICCAI 2020*.

#### 5.2.2 Diffusion Models for Inverse Problems (Chung et al., 2022-2023)

Score-based diffusion model을 사전 분포로 사용하여 역문제를 해결하는 접근법이다. Learned Primal-Dual와 비교하면:

- **장점**: 더 풍부한 사전 분포 표현, 우수한 perceptual quality, 다양한 역문제에 대한 유연성
- **단점**: 추론 시간이 훨씬 길고(수백~수천 step), 임상 적용에는 속도 제한이 있음

> 참고: Chung, H., et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems," *ICLR 2023*.

#### 5.2.3 Deep Equilibrium Models (Gilton et al., 2021)

고정 반복 횟수 대신 **고정점(fixed point)**을 implicit하게 구하는 방법으로, 메모리 사용량이 반복 횟수에 의존하지 않는다는 장점이 있다. 일반화 측면에서 무한 깊이 네트워크의 고정점이 더 안정적인 해를 제공할 가능성이 있다.

$$f^* = \text{CNN}_\theta(f^*, g) \quad \text{(implicit fixed point)}$$

> 참고: Gilton, D., et al., "Deep Equilibrium Architectures for Inverse Problems in Imaging," *IEEE TCI, 2021*.

#### 5.2.4 수렴 보장이 있는 Learned Primal-Dual (Mukherjee et al., 2021)

원래 Learned Primal-Dual의 핵심 한계인 **이론적 수렴 보장 부재**를 해결하기 위해, 학습된 연산자에 convexity/monotonicity 조건을 부과하여 수렴을 보장하는 연구이다.

> 참고: Mukherjee, S., et al., "Learned convex regularizers for inverse problems," *arXiv:2008.02839, 2021*.

### 5.3 일반화 관점에서의 최신 동향 정리

| 접근법 | 일반화 전략 | Learned Primal-Dual 대비 장점 |
|--------|-----------|---------------------------|
| Data Consistency 강화 | 물리적 일관성을 명시적으로 부과 | 분포 이동 시에도 물리적 타당성 유지 |
| Weight Sharing (MoDL) | 파라미터 수 감소로 과적합 방지 | 더 적은 데이터로도 학습 가능 |
| Equivariant 아키텍처 | 물리적 대칭성을 아키텍처에 내재 | 데이터 효율성 및 일반화 향상 |
| Diffusion Prior | 풍부한 사전 분포 학습 | 다양한 역문제에 유연하게 적용 |
| DEQ (고정점 모델) | 무한 깊이 수렴점 학습 | 메모리 효율적, 안정적 해 |
| 수렴 보장 조건 부과 | 이론적 보장으로 신뢰성 확보 | 임상 적용 시 신뢰성 제고 |

---

## 참고자료

1. **Adler, J. and Öktem, O.**, "Learned Primal-dual Reconstruction," *IEEE Transactions on Medical Imaging*, 2018. (arXiv:1707.06474v3)
2. **Chambolle, A. and Pock, T.**, "A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging," *Journal of Mathematical Imaging and Vision*, vol. 40, no. 1, pp. 120–145, 2010.
3. **Adler, J. and Öktem, O.**, "Solving ill-posed inverse problems using iterative deep neural networks," *Inverse Problems*, vol. 33, no. 12, 2017.
4. **Sriram, A., et al.**, "End-to-End Variational Networks for Accelerated MRI Reconstruction," *MICCAI 2020*.
5. **Chung, H., Kim, J., Mccann, M.T., Klasky, M.L., and Ye, J.C.**, "Diffusion Posterior Sampling for General Noisy Inverse Problems," *ICLR 2023*.
6. **Gilton, D., Ongie, G., and Willett, R.**, "Deep Equilibrium Architectures for Inverse Problems in Imaging," *IEEE Transactions on Computational Imaging*, 2021.
7. **Aggarwal, H.K., Mani, M.P., and Jacob, M.**, "MoDL: Model-Based Deep Learning Architecture for Inverse Problems," *IEEE Transactions on Medical Imaging*, vol. 38, no. 2, 2019.
8. **Mukherjee, S., Dittmer, S., Shumaylov, Z., Lunz, S., Öktem, O., and Schönlieb, C.-B.**, "Learned convex regularizers for inverse problems," *arXiv:2008.02839*, 2021.
9. **Putzky, P. and Welling, M.**, "Recurrent inference machines for solving inverse problems," *arXiv:1706.04008*, 2017.
10. **Jin, K.H., McCann, M.T., Froustey, E., and Unser, M.**, "Deep Convolutional Neural Network for Inverse Problems in Imaging," *IEEE Transactions on Image Processing*, vol. 26, no. 9, 2016.
11. **Ronneberger, O., Fischer, P., and Brox, T.**, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI 2015*.
12. **He, K., Zhang, X., Ren, S., and Sun, J.**, "Deep Residual Learning for Image Recognition," *CVPR 2016*.
13. **Hornik, K.**, "Approximation capabilities of multilayer feedforward networks," *Neural Networks*, vol. 4, no. 2, 1991.
