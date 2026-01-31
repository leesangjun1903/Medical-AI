
# Transformer‑based hand gesture recognition from instantaneous to fused neural decomposition of high‑density EMG signals

## 1. 논문의 핵심 주장 및 기여

### 1.1 주요 기여

본 논문의 핵심 기여는 **Compact Transformer-based Hand Gesture Recognition (CT-HGR)** 프레임워크로서 세 가지 획기적 성과를 달성했습니다.

**첫째, HD-sEMG 신호에 대한 최초의 Vision Transformer(ViT) 기반 아키텍처**입니다. 기존의 CNN-RNN 하이브리드 모델의 복잡성을 제거하고, 순수 Transformer 기반의 독립형 프레임워크로 직접 학습이 가능함을 보여주었습니다. 이는 전이 학습이나 데이터 증강이 필요 없으면서도 고정확도를 달성할 수 있다는 점에서 획기적입니다.

**둘째, 순간적(instantaneous) HD-sEMG 이미지만으로 89.13% 정확도를 달성**했다는 점입니다. 이는 단일 프레임의 2D sEMG 이미지로도 의미 있는 패턴 인식이 가능함을 입증하며, 실시간 HMI 시스템 개발의 기초를 마련했습니다.

**셋째, 거시적(macroscopic) 신경 정보와 미시적(microscopic) 신경 정보의 통합 프레임워크**를 최초로 제안했습니다. 원본 HD-sEMG 신호와 Blind Source Separation을 통해 추출한 Motor Unit Spike Trains(MUSTs)를 하이브리드 구조로 결합하여 94.86% 정확도를 달성했습니다.

### 1.2 해결하고자 한 문제

기존 깊은 학습 모델들의 주요 제약을 해결합니다:

1. **높은 모델 복잡성**: CNN-RNN 하이브리드 모델의 메모리 사용량(14.81GB) 대비 CT-HGR은 더 적은 파라미터(95,682-99,266개, V1 기준)로 우수한 성능 달성
2. **피처 엔지니어링 의존성**: SVM/LDA 같은 전통적 방법의 수동 피처 추출 부담 제거
3. **시공간 정보 동시 추출 불가**: 기존 CNN은 공간 특징만, RNN은 시간 특징만 추출하는 한계 극복
4. **대규모 학습 데이터 필요**: 5회 반복 실험으로도 안정적 성능 달성

***

## 2. 제안 방법 및 모델 구조

### 2.1 데이터 전처리 파이프라인

시스템 초기 단계에서 원본 HD-sEMG 신호는 다음과 같은 전처리를 거칩니다:

**µ-law 정규화** (식 1):
$$F(x_t) = \text{sign}(x_t) \ln\left(\frac{1 + \mu|x_t|}{\ln(1 + \mu)}\right)$$

여기서 $x_t$는 각 채널의 시간 영역 sEMG 신호이고, µ는 경험적으로 결정된 스케일링 매개변수입니다. 이 변환은 신호의 동적 범위의 급격한 변화를 완화하여, 신경망이 제스처를 더 효과적으로 학습하도록 합니다.

### 2.2 CT-HGR 아키텍처

#### 2.2.1 Patch Embedding
3D 신호 $W \times N_{ch} \times N_{cv}$는 $N$개의 작은 패치로 분할됩니다:

```math
\text{Linear\ Projection:\ }\mathbf{x}_{p}^{i}\mathbf{E}
```

```math
\mathbf{z}_{0}=[\mathbf{x}_{\text{class}};\mathbf{x}_{p}^{1}\mathbf{E};\mathbf{x}_{p}^{2}\mathbf{E};\dots ;\mathbf{x}_{p}^{N}\mathbf{E}]+\mathbf{E}_{pos}
```

패치 임베딩 행렬 $\mathbf{E}$는 크기 $d$의 모델 임베딩 차원으로 투영합니다. 클래스 토큰 $\mathbf{x}_p^0$이 추가되어 최종 수열 길이는 $N+1$입니다.

#### 2.2.2 Position Embedding
1D 학습 가능한 위치 임베딩 벡터를 추가:

```math
\mathbf{z}_{0}=[\mathbf{x}_{p}^{0};\mathbf{x}_{p}^{1}\mathbf{E};\mathbf{x}_{p}^{2}\mathbf{E};\dots ;\mathbf{x}_{p}^{N}\mathbf{E}]+\mathbf{E}_{pos}
```

여기서 $\mathbf{E}_{pos}$는 $(N+1) \times d$ 행렬로서 $d$차원 벡터에서 각 패치의 상대 위치를 보유합니다.

#### 2.2.3 Multi-head Self Attention (MSA)
스케일된 닷-프로덕트 어텐션 메커니즘:

$$\text{Attention} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$$

$L$ 개의 동일한 Transformer 인코더 계층에서:

$$\mathbf{z}'_l = \text{MSA}(\text{LayerNorm}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$

$$\mathbf{z}_l = \text{MLP}(\text{LayerNorm}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

최종 출력:

$$\mathbf{z}_L = [\mathbf{z}^p_{L0}; \mathbf{z}^p_{L1}; \ldots; \mathbf{z}^p_{LN}]$$

#### 2.2.4 분류 헤드
클래스 토큰 벡터 $\mathbf{z}^p_{L0}$을 선형 레이어로 전달:

$$\mathbf{y}_{predicted} = \text{Linear}(\mathbf{z}^p_{L0})$$

### 2.3 CT-HGR-V3: 신경 분해 기반 미시 경로

Motor Unit Spike Trains(MUSTs) 추출 후, Spike-Triggered Averaging(STA)으로 Motor Unit Action Potentials(MUAPs) 재구성:

$$\text{MUAP}_{i,j} = \frac{1}{N_{spikes}} \sum_{k=1}^{N_{spikes}} \mathbf{x}[t_k - w : t_k + w]$$

Peak-to-peak 값으로 2D 이미지 생성: $N_{ch} \times N_{cv}$

### 2.4 하이브리드 퓨전 모델

거시(Macro) 경로와 미시(Micro) 경로를 병렬로 실행 후 클래스 토큰 결합:

$$\mathbf{f}_{fused} = [\mathbf{z}^p_{L0,\text{Macro}}; \mathbf{z}^p_{L0,\text{Micro}}] \rightarrow \text{FC Layers} \rightarrow \text{Softmax}$$

***

## 3. 성능 분석 및 결과

### 3.1 CT-HGR-V1 기본 모델 성능

| 채널 수 | 윈도우 크기(ms) | 평균 정확도(%) | 표준편차(%) |
|---------|------------------|----------------|-----------|
| 32      | 31.25           | 86.23          | 2.94      |
| 64      | 62.5            | 88.93          | 2.61      |
| 128     | 125             | 91.35          | 2.31      |
| 128     | 250             | **91.98**      | 2.22      |

CT-HGR-V1은 128개 채널, 250ms 윈도우에서 최고 성능을 달성했습니다.

### 3.2 순간 인식 성능

| 항목 | 결과 |
|------|------|
| 윈도우 크기 | 1 샘플 (단일 프레임) |
| 채널 수 | 64 |
| 평균 정확도 | **89.13%** |
| 표준편차 | 2.61% |

이는 단일 시점의 2D sEMG 이미지만으로도 89% 이상의 인식이 가능함을 의미합니다.

### 3.3 모델 비교 분석

|모델|정확도(%)|메모리(GB)|학습시간(s)|
|----|---------|---------|---------|
|CT-HGR-V1|90.02|14.80|382.9|
|3D CNN|87.45|14.81|1228.9|
|SVM-V1|90.71|40.60|203.2|
|LDA-V1|90.85|40.60|149.3|

CT-HGR-V1은 SVM 대비 메모리는 2.7배 적으면서 비교 가능한 정확도를 달성했습니다.

### 3.4 하이브리드 퓨전 모델 성능

| 모델 | 정확도(%) |
|-----|----------|
| CT-HGR-V1 (거시 경로) | 89.34 |
| CT-HGR-V3 (미시 경로) | 86.64 |
| **Fused Model** | **94.86** |

퓨전 모델은 개별 모델 대비 5.5% 성능 향상을 달성했습니다.

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 현재 일반화 성능

**5-fold 교차 검증(개인 내):**
- 최고 성능: 91.98% (128 채널, 250ms)
- Fold 간 분산: 표준편차 2.22%

**셔플된 데이터셋 (모든 반복 혼합):**

| 윈도우 크기(ms) | 정확도(%) |
|-----------------|----------|
| 31.25          | 98.05    |
| 62.5           | 98.43    |
| 125            | 98.79    |

데이터를 완전히 셔플했을 때 98% 이상의 성능을 달성하여, 개인 내 학습은 우수하지만 **개인 간 일반화에는 제한**이 있음을 시사합니다.

### 4.2 일반화 성능 향상 메커니즘

#### 4.2.1 아키텍처 측면
1. **어텐션 메커니즘의 글로벌 수용 필드**: Transformer는 CNN의 로컬 수용 필드 제약을 극복하여, 전체 HD-sEMG 그리드의 장거리 의존성을 포착할 수 있습니다.

2. **입력 병렬화**: RNN과 달리 모든 패치를 병렬로 처리하므로, 신호의 미묘한 시간적 변이를 더 잘 캡처합니다.

#### 4.2.2 데이터 측면
1. **작은 스킵 스텝 (32 샘플)**: 15.3ms 간격으로 예측을 수행하여 모델이 더 많은 학습 샘플을 보게 되고, 신호 변이성에 더 강건해집니다.

2. **전체 신호 윈도우 포함**: 과도(transient)와 정상(steady) 상태를 모두 포함하여 더 완전한 제스처 표현을 학습합니다.

#### 4.2.3 하이브리드 퓨전의 역할
거시 정보(원본 sEMG)와 미시 정보(MUST)의 결합은:
- 상호 보완적 특징 추출
- 일관성 없는 신호에 대한 강건성 증가
- 개인 간 변이성 완화

### 4.3 개인 간 일반화의 한계와 개선 방안

**현재 한계:**
- 논문은 각 피험자에 대해 **독립적으로** 모델을 학습하므로, 엄격한 의미의 개인 간 크로스 검증이 부재합니다.
- 건강한 피험자만 포함되어 절단자 등 병리적 특성 미반영

**개선 가능성:**
1. **Few-Shot Learning**: 제한된 데이터로도 새로운 사용자 적응 가능
2. **Domain Adaptation**: 개인 간 EMG 신호 변이성 대응
3. **Meta-Learning**: 다양한 피험자 데이터로 메타 모델 학습
4. **Transfer Learning의 체계적 적용**: 기존 우리 결과 대비 80-95% 정확도 달성 가능(최신 연구 기준)

***

## 5. 한계 및 제약사항

### 5.1 데이터 및 피험자 관련

- **건강한 피험자만 대상**: 20명의 건강한 개인으로 제한, 절단자나 근골격계 질환자 미포함
- **정적 등척성 제스처**: 동적 동작이나 실제 손 움직임 미포함
- **제한된 환경**: 실험실 환경에서 획득, 일상적 노이즈/간섭 미고려

### 5.2 방법론 관련

**하이브리드 모델의 구현 제약:**
- MUST 분해는 오프라인 처리로만 가능하여, 온라인 실시간 HMI 시스템에 부적합
- Blind Source Separation의 매개변수(반복 횟수 7회, 실루엣 임계값 0.92)가 경험적으로 설정되어 객관적 기준 부족

**설명 가능성 부족:**
- Transformer 내부의 어텐션 맵이 신경생리학적으로 해석되지 않음
- 어떤 근육 신호가 특정 제스처 분류에 기여하는지 불명확

### 5.3 성능 평가 측면

- **Wilcoxon 검정의 제한적 개선**: CT-HGR-V1과 V2 간 유의미한 차이 없음 (일부 조건에서 ns: p > 0.05)
- **불균형한 제스처 성능**: 복잡한 2-DoF 제스처(정확도 82-92%)는 단순 1-DoF 제스처(98%+)보다 낮음

***

## 6. 최신 관련 연구 비교 분석 (2020-2025)

### 6.1 Transformer 기반 접근

#### 2021: TEMGNet (Rahimian et al.)
- **방법**: ViT 기반 아키텍처, sparse sEMG 신호 사용
- **성능**: 300ms 윈도우에서 82.93%, 200ms에서 82.05%
- **비교**: CT-HGR (125ms 동등 조건 ~90%) 대비 약 7% 우수하나, HD-sEMG 사용에 따른 차이 있음

#### 2022: TraHGR (Rahimian et al.)
- **특징**: Hybrid CNN-Transformer 아키텍처, 두 개의 병렬 경로
- **장점**: 다중 신호 처리의 유연성
- **비교**: CT-HGR의 순수 Transformer 접근보다 더 복잡한 구조

#### 2023: EMGTFNet
- **혁신**: Fuzzy Neural Block(FNB) 통합으로 확률적 sEMG 신호의 불확실성 처리
- **성능**: 데이터 증강/전이 학습 없이도 고정확도 달성
- **특징**: CT-HGR과 유사한 경량화 지향

#### 2025: WaveFormer
- **최신 기술**: 학습 가능한 웨이블릿 변환(Learnable Wavelet Transform) 통합
- **성능**: EPN612 데이터셋 95%, DB6 81.93% (inter-session protocol)
- **효율성**: 3.10M 파라미터로 대규모 모델 능가
- **특징**: 시간-주파수 도메인 특징 자동 추출

### 6.2 신경 분해 기반 접근 (최신)

#### 2024: cwCST-CNN (Chen et al.)
- **방법**: Channel-wise Cumulative Spike Train 이미지 + CNN
- **핵심**: Motor Unit의 공간 활성화 패턴으로 제스처 구별
- **성능**: 개별 운동 단위 정보의 높은 생리학적 해석성

#### 2025: Neuromorphic 구현 (Mixed-Signal Processor)
- **혁신**: DYNAP-SE 신경형 칩에서 직접 MU 분해 및 디코딩
- **성능**: 손가락 힘 회귀에서 8.16 ± 1.29% MVC RMSE
- **장점**: 초저전력 온라인 추론 (수십 mW)

### 6.3 강건성 및 일반화 개선

#### MoEMba (2025): Mamba 기반 Mixture of Experts
- **기술**: Selective StateSpace Model(SSM) + 채널 어텐션
- **성능 개선**: 이전 대비 40% 성능 향상 달성
- **특징**: 세션 간 신호 변이성에 강건

#### Domain Adaptation 접근
- **메타-러닝**: Few-shot 학습으로 새 사용자 적응 (73-85.94% → improved)
- **전이 학습**: 충분한 데이터로 80-95% cross-subject 정확도 달성 가능
- **기하학적 심화 학습**: Riemannian manifold 상에서의 TMKNet (covariance 기반)

### 6.4 종합 비교표

| 논문 | 연도 | 방법 | 성능 | 강점 | 약점 |
|-----|------|------|------|------|------|
| CT-HGR | 2023 | ViT + BSS | 91.98%(HD-sEMG) | 우수한 아키텍처, 퓨전 | 오프라인 분해 |
| WaveFormer | 2025 | Wavelet ViT | 95%(EPN612) | 최고 성능, 경량 | 실시간 미검증 |
| MoEMba | 2025 | Mamba-MOE | +40% 개선 | SSM의 시간적 효율 | 초기 모델 필요 |
| cwCST-CNN | 2024 | MUST+CNN | 높은 해석성 | 생리학적 기반 | CNN의 제약 |
| EMGTFNet | 2023 | Fuzzy ViT | 안정적 | 불확실성 처리 | 성능 수치 부제 |

***

## 7. 학술적 영향 및 향후 연구 방향

### 7.1 학술 기여

1. **패러다임 전환**: CNN-RNN 하이브리드에서 순수 Transformer로의 이동 선도
2. **신경 정보 통합**: 거시-미시 신경 정보 이원화의 개념적 기초 확립
3. **HD-sEMG 활용 확대**: 고밀도 신호의 장점을 처음 체계적으로 활용

### 7.2 임상 적용 가능성

- **보철 제어**: 절단자를 위한 다자유도 보철손 제어의 기초
- **신경재활**: 뇌졸중 환자의 운동 의도 해석
- **Human-Machine Interface**: XR/로봇 제어

### 7.3 향후 우선 연구 과제

#### 단기 (1-2년)
1. **온라인 MUST 분해**: 실시간 하이브리드 모델 구현
   - 추천: Spiking Neural Networks(SNNs)로 저전력 구현
   
2. **개인 간 일반화 강화**
   - Meta-learning 프레임워크 적용
   - Domain adversarial training (DANN)
   
3. **병리적 피험자 포함**
   - 절단자 데이터셋 구성
   - 근육 약화 조건 모의

#### 중기 (2-4년)
1. **설명 가능 AI 통합**
   - Attention map의 신경생리학적 해석
   - LIME/SHAP 등 해석 가능성 방법 적용
   
2. **다중 신호 융합**
   - EMG + 초음파 (Ultrasound)
   - EMG + IMU + 촉각 센서
   
3. **임상 시험**
   - 보철 사용자를 위한 파일럿 연구
   - 감각 피드백 통합 시스템

#### 장기 (4년 이상)
1. **신경형 칩 구현**: 주문형 저전력 칩셋 개발
2. **Continual Learning**: 온라인 적응 알고리즘
3. **Bidirectional Interface**: 운동 의도 + 감각 피드백

### 7.4 연구 설계 시 고려사항

**방법론적 엄격성:**
- ✓ 완전 개인 간 크로스 검증 (truly leave-one-subject-out)
- ✓ 다양한 환경 조건 포함 (노이즈, 피로도, 전극 이동)
- ✓ 정적 제스처 넘어 동적 동작 포함

**보고의 투명성:**
- ✓ 하이퍼파라미터 선정 근거 명시
- ✓ 실패 사례 및 한계 명확히 기술
- ✓ 재현성을 위한 코드 공개

**통계적 검증:**
- ✓ 다중 비교 보정 (Bonferroni correction)
- ✓ 효과 크기(Effect size) 보고
- ✓ 신뢰도 및 타당도 분석

***

## 결론

CT-HGR은 HD-sEMG 신호 처리에서 **중요한 기술적 도약**을 표현합니다. 순수 Transformer 기반 아키텍처, 순간적 인식 능력, 거시-미시 신경 정보의 통합은 학술적으로 의미 깊은 기여입니다. 

그러나 **일반화 성능**에서는 개인 내 성능(91.98%)과 개인 간 성능 간의 현저한 격차가 존재하며, 이는 향후 연구에서 반드시 해결해야 할 핵심 과제입니다. 2024-2025년의 최신 연구들(WaveFormer의 95% 성능, MoEMba의 40% 개선)은 웨이블릿 변환, SSM 아키텍처, 강화된 도메인 적응 기법을 통해 이러한 한계를 극복하고 있습니다.

**임상 실용화 관점**에서는 온라인 신경 분해, 절단자 기반 검증, 설명 가능성 확보가 필수적이며, 신경형 칩 기반의 초저전력 구현이 차세대 방향입니다. 이러한 발전들이 결합될 때, 직관적이고 적응형인 차세대 보철 및 신경재활 시스템의 실현이 가능할 것으로 예상됩니다.

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_5][^1_6][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: s41598-023-36490-w.pdf

[^1_2]: https://www.nature.com/articles/s41598-023-36490-w

[^1_3]: https://ieeexplore.ieee.org/document/10580881/

[^1_4]: https://ieeexplore.ieee.org/document/10956022/

[^1_5]: https://ieeexplore.ieee.org/document/10669579/

[^1_6]: https://ieeexplore.ieee.org/document/10197329/

[^1_7]: https://ieeexplore.ieee.org/document/10308789/

[^1_8]: https://ieeexplore.ieee.org/document/10385224/

[^1_9]: https://www.mdpi.com/2673-1592/5/4/88

[^1_10]: https://www.mdpi.com/2079-9292/13/20/4134

[^1_11]: https://ieeexplore.ieee.org/document/10341955/

[^1_12]: https://arxiv.org/pdf/2203.16336.pdf

[^1_13]: https://arxiv.org/pdf/2109.12379.pdf

[^1_14]: https://arxiv.org/pdf/2310.03754.pdf

[^1_15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10329032/

[^1_16]: https://arxiv.org/html/2404.15360

[^1_17]: http://arxiv.org/pdf/2502.17457.pdf

[^1_18]: https://www.mdpi.com/1424-8220/23/4/2065/pdf?version=1676181910

[^1_19]: https://www.frontiersin.org/articles/10.3389/fnbot.2023.1127338/pdf

[^1_20]: https://arxiv.org/pdf/2410.00586.pdf

[^1_21]: https://arxiv.org/pdf/2309.11086.pdf

[^1_22]: https://arxiv.org/pdf/2507.23474.pdf

[^1_23]: https://arxiv.org/html/2510.02000v1

[^1_24]: https://www.arxiv.org/pdf/2510.17660.pdf

[^1_25]: https://pubmed.ncbi.nlm.nih.gov/37419881/

[^1_26]: https://arxiv.org/pdf/2506.11168.pdf

[^1_27]: https://pdfs.semanticscholar.org/62c5/38ef20d967d4ce60219c5fc38e5a409eb25f.pdf

[^1_28]: https://pdfs.semanticscholar.org/d6c3/9f100a2998e41a5c6d705d308dbe329f479c.pdf

[^1_29]: https://www.semanticscholar.org/paper/Transformer-based-hand-gesture-recognition-from-to-Montazerin-Rahimian/dc279c52e3d7a132da718e47cefa88b82a23e831

[^1_30]: https://arxiv.org/html/2510.17660v1

[^1_31]: https://www.semanticscholar.org/paper/Real-Time-Hand-Gesture-Recognition-by-Decoding-Unit-Chen-Yu/596c6c8be703b95b3352e05747de65b34bfabee0

[^1_32]: https://arxiv.org/html/2509.23359v1

[^1_33]: https://arxiv.org/html/2411.15655v1

[^1_34]: https://www.semanticscholar.org/paper/87e7f342a34e509aba4ce9fff978352be7a313bb

[^1_35]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12069338/

[^1_36]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11927004/

[^1_37]: https://www.ijcai.org/proceedings/2024/0668.pdf

[^1_38]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12794296/

[^1_39]: https://spj.science.org/doi/10.34133/cbsystems.0219

[^1_40]: https://www.jait.us/uploadfile/2024/JAIT-V15N2-255.pdf

[^1_41]: https://www.nature.com/articles/s41598-025-28215-y

[^1_42]: https://www.sciencedirect.com/science/article/abs/pii/S0169260720314760

[^1_43]: https://www.sciencedirect.com/science/article/abs/pii/S1050641121000353

[^1_44]: https://www.biorxiv.org/content/10.1101/2025.09.10.675419v1.full.pdf

