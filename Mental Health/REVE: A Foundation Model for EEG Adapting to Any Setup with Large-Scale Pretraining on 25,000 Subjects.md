# REVE: A Foundation Model for EEG Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects

---

## 1. 핵심 주장과 주요 기여 요약

**REVE (Representation for EEG with Versatile Embeddings)**는 EEG 신호의 이질성(heterogeneity) 문제를 극복하고 **임의의 전극 배치와 시간 길이에 적응할 수 있는 범용 EEG 파운데이션 모델**이다. 핵심 주장과 기여는 다음과 같다:

1. **4D 위치 인코딩(4D Positional Encoding)**: 전극의 3D 공간 좌표와 시간 축을 결합한 Fourier 기반 4D 위치 인코딩을 제안하여, 고정된 전극 몽타주(montage)에 의존하지 않고 임의의 EEG 설정을 처리할 수 있음.
2. **역대 최대 규모의 EEG 사전학습 코퍼스**: 92개 데이터셋, 25,000명 이상의 피험자, 60,000시간 이상의 EEG 데이터를 수집·활용하여 가장 대규모의 EEG 사전학습을 수행함.
3. **SOTA 성능**: 10개 다운스트림 태스크에서 기존 파운데이션 모델 대비 평균 +2.5%의 balanced accuracy 향상을 달성하고, **linear probing에서 최대 17% 향상**을 보여 고품질 표현 학습 능력을 입증함.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

EEG 데이터는 다음과 같은 고유한 이질성 문제를 가짐:
- **전극 배치의 다양성**: 데이터셋마다 채널 수(4~128개)와 전극 배열이 상이함
- **시간 길이의 가변성**: 녹화 시간이 수 초에서 수 시간까지 다양함
- **프로토콜·장비 차이**: 샘플링 레이트, 기기, 기록 조건이 각기 다름
- **낮은 신호 대 잡음비(SNR)**: EEG 신호 자체의 노이즈가 높음

기존 EEG 파운데이션 모델(BIOT, LaBraM, CBraMod, NeuroGPT 등)은 주로 TUH 데이터베이스의 고정된 19/21 채널 몽타주에서만 사전학습하여 **다른 전극 레이아웃이나 기록 설정에서 일반화 실패**. 특히 **linear probing** 성능이 크게 떨어지며, 이는 사전학습된 표현의 품질이 불충분함을 시사함.

### 2.2 제안하는 방법

#### 2.2.1 EEG 표현 및 패치 임베딩

다채널 EEG 데이터를 $\mathbf{X} \in \mathbb{R}^{C \times T}$로 표현하며, 전극 위치는 $\mathbf{P} \in \mathbb{R}^{C \times 3}$이다. 각 채널을 크기 $w$, 오버랩 $o$의 패치로 분할:

$$p = \left\lceil \frac{T - w}{w - o} \right\rceil + \mathbb{1}[(T - w) \bmod (w - o) \neq 0]$$

이를 통해 $\mathbf{X}_p \in \mathbb{R}^{C \times p \times w}$로 변환하고, 선형 임베딩을 적용하여 $\mathbf{E} \in \mathbb{R}^{C \times p \times D_E}$를 얻음.

#### 2.2.2 4D 위치 인코딩 전략

기존 모델의 학습 가능한 위치 임베딩 테이블과 달리, **전극의 실제 3D 좌표와 시간 인덱스를 직접 활용**:

1. **공간 좌표 확장**: $\mathbf{P} \in \mathbb{R}^{C \times 3}$에 가우시안 노이즈($\sigma_{\text{noise}}$)를 추가하여 일반화 강화 후, 시간 차원을 추가하여 $\mathbf{P}_{\text{ext}} \in \mathbb{R}^{C \times p \times 4}$를 생성

2. **4D Fourier 기반 위치 인코딩**: $(x, y, z, t)$ 각 성분을 $n_{\text{freq}}$개의 주파수로 다주파수 공간에 투영. Cartesian product 구조를 따라 $n_{\text{freq}}^4$ 차원의 벡터를 생성하고, 사인/코사인 변환으로 $2 \cdot n_{\text{freq}}^4$ 차원의 임베딩 $\mathbf{F}_{\text{pe}} \in \mathbb{R}^{C \times p \times D_E}$를 얻음

3. **최종 위치 인코딩**: Fourier 특징과 학습 가능한 선형 표현을 결합:

$$\mathbf{P}_{\text{enc}} = \text{LayerNorm}(\mathbf{F}_{\text{pe}} + \mathbf{F}_{\text{lin}})$$

여기서 $\mathbf{F}\_{\text{lin}}$은 $\mathbf{P}_{\text{ext}}$를 선형 레이어 + GELU + LayerNorm으로 처리한 결과.

#### 2.2.3 시공간 블록 마스킹 전략

기존의 랜덤 마스킹 대신, **공간·시간 차원 모두에서 연속적인 블록을 마스킹**:
- 마스킹 비율 $M_r = 55\%$
- 공간 마스킹 반경 $R_s$, 시간 마스킹 반경 $R_t$
- 드롭아웃 비율 $D_r$, 드롭아웃 반경 $R_d$

이진 마스크 $\mathbf{B} \in \mathbb{R}^{C \times p}$를 생성하여 $N_m = \lfloor (1 - M_r) \cdot C \cdot p \rfloor$개의 마스킹 엔트리를 결정함.

#### 2.2.4 손실 함수

**주(Primary) 손실**: 마스킹된 패치의 원본 EEG 신호를 L1 손실로 복원:

$$\mathcal{L} = \frac{1}{|\mathbf{P}_m|} \sum_{i \in \mathbf{P}_m} \left\| \hat{\mathbf{P}}_m^{(i)} - \mathbf{P}_m^{(i)} \right\|_1$$

L2 대신 L1 손실을 사용한 이유는 EEG의 높은 노이즈 특성에 대한 강건성 확보를 위함.

**부(Secondary) 손실**: 인코더의 모든 MHA 레이어 출력에 대해 attention pooling을 적용하여 학습된 쿼리 토큰으로 글로벌 표현을 생성하고, 이 압축된 표현으로부터 마스킹된 패치를 복원:

$$\text{Loss} = \text{Primary Loss} + \lambda \cdot \text{Secondary Loss}$$

($\lambda = 0.1$). 이 보조 손실은 정보 병목(information bottleneck) 역할을 하여 인코더가 모든 레이어에 걸쳐 유용한 정보를 분산시키도록 유도하고, 최종 레이어의 과적합을 방지함.

### 2.3 모델 구조

REVE는 MAE(Masked Autoencoder) 구조를 기반으로 하며, 주요 구성 요소는:

| 구성 요소 | 세부 사항 |
|---|---|
| **인코더** | 깊은 Transformer (Base: 22층, dim=512, 8 heads, 69M 파라미터) |
| **디코더** | 경량 Transformer (사전학습 시에만 사용, 추론 시 폐기) |
| **정규화** | RMSNorm (LayerNorm 대체) |
| **활성화 함수** | GEGLU (GELU 대체, 게이팅 메커니즘으로 더 표현력 있음) |
| **FFN 확장 비율** | $\frac{8}{3}$ (LLaMA/Mistral 설계 따름) |
| **어텐션** | Flash Attention v2 (메모리·계산 효율) |
| **편향(Bias)** | 최종 디코더 레이어 제외 모든 선형 레이어에서 제거 |

**모델 크기별 설정** (Table 6):

| Size | Depth | Heads | Dim | Params | $n_{\text{freq}}$ |
|---|---|---|---|---|---|
| Small | 4 | 8 | 512 | 12M | 4 |
| Base | 22 | 8 | 512 | 69M | 4 |
| Large | 22 | 19 | 1,250 | 408M | 5 |

### 2.4 성능 향상

**Table 2** (9개 태스크 평균 balanced accuracy):

| 모델 | 평균 |
|---|---|
| EEGNet | 0.5941 |
| BIOT | 0.6438 |
| LaBraM-Base | 0.6653 |
| CBraMod | 0.6898 |
| **REVE-Base** | **0.7150** |

주요 성능 향상 포인트:
- **BCI-IV-2a (Motor Imagery)**: CBraMod 0.5138 → REVE 0.6396 (**+12.6%p**)
- **MAT (Mental Stress)**: CBraMod 0.7256 → REVE 0.7660 (**+4.0%p**)
- **Linear Probing** (Table 4): REVE-Large 평균 0.654 vs. CBraMod 0.501 (**+15.3%p**)
- **Frozen backbone** (Table 3): PhysioNet-MI에서 REVE 0.5371 vs. CBraMod 0.3845, BIOT 0.3698, LaBraM 0.3715 (**약 +15%p 이상**)

### 2.5 한계

1. **입력 길이 제한**: 최소 1초, 1초 단위 배수의 신호만 처리 가능
2. **데이터 큐레이션 부재**: 92개 데이터셋을 단순 집합하여 저품질 녹화의 선별적 제거 미실시
3. **인구 통계학적 편향**: 대부분의 공개 EEG 데이터가 북미·유럽 출처로, 인구 다양성 제한
4. **정밀한 스케일링 법칙 미도출**: 스케일링 효과는 관찰되나 정확한 법칙 미규명
5. **단순한 SSL 접근**: 기본 MAE와 표준 Transformer 사용, 더 고급 SSL 기법이나 아키텍처 미탐색

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

REVE의 일반화 성능은 다음 네 가지 핵심 메커니즘에 의해 달성되며, 각각은 향후 확장 가능성을 내포한다.

### 3.1 4D Fourier 위치 인코딩의 일반화 기여

기존 모델들이 학습 가능한 위치 임베딩 테이블을 사용하여 **사전학습 시 관찰한 전극/시간 구성에만 한정**되는 반면, REVE의 4D Fourier 인코딩은:

- **임의의 전극 배치**: 3D 좌표 기반이므로 학습 시 보지 못한 전극 레이아웃에도 즉시 적용 가능
- **임의의 시간 길이**: 연속적 시간 인코딩으로 사전학습(10초)보다 긴 입력(30초 sleep staging)에도 일반화 (ISRUC, HMC에서 검증)
- **바이폴러 몽타주 대응**: TUEV의 바이폴러 설정(학습 중 미관찰)에도 성공적 전이

**공간 노이즈 증강** ($\sigma_{\text{noise}} = 0.25\text{cm}$)은 머리 크기·전극 배치의 변동성에 대한 강건성을 제공하며, 이를 제거하면 성능이 약 4.7%p 하락 (Table 19: 0.707 → 0.660).

### 3.2 대규모 다양성 있는 사전학습 데이터의 효과

92개 데이터셋의 구성 (Table 7):
- **카테고리**: BCI(28), 인지(56), 임상(8)
- **채널 범위**: 3~129개
- **총 396개 고유 전극명**

이 다양성이 일반화에 미치는 핵심 효과는 **Table 3**에서 드러남:
- REVE는 사전학습으로 **11%p 성능 향상** (0.5409 → 0.6480)
- CBraMod는 사전학습으로 **2%p 향상**에 불과 (0.6196 → 0.6417)

이는 REVE가 사전학습의 이점을 더 크게 받는, 즉 **데이터 다양성에 비례하여 표현 품질이 향상**되는 구조임을 시사.

### 3.3 Secondary Loss와 표현 품질

Attention pooling 기반 보조 손실은 **information bottleneck** 역할을 하여:
- 인코더 전 레이어에 걸쳐 정보를 분산시킴
- 최종 레이어의 재구성 태스크 과적합 방지
- **Frozen-feature 시나리오에서 특히 효과적** (Table 17: 평균 LP 0.523 → 0.558, FT 0.612 → 0.665)

### 3.4 스케일링과 일반화

모델 크기 증가에 따른 일관된 일반화 향상 (Table 4):
- REVE-Base Linear Probing 평균: 0.609
- REVE-Large Linear Probing 평균: **0.654** (+4.5%p)

NLP의 스케일링 법칙 $\eta \propto D^{\alpha_D}$ ($\alpha_D = -0.90$)을 차용하여 학습률을 조정하고, 모델 크기 확대 시 **아키텍처 일관성**을 유지 (고정된 FFN 비율, 깊이/너비/헤드 수 조정).

### 3.5 Sparse Setup 및 Few-shot에서의 일반화

**Sparse setup** (Table 21): PhysioNet-MI L-R 태스크에서 64채널 → 1채널로 줄여도 0.824 → 0.660으로 graceful degradation.

**Few-shot** (Table 22): BCI-IV-2a Left-Right에서:
- 1-shot: 0.588 (사전학습만), 0.605 (cross-dataset fine-tuning 후)
- 20-shot: 0.723 → 0.817 (XFT 후 +10%p)

이는 REVE의 임베딩이 **레이블 없이도 즉시 활용 가능한 품질**을 가짐을 보여줌.

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

1. **EEG 연구의 표준화**: REVE가 코드, 사전학습 가중치, 튜토리얼을 공개함으로써, 이종 EEG 설정 간 **통합된 벤치마크와 임베딩 표준**의 기반을 마련함
2. **임상 적용 가속화**: Cross-site 배포를 가능케 하여 다기관 임상 연구에서의 EEG 모델 활용성을 대폭 향상
3. **BCI 캘리브레이션 단축**: Few-shot 및 linear probing 성능이 우수하여, BCI 시스템의 신규 사용자 적응 시간을 크게 단축할 수 있음
4. **스케일링 법칙의 EEG 확장**: NLP/CV에서 확인된 스케일링 법칙이 EEG 도메인에서도 작동함을 초기적으로 입증하여, 향후 더 큰 모델과 데이터셋 구축의 동기를 제공

### 4.2 향후 연구 시 고려할 점

1. **데이터 큐레이션 전략**: 단순 양적 확대를 넘어 저품질 녹화 제거, 분포 균형, 대표 서브셋 식별 등 **질적 데이터 큐레이션** 필요
2. **인구 다양성 확보**: 현재 북미·유럽 편중 데이터에서 벗어나 글로벌하게 공정한(equitable) 데이터 수집 필요
3. **멀티모달 확장**: MEG, iEEG, OPM-MEG 등 다른 뇌 신호 모달리티로의 확장 가능성
4. **고급 SSL 기법**: 현재 기본 MAE를 사용하고 있으나, 대조 학습(contrastive learning), JEPA 류, 또는 생성 모델 기반 SSL로의 발전 가능
5. **정밀 스케일링 법칙 도출**: 모델 크기, 데이터 양, 다운스트림 성능 간의 상호작용을 정확히 포착하는 EEG 특화 스케일링 법칙 필요
6. **Zero-shot/Few-shot 체계적 평가**: 더 다양한 태스크와 조건에서의 제로/퓨샷 레짐 평가 필요
7. **프라이버시 고려**: 디코더의 EEG 원본 신호 재구성 능력이 프라이버시 위험을 야기할 수 있으므로, 모델 배포 시 디코더 미공개 등 safeguard 유지 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 사전학습 데이터 | 위치 인코딩 | SSL 방법 | 주요 특징 | 제한사항 |
|---|---|---|---|---|---|---|
| **BIOT** (Yang et al., 2024) | 2024 | TUH (고정 채널) | 학습 가능한 절대 PE | 자기지도 학습 | 바이오시그널 cross-data 전이 | 고정 몽타주에 제한, linear probing 약함 |
| **LaBraM** (Jiang et al., 2024) | 2024 | TUH (~2,534h) | 학습 가능한 절대 PE | MAE 기반 | 대규모 EEG 표현 학습 | 고정 전극 배열, 사전학습 데이터 규모 제한 |
| **CBraMod** (Wang et al., 2024b) | 2024 | TUH (~9,000h, 100μV 이하 필터) | Convolutional PE | Criss-cross MAE | 교차 차원 모델링 | 100μV 초과 신호 제거, 공간 다양성 부족 |
| **NeuroGPT** (Cui et al., 2024) | 2024 | TUH | - | GPT 스타일 | LLM 방식 적용 시도 | 단일 데이터소스, 제한된 일반화 |
| **BrainWave** (Yuan et al., 2024a) | 2024 | ~40,000h (주로 iEEG) | - | - | 임상 응용 중심 | 두피 EEG가 아닌 침습적 iEEG 중심 |
| **Brant-2** (Yuan et al., 2024b) | 2024 | 대규모 뇌 신호 | - | 파운데이션 모델 | 뇌 신호 범용 모델 | 비침습 EEG 특화 부족 |
| **EEGPT** (Wang et al., 2024a) | 2024 | EEG 데이터 | - | Pretrained Transformer | 범용 EEG 표현 | REVE 대비 제한된 벤치마크 |
| **EEG2Rep** (Mohammadi Foumani et al., 2024) | 2024 | EEG | 공간 마스킹 | Informative masked inputs | 자기지도 표현 강화 | 시간 블록 마스킹 미포함 |
| **S-JEPA** (Guetschel et al., 2024) | 2024 | 다중 EEG | Dynamic spatial attention | JEPA 변형 | Cross-dataset 전이를 위한 동적 공간 어텐션 | 데이터 규모 제한 |
| **MaEEG** (Chien et al., 2022) | 2022 | EEG | - | 랜덤 마스킹 MAE | EEG에 MAE 최초 적용 | 구조적 마스킹 미사용 |
| **REVE** (본 논문, 2025) | 2025 | **92 데이터셋, 25,000명, 60,000h** | **4D Fourier PE + 노이즈 증강** | **블록 마스킹 MAE + 보조 손실** | **임의 전극/시간에 적응, SOTA** | 최소 1초 제한, 인구 편향 |

**핵심 차별화 요소**: REVE는 (1) 위치 인코딩의 유연성, (2) 사전학습 데이터의 규모와 다양성, (3) 보조 손실에 의한 표현 품질 모두에서 기존 모델들을 체계적으로 능가함. 특히 **linear probing에서의 압도적 우위**(평균 +15%p 이상 vs. CBraMod)는 REVE가 fine-tuning 없이도 활용 가능한 고품질 범용 표현을 학습했음을 강력히 지지함.

---

## 참고자료

1. **El Ouahidi, Y., Lys, J., Thölke, P., et al.** "REVE: A Foundation Model for EEG Adapting to Any Setup with Large-Scale Pretraining on 25,000 Subjects." *arXiv:2510.21585v1*, NeurIPS 2025. ([arXiv 링크](https://arxiv.org/abs/2510.21585))
2. **Yang, C., Westover, M., & Sun, J.** "BIOT: Biosignal Transformer for Cross-Data Learning in the Wild." *NeurIPS*, 2024.
3. **Jiang, W., Zhao, L., & Lu, B.** "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI." *ICLR*, 2024.
4. **Wang, J., Zhao, S., et al.** "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding." *arXiv:2412.07236*, 2024.
5. **Cui, W., Jeong, W., et al.** "Neuro-GPT: Towards a Foundation Model for EEG." *IEEE ISBI*, 2024.
6. **He, K., Chen, X., et al.** "Masked Autoencoders Are Scalable Vision Learners." *CVPR*, 2022.
7. **Yuan, Z., Shen, F., et al.** "BrainWave: A Brain Signal Foundation Model for Clinical Applications." *arXiv:2402.10251*, 2024.
8. **Wang, G., Liu, W., et al.** "EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals." *NeurIPS*, 2024.
9. **Mohammadi Foumani, N., et al.** "EEG2Rep: Enhancing Self-Supervised EEG Representation Through Informative Masked Inputs." *ACM SIGKDD*, 2024.
10. **Guetschel, P., Moreau, T., & Tangermann, M.** "S-JEPA: Towards Seamless Cross-Dataset Transfer Through Dynamic Spatial Attention." *arXiv:2403.11772*, 2024.
11. **Chien, H.Y.S., et al.** "MaEEG: Masked Auto-Encoder for EEG Representation Learning." *NeurIPS Workshop on Learning from Time Series for Health*, 2022.
12. **프로젝트 페이지**: [https://brain-bzh.github.io/reve/](https://brain-bzh.github.io/reve/)

> **참고**: 본 분석은 제공된 원논문(arXiv:2510.21585v1)의 내용을 기반으로 작성되었으며, 논문에 직접 기술된 수치와 방법론을 충실히 반영하였습니다. 비교 분석의 관련 연구들은 해당 논문에서 인용된 참고문헌을 기반으로 정리하였습니다.
