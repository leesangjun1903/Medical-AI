
# Learning to Distill Global Representation for Sparse-View CT

> **논문 정보**
> - **제목**: Learning to Distill Global Representation for Sparse-View CT
> - **저자**: Zilong Li, Chenglong Ma, Jie Chen, Junping Zhang, Hongming Shan
> - **학회**: ICCV 2023 (pp. 21196–21207)
> - **arXiv**: [2308.08463](https://arxiv.org/abs/2308.08463)
> - **코드**: [https://github.com/longzilicart/GloReDi](https://github.com/longzilicart/GloReDi)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

Sparse-view CT는 소수의 투영(projection)만으로 단층 촬영 재건을 수행하여 환자에 대한 방사선 선량을 크게 줄이고 데이터 획득을 가속화할 수 있다. 그러나 재건된 이미지는 강한 아티팩트(artifact)로 인해 진단 가치가 크게 제한된다.

현재의 추세는 Sparse-view CT를 위해 원시 데이터(raw data)를 활용하는 방향으로 나아가고 있다. 그러나 이로부터 파생된 **듀얼 도메인(dual-domain) 방법들**은 특히 ultra-sparse view 시나리오에서 2차 아티팩트 문제를 겪으며, 다른 스캐너/프로토콜로의 일반화가 크게 제한된다.

이에 대해 본 논문은 이렇게 반박한다:

**이미지 후처리(image post-processing) 방법이 한계에 도달했는가?** 저자들의 답변은 "아직 아니다"이며, 뛰어난 유연성을 이유로 이미지 후처리 방법을 고수하면서 Sparse-view CT를 위한 **전역 표현(Global Representation, GloRe) 증류 프레임워크**인 **GloReDi**를 제안한다.

---

### 📌 주요 기여 4가지


1. **GloRe 학습**: Fourier 합성곱을 이용한 전역 표현 학습을 sparse-view CT에 최초로 제안 (이미지 후처리 단계에서의 표현 학습을 강조한 최초의 연구)
2. **GloRe 증류 프레임워크**: 중간 뷰(intermediate-view) 재건 이미지로부터 추가 감독 신호를 활용하는 새로운 증류 프레임워크 제안
3. **두 가지 핵심 증류 모듈**: 표현 방향 정렬을 위한 **Representation Directional Distillation**과 임상적으로 중요한 세부 정보를 학습하는 **Band-pass-specific Contrastive Distillation** 제안
4. 정량적 지표 및 시각적 비교에서 최신 Sparse-view CT 재건 방법들(이중 도메인 포함)에 대한 우수성 입증


---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

**① 이미지 후처리 방법의 제한적 수용 영역(receptive field)**

기존 CNN 기반 방법들은 제한된 수용 영역으로 인해 전역 정보 처리에 한계가 있다. FFC(Fast Fourier Convolution)를 이용한 GloRe 학습을 통해 각 요소가 이미지 전체에 걸친 수용 영역을 갖도록 하며, 이 전역적 특성은 전체 이미지에 걸쳐 퍼져 있는 아티팩트와 정보를 더 잘 모델링할 수 있게 한다.

**② 감독 신호(supervision signal)의 한계**

기존 방법들은 Full-view 이미지만을 감독 신호로 사용하는 반면, GloReDi는 이미 쉽게 획득 가능하나 이전 문헌에서는 탐구되지 않은 **중간 뷰(intermediate-view) 재건 이미지**로부터 GloRe를 증류(distill)한다.

**③ 듀얼 도메인 방법의 일반화 문제**

현재의 추세는 정보 복구를 위해 원시 데이터를 활용하는 방향이다. 그러나 이 방법들은 ultra-sparse view 시나리오에서 2차 아티팩트를 유발하며, 다른 스캐너/프로토콜에 대한 일반화가 크게 제한된다.

---

### 2.2 제안 방법 (수식 포함)

#### (A) Fast Fourier Convolution (FFC)를 활용한 GloRe 학습

핵심 기술은 FFC를 이용한 전역 표현 학습이다. FFC는 먼저 실수 푸리에 변환을 적용하여 주파수 특징 맵을 얻고, 주파수 성분에 대한 합성곱을 수행한 후 역변환한다.

FFC 연산을 수식으로 표현하면:

$$\mathcal{F}_{FFC}(x) = \mathcal{F}^{-1}\left( \text{Conv}_{freq}\left(\mathcal{F}(x)\right) \right)$$

- $\mathcal{F}$: 실수 푸리에 변환 (Real FFT)
- $\text{Conv}_{freq}$: 주파수 도메인에서의 합성곱
- $\mathcal{F}^{-1}$: 역 푸리에 변환

FFC의 핵심 특성은 **이미지 전체 영역의 수용 영역(image-wide receptive field)**을 가진다는 것이다. 이는 주파수 도메인에서 전역 정보가 자연스럽게 인코딩되기 때문이다:

$$\text{GloRe}(I_S) = \text{Encoder}_{FFC}(I_S)$$

여기서 $I_S$는 Sparse-view CT 재건 이미지이다.

---

#### (B) 증류 프레임워크 (GloReDi)

구체적으로, 먼저 중간 뷰 재건 이미지를 활용하여 **교사 네트워크(teacher network)**를 훈련하고, 이 교사 네트워크를 사용하여 **학생 모델(student model)**, 즉 sparse-view CT의 학습을 가이드한다.

전체 프레임워크를 수식으로 요약하면:

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{RDD} + \lambda_2 \mathcal{L}_{BPCD}$$

- $\mathcal{L}_{recon}$: 재건 손실 (예: L1 또는 L2 loss)
- $\mathcal{L}_{RDD}$: Representation Directional Distillation 손실
- $\mathcal{L}_{BPCD}$: Band-pass-specific Contrastive Distillation 손실
- $\lambda_1, \lambda_2$: 각 손실 항의 가중치 계수

---

#### (C) Representation Directional Distillation (RDD)

표현 방향 증류(RDD)는 학생과 교사 GloRe 간의 방향을 정렬하는 것으로, 서로 다른 뷰의 CT 이미지 간 도메인 격차(domain gap)로 인한 대규모 정보 손실을 고려하여 적절한 감독 신호를 제공한다.

RDD 손실의 수식적 형태:

$$\mathcal{L}_{RDD} = 1 - \frac{\langle \mathbf{g}_S, \mathbf{g}_T \rangle}{\|\mathbf{g}_S\| \cdot \|\mathbf{g}_T\|}$$

여기서:
- $\mathbf{g}_S = \text{GloRe}(I_S)$: 학생 네트워크의 전역 표현
- $\mathbf{g}_T = \text{GloRe}(I_T)$: 교사 네트워크의 전역 표현
- $\langle \cdot, \cdot \rangle$: 내적(inner product)

> ⚠️ **주의**: 위 수식은 논문의 개념적 설명을 바탕으로 코사인 유사도 기반의 방향 정렬 손실을 표현한 것입니다. 논문 내 정확한 수식 표기는 원문 PDF를 통해 확인하시기 바랍니다.

---

#### (D) Band-pass-specific Contrastive Distillation (BPCD)

밴드패스 특화 대조 증류(BPCD)는 각 CT 이미지의 특정 임상적 가치를 재건 정확도를 저해하지 않으면서 증류하기 위해 **밴드패스 성분에만** 대조 학습(contrastive learning)을 활용한다.

BPCD 손실의 개념적 수식:

$$\mathcal{L}_{BPCD} = -\log \frac{\exp\left(\text{sim}(\mathbf{g}_S^{bp}, \mathbf{g}_T^{bp}) / \tau\right)}{\sum_{k} \exp\left(\text{sim}(\mathbf{g}_S^{bp}, \mathbf{g}_k^{bp}) / \tau\right)}$$

여기서:
- $\mathbf{g}^{bp}$: 밴드패스 필터링된 표현 벡터
- $\text{sim}(\cdot, \cdot)$: 코사인 유사도
- $\tau$: 온도 파라미터 (temperature)

---

### 2.3 모델 구조 (Architecture)

GloReDi는 주로 네 가지 파트로 구성된다:
- **두 개의 인코더(Encoder)**: Sparse-view 및 Intermediate-view 재건 이미지 각각에서 GloRe를 학습
- **두 개의 디코더(Decoder)**: GloRe로부터 최종 처리된 이미지를 출력
- **RDD 모듈**: 학생-교사 GloRe 간 방향 정렬
- **BPCD 모듈**: 임상적으로 중요한 특징 증류

구현상의 두 가지 핵심 컴포넌트는 (1) CUDA로 구현된 Sparse-view CT 시뮬레이터 프로토콜(다양한 네트워크 및 데이터셋과 호환 가능한 래퍼)과 (2) Fourier 네트워크 및 증류 프레임워크(이미지 도메인 전용 방법)로 이루어진다.

전체 아키텍처 흐름:

```
[Sparse-view CT 입력 I_S] ──→ [Student Encoder (FFC)]──→ GloRe_S ──→ [Student Decoder]──→ [최종 출력]
                                                              ↕ RDD + BPCD
[Intermediate-view CT I_T]──→ [Teacher Encoder (FFC)]──→ GloRe_T ──→ [Teacher Decoder]──→ (학습 시 보조)
```

데이터셋: DeepLesion 데이터셋 및 AAPM-Myo 데이터셋이 사용되었다.

---

### 2.4 성능 향상

GloRe 증류의 성공은 두 핵심 컴포넌트인 GloRe 방향 정렬을 위한 **Representation Directional Distillation**과 임상적으로 중요한 세부 정보 획득을 위한 **Band-pass-specific Contrastive Distillation**에 기인한다. 광범위한 실험에서 제안된 GloReDi가 이중 도메인 방법들을 포함한 최신 기법들에 대한 우수성을 입증하였다.

---

### 2.5 한계

논문의 내용 및 맥락에 기반한 주요 한계:

1. **중간 뷰 데이터 의존성**: 증류 프레임워크가 중간 뷰 재건 이미지를 필요로 하므로, 이를 구하기 어려운 일부 임상 환경에서의 적용이 제한될 수 있다.
2. **이미지 도메인 전용의 물리적 제약**: 원시 사이노그램(sinogram)이 상업적 프라이버시 문제로 인해 접근 불가능한 경우가 많으며, 사이노그램이 극도로 희소한 경우 이미지 도메인 방법이 이중 도메인 방법보다 성능이 낮을 수 있다.
3. **2D 처리 한계**: 일반적인 이미지 도메인 후처리 방법처럼 3D 볼륨 전체의 맥락 정보를 동시에 활용하는 데 있어 제약이 있을 수 있다.
4. **일반화 범위**: 훈련 데이터셋(DeepLesion, AAPM-Myo)을 기반으로 실험이 이루어졌으므로, 완전히 다른 해부학적 구조 또는 스캐너 환경에서의 일반화 성능은 추가 검증이 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 측면에서의 강점

**① 이미지 후처리 방법의 유연성**

현재 추세는 정보 복구를 위해 원시 데이터로 전환되고 있으나, 이중 도메인 방법들은 특히 ultra-sparse view 시나리오에서 2차 아티팩트를 겪으며 다른 스캐너/프로토콜에 대한 일반화가 크게 제한된다.

반면 GloReDi는 이미지 도메인에서만 동작하므로, 스캐너마다 다른 원시 데이터 형식이나 사이노그램 접근 제한에 구애받지 않는다. 이는 다양한 임상 환경에서의 배포 가능성을 높인다.

**② 이미지 전체 수용 영역(Image-wide Receptive Field)**

FFC를 이용한 GloRe 학습으로 각 요소가 이미지 전체의 수용 영역을 가지며, 이 전역적 특성은 전체 이미지에 걸쳐 퍼져 있는 아티팩트와 정보를 더 잘 모델링하고 서로 다른 뷰의 표현 정렬을 용이하게 한다.

이 점은 아티팩트 패턴이 뷰 수에 따라 달라지는 다양한 Sparse-view 시나리오에서도 강인한 성능을 발휘할 수 있는 기반을 제공한다.

**③ 중간 뷰를 활용한 표현 증류**

중간 뷰 이미지로부터 지식을 증류하는 병렬 교사 네트워크를 포함하여, Sparse-view CT 이미지의 GloRe 학습에 고품질의 적절한 가이드를 제공한다.

이 접근 방식은 훈련 시에만 교사 네트워크가 필요하며, 추론(inference) 단계에서는 학생 네트워크만 사용하므로 추가적인 연산 비용 없이 다양한 환경에서 배포 가능하다.

**④ 시뮬레이터 프로토콜의 범용성**

CUDA로 구현된 Sparse-view CT 시뮬레이터 프로토콜은 다양한 네트워크 및 데이터셋과 호환 가능한 사용하기 쉬운 래퍼로 설계되었다.

이는 새로운 스캐너 설정이나 임상 프로토콜에 대한 시뮬레이션을 통해 훈련 데이터를 증강할 수 있게 하며, 일반화 성능 향상에 기여한다.

---

### 3.2 일반화 측면에서의 도전 과제

딥러닝 방법들은 Sparse-view CT 재건에서 상당한 가능성을 보여주지만, 고품질의 쌍(paired) 데이터에 크게 의존하며 이러한 데이터는 종종 구하기 어렵다.

생성 모델 기반 방법들(예: 확산 모델)은 훈련 데이터 없이도 선형 역문제를 해결할 수 있으나, 훈련 데이터로부터의 편향된 사전 정보(biased prior)를 재건 이미지에 도입하여 추가적인 아티팩트를 유발할 수 있다. 또한 서로 다른 취득 장소(acquisition sites)와 조건 간 사용 시 일반화 문제가 발생한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 이미지 도메인 표현 학습의 재평가
GloReDi는 이미지 후처리 방법이 한계에 도달했다는 기존 인식에 반박하며, **전역 표현(representation) 학습**이라는 새로운 관점을 Sparse-view CT 연구에 도입했다. 이는 향후 이미지 도메인 방법에서도 표현 품질을 명시적으로 최적화하는 연구로 이어질 수 있다.

#### (2) 중간 감독 신호(Intermediate Supervision)의 활용
Full-view 이미지만을 감독 신호로 사용하는 방법들과 달리, GloReDi는 이전 문헌에서는 탐구되지 않았던 중간 뷰 재건 이미지를 활용한다. 이러한 **계층적·중간 감독** 전략은 의료 영상 전반(저선량 CT, MRI 재건 등)에서 광범위하게 적용될 수 있다.

#### (3) 주파수 도메인 학습과 CT의 결합
FFC의 활용은 CT 아티팩트가 주로 특정 주파수 대역에 분포한다는 특성을 효과적으로 활용한다. 이는 관련 후속 연구인 FreeSeed에서도 계승되었으며, **주파수 도메인 인식(frequency-band-aware) 학습**이 CT 재건의 중요한 방향임을 시사한다.

#### (4) 지식 증류(Knowledge Distillation)의 의료 영상 응용 확장
컴퓨터 비전에서 주로 사용되던 지식 증류 기법을 Sparse-view CT 재건에 성공적으로 적용한 것은, 증류 기반 프레임워크가 다양한 의료 영상 역문제(inverse problem) 해결에 적용될 수 있음을 보여준다.

---

### 4.2 앞으로 연구 시 고려할 점

#### ① 3D 볼륨 전역 표현으로의 확장
현재 GloReDi는 2D 이미지 슬라이스 단위로 동작한다. 3D FFC 또는 Volumetric 전역 표현을 통해 슬라이스 간 상관관계(inter-slice correlation)를 활용하면 성능을 더욱 향상시킬 수 있다.

#### ② 도메인 적응(Domain Adaptation) 강화
서로 다른 취득 장소 및 조건 간에 사용될 때 일반화 문제가 발생한다. 이를 해결하기 위해 테스트 시 적응(Test-time Adaptation) 또는 연속 학습(Continual Learning) 기법과 GloReDi의 결합을 고려할 수 있다.

#### ③ 자기지도 및 비지도 학습으로의 확장
INR(Implicit Neural Representation) 기반 방법들은 "자기지도(self-supervised)" 훈련과 분포 이탈(distribution-shift)에 대한 강인성 측면에서 유망한 접근법으로 인식된다. GloReDi의 중간 뷰 증류 개념을 자기지도 학습 프레임워크와 결합하면 레이블 의존성을 줄일 수 있다.

#### ④ 확산 모델과의 융합
확산 모델과 같은 생성 방법들은 쌍(paired) 데이터 없이도 선형 역문제를 해결할 수 있으나, 훈련 데이터로부터 편향된 사전 정보를 도입하여 추가적인 아티팩트를 유발할 수 있다. GloReDi의 전역 표현과 확산 모델의 생성 능력을 결합하는 하이브리드 접근법이 유망한 연구 방향이다.

#### ⑤ 임상 타당성 검증의 강화
실험이 DeepLesion 및 AAPM-Myo 데이터셋 위주로 이루어졌으므로, 실제 임상 환경의 다양한 해부학적 구조 및 병변 유형에 대한 광범위한 검증이 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 유형 | 핵심 기법 | 강점 | 한계 |
|---|---|---|---|---|
| **GloReDi** (ICCV 2023) | 이미지 도메인 | FFC + 중간 뷰 증류 | 전역 수용 영역, 스캐너 독립적 | 중간 뷰 데이터 필요 |
| **ProCT** (2023) | 이미지 도메인 (All-in-one) | Prompted Contextual Transformer | 다양한 불완전 뷰 설정 통합 처리 | 프롬프트 해석 가능성 부족 |
| **QN-Mixer** (2024) | 딥 언롤링(Deep Unrolling) | Quasi-Newton + MLP-Mixer | 빠른 수렴, 낮은 시간 복잡도 | ultra-sparse 데이터에서 수렴 이슈 |
| **DPMA** (2025) | 이중 도메인 | 잔차 정규화 + 멀티스케일 어텐션 | 데이터 일관성 보장 | 사이노그램 접근 필요 |
| **3DGR-CT** (2025) | INR (Implicit) | 3D Gaussian Splatting | 자기지도, 분포 이탈 강인성 | 학습 시간이 딥러닝 방법보다 길다 |
| **CoreDiff** (TMI 2024) | 확산 모델 | 상황 오류 변조 확산 모델 | 저선량 CT 일반화 강력 | 추론 속도 느림 |

불완전 뷰 CT는 투영 샘플링 방식에 따라 Sparse-view CT와 Limited-angle CT로 나뉘며, 기존 방법들은 이러한 설정들을 개별적으로 처리하여 높은 계산·저장 비용을 유발하고 새로운 설정에 대한 유연한 적응을 방해한다.

최근의 Sparse-view CT 재건 발전은 주로 두 가지 딥러닝 방법인 후처리(post-processing)와 이중 도메인 접근법을 활용하고 있으며, 후처리 방법에는 RedCNN, FBPConvNet, DDNet 등이 포함된다.

---

## 📚 참고 자료 및 출처

1. **Li, Z., Ma, C., Chen, J., Zhang, J., & Shan, H.** (2023). *Learning to Distill Global Representation for Sparse-View CT*. ICCV 2023, pp. 21196–21207.
   - arXiv: https://arxiv.org/abs/2308.08463
   - ICCV Open Access: https://openaccess.thecvf.com/content/ICCV2023/html/Li_Learning_to_Distill_Global_Representation_for_Sparse-View_CT_ICCV_2023_paper.html
   - IEEE Xplore: https://ieeexplore.ieee.org/document/10376896/
   - GitHub: https://github.com/longzilicart/GloReDi
   - DeepAI: https://deepai.org/publication/learning-to-distill-global-representation-for-sparse-view-ct
   - ar5iv (HTML 풀텍스트): https://ar5iv.labs.arxiv.org/html/2308.08463

2. **Gao, Q., et al.** (2024). *CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization*. IEEE Transactions on Medical Imaging.

3. **ProCT** (2023). *Universal Incomplete-View CT Reconstruction with Prompted Contextual Transformer*. arXiv: https://arxiv.org/html/2312.07846v1

4. **QN-Mixer** (2024). *QN-Mixer: A Quasi-Newton MLP-Mixer Model for Sparse-View CT Reconstruction*. arXiv: https://arxiv.org/html/2402.17951v3

5. **DPMA** (2025). *Dual-Domain deep prior guided sparse-view CT reconstruction with multi-scale fusion attention*. Scientific Reports. https://www.nature.com/articles/s41598-025-02133-5

6. **3DGR-CT** (2025). *3DGR-CT: Sparse-view CT reconstruction with a 3D Gaussian representation*. ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S136184152500132X

7. **Sidky, E., et al.** (2022). *Report on the AAPM deep-learning sparse-view CT Grand Challenge*. Medical Physics, 49(8). https://pmc.ncbi.nlm.nih.gov/articles/PMC9314462/

8. **FreeSeed** (2023). *FreeSeed: Frequency-band-aware and Self-guided Network for Sparse-view CT Reconstruction*. ResearchGate.

9. **Cheng, S., et al.** *Deep Learning for Sparse-View CT Reconstruction: A Survey*. SSRN. https://ssrn.com/abstract=5366340

---

> ⚠️ **정확도 고지**: 본 답변의 모델 구조 세부 사항 및 수식은 공개된 arXiv 논문(ar5iv HTML 버전), ICCV Open Access, GitHub 저장소를 기반으로 작성하였습니다. 일부 세부 수식(손실 함수의 정확한 표기 등)은 원문 PDF에서 확인하실 것을 권장합니다. 원문 논문의 수식 접근이 제한되어 개념적 수식으로 표현한 부분이 있음을 밝힙니다.
