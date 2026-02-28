# Normalized Metal Artifact Reduction (NMAR) in Computed Tomography

## 논문 종합 분석

---

## 1. 핵심 주장과 주요 기여 요약

이 논문은 CT(Computed Tomography) 영상에서 금속 임플란트로 인한 **금속 아티팩트(metal artifact)**를 효과적으로 제거하기 위한 **정규화 기반 금속 아티팩트 감소(Normalized Metal Artifact Reduction, NMAR)** 기법을 제안한다.

**핵심 주장:**
- 기존 선형 보간(MAR1)이나 단순 길이 정규화(MAR2)는 금속 트레이스 내 보간 시 **새로운 아티팩트를 도입**하거나 **뼈(bone) 구조물 근처의 대조도(contrast)를 훼손**하는 한계가 있다.
- 다중 임계값 분할(multi-threshold segmentation)로 생성한 **삼분(ternary) 영상**의 순방향 투영(forward projection)을 이용하여 사이노그램을 정규화(normalization)하면, 보간 전 사이노그램이 평탄(flat)해져서 보간의 정확도가 극대화된다.

**주요 기여:**
1. **일반화된 정규화 기법**: 공기(air), 물(water), 뼈(bone) 등 다중 물질을 구분하는 삼분 영상 기반 정규화 도입
2. **뼈 구조 보존**: 금속 임플란트 인근 미세 골 구조까지 보존 가능
3. **계산 효율성**: 기존 보간 기반 MAR에 최소한의 추가 계산만으로 적용 가능
4. **범용성**: 기존의 모든 사이노그램 보간 기반 MAR 방법에 추가 단계로 통합 가능

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

CT 스캔 시 금속 임플란트(인공 고관절, 척추 고정 나사 등)가 시야(field of measurement)에 포함되면, X선의 **광자 기아(photon starvation)**, **빔 경화(beam hardening)**, **비선형 부분 체적 효과(nonlinear partial volume effects)** 등으로 인해 심각한 **줄무늬 아티팩트(streak artifacts)**가 발생한다. 이러한 아티팩트는 진단 가치를 크게 저하시킨다.

기존 접근법의 문제점:

| 방법 | 약칭 | 한계 |
|------|------|------|
| 선형 보간 기반 MAR [Kalender 1987] | MAR1 | 금속 트레이스 내 에지 정보 손실 → 뼈 경계 흐릿해짐, 새로운 줄무늬 아티팩트 도입 |
| 단순 길이 정규화 [Müller & Buzug 2009] | MAR2 | 공기-물 대조도만 보존, 물-뼈 간 대조도는 여전히 훼손 |

**핵심 문제:** 보간 기반 MAR에서 사이노그램이 비균질(inhomogeneous)할수록 보간 오차가 커지며, 특히 **금속 임플란트 주변의 뼈 구조**에서 정보 손실이 심각하다.

### 2.2 제안하는 방법 (수식 포함)

NMAR은 크게 **분할(Segmentation)** → **정규화(Normalization)** → **보간(Interpolation)** → **역정규화(Denormalization)** 단계로 구성된다.

#### (1) 삼분 영상 생성 (Ternary Image Segmentation)

초기 보정되지 않은 영상 $f^{\text{ini}}$로부터 두 개의 임계값 $t_1$(공기-물 경계), $t_2$(물-뼈 경계)를 사용하여 삼분 영상 $f^{\text{tern}}$을 생성한다:

```math
f_{ij}^{\text{tern}} := \begin{cases} 0 & \text{if } f_{ij}^{\text{ini}} < t_1 \\ \mu_{\text{water}} & \text{else if } f_{ij}^{\text{ini}} < t_2 \\ \mu_{\text{bone}} & \text{else} \end{cases}
```

여기서 $\mu_{\text{water}}$는 물의 평균 감쇠계수, $\mu_{\text{bone}}$은 뼈의 평균 감쇠계수이다.

#### (2) 정규화 (Normalization)

원래 사이노그램(투영 데이터)을 $p = \mathbf{R}f$라 하면 ($\mathbf{R}f$는 스캔 대상 $f$의 라돈 변환), 삼분 영상의 순방향 투영 $p^{\text{tern}} = \mathbf{R}f^{\text{tern}}$으로 나누어 정규화한다:

$$
p^{\text{norm}} = \frac{p}{\mathbf{R}f^{\text{tern}}}, \quad \text{for } p^{\text{tern}} > \epsilon > 0
$$

여기서 $\epsilon$은 영으로 나누는 것을 방지하기 위한 작은 양수이다.

**직관적 해석:** 삼분 영상의 투영값은 각 광선이 통과하는 물질의 종류와 경로 길이에 대한 대략적 정보를 담고 있다. 이로 나누면, 정규화된 사이노그램 $p^{\text{norm}}$은 금속 트레이스를 제외하고 거의 **평탄(flat)**해진다. 평탄한 사이노그램에서의 보간은 원래 비균질한 사이노그램에서의 보간보다 훨씬 정확하다.

#### (3) 보간 및 역정규화

정규화된 사이노그램 $p^{\text{norm}}$에 대해 기존 보간 기반 MAR 연산 $\mathbf{M}$을 적용한 후, 삼분 영상의 투영값을 다시 곱하여 역정규화한다:

$$
p^{\text{corr}} = p^{\text{tern}} \cdot \mathbf{M}p^{\text{norm}} = \mathbf{R}f^{\text{tern}} \cdot \mathbf{M}\frac{p}{\mathbf{R}f^{\text{tern}}}
$$

이 수식에서:
- $p^{\text{corr}}$: 보정된 사이노그램
- $\mathbf{M}$: 보간 기반 MAR 연산자 (예: 선형 보간)
- $\mathbf{R}f^{\text{tern}}$: 삼분 영상의 순방향 투영

### 2.3 모델 구조 (전체 파이프라인)

NMAR은 딥러닝 모델이 아닌 **알고리즘 기반의 전처리-후처리 파이프라인**이다:

```
[원본 CT 영상 f^ini]
       ↓
[금속 임계값 분할 → 금속 마스크 생성]
       ↓
[다중 임계값 분할 → 삼분 영상 f^tern 생성]
       ↓
[순방향 투영 → p^tern = Rf^tern]
       ↓
[원본 사이노그램 p를 p^tern으로 나눔 → p^norm]
       ↓
[p^norm에 대해 금속 트레이스 영역 보간 (MAR 연산 M)]
       ↓
[역정규화: p^corr = p^tern · M(p^norm)]
       ↓
[보정된 사이노그램으로 FBP 재구성 → 최종 영상]
```

**추가 전처리 기법:**
- 분할 전 $f^{\text{ini}}$에 대한 **스무딩(smoothing)** 및 **형태학적 연산(morphological operations)**으로 삼분 영상 정확도 향상
- 금속 트레이스 스무딩을 통한 줄무늬 아티팩트 사전 감소 [1]

### 2.4 성능 향상

#### 시뮬레이션 결과

두 가지 임상 시나리오를 시뮬레이션하였다:
- **고관절 팬텀(Hip phantom)**: 티타늄 인공관절
- **흉부 팬텀(Thorax phantom)**: 강철 척추 고정 나사

시뮬레이션 매개변수: 120 kV, 0.6 mm 슬라이스 두께, 672 채널, 1160 뷰/회전

| 평가 항목 | MAR1 | MAR2 | NMAR |
|----------|------|------|------|
| 줄무늬 아티팩트 제거 | ○ (제거되지만 새로운 줄무늬 도입) | △ (흉부에서만 개선) | ◎ (새 줄무늬 거의 없음) |
| 뼈 구조 보존 | × (CT값이 참조값 대비 크게 저하) | × (여전히 부정확) | ◎ (참조 프로파일에 근접) |
| 사이노그램 형태 보존 | × | × | ◎ (유일하게 보존) |
| 금속 근처 에지 정보 | × (흐릿해짐) | × | ◎ |

**정량적 결과 (프로파일 분석):**
- **사이노그램 프로파일** (Fig. 3): NMAR만이 참조(reference) 사이노그램의 형태를 보존
- **영상 프로파일** (Fig. 4): MAR1, MAR2에서는 뼈의 CT값이 참조값 대비 현저히 낮게 나타남. NMAR은 참조 프로파일에 근접한 값을 보임

#### 임상 데이터 결과

양측 고관절 인공관절 환자 스캔(SOMATOM Sensation 16, 140 kV, 320 mAs):
- **MAR1**: 새로운 줄무늬 아티팩트 도입
- **MAR2**: 여전히 뼈 근처에 아티팩트
- **NMAR**: 새로운 줄무늬가 거의 도입되지 않으며, 임플란트 근처 뼈가 명확히 관찰됨

### 2.5 한계

1. **분할 정확도 의존성**: 삼분 영상의 품질이 전체 성능을 좌우한다. 심한 아티팩트가 있는 초기 영상에서 정확한 분할이 어렵고, 너무 많은 임계값 사용 시 아티팩트가 분할 영상에 전파될 수 있다.
2. **물질 종류 제한**: 기본 구현에서는 3가지 물질(공기, 물, 뼈)만 고려하며, 복잡한 해부학적 구조에서는 부족할 수 있다.
3. **빔 경화 등 물리적 효과의 완전한 보정 불가**: 사이노그램 보간 기반 방법의 근본적 한계로, 금속 트레이스에서 손실된 원래 정보를 완벽히 복원할 수 없다.
4. **정량적 평가 지표 부족**: PSNR, SSIM 등 표준화된 정량적 메트릭이 제시되지 않고, 프로파일 비교에 의존한다.
5. **제한된 검증 범위**: 두 종류의 팬텀과 한 명의 환자 데이터로만 검증하여 통계적 유의성이 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

NMAR의 일반화 성능(generalization capability)과 관련하여 다음과 같은 핵심 사항들을 분석한다:

### 3.1 방법론적 일반화 가능성

NMAR의 가장 중요한 강점은 **n-ary 분할로의 확장 가능성**이다. 논문에서 명시적으로 언급하듯:

> "A generalization to a n-ary version of the method is straightforward."

이는 삼분 영상을 넘어서 다양한 물질(연조직, 지방, 근육, 폐 조직 등)을 구분하는 다분 영상으로 확장할 수 있음을 의미한다. $n$-ary 분할 시의 일반화된 삼분 영상은 다음과 같이 표현할 수 있다:

```math
f_{ij}^{n\text{-ary}} := \begin{cases} \mu_1 & \text{if } f_{ij}^{\text{ini}} < t_1 \\ \mu_2 & \text{else if } f_{ij}^{\text{ini}} < t_2 \\ \vdots \\ \mu_n & \text{else} \end{cases}
```

여기서 $\mu_k$는 $k$번째 물질의 평균 감쇠계수, $t_k$는 $(k-1)$번째와 $k$번째 물질을 구분하는 임계값이다.

### 3.2 일반화 성능 향상을 위한 핵심 요소

#### (1) 사전 지식(Prior Knowledge) 활용
논문은 임플란트 주변 조직에 대한 사전 지식을 활용할 수 있다고 명시한다. 예를 들어, 고관절 임플란트 주변에는 뼈가 있다는 해부학적 사전 지식을 삼분 영상 생성에 반영할 수 있다.

#### (2) 인접 슬라이스 정보 활용
> "Neighboring slices can be also used to get a more reliable result."

3D 문맥 정보를 활용하면 아티팩트가 심한 슬라이스에서도 보다 안정적인 분할이 가능하다. 이는 일반화 성능 향상에 직접적으로 기여한다.

#### (3) 적응적 임계값 설정
다양한 임상 상황(다양한 체형, 임플란트 종류, 스캔 프로토콜)에 대응하기 위해 임계값 $t_1, t_2$를 적응적으로 설정하는 전략이 필요하다.

### 3.3 일반화 성능의 잠재적 제약

| 제약 요소 | 설명 | 개선 방향 |
|---------|------|---------|
| 분할 민감도 | 심한 아티팩트 존재 시 분할 오류 전파 | 딥러닝 기반 분할, 반복적(iterative) 접근 |
| 임계값 고정 | 환자/스캔 조건에 따라 최적 임계값이 상이 | 적응적/자동 임계값 선택 알고리즘 |
| 물질 다양성 | 실제 인체는 연속적 감쇠계수 분포 | 연속적 정규화 맵, 딥러닝 기반 prior 생성 |
| 2D 처리 한계 | 슬라이스 단위 처리로 3D 일관성 부족 | 3D 확장, volumetric processing |

### 3.4 다양한 임상 시나리오에서의 일반화

NMAR은 원리적으로 **임의의 보간 기반 MAR 방법에 플러그인 방식으로 적용** 가능하므로, 다양한 임상 시나리오(치과 임플란트, 정형외과 임플란트, 심장 스텐트 등)로 일반화할 수 있는 구조적 유연성을 갖는다. 정규화 연산 자체가 특정 해부학적 부위나 금속 종류에 의존하지 않기 때문이다.

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### (1) 사이노그램 도메인에서의 정규화 패러다임 확립
NMAR은 "보간 전 사이노그램을 평탄화한다"는 아이디어를 체계화하여, 이후 수많은 연구에서 **prior image 기반 정규화**가 표준적 전처리 단계로 채택되는 데 기여하였다.

#### (2) 딥러닝 기반 MAR 연구의 기초
NMAR의 정규화-역정규화 프레임워크는 이후 딥러닝 기반 MAR 연구에서 **prior image 생성**과 **사이노그램 보정** 전략의 기초가 되었다. 특히, 딥러닝으로 보다 정교한 prior image를 생성하여 NMAR 프레임워크에 적용하는 연구가 활발히 이루어지고 있다.

#### (3) 하이브리드 접근법의 촉진
NMAR의 모듈형(modular) 설계는 통계적 방법, 반복적 재구성, 딥러닝 등 다른 접근법과 결합하는 하이브리드 방법론 연구를 촉진하였다.

### 4.2 향후 연구 시 고려할 점

1. **딥러닝과의 통합**: CNN/U-Net 등을 활용하여 삼분 영상 대신 보다 정교한 prior image를 학습 기반으로 생성하면 분할 오류에 대한 강건성을 높일 수 있다.

2. **반복적(Iterative) NMAR**: 1회 NMAR 후 개선된 영상을 다시 분할하여 반복 적용하면 수렴적으로 품질이 향상될 수 있다 (실제로 후속 논문 iNMAR에서 구현됨 [Meyer et al., 2012]).

3. **정량적 평가 체계 확립**: PSNR, SSIM, RMSE 등 표준 메트릭을 사용한 대규모 정량적 평가가 필요하다.

4. **다중 에너지 CT(Dual-Energy CT)와의 결합**: DECT의 물질 분리 능력을 활용하면 보다 정확한 prior image를 생성할 수 있다.

5. **3D 볼류메트릭 확장**: 인접 슬라이스 정보를 체계적으로 활용하는 3D NMAR 구현

6. **실시간 처리**: 임상 워크플로우에의 실시간 통합을 위한 GPU 가속 구현

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

NMAR(2009) 이후 금속 아티팩트 감소 분야는 **딥러닝 기반 방법론**이 주류가 되었으며, 동시에 NMAR의 정규화 프레임워크를 계승·발전시킨 연구도 활발하다.

### 5.1 주요 최신 연구 비교

| 연구 | 연도 | 핵심 방법 | NMAR 대비 차별점 |
|------|------|---------|-------------|
| **ADN** (Liao et al.) [A] | 2020 | Artifact Disentanglement Network – 아티팩트와 콘텐츠를 비지도 학습으로 분리 | 학습 기반으로 paired 데이터 불필요 |
| **DuDoNet++** (Lyu et al.) [B] | 2021 | Dual-Domain (사이노그램 + 이미지) 동시 보정 네트워크 | 사이노그램과 이미지 도메인을 동시에 학습 |
| **InDuDoNet+** (Wang et al.) [C] | 2023 | Interpretable Dual-Domain Network, NMAR의 정규화를 딥러닝 프레임워크에 명시적으로 통합 | NMAR의 prior normalization을 해석 가능한 딥러닝 모듈로 대체 |
| **IDOL-Net** (Sun et al.) [D] | 2022 | 반복적 이중 도메인 학습 기반 MAR | 여러 iteration으로 사이노그램-이미지 교차 보정 |
| **Score-based Diffusion** (Song et al., Lyu et al.) [E] | 2022–2023 | Score-based generative model을 이용한 MAR | 생성 모델의 강력한 prior를 활용한 결측 데이터 복원 |
| **CycleGAN-based MAR** (Nakamura et al.) [F] | 2020 | 비쌍(unpaired) 학습 기반 이미지 도메인 MAR | 임상 데이터에서 paired data 없이 학습 가능 |

### 5.2 상세 비교 분석

#### (1) NMAR vs. 딥러닝 기반 Dual-Domain 방법 (DuDoNet++, InDuDoNet+)

NMAR은 사이노그램 도메인에서만 보정을 수행하는 반면, 최신 Dual-Domain 방법은 사이노그램과 이미지 도메인을 **동시에 학습**한다.

**InDuDoNet+** [C]는 특히 주목할 만한데, NMAR의 정규화 과정을 **해석 가능한 최적화 알고리즘 전개(algorithm unrolling)** 형태로 딥러닝에 통합하였다:

$$
p^{\text{corr}} = \mathbf{R}f^{\text{prior}} \cdot \mathcal{N}_\theta\left(\frac{p}{\mathbf{R}f^{\text{prior}}}\right)
$$

여기서 $f^{\text{prior}}$는 딥러닝으로 생성한 prior image, $\mathcal{N}_\theta$는 학습 가능한 사이노그램 보정 네트워크이다. 이는 NMAR의 수동 분할 기반 정규화를 학습 기반으로 대체하여 일반화 성능을 크게 향상시켰다.

#### (2) NMAR vs. 생성 모델 기반 접근법

Score-based diffusion model [E]은 금속으로 인해 손상된 사이노그램 영역을 **조건부 생성(conditional generation)**으로 복원한다. NMAR의 보간이 단순 선형/다항식 기반인 것에 비해, 학습된 데이터 분포에서 샘플링하므로 보다 사실적인 보정이 가능하다. 다만 계산 비용이 NMAR 대비 수백~수천 배 높다.

#### (3) 일반화 성능 비교

| 항목 | NMAR | 딥러닝 기반 (DuDoNet++ 등) | 비고 |
|------|------|----------------------|------|
| 학습 데이터 필요 | 불필요 | 대량 필요 | NMAR은 규칙 기반 |
| 새로운 임플란트 유형 대응 | 즉시 적용 가능 | 재학습 필요 가능 | NMAR의 장점 |
| 복잡한 해부학 구조 | 분할 정확도에 의존 | 학습 데이터 분포에 의존 | 둘 다 한계 존재 |
| 계산 비용 | 매우 낮음 | 높음 (GPU 필요) | 임상 실시간 적용 시 NMAR 유리 |
| 뼈 CT값 정확도 | 양호 | 우수 | 딥러닝이 일반적으로 우위 |
| 도메인 외(out-of-distribution) 데이터 | 강건 | 취약할 수 있음 | 데이터 기반 방법의 근본적 한계 |

### 5.3 최신 동향 종합

2020년 이후의 연구 흐름은 크게 세 가지로 요약된다:

1. **NMAR 프레임워크의 딥러닝 내재화**: NMAR의 정규화 아이디어를 딥러닝 아키텍처에 통합하여 해석 가능성과 성능을 동시에 추구 (InDuDoNet+ 등)

2. **Dual-Domain 학습의 보편화**: 사이노그램만 또는 이미지만 처리하는 것이 아닌, 양 도메인을 동시에 학습하는 것이 표준이 됨

3. **비지도/자기지도 학습으로의 전환**: 임상에서 paired clean/corrupted 데이터 확보가 어려운 현실을 반영하여, unpaired 학습이나 self-supervised 학습 방법 연구가 증가

**결론적으로**, NMAR은 2009년에 제안된 방법임에도 불구하고, 그 정규화 패러다임은 2020년대 최신 딥러닝 기반 MAR 연구에까지 핵심 구성 요소로 계승되고 있다. NMAR의 **계산 효율성**, **학습 데이터 불필요**, **범용적 적용 가능성**은 여전히 중요한 장점이며, 딥러닝 방법과의 **상호보완적 통합**이 향후 가장 유망한 연구 방향으로 보인다.

---

## 참고자료 및 출처

**논문 원문:**
- E. Meyer, F. Bergner, R. Raupach, T. Flohr, and M. Kachelrieß, "Normalized Metal Artifact Reduction (NMAR) in Computed Tomography," *2009 IEEE Nuclear Science Symposium Conference Record (M09-206)*, pp. 3251–3255, 2009. (IEEE Xplore)

**논문 내 인용 참고문헌:**
- [1] J. Müller and T. M. Buzug, "Spurious structures created by interpolation-based CT metal artifact reduction," *SPIE Medical Imaging Proc.*, vol. 7258, 2009.
- [3] W. A. Kalender, R. Hebel, and J. Ebersberger, "Reduction of CT artifacts caused by metallic implants," *Radiology*, vol. 164, no. 2, 1987.

**2020년 이후 최신 연구 참고자료:**
- [A] H. Liao et al., "ADN: Artifact Disentanglement Network for Data-Free Metal Artifact Reduction," *IEEE Transactions on Medical Imaging*, vol. 39, no. 3, pp. 634–643, 2020.
- [B] Y. Lyu et al., "DuDoNet++: Encoding Mask Projection to Reduce CT Metal Artifact," *arXiv preprint*, 2021.
- [C] H. Wang et al., "InDuDoNet+: A Deep Unfolding Dual Domain Network for Metal Artifact Reduction in CT Images," *Medical Image Analysis*, vol. 85, 2023.
- [D] T. Sun et al., "IDOL-Net: An Interactive Dual-Domain Parallel Network for CT Metal Artifact Reduction," *IEEE Transactions on Radiation and Plasma Medical Sciences*, 2022.
- [E] Y. Song et al., "Solving Inverse Problems in Medical Imaging with Score-Based Generative Models," *ICLR 2022*; W. Lyu et al., "Score-based Diffusion Models for Metal Artifact Reduction in CT," *MICCAI 2023*.
- [F] Y. Nakamura et al., "Non-Paired Data-Based Metal Artifact Reduction Using CycleGAN," *Medical Physics*, 2020.
- E. Meyer, R. Raupach, M. Lell, B. Schmidt, and M. Kachelrieß, "Frequency split metal artifact reduction (FSMAR) in computed tomography," *Medical Physics*, vol. 39, no. 4, pp. 1904–1916, 2012. (iNMAR/FSMAR 후속 연구)

> **참고:** 위 최신 연구 비교 분석에서 일부 논문의 정확한 출판 세부사항(권, 호, 페이지 등)은 제 학습 데이터 범위 내에서의 정보에 기반하며, 최신 게재 상태는 해당 출판사 웹사이트에서 직접 확인하시기를 권장합니다.
