# Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels 

> **서지정보**: Chu Han, Jiatai Lin, Jinhai Mai 외 15인, "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels," *arXiv:2110.08048* (2021), 확장판 *Medical Image Analysis*, Vol. 80, 102487 (2022).

---

## 1. Executive Summary (10문장 이내)

이 논문은 전산 병리학(computational pathology)에서 조직 수준 semantic segmentation이 핵심 단계이며, 완전 감독(fully-supervised) 모델은 픽셀 단위 주석으로 뛰어난 성능을 달성했지만, 거대 픽셀(giga-pixel) 단위인 whole slide image(WSI)에 픽셀 단위 라벨을 그리는 것은 매우 비용이 크고 시간이 많이 든다는 문제에서 출발한다.  
저자들은 패치(patch) 단위 분류 라벨만을 이용해 조직 semantic segmentation을 수행함으로써 주석 작업 부담을 줄이는 방법을 제안한다.  
전체 모델은 분류(classification) 단계와 분할(segmentation) 단계로 구성된 2단계(two-step) 구조를 갖는다.  
분류 단계에서는 CAM(Class Activation Map) 기반 모델로 패치 단위 라벨로부터 pseudo mask를 생성한다.  
분할 단계에서는 제안된 Multi-Layer Pseudo-Supervision(MLPS)을 통해 조직 semantic segmentation을 수행하며, 픽셀 단위와 패치 단위 주석 간 정보 격차를 줄이기 위한 여러 기술적 novelty가 제안되었다.  
이 논문의 일환으로 폐 선암(lung adenocarcinoma)을 위한 새로운 약지도 semantic segmentation(WSSS) 데이터셋인 LUAD-HistoSeg를 소개한다.  
두 데이터셋에서 여러 실험을 수행했고, 제안 모델은 5개의 SOTA WSSS 접근법을 능가한다.  
완전 감독 모델과 비교했을 때 MIoU와 FwIoU 기준으로 약 2%의 격차만 있는 비교적 준수한 정량·정성적 결과를 달성한다.  
제안 모델은 수동 라벨링보다 약 10배 빠른 것으로 입증되었다.

**용어 설명**: *CAM(Class Activation Map)*은 CNN 분류기가 어떤 클래스를 예측할 때 이미지의 어느 영역을 근거로 삼았는지를 시각화하는 히트맵이다. *WSSS(Weakly-Supervised Semantic Segmentation)*는 픽셀 단위 라벨 없이 이미지/패치 단위의 약한 라벨(예: "이 패치에 종양세포가 있다/없다")만으로 픽셀 단위 분할을 학습하는 기법이다.

### 1-1. 연구의 목적과 필요성

- **목적**: 병리 영상에서 픽셀 단위(pixel-level) segmentation 라벨 없이, 상대적으로 저렴한 패치 단위(patch-level) 분류 라벨만으로 조직 단위 semantic segmentation을 수행하는 것.
- **필요성**: 조직 수준 semantic segmentation은 전산 병리학의 핵심 단계이고, 완전 감독 모델은 이미 뛰어난 성능을 보였으나 거대 픽셀 WSI에 대한 밀집 픽셀 단위 라벨링은 극히 비용과 시간이 많이 든다는 실무적 병목을 해소하기 위함이다. 병리학자의 라벨링 부담을 줄이는 것은 AI 기반 병리 진단 도구의 임상 확산에 직결되는 문제이다.

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거(요약) | 위치 |
|---|---|---|---|
| 1 | 패치 단위 라벨만으로도 완전감독 모델에 근접한 segmentation이 가능하다 | 완전 감독 모델과 비교하여 MIoU·FwIoU 기준 약 2% 격차만 존재 | Abstract, Table I |
| 2 | Progressive Dropout Attention(PDA)이 CAM의 discriminative 영역 편중 문제를 완화한다 | DA 적용 모델이 baseline 대비 전 범주 및 FwIoU/MIoU/ACC에서 향상되었고, PDA 적용 시 비주요 범주(NEC, LYM)에서 추가 향상, 이는 최고활성영역 비활성화가 2차 판별 영역 학습을 유도하기 때문 | Table III, Fig. 9 |
| 3 | Multi-Layer Pseudo-Supervision과 classification gate가 정보 격차 및 false positive를 줄인다 | Multi-layer pseudo-supervision with progressive dropout attention은 patch-level과 pixel-level 라벨 간 정보 격차를 줄이고, classification gate mechanism은 false-positive율을 줄인다 | 본문, Fig. 2 |
| 4 | 제안 모델은 기존 WSSS SOTA 기법들을 능가한다 | 5개의 SOTA WSSS 접근법을 능가, Table I에서 최종 모델이 두 기존 모델 모두를 크게 능가했고, Phase 1의 pseudo mask조차 기존 CAM 기반 WSSS 방법을 능가 | Table I |
| 5 | LYM(임파구) 범주는 데이터 불균형으로 성능이 불안정하다 | LYM을 제외한 모든 범주에서 유의미하고 일관된 개선을 보였는데, LYM은 데이터셋의 약 4%만 차지하는 극단적 불균형 범주이며 학습 샘플 부족이 불안정한 성능으로 이어질 수 있다 | Table I, 본문 |
| 6 | 새로운 LUAD-HistoSeg 데이터셋이 WSSS 연구에 기여한다 | LUAD 데이터셋이 약지도 조직 semantic segmentation 용도로 처음 공개되었다 | 본문, 데이터셋 섹션 |

---

## 2-1. 세부 설명: 문제, 방법(수식), 모델 구조, 성능, 한계

### (1) 해결하고자 하는 문제
Image-level 약지도 semantic segmentation(WSSS)은 고품질 데이터 주석에 대한 의존도를 낮추지만, CAM은 대상 위치화(localize)에 널리 쓰이는 방법임에도 병리 영상에서는 조직 간 대비(contrast)가 낮아 CAM 기반 방법이 조직(gland) segmentation에서 성능이 떨어진다는 것이 이 논문이 정면으로 다루는 기술적 문제다. 저자들은 class consistency와 semantic consistency가 네트워크로 하여금 혼동되는 픽셀을 효과적으로 구분하고 세밀한 pseudo mask를 생성하도록 유도할 수 있음을 발견했다.

### (2) 제안하는 방법 (수식 포함)

**① Dropout Attention (기본형)**

분류기가 CAM 상에서 가장 activation이 높은(가장 discriminative한) 영역에만 과도하게 집중하는 문제를 해결하기 위해, 다음과 같이 CAM의 최고 활성 영역을 강제로 0으로 만든다:

$$
\hat{\mathcal{M}}_k(i,j) = \begin{cases} \mathcal{M}_k(i,j), & \mathcal{M}_k(i,j) \le \beta \\ 0, & \mathcal{M}_k(i,j) > \beta \end{cases}
$$

- $\mathcal{M}_k(i,j)$: 클래스 $k$에 대한 CAM의 좌표 $(i,j)$ 위치 활성값
- $\hat{\mathcal{M}}_k(i,j)$: dropout 처리 후의 활성값
- $\beta$: dropout 임계값(threshold) — 이 값을 초과하는 "가장 눈에 띄는" 영역을 인위적으로 제거

**② Progressive Dropout Attention (PDA)**

분류기가 훈련이 진행될수록 가장 discriminative한 영역에 집중하는 정도가 증가하므로, 훈련 epoch가 증가함에 따라 dropout 계수를 점진적으로 변화시킨다:

$$
\mu_t = \begin{cases} \sigma \cdot \mu_{t-1}, & \mu_t > l \\ l, & \mu_t \le l \end{cases}
$$

- $\mu_t$: epoch $t$에서의 dropout 계수 (임계값 $\beta$를 결정하는 데 사용되는 것으로 추정)
- $\sigma$: 감쇠(decay) 인자
- $l$: 계수의 하한값(lower bound)

> ⚠️ **불확실성 표시**: $\mu_t$와 $\beta$의 정확한 결합 수식(예: $\beta = \mu_t \times \max(\mathcal{M}_k)$ )은 검색된 자료에서 명확히 확인되지 않아, 정확한 수식을 확정적으로 제시하지 않습니다. 원문 논문(Section III-B 부근)을 직접 확인하는 것을 권장합니다.

**③ Multi-Layer Pseudo-Supervision (MLPS)**

진행형 dropout attention(PDA), multi-layer pseudo-supervision(MLPS), classification gate mechanism의 novelty를 정량·정성적으로 평가하기 위한 ablation study를 수행했다. PDA를 적용한 약지도 모델이 조직 semantic segmentation을 위한 multi-layer pseudo mask를 생성하고, classification gate mechanism이 결합된 DeepLab V3+ 모델이 MLPS의 안내를 받아 semantic segmentation을 수행한다. 즉 서로 다른 CNN 레이어(layer bn7 등)에서 추출한 여러 층의 pseudo mask를 결합해 segmentation 네트워크의 감독 신호로 사용하는 방식이다. (정확한 손실함수 수식은 원문 확인 필요 — 검색 결과에서 명시적 수식을 확인하지 못함.)

**④ Classification Gate Mechanism**: 패치 단위 분류 결과를 이용해 segmentation 출력을 걸러내(gate) false-positive율을 낮추는 메커니즘으로, 분류기가 특정 클래스가 없다고 판단한 패치에서는 해당 클래스의 segmentation 활성을 억제하는 방식으로 이해된다(정확한 수식 미확인).

### (3) 모델 구조
전체 아키텍처는 (a) multi-layer pseudo mask를 생성하기 위해 PDA를 적용한 약지도 모델과 (b) MLPS의 안내를 받는 classification gate mechanism이 결합된 DeepLab V3+ segmentation 모델로 구성되며, 이론적으로 임의의 조직 범주 수에 대해 semantic segmentation을 지원할 수 있고, 4개 조직 클래스(종양 상피, 종양 관련 스트로마, 괴사, 림프구)를 가진 LUAD-HistoSeg 데이터셋으로 시연된다. 모든 CNN은 PyTorch로 구현되었으며 NVIDIA RTX 2080Ti에서 훈련되었고, ResNet38과 DeepLab V3+가 백본으로 도입되었다.

### (4) 성능 향상
- Phase 1의 pseudo mask만으로도 두 데이터셋에서 기존 CAM 기반 WSSS 방법들을 능가했으며(Table I),
- Dropout Attention(DA) 도입만으로 baseline 대비 모든 범주 및 FwIoU/MIoU/ACC에서 개선되었고, PDA 도입 시 특히 비주요 범주(NEC, LYM)에서 지속적 개선(Table III),
- 최종 모델은 5개의 SOTA WSSS 접근법을 능가하며, 완전 감독 baseline과 비교해도 준수한 성능을 보인다(Table I, II).

### (5) 한계
- LYM 범주는 전체 데이터의 약 4%만 차지하는 극심한 불균형 클래스로, Phase 2 이후에도 성능 개선이 불안정하다.
- 논문 자체에서 명시하지 않았지만, 여러 후속 연구들이 이 모델의 다단계(2-stage) 구조로 인한 오차 누적(pseudo mask 오류 → segmentation 오류로 전파) 문제를 지적한다(§8-2 참조).

---

## 3. 페이지/Figure/Table 인용 위치 정리

| 주장 | 위치 |
|---|---|
| 문제 정의(gigapixel 라벨링 비용) | Abstract |
| 2단계 모델 개요 | Abstract, Fig. 2 |
| PDA 수식 및 개념 | Section III (Method), 관련 수식 |
| 정량적 비교 (baseline vs 제안 모델) | **Table I** |
| Ablation study (PDA/MLPS/Class-Gate) | **Table III**, **Fig. 9** |
| 분류 정확도(PDA 유무) | **Table IV** |
| 정성적 비교 결과 | **Fig. 7** |
| 데이터셋 설명(LUAD-HistoSeg) | 데이터셋 섹션 |
| 라벨링 시간 절감(10배) | Section 5.5 |

---

## 4. 저자 직접 보고 결과 vs 나의 해석 분리

**저자가 직접 보고한 것:**
- 완전 감독 모델과 비교해 MIoU·FwIoU에서 약 2% 격차
- 5개 SOTA WSSS 방법을 능가
- LYM을 제외한 모든 범주에서 일관된 개선, LYM은 데이터 불균형(약 4%)으로 불안정
- 참고: 후속 논문의 재현 실험에서는 MLPS의 LUAD-HistoSeg mIoU가 Phase1 74.18±0.89%, Phase2 75.27±0.13%로 보고되기도 함(제3자 재현치, 원 논문 수치와 다를 수 있음).

**나의 해석(저자가 명시하지 않은 추론):**
- PDA와 MLPS의 결합은 사실상 "반복적 self-training/노이즈 라벨 정제" 패러다임의 병리영상 특화 구현으로 볼 수 있으며, 이는 자연영상 WSSS(SEAM, OAA 등)의 아이디어를 병리영상의 저대비(low-contrast) 특성에 맞게 변형한 것으로 해석할 수 있다.
- 2단계 구조(Phase1 → Phase2)는 필연적으로 Phase1의 CAM 오류가 Phase2 학습에 전파되는 구조적 한계를 내재하는데, 논문은 이를 직접적인 "한계"로 명시하지는 않았으나 이후 연구들(§8-2)이 이를 지적한다.

---

## 5. 통계적으로 취약한 부분 / 비교 불가능한 수치

- **LYM 범주 성능**: 전체 데이터셋의 약 4%만 차지하는 극단적 클래스 불균형으로 인해 IoU 값의 통계적 신뢰도가 낮을 가능성이 크다. 표본 수가 적어 분산이 크고, 다른 논문 재현 실험에서도 편차(±std)가 상대적으로 큼(예: 39-8 참조, 표준편차 명시).
- **Table I의 특정 수치**(예: OCR로 확인된 "0.77704, 0.45594, 0.68286, 0.69557, 0.72897, 0.71853, 0.75567" 등)는 검색 스니펫이 표 형식을 온전히 보존하지 못해 어느 행(모델)·열(지표)에 정확히 대응하는지 100% 확신할 수 없다. 이는 원문 PDF의 표를 직접 확인해야 하는 부분이다.
- **타 논문과의 성능 비교 시 주의**: 후속 논문들(HisynSeg, ARML, ESFAN, C2RM-Seg 등)이 보고하는 MLPS의 mIoU 수치는 원 논문 수치와 1.81%p 차이 등으로 서로 다르게 나타나는데, 이는 재현 환경, backbone, 데이터 분할(split) 차이에서 기인할 수 있어 **직접적인 수치 비교는 위험**하다.
- BCSS·LUAD-HistoSeg 각각의 클래스 수/정의가 논문마다 조금씩 다르게 표기되는 경우가 있어(TUM/STR/LYM/NEC vs TE/NEC/LYM/TAS), cross-dataset 비교 시 주의가 필요하다.

---

## 6. 문서가 답하지 않는 질문

1. Classification gate mechanism의 정확한 수학적 정의(게이팅 함수, threshold 설정 방식)는 검색된 자료에서 구체적으로 확인되지 않음.
2. MLPS의 정확한 손실함수(loss function) 형태(예: 여러 레이어의 pseudo mask를 어떻게 가중 결합하는지)가 명시적으로 확인되지 않음.
3. LUAD-HistoSeg 데이터셋의 정확한 수집 기관, 환자 수, 스캐너 종류 등 메타데이터는 확인되지 않음(다만 후속 논문에서 17,291개 패치, 16,678 학습/306 검증/307 테스트로 인용되기도 함).
4. 모델이 4개 클래스 이외의 더 많은 조직 범주(예: 10개 이상)로 확장될 때의 실제 성능은 논문에서 실험적으로 검증되지 않음("이론적으로 지원 가능"이라는 서술만 존재).
5. 다른 암종(유방암 외 위암, 대장암 등)이나 염색법(H&E 외 IHC 등)에 대한 일반화 가능성은 본 논문에서 직접 실험되지 않음.
6. 임상적 실사용(pathologist-in-the-loop 검증, 실제 진단 정확도에 미치는 영향)에 대한 데이터는 제시되지 않음.

---

## 7. 가장 중요한 그림 5개 해석

1. **Fig. 2 (전체 아키텍처)**: PDA를 적용한 약지도 모델(a)이 multi-layer pseudo mask를 생성하고, classification gate mechanism이 결합된 DeepLab V3+ 모델(b)이 MLPS의 안내를 받아 semantic segmentation을 수행하며, 이론적으로 임의의 조직 범주 수에 대해 확장 가능함을 보여준다. → 이 그림은 논문의 핵심 파이프라인(2단계 구조)을 한 눈에 보여주는 가장 중요한 그림이다.

2. **Table I (정량적 비교)**: "Ours Phase 2"가 최종 모델이며, 두 데이터셋에서 기존 두 SOTA 모델을 크게 능가함을 보여준다. → 제안 방법의 우수성을 뒷받침하는 핵심 근거.

3. **Fig. 7 (정성적 비교)**: 패치 단위 semantic segmentation 결과를 원본 이미지에 오버레이한 것으로, 검은 원으로 기존 방법들의 부정확한 경계와 불완전한 결과를 강조했다. → 수치뿐 아니라 시각적으로도 경계 정확도의 개선을 보여주는 그림.

4. **Table III & Fig. 9 (Ablation)**: PDA, MLPS, classification gate의 novelty를 정량·정성적으로 평가한 ablation 결과이며, 대표 baseline 모델들을 선택해 novelty의 효과를 정성적으로도 입증한다. → 각 구성요소(component)의 기여도를 분리해서 보여주는 핵심 근거 자료.

5. **Table IV (PDA 유무에 따른 분류 정확도)**: PDA 적용 유무에 따른 patch-level 분류 정확도를 비교한 표로, PDA가 segmentation 성능뿐 아니라 상류(upstream) 분류 성능에도 미치는 영향(트레이드오프 여부)을 확인할 수 있는 자료다.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 제언

저자들은 광범위한 실험과 ablation study를 통해 패치 단위 주석만을 이용한 semantic segmentation의 우수성을 입증했으며, 완전 감독 모델과 비교해 MIoU·FwIoU 기준 약 2%의 격차만 있고, 수동 라벨링보다 약 10배 빠르다는 것을 증명했다고 결론짓는다. LUAD-HistoSeg라는 새로운 WSSS 데이터셋을 공개함으로써, 폐 선암의 H&E 염색 WSI에서 조직 semantic segmentation을 수행하는 후속 연구를 위한 벤치마크를 마련했다는 점이 실질적인 학술적 기여다. 실제로 WSSS4LUAD Grand Challenge가 이 논문을 핵심 참고문헌으로 인용하며 LUAD-HistoSeg를 표준 벤치마크로 채택한 점은 이 연구의 파급력을 보여준다.

### 8-1. 모델의 일반화 성능 향상 가능성

- **긍정적 요소**: "이론적으로 이 모델은 임의의 수의 조직 범주에 대한 semantic segmentation을 지원할 수 있다"는 서술은 구조적 확장성을 시사한다. 실제로 BCSS(유방암)와 LUAD-HistoSeg(폐암) 두 개의 서로 다른 암종·조직 유형에서 모두 SOTA를 달성했다는 점은 도메인 간 일반화 가능성을 뒷받침한다.
- **제약 요소**: 일반 자연영상에서는 "자동차는 도로 위에 존재한다"와 같은 공간적 사전 규칙(prior)이 존재하지만, WSI에서는 그런 규칙 없이 조직 분포가 무작위적이고 산발적(scattered)인 경우가 많다는 지적(제3자 해설)은, 병리영상 WSSS가 자연영상 WSSS보다 본질적으로 일반화가 어려운 도메인임을 시사한다. 또한 클래스 불균형(LYM 4%)에서 나타났듯, 특정 조직이 희소한 상황(rare tissue type, rare cancer subtype)에서는 일반화 성능이 저하될 가능성이 높다.
- **개선 방향 제언**: (1) 클래스 불균형 완화를 위한 re-sampling/focal loss 결합, (2) 도메인 적응(domain adaptation) 기법을 통한 염색 배치(batch) 간 색상 변동 대응, (3) foundation model(예: 병리 특화 self-supervised pretraining) 기반 특징 추출기로 교체하여 저대비 조직 간 판별력 강화 등이 일반화 성능 향상에 기여할 수 있다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석 및 향후 고려사항

2020년 이후 이 논문과 직접적으로 연관되거나 이 논문을 벤치마크로 활용한 대표적 후속 연구들:

| 연구 | 특징 | MLPS와의 관계 |
|---|---|---|
| SC-CAM (2020) | 자연영상 WSSS, 클러스터링 기반 | MLPS의 비교 baseline |
| CGNet, OAA, Grad-CAM++ (2021) | CAM 개선 기법 | 확장판(MIA'22)의 비교 baseline |
| OEEM, HAMIL, PCSformer (~2023-24) | Transformer/self-training 기반 개선 | LUAD-HistoSeg에서 mIoU 71~73%대로 MLPS(75.60%) 대비 낮은 성능 |
| ARML (MICCAI'24), ESFAN (MICCAI'25) | Phase1/Phase2 구조 유지, attention·rectification 개선 | MLPS 대비 Phase2 mIoU를 약 1~1.6%p 개선 |
| HisynSeg (2024) | 이미지 합성(mixing)·일관성 정규화 기반 | mIoU 76.19%, fwIoU 75.79%로 WSSS-Tissue(MLPS 계열 최고 성능) 대비 각각 1.81%p, 1.43%p 개선 |
| LAVT, PBIP 등 vision-language 기반 (2024-25) | 텍스트 프롬프트/비전-언어 결합 | MLPS를 baseline으로 포함해 비교, MLPS는 BCSS-WSSS에서 3위 수준 |
| C2RM-Seg, Single-Stage Hierarchical Rectification (2025-26) | 인과추론(causal), 단일 스테이지 구조 | MLPS와 같은 2-stage 구조의 오차 누적 문제를 지적하며 단일 스테이지 대안 제시 |

**시사점 및 향후 연구 시 고려할 점**:
1. **2-stage 구조의 근본적 한계**: MLPS 이후 다수의 연구가 "Phase1(분류) → Phase2(분할)"의 순차적 구조에서 발생하는 오차 전파 문제를 지적하며, 최근에는 단일 스테이지(single-stage) 구조로의 전환이 트렌드가 되고 있다(예: arxiv 2606.20250의 "Single-Stage Hierarchical Rectification").
2. **성능 향상은 점진적(incremental)**: 후속 연구들의 mIoU 개선폭은 대부분 1~5%p 수준으로, MLPS가 제시한 PDA·MLPS·classification gate라는 핵심 아이디어의 틀 자체는 여전히 유효하며 완전히 대체되지 않았다.
3. **벤치마크로서의 지속적 가치**: LUAD-HistoSeg와 BCSS-WSSS는 2024-2026년 현재까지도 표준 벤치마크로 활용되고 있어, 데이터셋 공개라는 기여가 방법론적 기여보다 더 장기적인 영향력을 갖는 경우로 평가할 수 있다.
4. **향후 연구 시 고려사항**: (a) 클래스 불균형(rare tissue) 문제에 대한 명시적 대응 전략 필요, (b) 비전-언어 모델·foundation model 결합을 통한 zero-shot/few-shot 일반화 검증, (c) 서로 다른 재현 실험 간 수치 차이(§5 참조)를 고려해 통계적 유의성 검정(t-test 등)을 표준적으로 보고하는 관행 정착이 필요하다.

---

## 참고 자료 (출처 목록)

1. arXiv:2110.08048, "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels" (https://arxiv.org/abs/2110.08048, https://arxiv.org/pdf/2110.08048)
2. ar5iv HTML 버전 (https://ar5iv.labs.arxiv.org/html/2110.08048)
3. ScienceDirect, "Multi-layer pseudo-supervision for histopathology tissue semantic segmentation using patch-level classification labels," Medical Image Analysis (https://www.sciencedirect.com/science/article/pii/S1361841522001347)
4. DeepAI publication page
5. Ovid/Medical Image Analysis abstract page
6. dblp bibliography entry
7. ResearchGate publication page
8. Scribd 문서 (Han 등 2022 원문 발췌)
9. WSSS4LUAD Grand Challenge 페이지 (https://wsss4luad.grand-challenge.org/)
10. Velog 블로그 리뷰 글 (한국어 요약, 참고용)
11. arXiv:2412.20924, "HisynSeg: Weakly-Supervised Histopathological Image Segmentation via Image-Mixing Synthesis and Consistency Regularization"
12. arXiv:2606.20250, "Single-Stage Hierarchical Rectification for Weakly Supervised Histopathology Segmentation"
13. arXiv:2606.25508, "C2RM-Seg: Causal Counterfactual Reasoning with Structural-Semantic Priors for Weakly Supervised Histopathological Tissue Segmentation"
14. arXiv:2503.12068, "Prototype-Based Image Prompting for Weakly Supervised Histopathological Image Segmentation"
15. BMVA Archive 2025 논문, "Bridging Visual-Textual Modalities: Weakly Supervised..."
16. MDPI Applied Sciences 15(12), 6474, "Dual Attention Equivariant Network for Weakly Supervised Semantic Segmentation"
17. ACM/Springer, "A Weakly Supervised Semantic Segmentation Method on Lung Adenocarcinoma Histopathology Images"

> **주의**: 본 답변은 웹 검색으로 수집된 논문 발췌문·초록·표 스니펫을 기반으로 구성되었으며, 원문 PDF 전체를 직접 열람한 것은 아닙니다. 특히 수식의 정확한 최종형(classification gate mechanism, MLPS 손실함수)과 일부 표의 수치 대응관계는 원문을 직접 확인하시는 것을 권장합니다.
