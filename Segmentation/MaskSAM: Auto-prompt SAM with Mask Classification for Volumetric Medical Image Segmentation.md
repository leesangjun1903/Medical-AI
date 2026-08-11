# MaskSAM: Auto-prompt SAM with Mask Classification for Volumetric Medical Image Segmentation

> **참고 자료:**
> - Xie et al., "MaskSAM: Auto-prompt SAM with Mask Classification for Volumetric Medical Image Segmentation," arXiv:2403.14103v2, 2025.
> - 논문 내 인용 문헌 전체 (References [1]–[47])
> - 관련 배경 지식: SAM (Kirillov et al., 2023), MaskFormer (Cheng et al., NeurIPS 2021), DETR (Carion et al., ECCV 2020), nnUNet (Isensee et al., 2019)

---

## 1. Executive Summary (10문장 이내)

MaskSAM은 자연 이미지용 파운데이션 모델인 SAM(Segment Anything Model)을 의료 영상 분할에 적용할 때 발생하는 세 가지 핵심 문제(시맨틱 레이블 예측 불가, 수동 프롬프트 의존, 성능 저하)를 해결하기 위해 설계된 프레임워크이다.  
본 연구는 SAM의 모든 구조와 가중치를 유지한 채, 경량 어댑터와 프롬프트 생성기만을 삽입하는 방식으로 효율적 파인튜닝을 달성한다.  
이미지 인코더에서 추출한 다중 레벨 특징맵을 기반으로 보조 이진 마스크, 바운딩 박스, 분류기 토큰을 자동 생성하는 프롬프트 생성기를 도입한다.  
MaskFormer에서 영감을 받은 학습 가능한 전역 분류기 토큰과 보조 분류기 토큰의 합산으로 시맨틱 레이블 예측 기능을 부여한다.  
3D 의료 볼류메트릭 데이터 처리를 위해 DConvAdapter(이미지 임베딩용)와 DMLPAdapter(프롬프트 임베딩용) 두 종류의 3D 깊이 어댑터를 설계하였다.  
DETR 방식의 이분 매칭(Bipartite Matching)을 통해 예측 마스크와 정답 간 최적 할당을 수행하며 손실을 계산한다.  
AMOS2022 챌린지에서 Dice 90.52%를 달성하여 nnUNet 대비 2.7% 향상된 SOTA 성능을 기록하였다.  
ACDC 데이터셋에서 93.39%, Synapse 데이터셋에서 87.23%의 Dice 점수를 달성하였다.  
전체 파라미터 91M+14M 중 단 14M만 학습 가능하여 계산 효율성이 높다.  
본 연구는 SAM 기반 모델이 의료 영상 분할 챌린지에서 SOTA를 달성한 최초 사례 중 하나로서 의미가 크다.

### 1-1. 연구의 목적과 필요성

**목적:** SAM을 의료 볼류메트릭 영상 분할에 프롬프트 없이 적용할 수 있는 범용 프레임워크 개발.

**필요성:**
| 문제 | 기존 한계 | MaskSAM의 필요성 |
|------|-----------|-----------------|
| 시맨틱 레이블 부재 | SAM은 단순 이진 마스크만 예측 | 다중 장기 분할에는 클래스 정보 필수 |
| 수동 프롬프트 의존 | 추론 시 GT 박스 사용 → 실제 적용 불가 | 완전 자동화된 임상 워크플로우 요구 |
| 3D 정보 처리 불가 | SAM은 2D 자연 이미지 기반 | CT/MRI는 3D 볼륨 데이터 |
| 파인튜닝 효율성 | SAM 전체 파인튜닝 시 과도한 계산 비용 | 경량 어댑터 기반 효율적 적응 필요 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| 1 | SAM을 직접 의료 영상에 적용하면 성능 저하 | SAM Dice 53.8% vs nnUNet 87.8% on AMOS | Table 1, p.5 |
| 2 | 프롬프트 생성기로 수동 프롬프트 불필요 | MaskAvgBBoxPG가 최고 성능(91.45% vs 89.53%) | Table 4, p.8 |
| 3 | 시맨틱 레이블 예측 가능 | 학습 가능한 전역 분류기 토큰 + 보조 분류기 토큰 합산 | Sec 2.6, p.6 |
| 4 | DConvAdapter가 3D 공간 정보 학습 | B8→B9 개선: 93.10%→93.39% | Table 4, p.8 |
| 5 | DMLPAdapter가 깊이 정보 학습 | B5→B8 개선: 91.61%→93.10% | Table 4, p.8 |
| 6 | AMOS2022 SOTA 달성 | MaskSAM 90.52% vs nnUNet 87.8% | Table 1, p.5 |
| 7 | 14M 튜닝 파라미터로 효율적 | SAMed 19M, MedSAM 91M 대비 적음 | Table 5, p.8 |
| 8 | 이분 매칭으로 가변 길이 GT 처리 가능 | N ≥ N_gt 설정 후 최적 할당 | Sec 2.7, p.6 |

### 2-1. 상세 설명

#### 해결하고자 하는 문제 (p.2, Introduction)

1. **시맨틱 레이블 예측 불가:** SAM은 프롬프트당 1개의 이진 마스크만 출력하며, 클래스 정보 없음
2. **수동 프롬프트 의존:** 추론 시 GT 기반 박스/점 프롬프트 필요 → 실제 배포 불가
3. **3D 의료 데이터 처리 한계:** SAM은 2D RGB 이미지 기반 설계

---

#### 제안하는 방법 및 수식

##### (A) 전체 손실 함수 (Sec 2.7, p.6)

$$\mathcal{L}_{\text{mask-box-cls}} = \sum_{j=1}^{N} \left[ -\log p_{\sigma(j)}(c_j^{gt}) + \mathbf{1}_{c_j^{gt} \neq \varnothing} \mathcal{L}_{\text{mask}}^{\text{aux}}(m_{\sigma(j)}^{\text{aux}}, m_j^{gt}) + \mathbf{1}_{c_j^{gt} \neq \varnothing} \mathcal{L}_{\text{box}}(b_{\sigma(j)}, b_j^{gt}) + \mathbf{1}_{c_j^{gt} \neq \varnothing} \mathcal{L}_{\text{mask}}^{\text{final}}(m_{\sigma(j)}^{\text{final}}, m_j^{gt}) \right]$$

**기호 설명:**

| 기호 | 설명 |
|------|------|
| $N$ | 모델이 생성하는 총 예측 수 (N ≥ N_gt) |
| $j$ | 예측 인덱스 (1부터 N까지) |
| $\sigma$ | 이분 매칭으로 결정된 최적 순열 함수 |
| $p_{\sigma(j)}(c_j^{gt})$ | 매칭된 예측의 정답 클래스 $c_j^{gt}$에 대한 확률 |
| $c_j^{gt}$ | j번째 GT 클래스 레이블 ($c_j^{gt} \in \{1,...,K\}$) |
| $\varnothing$ | "객체 없음" 레이블 (no-object token) |
| $\mathbf{1}_{c_j^{gt} \neq \varnothing}$ | GT가 실제 객체일 때만 해당 손실 활성화하는 지시 함수 |
| $\mathcal{L}_{\text{mask}}^{\text{aux}}$ | 보조 이진 마스크 손실 (BCE + Dice 결합) |
| $m_{\sigma(j)}^{\text{aux}}$ | 매칭된 j번째 보조 예측 이진 마스크 |
| $m_j^{gt}$ | j번째 GT 이진 마스크 ($m_j^{gt} \in \{0,1\}^{H \times W}$) |
| $\mathcal{L}_{\text{box}}$ | 박스 예측 손실 (L1 + Generalized IoU 결합) |
| $b_{\sigma(j)}$ | 매칭된 j번째 예측 바운딩 박스 ($b \in [0,1]^4$) |
| $b_j^{gt}$ | j번째 GT 바운딩 박스 |
| $\mathcal{L}_{\text{mask}}^{\text{final}}$ | 최종 이진 마스크 손실 (BCE + Dice 결합) |
| $m_{\sigma(j)}^{\text{final}}$ | 최종 예측 이진 마스크 ($m \in [0,1]^{H \times W}$) |

> **📘 용어 설명**
> - **이분 매칭(Bipartite Matching):** 두 집합(예측과 GT) 사이의 최소 비용 일대일 대응을 찾는 알고리즘. DETR에서 도입된 방식으로, 헝가리안 알고리즘으로 구현됨.
> - **Generalized IoU (GIoU):** 두 박스가 겹치지 않을 때도 기울기가 존재하는 개선된 IoU 손실 함수 [Rezatofighi et al., CVPR 2019].

---

##### (B) 최종 출력 정의

$$z = \{(p_i, m_i)\}_{i=1}^{N}, \quad m_i^{\text{final}} \in [0,1]^{H \times W}, \quad p_i \in \Delta^{K+1}$$

$$z_{\text{aux}} = \{(b_i^{\text{aux}}, m_i^{\text{aux}}) \mid b_i^{\text{aux}} \in [0,1]^4, m_i^{\text{aux}} \in \{0,1\}^{H \times W}\}_{i=1}^{N}$$

$$z^{gt} = \{(c_i^{gt}, b_i^{gt}, m_i^{gt}) \mid c_i^{gt} \in \{1,...,K\}, b_i^{gt} \in [0,1]^4, m_i^{gt} \in \{0,1\}^{H \times W}\}_{i=1}^{N_{gt}}$$

**기호 설명:**

| 기호 | 설명 |
|------|------|
| $\Delta^{K+1}$ | K개 카테고리 + "no-object" 포함 K+1차원 확률 심플렉스 |
| $K$ | 분할 카테고리 수 |
| $N_{gt}$ | 현재 이미지의 실제 GT 세그먼트 수 |

> **📘 용어 설명**
> - **확률 심플렉스 $\Delta^{K+1}$:** 모든 원소가 0 이상이고 합이 1인 K+1차원 벡터 공간. 다중 클래스 분류 확률 분포를 표현함.

---

#### 모델 구조 (Figure 2, 3, p.2-3)

```
Raw Images
    │
    ▼
┌─────────────────────────────────────┐
│   Modified Image Encoder (ViT-B)    │
│   + DConvAdapter (각 블록마다)       │
│   + 학습 가능 깊이 위치 임베딩        │
│   + 모달리티 적응 Conv 레이어        │
└──────────┬──────────────────────────┘
           │ 다중 레벨 특징맵
           ▼
┌─────────────────────────────────────┐
│         Prompt Generator            │
│ → N개 보조 이진 마스크               │
│ → N개 보조 바운딩 박스 (평균화)      │
│ → N개 보조 분류기 토큰              │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
Prompt Encoder  Modified Mask Decoder
(고정)          + DMLPAdapter (프롬프트 임베딩)
                + DConvAdapter (이미지 임베딩)
                + 학습 가능 전역 분류기 토큰
                    │
                    ▼
           N × Binary Masks + N × Classifiers
                    │
              Bipartite Matching
                    │
           Multi-class Segmentation Output
```

> **📘 용어 설명**
> - **ViT (Vision Transformer):** 이미지를 패치로 분할하여 트랜스포머로 처리하는 이미지 인코더 [Dosovitskiy et al., 2020].
> - **DConvAdapter:** 원래 어댑터의 중간에 3D 깊이방향(depth-wise) 합성곱 레이어를 추가한 구조. 이미지 임베딩의 공간적 관계를 3D로 학습.
> - **DMLPAdapter:** 깊이 차원에서 역병목(invert-bottleneck) MLP를 추가한 어댑터. 프롬프트 임베딩의 깊이 정보를 학습.
> - **역병목(Invert-bottleneck):** 입력 차원을 먼저 확장 후 축소하는 구조 (표준 bottleneck은 반대).

---

#### 성능 향상 (Tables 1-4)

| 데이터셋 | MaskSAM | nnUNet | 향상폭 |
|---------|---------|--------|--------|
| AMOS2022 | 90.52% | 87.8% | +2.7% |
| ACDC | 93.39% | 91.61% | +1.7% |
| Synapse | 87.23% | 86.21% | +1.0% |

#### 한계

- 소규모 장기(담낭 GB, 식도 Eso.)에서 상대적으로 낮은 성능 (Synapse GB: 72.20%)
- Synapse에서 nnFormer 대비 0.7% 차이로 마진이 작음
- 학습 데이터 외 도메인(다른 병원, 다른 스캐너) 일반화 실험 부재
- 추론 속도 및 메모리 사용량 비교 실험 없음

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 위치 |
|------|------|
| SAM의 세 가지 한계 | p.2, Introduction |
| 전체 아키텍처 개요 | Figure 2 (p.2) |
| 프롬프트 생성기 변형 비교 | Figure 5 (p.4) |
| 어댑터 구조 | Figure 4 (p.3) |
| 이미지 인코더/디코더 상세 | Figure 3 (p.3) |
| AMOS 결과 | Table 1 (p.5) |
| Synapse 결과 | Table 2 (p.6) |
| ACDC 결과 | Table 3 (p.7) |
| 어블레이션 연구 | Table 4 (p.8) |
| 파라미터 효율성 | Table 5 (p.8) |
| 정성적 비교 (Synapse) | Figure 6 (p.6) |
| 정성적 비교 (ACDC) | Figure 7 (p.7) |
| 손실 함수 | Sec 2.7, p.6 |

---

## 4. 저자 보고 결과 vs. 분석자 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제:**
> "We propose MaskSAM, a novel mask classification prompt-free SAM adaptation framework for medical image segmentation." (Abstract)

**방법 (저자 원문):**
> 손실 함수:

$$\mathcal{L}_{\text{mask-box-cls}} = \sum_{j=1}^{N}\left[-\log p_{\sigma(j)}(c_j^{gt}) + \mathbf{1}_{c_j^{gt}\neq\varnothing}\mathcal{L}_{\text{mask}}^{\text{aux}} + \mathbf{1}_{c_j^{gt}\neq\varnothing}\mathcal{L}_{\text{box}} + \mathbf{1}_{c_j^{gt}\neq\varnothing}\mathcal{L}_{\text{mask}}^{\text{final}}\right]$$

**결과 (저자 원문):**
- "Our method achieves state-of-the-art performance on AMOS2022, 90.52% Dice, which improved by 2.7% compared to nnUNet." (Abstract)
- "MaskSAM surpasses nnUNet by 1.7% on ACDC and 1.0% on Synapse datasets." (Abstract)
- "With only 14M tunable parameters, effectively adapts the 91M ViT-base SAM model." (Table 5 설명, p.8)

---

### 4-2. 분석자 해석 (저자 주장과 구분)

> ⚠️ 아래는 논문 내용에 기반한 분석자의 해석입니다.

**해석 1:** MaskSAM의 AMOS2022 성능 향상(2.7%)은 인상적이나, 이 수치가 5-fold CV 기반임을 고려하면, **데이터셋 크기(200 케이스)가 상대적으로 작아 통계적 분산이 존재**할 수 있음.

**해석 2:** GT 프롬프트를 사용한 MaskSAM bbox 결과(Synapse 94.70%)는 프롬프트 없는 버전(87.23%)과 7.47%p 차이가 있어, **프롬프트 생성기가 여전히 완벽한 GT 프롬프트를 대체하지 못함**을 시사.

**해석 3:** DConvAdapter가 B8→B9에서 단 0.29% 개선을 보이는 것은 **깊이 방향 합성곱의 한계적 이득**을 시사하며, 3D 구조 이해의 근본적 개선보다 미세 조정에 가까울 수 있음.

**해석 4:** Synapse에서 담낭(GB: 72.20%)과 췌장(Panc: 79.62%)의 낮은 성능은 소형·불규칙 장기에 대한 **프롬프트 생성기의 구조적 한계**를 보여줌.

---

## 5. 통계적 취약점 및 비교 불가능한 수치 ⚠️

| 항목 | 문제점 | 위치 |
|------|--------|------|
| AMOS2022 테스트셋 크기 | 200 케이스 — 통계적 유의성 검증(p-value, CI) 미제시 | Table 1 |
| Synapse 12 케이스 검증 | 검증 세트가 극히 작아 분산이 큼 | Sec 3, p.7 |
| GT 프롬프트 MaskSAM bbox vs. 프롬프트 없는 버전 비교 | **조건이 다름** → 직접 비교 불공정 | Table 2 |
| SAM, MedSAM 비교 | nnUNet 예측으로 박스 생성 → **외부 모델 의존** | Table 1, p.7 |
| 5-fold CV 표준편차 미보고 | 평균 Dice만 보고, 분산 미제시 | Table 1, 2, 3 |
| 추론 시간/메모리 비교 없음 | 임상 적용 가능성 평가 불충분 | Table 5 |
| SAM2 비교 | AMOS에만 포함, ACDC·Synapse 미포함 → **불완전 비교** | Table 1 |

> ⚠️ **비교 불가능한 수치:** Table 2의 "MaskSAM (Ours) bbox" 행(94.70%)은 GT 박스를 사용한 상한선이며, 동일 행의 프롬프트 없는 "MaskSAM (Ours) 87.23%"와 직접 비교하는 것은 조건이 다르므로 부적절함.

---

## 6. 논문이 답하지 않는 질문

| # | 미답변 질문 | 중요도 |
|---|-----------|--------|
| 1 | 다른 기관/병원 데이터(외부 도메인)에서의 일반화 성능은? | 🔴 높음 |
| 2 | MRI 외 다른 모달리티(초음파, PET 등)에 적용 가능한가? | 🔴 높음 |
| 3 | 추론 속도(FPS)와 GPU 메모리 사용량은 타 방법 대비 어느 수준인가? | 🔴 높음 |
| 4 | N(생성 프롬프트 수)의 최적값 선택 기준과 민감도는? | 🟡 중간 |
| 5 | 통계적 유의성 검증(paired t-test 등)이 없는데 성능 차이가 유의한가? | 🟡 중간 |
| 6 | 프롬프트 생성기가 실패하는 케이스(OOD 이미지)의 특성은? | 🟡 중간 |
| 7 | SAM2 대비 성능이 ACDC, Synapse에서도 우월한가? | 🟡 중간 |
| 8 | 학습 데이터 크기 변화에 따른 성능 변화(few-shot 시나리오)는? | 🟡 중간 |
| 9 | 어댑터 삽입 위치(어느 레이어)가 성능에 얼마나 영향을 주는가? | 🟢 낮음 |
| 10 | 불균형 클래스(소형 장기 vs 대형 장기) 문제에 대한 별도 처리 전략은? | 🟡 중간 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — 보조 박스 프롬프트 시각화

**해석:** 상단 행은 GT와 GT 박스, 하단 행은 예측 마스크와 MaskSAM이 자동 생성한 박스를 보여준다. 생성된 박스들이 실제 장기 경계보다 다소 넓게 설정되는 경향이 있으나, 복잡한 해부학적 구조에서 다수의 프롬프트를 생성하여 마스크 예측을 보조한다. 특히 "No-object" 쿼리들이 자동으로 배제되어 유효 프롬프트만 활성화되는 원리를 시각적으로 확인할 수 있다.

> **📘 용어 설명**
> - **No-object 쿼리:** N개의 예측 중 실제 객체와 매칭되지 않은 예측으로, 학습 과정에서 "빈 객체"로 분류되어 손실 계산에서 제외됨.

---

### Figure 2 (p.2) — MaskSAM 전체 아키텍처

**해석:** 전체 파이프라인의 세 핵심 경로를 보여준다: ① 수정된 이미지 인코더 → 프롬프트 생성기 → 프롬프트 인코더 → 수정된 마스크 디코더의 순방향 경로, ② 하단의 데이터셋 매핑 파이프라인(다중 클래스 마스크 → 이진 마스크 집합), ③ 이분 매칭을 통한 예측-GT 연결. 파란색(고정)과 나머지(학습) 블록의 구분이 파라미터 효율성의 핵심임을 시각화한다. SAM의 기존 구조(흰 박스)를 그대로 유지하면서 새로운 블록(색상 박스)만 추가한 설계 철학이 명확히 드러난다.

---

### Figure 3 (p.3) — 세부 컴포넌트 구조

**해석:**
- **(a) 수정된 이미지 인코더:** Conv 레이어(모달리티 적응) → Patch Embedding → 4그룹의 ViT 블록(각 3개) + DConvAdapter. 각 그룹 출력이 프롬프트 생성기로 연결되는 다중 스케일 특징 추출 구조가 핵심.
- **(b) 프롬프트 생성기:** ConvTranspose로 업샘플링, AdaptiveAvgPool로 (2,2) 및 (1,1) 크기로 변환하여 각각 박스와 분류기 토큰 생성. FPN(Feature Pyramid Network) 유사 설계.
- **(c) 수정된 마스크 디코더:** DMLPAdapter(프롬프트 임베딩 처리)와 DConvAdapter(이미지 임베딩 처리)가 각 트랜스포머 블록 내 삽입 위치를 명확히 보여줌.

> **📘 용어 설명**
> - **FPN (Feature Pyramid Network):** 다양한 해상도의 특징맵을 계층적으로 결합하는 구조. 소형 객체 검출에 효과적.
> - **AdaptiveAvgPool:** 출력 크기를 고정하는 평균 풀링 레이어.

---

### Figure 5 (p.4) — 프롬프트 생성기 변형 4가지

**해석:** 어블레이션 설계의 시각적 요약. (a) 마스크→박스 파생만 사용, (b) 학습 가능한 박스만 사용, (c) 마스크+박스 독립 학습, (d) 마스크 기반 박스와 학습 박스의 평균 사용. Table 4에서 (d)가 91.45%로 최고 성능을 보이는 이유는 두 박스 소스의 보완적 정보를 결합하기 때문으로 해석된다. 마스크 기반 박스는 정밀하지만 노이즈에 민감하고, 학습 박스는 더 안정적이어서 평균화가 강인성을 높인다.

---

### Figure 6 (p.6) — Synapse 데이터셋 정성적 비교

**해석:** 8개 방법(MaskSAM, SAM3D, SAMed, MedSAM-bbox, SAM-bbox, nnFormer, nnUNet, UNETR)의 세그멘테이션 결과를 행별로 비교. MaskSAM은 간(Liver), 비장(Spleen), 위(Stomach) 같은 대형 장기에서 가장 정확한 경계를 보이며 이상치(outlier) 영역이 가장 적다. 반면 SAM-bbox는 장기 경계가 불규칙하고, SAM3D는 작은 장기에서 누락이 발생하는 패턴이 관찰된다. 특히 MaskSAM의 위(Stomach) 분할이 nnUNet 대비 더 완전한 형태를 보이는 점이 주목할 만하다.

---

## 8. 결론 및 후속 연구

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 제시 시사점 (p.8-9, Conclusion):**
1. SAM의 모든 구조와 가중치를 유지하면서 의료 분야에 적응 가능함을 증명
2. 마스크 분류 + 프롬프트 자동 생성의 결합이 의료 챌린지 수준의 성능을 달성 가능
3. 14M의 경량 튜닝으로 91M 파라미터 모델을 효율적으로 적응 가능
4. 3D 어댑터 설계가 2D 파운데이션 모델의 3D 의료 영상 적응에 효과적

**후속 연구 계획:** 논문에서 명시적으로 제시된 후속 연구 계획은 없음 (⚠️ 확인 불가 영역 — 억지로 지어내지 않음).

---

### 8-1. 모델 일반화 성능 향상 가능성 (중점)

현재 MaskSAM의 일반화 한계와 개선 방향:

| 일반화 도전 과제 | 현재 상태 | 제안 방향 |
|----------------|-----------|-----------|
| **도메인 이동** (다른 병원, 스캐너) | 미검증 | 도메인 적응 어댑터 추가, Test-Time Adaptation |
| **다중 모달리티** (CT→MRI→초음파) | CT 중심 평가 | 모달리티 불가지론적(modality-agnostic) 인코더 설계 |
| **소형 장기** (담낭, 췌장) | 상대적 저성능 | 계층적 주의(hierarchical attention) 강화 |
| **희귀 해부학적 변이** | 미평가 | 메타 학습(Meta-Learning) 결합 |
| **데이터 부족 시나리오** | 미평가 | Few-shot 파인튜닝 실험 필요 |

**일반화 향상을 위한 구체적 제안:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mask-box-cls}} + \lambda_{\text{domain}} \mathcal{L}_{\text{domain-align}} + \lambda_{\text{consistency}} \mathcal{L}_{\text{consistency}}$$

여기서 $\mathcal{L}\_{\text{domain-align}}$은 소스-타겟 도메인 간 특징 분포 정렬 손실 (예: MMD, CORAL), $\mathcal{L}_{\text{consistency}}$는 데이터 증강에 대한 예측 일관성 손실.

> **📘 용어 설명**
> - **Test-Time Adaptation (TTA):** 추론 시 모델을 테스트 데이터에 동적으로 적응시키는 기법.
> - **MMD (Maximum Mean Discrepancy):** 두 분포 간 차이를 측정하는 통계량으로, 도메인 적응에 활용.
> - **Meta-Learning:** 적은 데이터로 빠르게 새 태스크에 적응하는 "학습하는 방법을 학습"하는 기법.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | AMOS DSC | 일반화 | 프롬프트 자유 |
|------|------|------|---------|--------|--------------|
| nnUNet [22] | 2019 | CNN 앙상블 자동화 | 87.8% | 중간 | ✅ |
| SwinUNETR [16] | 2021 | Swin Transformer | 88.0% | 중간 | ✅ |
| MedSAM [31] | 2024 | SAM 파인튜닝 | 60.0% (bbox) | 낮음 | ❌ |
| SAMed [44] | 2023 | LoRA + SAM | 77.2% | 낮음 | ✅ |
| 3DSAM-Adapter [15] | 2023 | 3D 어댑터 | 미보고 | 중간 | ❌ |
| **MaskSAM (본 논문)** | 2025 | 마스크 분류 + 자동 프롬프트 | **90.52%** | 미검증 | ✅ |

> ⚠️ 위 표의 일부 수치(특히 3DSAM-Adapter의 AMOS 결과)는 직접 해당 논문에서 확인하지 못하였으므로, 정확한 수치는 원 논문 참조 권장.

**MaskSAM이 앞으로 연구에 미치는 영향:**

1. **SAM 어댑테이션 패러다임 확장:** 프롬프트 없는 SAM 적용이 가능함을 증명하여, 향후 임상 자동화 워크플로우 설계에 직접적 기여.

2. **마스크 분류 패러다임의 의료 영상 도입:** MaskFormer의 설계 철학을 의료 영상에 성공적으로 이식, 후속 연구의 기준점 제시.

3. **파운데이션 모델 경량 적응의 선례:** 14M 튜닝 파라미터로 SOTA를 달성하여, 계산 자원이 제한된 의료 기관에서도 파운데이션 모델 활용 가능성 제시.

**앞으로 연구 시 고려할 점:**

| 고려 사항 | 세부 내용 |
|----------|-----------|
| **재현성** | 코드와 사전학습 가중치 공개 여부 확인 필요 |
| **공정한 비교** | GT 프롬프트 사용 결과와 자동 결과를 항상 분리 보고 |
| **통계적 유의성** | Dice 차이에 대한 paired t-test 또는 Wilcoxon 검정 필수 |
| **임상 평가** | Dice 외 HD95(95번째 백분위 하우스도르프 거리) 등 임상 관련 지표 추가 |
| **대용량 데이터** | TotalSegmentator 등 더 큰 데이터셋으로 일반화 검증 |
| **SAM2 통합** | SAM2의 동영상 추적 능력과 MaskSAM의 3D 처리 결합 가능성 탐색 |
| **경량화** | 모바일/엣지 의료기기 배포를 위한 지식 증류(Knowledge Distillation) 적용 |

> **📘 용어 설명**
> - **HD95 (95th Hausdorff Distance):** 두 세그멘테이션 간 경계 거리의 95번째 백분위 값. 경계 정확도를 측정하는 임상적으로 중요한 지표.
> - **지식 증류(Knowledge Distillation):** 큰 교사 모델의 지식을 작은 학생 모델로 전달하는 압축 기법.
> - **TotalSegmentator:** 104개 해부학적 구조를 레이블한 대규모 공개 CT 데이터셋 (Wasserthal et al., 2023).

---

**추가 후속 연구 방향 (분석자 제안):**

1. **MaskSAM + SAM2:** SAM2의 메모리 메커니즘을 활용하여 연속 슬라이스 간 시간적(temporal) 일관성 강화
2. **불확실성 정량화:** 예측 마스크에 대한 베이지안 불확실성 추정으로 임상 신뢰도 제공
3. **능동 학습(Active Learning):** 불확실성이 높은 샘플을 선별하여 레이블링 효율화
4. **연합 학습(Federated Learning):** 다기관 데이터를 프라이버시 보호하며 학습하여 일반화 향상
5. **병리 영상 확장:** CT/MRI에서 조직 병리(Histopathology) 슬라이드로 도메인 확장 가능성 탐색
