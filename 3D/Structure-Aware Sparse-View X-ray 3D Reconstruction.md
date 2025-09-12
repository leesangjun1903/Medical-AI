# Structure-Aware Sparse-View X-ray 3D Reconstruction

## 핵심 주장과 주요 기여

이 논문은 기존 NeRF 기반 X선 3D 재구성 방법들이 X선 영상의 본질적 특성을 간과했다는 문제점을 제기하며, **Structure-Aware X-ray Neural Radiodensity Fields (SAX-NeRF)**를 제안합니다. 주요 기여는 다음과 같습니다:[1]

- **Line Segment-based Transformer (Lineformer)**: X선 내부의 구조적 의존성을 포착하는 새로운 Transformer 아키텍처 설계
- **Masked Local-Global (MLG) ray sampling**: 2D 투영에서 기하학적/맥락적 정보를 효과적으로 추출하는 샘플링 전략
- **X3D 데이터셋**: 의학, 생물학, 보안, 산업 등 4개 분야 15개 장면을 포함한 대규모 벤치마크 구축
- **성능 향상**: 기존 SOTA 방법 대비 평균 12.56 dB PSNR 개선 달성[1]

## 해결하고자 하는 문제

### 기존 방법의 한계점

1. **구조적 특성 무시**: 기존 NeRF 기반 방법들이 단순한 MLP를 사용하여 X선이 통과하는 각 점을 동등하게 처리[1]
2. **비효율적 샘플링**: 무작위 픽셀 기반 샘플링으로 인한 맥락 정보 손실과 비정보적 영역에서의 비효율적 학습[1]
3. **제한된 적용 범위**: 주로 의료 영상에 국한된 연구로 다양한 X선 응용 분야에서의 성능 미검증[1]

### 가시광선과 X선 영상의 차이점

- **가시광선**: 객체 표면에서의 반사에 의존하여 외부 특징을 주로 포착[1]
- **X선**: 객체를 관통하며 감쇠되어 내부 구조를 드러냄, 이는 3D 재구성의 핵심 단서 제공[1]

## 제안하는 방법

### 1. Neural Radiodensity Fields 모델링

기존 RGB NeRF의 색상 정보와 달리, X선은 방사밀도(radiodensity) 특성만 기록합니다:

$$F_{\Theta_L}: (x, y, z) \rightarrow \rho$$

여기서 $\rho$는 방사밀도를 나타냅니다.[1]

Beer-Lambert 법칙에 따라 X선 강도는 다음과 같이 계산됩니다:

$$I_{gt}(\mathbf{r}) = I_0 \cdot \text{exp}\left(-\int_{t_n}^{t_f}\rho(\mathbf{r}(t)) dt\right)$$

이를 이산화하면:

$$I_{pred}(\mathbf{r}) = I_0 \cdot \text{exp}\left(-\sum_{i=1}^{N} \rho_i \delta_i\right)$$

### 2. Line Segment-based Transformer (Lineformer)

#### 구조적 특징
- X선을 M개의 선분으로 분할하여 각 선분 내에서 self-attention 계산[1]
- 4개의 Line Segment-based Attention Block (LSAB) 사용
- Hash encoding과 skip connection 적용

#### Line Segment-based Multi-head Self-Attention (LS-MSA)

입력 특징 $\mathbf{X} \in \mathbb{R}^{N \times C}$를 M개 세그먼트로 분할:

$$\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2, \cdots, \mathbf{X}_M]^T$$

각 세그먼트 $\mathbf{X}_i$에 대해 Query, Key, Value 계산:

$$\mathbf{Q}_i = \mathbf{X}_i\mathbf{W}^{\mathbf{Q}_i}, \quad \mathbf{K}_i = \mathbf{X}_i\mathbf{W}^{\mathbf{K}_i}, \quad \mathbf{V}_i = \mathbf{X}_i\mathbf{W}^{\mathbf{V}_i}$$

Self-attention 계산:

$$\mathbf{H}_i^j = \mathbf{V}_i^j \cdot \text{softmax}\left(\frac{{\mathbf{K}_i^j}^T\mathbf{Q}_i^j}{\alpha_i^j}\right)$$

#### 계산 복잡도 분석

- **LS-MSA**: $\mathcal{O}(\frac{2NC^2}{k})$ (선형)
- **Vanilla Transformer**: $\mathcal{O}(2N^2C)$ (이차)

LS-MSA는 vanilla Transformer 대비 **3.41%**의 계산 비용으로 **5.30 dB** 성능 향상을 달성했습니다.[1]

### 3. Masked Local-Global (MLG) Ray Sampling

#### 전략 구성
1. **마스크 생성**: 임계값 T를 사용하여 전경 영역 분할
2. **패치 레벨 샘플링**: $N_l$개 윈도우에서 지역적 맥락 정보 추출
3. **픽셀 레벨 샘플링**: 전역적 기하학적 형태 인식을 위한 $N_g$개 픽셀 샘플링

최종 훈련 X선 배치:
$$\mathcal{R} = \mathcal{R}_l \bigcup \mathcal{R}_g$$

## 모델 구조

SAX-NeRF는 다음과 같은 파이프라인으로 구성됩니다:[1]

1. **MLG 샘플링**: 정보가 풍부한 X선 배치 $\mathcal{R}$ 생성
2. **점 샘플링**: 각 X선 $r \in \mathcal{R}$에서 N개 점 위치 $\mathbf{P}$ 샘플링
3. **Lineformer 처리**: Hash encoding → 4개 LSAB → 방사밀도 $\mathbf{D}$ 예측
4. **볼륨 렌더링**: Beer-Lambert 법칙에 따른 X선 강도 계산

## 성능 향상

### Novel View Synthesis (NVS) 결과
- **평균 PSNR**: 51.37 dB (기존 최고 대비 +12.56 dB)[1]
- **응용별 개선도**: 의학(+10.91 dB), 생물학(+15.03 dB), 보안(+5.13 dB), 산업(+13.76 dB)[1]

### CT Reconstruction 결과
- **NAF 대비**: +2.49 dB PSNR 개선[1]
- **최적화 기반 방법 대비**: +4.92 dB 개선[1]
- **분석적 방법 대비**: +12.13 dB 개선[1]

### Ablation Study 결과
- **LS-MSA 기여도**: NVS에서 +9.98 dB, CT에서 +2.65 dB[1]
- **MLG 샘플링 기여도**: NVS에서 +5.54 dB, CT에서 +1.09 dB[1]
- **결합 효과**: NVS에서 +13.40 dB, CT에서 +3.04 dB[1]

## 일반화 성능 향상 가능성

### 강건성 분석
- **적은 투영 수에서도 우수한 성능**: 60%의 훈련 투영만으로도 다른 알고리즘을 능가[1]
- **다양한 매개변수에서 안정적 성능**: 세그먼트 수 M과 패치 크기 S 변화에도 일관된 개선 효과[1]

### 도메인 적응성
- **4개 응용 분야**: 의학, 생물학, 보안, 산업 전반에서 일관된 성능 향상[1]
- **CT 데이터 불필요**: 투영 데이터만으로 훈련 가능하여 데이터 수집 부담 경감[1]
- **확장 가능한 아키텍처**: Transformer 기반 구조로 다양한 크기의 입력에 적응 가능[1]

## 한계점

### 기술적 한계
1. **고정된 기하학적 설정**: 원형 콘빔 스캔에 특화된 구조로 다른 스캔 방식에 대한 적응성 미검증
2. **실시간 처리**: 3000 iteration 훈련 필요로 실시간 응용에는 제약
3. **메모리 요구량**: Hash encoding과 다중 LSAB로 인한 메모리 사용량 증가 가능성

### 실험적 한계
1. **시뮬레이션 데이터**: TIGRE로 생성된 합성 투영 데이터 사용으로 실제 X선 장비에서의 성능 검증 필요
2. **제한된 노이즈 조건**: 3% 가우시안 노이즈만 고려하여 실제 임상 환경의 다양한 노이즈에 대한 강건성 미확인

## 미래 연구에 미치는 영향

### 긍정적 영향
1. **패러다임 전환**: X선 영상의 구조적 특성을 고려한 새로운 NeRF 설계 방향 제시
2. **효율적 Transformer**: 선형 복잡도의 LS-MSA로 3D 의료 영상에서 Transformer 활용 가능성 확대
3. **샘플링 전략 개선**: MLG 샘플링이 다른 희소 뷰 재구성 문제에도 적용 가능
4. **벤치마크 확장**: X3D 데이터셋으로 다양한 응용 분야에서의 비교 연구 촉진

### 후속 연구 고려사항

#### 기술적 개선 방향
1. **실시간 최적화**: 훈련 시간 단축을 위한 지식 증류나 프루닝 기법 적용
2. **적응형 샘플링**: 동적으로 패치 크기와 샘플링 비율을 조정하는 알고리즘 개발
3. **다중 모달 융합**: CT, MRI 등 다른 의료 영상 모달리티와의 융합 연구

#### 실용화 고려사항
1. **실제 데이터 검증**: 다양한 X선 장비와 임상 환경에서의 성능 평가 필요
2. **안전성 평가**: 의료 영상 진단에서의 오진 가능성과 신뢰성 확보 방안
3. **표준화**: 의료 영상 표준(DICOM) 호환성과 임상 워크플로우 통합

이 연구는 X선 3D 재구성 분야에서 구조 인식 기반 접근법의 중요성을 입증했으며, 향후 의료 AI와 3D 영상 재구성 연구의 새로운 방향성을 제시할 것으로 예상됩니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ad79b5e6-fc14-4877-aee0-34722763a4ed/2311.10959v3.pdf)
