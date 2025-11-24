# Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images

### 1. 핵심 주장과 주요 기여 요약

Swin UNETR은 계층적 Swin Transformer를 인코더로 활용하고 합성곱신경망(CNN) 기반 디코더를 연결한 U자형 신경망으로, 3D 다중모드 뇌종양 MRI 분할을 위한 새로운 접근방식을 제시합니다. 이 논문의 핵심 주장은 기존 FCNNs 기반 U-Net이 제한된 커널 크기로 인해 장거리 의존성을 효과적으로 모델링하지 못한다는 점에 있습니다. Swin UNETR은 이 문제를 다음과 같이 해결합니다:[1]

- **장거리 정보 모델링**: Vision Transformer의 자기주의(self-attention) 메커니즘을 활용하여 전역적 맥락 정보를 효과적으로 포착
- **다중 해상도 계층 구조**: Swin Transformer의 이동 윈도우(shifted window) 기법을 통해 5개의 서로 다른 해상도에서 특징 추출
- **효율적 계산**: 선형 계산 복잡도를 유지하면서도 전역적 정보 모델링 가능
- **BraTS 2021 도전 에서 최고 성능**: 검증 단계에서 상위 순위의 방법론 중 하나로 선정

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하는 문제

뇌종양 분할은 다음의 도전과제를 마주합니다:[1]

- **가변적 종양 크기**: 서로 다른 크기의 종양을 정확히 구분해야 함
- **다중모드 데이터**: T1, T1c, T2, FLAIR 등 4가지 MRI 모드를 동시에 처리
- **복잡한 경계**: 종양과 정상 조직의 불명확한 경계 구분
- **지역적 편향 문제**: CNN의 제한된 수용장(receptive field)

#### 2.2 제안하는 방법

**패치 분할 및 임베딩:**

입력 $$X \in \mathbb{R}^{H \times W \times D \times S}$$는 먼저 패치 크기 $$(H', W', D')$$로 분할되어 토큰 개수 $$\lceil \frac{H}{H'} \rceil \times \lceil \frac{W}{W'} \rceil \times \lceil \frac{D}{D'} \rceil$$로 변환되고, 이는 차원 C의 임베딩 공간으로 투영됩니다.[1]

**이동 윈도우 자기주의(Shifted Window Multi-Head Self-Attention):**

엔코더의 l 번째와 l+1 번째 계층의 출력은 다음과 같이 계산됩니다:[1]

$$\hat{z}_l = \text{W-MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$

$$z_l = \text{MLP}(\text{LN}(\hat{z}_l)) + \hat{z}_l$$

$$\hat{z}_{l+1} = \text{SW-MSA}(\text{LN}(z_l)) + z_l$$

$$z_{l+1} = \text{MLP}(\text{LN}(\hat{z}_{l+1})) + \hat{z}_{l+1}$$

여기서 W-MSA는 정규 윈도우 분할 자기주의, SW-MSA는 이동 윈도우 자기주의이며, LN은 계층 정규화(Layer Normalization), MLP는 다층 퍼셉트론을 나타냅니다.

**자기주의 계산:**

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

여기서 Q, K, V는 각각 쿼리, 키, 값이고, d는 쿼리와 키의 크기입니다.[1]

**손실함수:**

소프트 Dice 손실함수를 사용하여 모든 클래스에 대해 계산합니다:[1]

$$L(G, Y) = 1 - \frac{2}{J} \sum_{j=1}^{J} \frac{\sum_{i=1}^{I} G_{i,j}Y_{i,j}}{\sum_{i=1}^{I} G_{i,j}^2 + \sum_{i=1}^{I} Y_{i,j}^2}$$

여기서 I는 복셀(voxel) 개수, J는 클래스 개수, $$Y_{i,j}$$와 $$G_{i,j}$$는 각각 복셀 i에서 클래스 j의 출력 확률과 원-핫 인코딩된 정답입니다.

#### 2.3 모델 구조

**인코더 구조:**[1]
- **4단계 계층 구조**: 각 단계에서 2개의 Transformer 블록 (총 8개 계층)
- **해상도 단계**: H/2 × W/2 × D/2 (48차원), H/4 × W/4 × D/4 (96차원), H/8 × W/8 × D/8 (192차원), H/16 × W/16 × D/16 (384차원)
- **패치 병합**: 각 단계 끝에서 해상도를 2배 감소
- **윈도우 크기**: M × M × M (효율적인 계산을 위해 3D 주기적 이동 활용)

**디코더 구조:**[1]
- **잔차 블록**: 인코더 각 해상도에서 나온 특징 맵을 받아 3×3×3 합성곱층으로 처리
- **스킵 연결**: 인코더의 여러 해상도에서의 출력이 디코더의 각 단계로 전달
- **역합성곱(Deconvolution)**: 해상도를 2배 증가
- **최종 출력**: 1×1×1 합성곱층과 시그모이드 활성화 함수로 3개 채널(ET, WT, TC) 생성

**모델 사양:**[1]
- **매개변수 수**: 61.98M
- **연산량(FLOPs)**: 394.84G
- **입력 크기**: 128 × 128 × 128
- **배치 크기**: GPU당 1

#### 2.4 성능 향상

**내부 검증 결과 (5-fold 교차 검증):**[1]

| 방법 | ET(%) | WT(%) | TC(%) | 평균(%) |
|------|-------|-------|-------|---------|
| Swin UNETR | 89.1 | 93.3 | 91.7 | 91.3 |
| nnU-Net | 88.3 | 92.7 | 91.3 | 90.8 |
| SegResNet | 88.3 | 92.7 | 91.3 | 90.7 |
| TransBTS | 86.8 | 91.1 | 89.8 | 89.1 |

Swin UNETR은 모든 종양 부분류(ET, WT, TC)에서 경쟁 방법들을 능가했습니다.[1]

**BraTS 2021 검증 세트 성능:**[1]

| 메트릭 | ET | WT | TC |
|--------|----|----|-----|
| Dice | 0.858 | 0.926 | 0.885 |
| Hausdorff (mm) | 6.016 | 5.831 | 3.770 |

**테스트 세트 성능:**[1]

| 메트릭 | ET | WT | TC |
|--------|----|----|-----|
| Dice | 0.853 | 0.927 | 0.876 |
| Hausdorff (mm) | 16.326 | 4.739 | 15.309 |

#### 2.5 한계점

1. **테스트-검증 성능 갭**: 종양 코어(TC)에서 0.9%의 성능 저하 관찰[1]
2. **도메인 일반화**: 서로 다른 MRI 스캐너와 프로토콜에서의 성능 감소 우려
3. **계산 비용**: 큰 모델 크기로 인한 높은 연산량(394.84G FLOPs)
4. **앙상블 의존성**: 최종 결과는 10개 모델의 앙상블로 달성[1]

### 3. 일반화 성능 향상 가능성

#### 3.1 현재 한계와 문제점

최신 연구에 따르면 의료영상 분할의 주요 일반화 문제는 다음과 같습니다:[2][3][4]

- **도메인 시프트**: 병원, 스캐너 제조사, 촬영 프로토콜 간의 분포 차이
- **외부 검증 성능 저하**: 내부 테스트에서 우수한 성능도 다른 데이터로부터의 성능 급격히 감소[5]
- **스캐너별 편향**: 모델이 스캔 위치를 추론할 수 있을 정도로 스캐너별 신호 학습[5]

#### 3.2 일반화 성능 향상 전략

**1) 자기감독 학습(Self-Supervised Learning) 및 사전학습:**

최근 연구에서 자기감독 학습으로 사전학습된 Swin Transformer는 일반화 성능을 크게 향상시킵니다. 특히:[6][7][8]

- 대규모 미표지 의료영상 데이터로 사전학습 후 목표 작업에 파인튜닝 시 성능 향상
- 자기감독 학습 방식(대조 학습, 생성 모델, 자기예측)이 지도학습 단독 방식 대비 평균 6.35% 성능 개선[8]
- 의료 도메인 특화 전략: 종양 영역의 공간 정보를 활용한 양성 쌍 마이닝[8]

**2) 도메인 일반화(Domain Generalization) 기법:**

- **데이터 증강**: 광범위한 이미지 변환으로 "예상되는" 도메인 시프트 모의[4][9]
- **스타일 전이**: 랜덤 스타일 전이를 통한 도메인 증강으로 unknown target 도메인에 대한 강건성 강화[10]
- **대조학습**: 출처별 특징 임베딩 정렬로 도메인 불변 표현 학습[7][11]
- **메타학습**: 에피소딕 훈련으로 시뮬레이션된 도메인 시프트로부터 지식 전이[12]

**3) 고급 정규화 기법:**

- **Style Feature Whitening**: 도메인 특이적 스타일을 고차 공분산 통계로부터 분리[13]
- **적응형 인스턴스 정규화**: 각 스캔의 특징을 정규화하여 스캐너별 편향 감소

**4) 최신 모델 개선안:**

최신 2024-2025년 연구들에서 Swin UNETR 기반의 개선 방법들이 제시되고 있습니다:[14][15][16][17][18]

- **DiffSwinTr(2024)**: 확산 모델(Diffusion Model)과 Swin Transformer 결합으로 노이즈와 인공물에 대한 강건성 향상[14]
- **SwinHCAD(2025)**: 계층적 채널별 주의(Hierarchical Channel-wise Attention) 디코더 추가로 신호 상세 보존 개선
- **A4-Unet(2024)**: 변형 가능한 대형 커널 주의(Deformable Large Kernel Attention)로 다중 크기 종양 포착 개선[16]
- **UnetTransCNN(2025)**: Vision Transformer와 CNN을 병렬로 결합하여 지역-전역 특징 균형[19]

### 4. 미래 연구에 미치는 영향 및 고려사항

#### 4.1 연구 영향

**1) Transformer 기반 의료영상 분할의 기초 확립:**

Swin UNETR은 BraTS 2021에서 처음으로 Transformer 기반 모델이 경쟁력 있는 성능을 달성한 사례입니다. 이후 다양한 Transformer 변형이 개발되었으며, 2024-2025년의 최신 논문들에서 대부분의 최고 성능 모델이 Transformer 구조를 기반으로 합니다.[20][21][1]

**2) 계층적 다중 해상도 처리의 중요성:**

Swin Transformer의 계층적 구조는 의료영상 분할에서 필수적임이 증명되었으며, 이는 CNN과 Transformer의 장점을 결합하는 방향으로 진화했습니다.[21][20]

**3) 다양한 응용 분야 확대:**

- BraTS-PEDs 2023(소아 뇌종양) 도전에서 앙상블 nnU-Net과 Swin UNETR이 최고 성능 기록[22]
- PET/CT 자동 분할에 자기감독 학습과 함께 적용[6]
- 뇌졸중 재활 효과 분석에 개선된 SWI-BITR-UNet 모델로 활용[17]

#### 4.2 향후 연구 시 고려사항

**1) 일반화 성능 개선의 우선순위:**

- **도메인 일반화 테스트**: 다양한 스캐너, 촬영 프로토콜, 환자 집단에서의 외부 검증 필수[2][21]
- **도메인 시프트 완화 기법 개발**: 단일 출처 도메인 일반화(SDG)를 위한 방법론 강화[11][7][13]
- **페더레이션 학습**: 의료 데이터 프라이버시 보호하면서 다중 센터 협력[21]

**2) 효율성과 해석성:**

- **모델 경량화**: 임상 배포를 위한 추론 속도 및 메모리 최적화[21]
- **설명 가능한 인공지능(XAI)**: 임상의가 신뢰할 수 있는 분할 결정 근거 제시[21]

**3) 데이터 주석 부담 완화:**

- **반-감독 학습과 도메인 일반화 융합**: SSL과 DG를 결합한 SSL-DG 프레임워크 개발[11]
- **약한 감독 학습**: 전체 복셀 수준 주석 대신 약한 주석으로 학습 가능성[21]

**4) 파운데이션 모델의 역할:**

- **대규모 사전학습**: CLIP, SAM 같은 파운데이션 모델을 의료영상 분할에 적응[23][21]
- **분포 시프트 강건성**: 파운데이션 모델의 고유 일반화 능력 활용하되, 신뢰도 보정 필요[23]

**5) 자기감독 학습의 최적화:**

- **의료 도메인 특화 SSL 전략**: 해부학적 구조 제약과 임상 선행지식 통합[24][8]
- **그리드 마스크 이미지 모델링(GMIM)**: 3D 의료 영상 특성에 맞춘 유연한 자기감독 방법[24]

**6) 신경망 복잡도 측정:**

- **일반화 경계 분석**: PAC-Bayes 이론을 통한 이론적 일반화 성능 예측[25][23]
- **복잡도 측정 상관성 조사**: 25개 이상의 복잡도 측정과 실제 일반화 성능 관계 분석[25]

#### 4.3 임상 적용을 위한 과제

의료 이미징에서 실제 임상 배포를 위해서는:[21]

1. **데이터 이질성 극복**: 다양한 센터, 장비, 프로토콜로부터의 데이터 통합
2. **실시간 분할**: 수술 전후 평가를 위한 빠른 추론 속도 요구
3. **통합 장벽**: 기존 의료 영상 정보 시스템(PACS) 및 전자의무기록(EHR)과의 호환성
4. **규제 요구사항**: FDA 승인 등 의료기기 인증 프로세스

### 결론

Swin UNETR은 Transformer 기반 의료영상 분할의 혁신적인 접근방식으로, 계층적 윈도우 기반 자기주의 메커니즘을 통해 장거리 의존성을 효과적으로 모델링합니다. BraTS 2021 도전에서 경쟁력 있는 성능을 달성하였으나, 도메인 일반화 측면에서는 여전히 개선의 여지가 있습니다. 향후 연구는 자기감독 학습, 도메인 일반화 기법, 파운데이션 모델 활용, 그리고 임상 배포를 위한 효율성 개선에 집중해야 합니다. 특히 다양한 센터와 장비로부터의 데이터에 대한 강건성 확보가 임상 적용의 핵심 요소가 될 것입니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ecb15c02-3e7d-4668-a67a-4695475c0753/2201.01266v1.pdf)
[2](https://ieeexplore.ieee.org/document/11076674/)
[3](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12927/3008702/Reducing-the-impact-of-domain-shift-in-deep-learning-for/10.1117/12.3008702.full)
[4](https://ieeexplore.ieee.org/document/8995481/)
[5](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/)
[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC10903052/)
[7](https://ieeexplore.ieee.org/document/10688127/)
[8](https://www.nature.com/articles/s41746-023-00811-0)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676/)
[10](https://link.springer.com/10.1007/978-3-030-68107-4_21)
[11](https://arxiv.org/abs/2311.02583)
[12](https://www.sciencedirect.com/science/article/abs/pii/S0010482521009380)
[13](https://ieeexplore.ieee.org/document/10446700/)
[14](https://onlinelibrary.wiley.com/doi/10.1002/ima.23080)
[15](https://link.springer.com/10.1007/s11548-023-03024-8)
[16](https://ieeexplore.ieee.org/document/10821912/)
[17](https://dx.plos.org/10.1371/journal.pone.0317193)
[18](https://www.techscience.com/cmc/online/detail/24612/pdf)
[19](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2025.1467672/full)
[20](https://www.sciencedirect.com/science/article/abs/pii/S1746809423002240)
[21](https://www.scirp.org/journal/paperinformation?paperid=144861)
[22](https://melba-journal.org/2025:005)
[23](https://arxiv.org/abs/2507.09222)
[24](https://pubmed.ncbi.nlm.nih.gov/38728994/)
[25](https://arxiv.org/abs/2103.03328)
[26](https://probiologists.com/Article/recent-advances-and-challenges-in-brain-tumor-segmentation-utilizing-expansion-graph-cut)
[27](https://link.springer.com/10.1007/978-3-031-08999-2_22)
[28](https://www.scirp.org/journal/doi.aspx?doi=10.4236/jbise.2025.188024)
[29](https://www.mdpi.com/2076-3425/12/6/797)
[30](https://arxiv.org/abs/2402.07008)
[31](https://pmc.ncbi.nlm.nih.gov/articles/PMC10909192/)
[32](https://arxiv.org/pdf/2201.01266.pdf)
[33](https://arxiv.org/html/2412.06088v1)
[34](https://peerj.com/articles/cs-1867)
[35](https://arxiv.org/html/2409.08232)
[36](https://www.mdpi.com/2076-3417/14/22/10154)
[37](https://arxiv.org/abs/2409.00346)
[38](https://arxiv.org/abs/2201.01266)
[39](https://proceedings.mlr.press/v143/rajagopal21a.html)
[40](https://www.nature.com/articles/s41598-025-09833-y)
[41](https://link.springer.com/10.1007/s10489-023-05033-1)
[42](https://ojs.aaai.org/index.php/AAAI/article/view/33148)
[43](https://www.semanticscholar.org/paper/58ea57580b9cde6958e3e88e49ce70070ddb20ee)
[44](http://arxiv.org/pdf/2106.13292.pdf)
[45](https://arxiv.org/pdf/2311.02583.pdf)
[46](https://arxiv.org/pdf/2501.04741.pdf)
[47](http://arxiv.org/pdf/2401.02076.pdf)
[48](https://arxiv.org/pdf/2212.02078.pdf)
[49](https://arxiv.org/pdf/2310.20271.pdf)
[50](http://arxiv.org/pdf/2412.05572.pdf)
[51](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/)
[52](https://www.emergentmind.com/topics/swin-transformer)
[53](https://www.lightly.ai/blog/swin-transformer)
[54](https://github.com/Ziwei-Niu/Generalized_MedIA)
[55](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)
[56](https://arxiv.org/abs/2502.04748)
[57](https://www.nature.com/articles/s41598-025-95390-3)
[58](https://velog.io/@sanha9999/%EB%85%BC%EB%AC%B8%EC%9D%BD%EA%B8%B0-Swin-Transformer)
