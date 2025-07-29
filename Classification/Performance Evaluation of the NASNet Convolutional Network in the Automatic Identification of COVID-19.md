# Performance Evaluation of the NASNet Convolutional Network in the Automatic Identification of COVID-19

**개요**  
본 보고서는 2020년 국제 학술지 *International Journal on Advanced Science, Engineering and Information Technology* (Vol.10 No.2)에 게재된 “Performance Evaluation of the NASNet Convolutional Network in the Automatic Identification of COVID-19” 논문을 중심으로, 연구의 핵심 주장·기여를 간략히 요약한 뒤 문제 정의, 방법론(수식 포함), 모델 구조, 성능, 한계, 일반화 성능 향상 방안, 그리고 후속 연구에 미칠 영향까지 단계별로 상세하게 분석한다[1][2].  

## 핵심 주장과 주요 기여  

- **첫 NASNet-기반 COVID-19 X-ray 분류기 검증**: Google Brain의 Neural Architecture Search Network(NASNet)을 코로나19 흉부 X-ray 자동 진단에 처음 적용하여 97% 정확도를 달성했다고 주장[1].  
- **소형 공개 데이터 활용 효율성 입증**: 단 240장의 제한적 데이터(양성 120·정상 120)로도 고성능을 시현, 의료 영상 소규모 학습 가능성을 제시[1].  
- **고속·저가 의료 트리아지 목표**: X-ray 장비만으로 실시간 선별이 가능한 인공지능 도구의 임상 도입 가능성을 강조[1].  

## 연구 동기 및 문제 정의  

### RT-PCR 한계와 영상 진단 필요  
- RT-PCR 검사 비용·시간·거짓 음성 문제로 인해 신속 대안 필요[1].  
- 전 세계 의료기관에 보급된 흉부 X-ray 장비를 활용, 자동 분류기로 의료 인력 부족 해소 목표[1].  

### 연구 질문  
1. 제한된 공공 COVID-19 흉부 X-ray 데이터로 NASNet을 학습하면 임상적으로 유의미한 정확도를 낼 수 있는가?  
2. 기존 수작업 설계 CNN보다 NASNet의 자동 탐색 구조가 작은 데이터에서도 우수한가?  

## 데이터셋 및 전처리  

| 항목 | 세부 내용 | 근거 |
|------|-----------|------|
| 총 이미지 수 | 240장(양성 120·정상 120) | 240[1] |
| 자료 출처 | COVID-19: Cohen et al. GitHub, 정상: Kaggle Pediatric CXR | Cohen[1]; Kaggle[1] |
| 해상도 통일 | 256×256 px RGB 스케일링 | 256×256[1] |
| 정규화 | 픽셀 범위 0-255 → 0-1 | 0-1[1] |
| 학습/검증 분할 | 70%/30% 무작위 | 70/30[1] |
| 증강 | 좌우 반전·회전·노이즈 (논문 내 구체 비공개) | Aug.[1] |

## NASNet 아키텍처 선택 및 구조  

### 1. NASNet 검색 공간 개요  
NASNet은 Reinforcement Learning 기반 RNN-Controller가 **Normal Cell**과 **Reduction Cell**을 탐색·조합해 대형 네트워크를 구축한다[3].  

### 2. 논문 구현 세부  
- **총 층수**: 771  
- **입력**: 256×256×3  
- **출력 노드**: 2(정상·감염)  
- **파라미터 수**: 4,236,149(학습 가능) + 36,738(고정)[1].  

> 💡 **ScheduledDropPath** 등 NASNet 고유 regularizer는 명시되지 않았으나, 저자들은 *stochastic gradient descent* (SGD)와 *categorical cross-entropy* 손실을 사용해 10 epoch 학습[1].

### 3. 핵심 수식  
1. **Categorical Cross-Entropy**  

$$
\mathcal{L}\_{CE}
  = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}y_{i,c}\,\log \hat{y}_{i,c}\,[1]
$$

2. **Mean Squared Error**  

$$
\mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2[1]
$$

## 학습 설정 및 평가 지표  

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 학습률 | 0.01(SGD 기본) | 논문 명시 | 
| 배치 크기 | 32(추정) | 논문 미표기 |
| Epoch | 10 | 합리적 수렴[1] |
| 평가지표 | Loss, Accuracy, Precision, Recall, F1-score, ROC-AUC | 모델 성능 종합[1] |

## 실험 결과 분석  

### 1. 학습 곡선  
- **Loss**: 4epoch까지 급감, 이후 완만히 감소 후 수렴(과적합 징후 없음)[1].  
- **Accuracy**: 학습·검증 모두 0.97 부근에서 포화, 곡선 평행 유지[1].

### 2. 혼동 행렬 요약  

| 실제\예측 | 정상 | 감염 |
|-----------|------|------|
| 정상      | 35   | 1    |
| 감염      | 2    | 32   |

- **Precision**: 0.97  
- **Recall**: 0.97  
- **F1-score**: 0.97[1].

### 3. ROC-AUC  
- 클래스별 AUC≈0.99, 평균 AUC≈0.98(상세 그래프 Fig.9-10)[1].

### 4. 타 연구 대비  

| 연구 | 아키텍처 | 데이터 크기 | 정확도 | 비고 |
|------|----------|-------------|--------|------|
| Martínez et al. 2020 | NASNet | 240 | 97% | 본 논문[1] |
| Li et al. 2020 | ResNet-50 | 4,356 | 94.3% | 대규모 CT[4] |
| Turk & Kökver 2022 | DenseNet | 7,200 | 93.38% | 3-class[5] |
| Gozes et al. 2020 | ResNet-50 | 1,044 CT | 95% | CT기반[6] |

> 정확도만 보면 본 논문이 소규모 데이터 대비 준수한 성능을 보이나, **외부 검증 세트 부재**라는 명백한 한계가 존재한다.

## 모델 성능 한계와 일반화 향상 가능성  

### 한계 요약  
1. **데이터 부족**: 240장은 과적합 위험.  
2. **클래스 편향**: 두 클래스 균형이나 실제 임상은 다중 질환(폐렴, 결핵) 혼재.  
3. **해상도 축소**: 256×256으로 정보 손실 가능.  
4. **외부 기관 테스트 부재**: 도메인 시프트 미평가.  

### 일반화 성능 향상 방안  

| 범주 | 구체 전략 | 예상 효과 |
|------|----------|----------|
| 데이터 | - 대규모 공개 CXR(COVIDx8B·BIMCV) 병합- Semi-supervised pseudo-labeling | 다양성·샘플 수↑ → 분산 감소 |
| 모델 | - Transfer learning(NASNet-Mobile) fine-tune- Ensemble(EfficientNet+NASNet) | 특성 보완·분류 경계 안정화 |
| 최적화 | - ScheduledDropPath·Label Smoothing- Early Stopping + Cyclical LR | Regularization 강화 |
| 도메인 | - Test-Time Augmentation(TTA)- Unsupervised Domain Adaptation(MMD) | 병원 간 시프트 완화 |
| 설명성 | - Grad-CAM 기반 ROI 시각화- LIME 융합 NASNet-ViT 연구 적용[7] | 임상 신뢰도 확보 |

## 연구의 학문·산업적 영향  

1. **AutoML 기반 의료 AI 가능성 부각**: NASNet의 자동 설계 블록을 임상의 문제에 직접 이식 가능함을 증명해 “작은 데이터-대규모 모델” 패러다임 확장[3].  
2. **데이터 스케일 대응 연구 촉진**: 후속 연구들은 대형 CXR 코호트와 Transformer 혼합 구조(NASNet-ViT)[8] 등으로 일반화 연구를 진전.  
3. **임상 트리아지 도구 개발**: 저비용 X-ray 판독 보조 시스템 상용화 논의 촉발[9][10].  
4. **설명가능성·윤리 이슈**: 제한적 데이터·불투명 결정 과정의 안전성, 편향 검증 프레임워크 필요성을 제기[11].

## 후속 연구 시 고려 사항  

- **다중 질환 다중 클래스 확장**: COVID-19·폐렴·결핵·정상 등 현실 진단 시나리오 반영.  
- **대륙·연령·기기 다양성 확보**: 외부 테스트에서 민감도 드롭 방지.  
- **CT·X-ray 멀티모달 융합**: 하나의 NASNet 백본에 2D-3D 혼합 입력 실험.  
- **경량화·엣지 배포**: NASNet-Mobile, pruning, Quantization-aware training으로 모바일 촬영기 연동.  
- **설명가능 AI(E-XAI)**: Grad-CAM, Layer-wise Relevance Propagation(LRP)로 병변 시각 근거 제공.  
- **규제·윤리 준수**: GDPR·K-HIPAA 등 의료 데이터 프라이버시, 임상 시험(ISO 14155) 준수.

### 결론적 시사점  

본 논문은 **NASNet 자동 탐색 셀**이 제한적 흉부 X-ray에서도 **97% 정확도**를 달성함을 입증하여, **소규모 데이터 환경에서 AutoML의 의료 영상 적용 가능성**을 선도적으로 제시했다[1]. 다만 **데이터 규모·외부 검증·설명성** 한계가 뚜렷하므로, 후속 연구는 대규모 다기관 데이터·일반화 기법·임상 현장 사용성 검증을 통해 **신뢰할 수 있는 의사결정 지원 시스템**으로 발전시켜야 한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/768bb5ca-af26-49b0-890b-34d2c4c5a7b7/sriatmaja-32.-Fredy-MartA_nez-11446-AAP.pdf
[2] https://ijaseit.insightsociety.org/index.php/ijaseit/article/view/11446
[3] https://arxiv.org/abs/1707.07012
[4] https://velog.io/@cosmos42/NASNet-Learning-Transferable-Architectures-for-Scalable-Image-Recognition
[5] http://saucis.sakarya.edu.tr/en/download/article-file/2301158
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC8614951/
[7] https://www.sciencedirect.com/science/article/abs/pii/S1746809424001721
[8] https://arxiv.org/pdf/2502.20570.pdf
[9] https://link.springer.com/10.1007/s00530-022-00917-7
[10] https://www.nature.com/articles/s41598-024-80826-z
[11] https://www.nature.com/articles/s41598-023-47038-3
[12] https://link.springer.com/10.1007/978-3-030-71187-0_59
[13] https://link.springer.com/10.1007/s42979-022-01545-8
[14] https://journals.sagepub.com/doi/10.1177/14604582241290724
[15] https://link.springer.com/10.1007/s10278-022-00754-0
[16] https://www.frontiersin.org/articles/10.3389/fpubh.2023.1297909/full
[17] https://ieeexplore.ieee.org/document/10604592/
[18] https://ieeexplore.ieee.org/document/10151051/
[19] https://www.jmir.org/2021/4/e27468
[20] https://pmc.ncbi.nlm.nih.gov/articles/PMC8078376/
[21] http://www.insightsociety.org/ojaseit/index.php/ijaseit/article/download/11446/2344
[22] https://pmc.ncbi.nlm.nih.gov/articles/PMC8382139/
[23] https://pmc.ncbi.nlm.nih.gov/articles/PMC7831681/
[24] https://www.worldscientific.com/doi/10.1142/S1793557122502400
[25] https://pubmed.ncbi.nlm.nih.gov/33162872/
[26] https://pmc.ncbi.nlm.nih.gov/articles/PMC9088453/
[27] https://www.cognex.com/ko-kr/blogs/deep-learning/research/learning-transferable-architectures-scalable-image-recognition-review
[28] https://www.worldscientific.com/doi/pdf/10.1142/S1793557122502400
[29] https://pure.kaist.ac.kr/en/publications/deep-learning-covid-19-features-on-cxr-using-limited-training-dat
[30] https://hongl.tistory.com/52
[31] https://www.sciencedirect.com/science/article/pii/S2667099222000172
[32] https://arxiv.org/abs/2004.02060
[33] https://www.mdpi.com/2504-4990/2/4/27/pdf
[34] https://arxiv.org/pdf/2108.03131.pdf
[35] https://pmc.ncbi.nlm.nih.gov/articles/PMC10975406/
[36] https://pmc.ncbi.nlm.nih.gov/articles/PMC7384689/
[37] https://www.frontiersin.org/articles/10.3389/fcvm.2025.1450470/full
[38] https://s3.ca-central-1.amazonaws.com/assets.jmir.org/assets/preprints/preprint-45367-accepted.pdf
[39] https://academic.oup.com/nar/article-pdf/49/W1/W619/38841899/gkab417.pdf
[40] https://www.mdpi.com/1660-4601/18/6/3195/pdf
