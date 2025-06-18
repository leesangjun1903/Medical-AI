# 논문 요약: DeBoNet — 흉부 X선 영상에서 질병 검출 향상을 위한 딥러닝 기반 뼈 억제 모델 앙상블

## 연구 배경 및 목적

- 흉부 X선(CXR) 영상에서 갈비뼈와 쇄골 같은 뼈 구조가 폐 조직의 이상 소견(예: COVID-19, 결핵 등)을 가려 진단 정확도를 저해할 수 있음[1][2].
- 본 논문은 딥러닝 기반 뼈 억제 모델 앙상블(DeBoNet)을 개발하여, X선 영상에서 뼈 구조를 효과적으로 억제하고 질병 검출 성능을 높이는 것을 목표로 함[1][2].

## 연구 방법

### 데이터셋

- NIH Clinical Center에서 수집한 듀얼 에너지 서브트랙션(DES) X-ray 데이터(학습 및 평가용), 공개 COVID-19 CXR 데이터셋, RSNA 정상 CXR 데이터셋 등 다양한 데이터를 활용함[1].
- 데이터 증강(회전, 이동, 확대, 블러 등) 및 전처리(명암 대비 향상, 정규화) 적용[1].

### 뼈 억제 모델

- 다양한 딥러닝 구조(Autoencoder, U-Net, Feature Pyramid Network(FPN), ResNet 등)로 뼈 억제 모델을 설계 및 학습[1].
- 모델 성능은 PSNR, SSIM, MS-SSIM, 상관계수, 히스토그램 교차, 카이제곱, Bhattacharyya 거리 등 다양한 지표로 평가[1].
- 최상위 3개 모델(FPN-EfficientNetB0, FPN-ResNet18, U-Net-ResNet18)을 선정하여, 각 모델의 예측 결과를 소블록 단위로 분할(M×M) 후 MS-SSIM 기반 다수결로 최종 합성 이미지를 생성하는 앙상블 방식(DeBoNet) 적용[1].

### 질병 분류 모델

- 뼈 억제 모델의 encoder를 활용해 폐 영역을 분할하고, 폐 ROI만을 대상으로 COVID-19 등 질병 분류를 수행[1].
- 뼈 억제 이미지와 원본 이미지를 각각 학습시켜 분류 성능 비교[1].

## 주요 결과

### 뼈 억제 성능

- FPN-EfficientNetB0 기반 모델이 단일 모델 중 최고 성능(PSNR 36.55, MS-SSIM 0.984 등)을 기록[1].
- DeBoNet(앙상블)은 소블록 크기 4×4에서 PSNR 36.80, MS-SSIM 0.9848 등 단일 모델 대비 모든 지표에서 우수한 성능을 보임[1].
- 통계적으로도 DeBoNet이 단일 모델 대비 유의미하게 낮은 카이제곱 값을 기록, 실제 뼈 억제 이미지가 정답(soft-tissue) 이미지와 가장 유사함[1].

### 질병 분류 성능

- 뼈 억제 이미지를 사용한 분류 모델이 원본 이미지를 사용한 모델 대비 모든 지표(정확도, AUROC, 정밀도, 재현율, F1-score, MCC 등)에서 유의하게 우수함[1].
- 예를 들어, COVID-19 검출에서 뼈 억제 모델의 MCC는 0.9645(95% CI: 0.9510~0.9780)로, 원본 이미지 모델(MCC 0.7961, 95% CI: 0.7667~0.8255)보다 크게 향상됨[1].

| 모델           | PSNR     | MS-SSIM  | MCC (COVID-19 분류) |
|----------------|----------|----------|---------------------|
| FPN-EfficientNetB0 | 36.55±1.69 | 0.9840±0.0081 | -                   |
| DeBoNet (4×4)  | 36.80±1.62 | 0.9848±0.0073 | 0.9645              |
| 원본 이미지    | -        | -        | 0.7961              |

## 결론 및 의의

- DeBoNet 앙상블은 흉부 X선 영상에서 뼈 구조를 효과적으로 억제하여, 폐 조직의 미세 병변 가시성과 질병 분류 성능을 크게 향상시킴[1][2].
- 뼈 억제 기술은 COVID-19뿐 아니라 다양한 폐 질환의 자동 진단 보조 시스템에 활용 가능성이 높음[1].
- 논문에서 제안한 코드와 모델은 오픈소스로 공개되어 있어, 연구 재현 및 확장 연구에 활용할 수 있음[1].

---

**참고:** 논문 전체 및 오픈소스 코드는 [여기](https://github.com/sivaramakrishnan-rajaraman/Bone-Suppresion-Ensemble)에서 확인할 수 있음[1].

[1][2]

[1] https://dx.plos.org/10.1371/journal.pone.0265691
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC8970404/
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC8611463/
[4] https://arxiv.org/pdf/1810.07500.pdf
[5] https://arxiv.org/pdf/2002.03073.pdf
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC6510604/
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC9246721/
[8] http://arxiv.org/pdf/2111.03404.pdf
[9] https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0265691
[10] https://pubmed.ncbi.nlm.nih.gov/35358235/
[11] https://github.com/sivaramakrishnan-rajaraman/Bone-Suppresion-Ensemble
[12] https://www.ntis.go.kr/outcomes/popup/srchTotlPapr.do?cmd=get_contents&rstId=JNL-2022-00112779548
[13] https://pubmed.ncbi.nlm.nih.gov/27589577/
[14] https://pubmed.ncbi.nlm.nih.gov/32275586/
[15] https://pmc.ncbi.nlm.nih.gov/articles/PMC9890286/
[16] https://arxiv.org/abs/2104.04518
[17] https://journals.plos.org/plosone/article/figures?id=10.1371%2Fjournal.pone.0265691
[18] https://www.sciencedirect.com/science/article/abs/pii/S0169260722004060
[19] https://escholarship.org/uc/item/6zq7w1j2
