# Chest X-ray bone suppression toward improving classification and detection of Tuberculosis-consistent findings

## 연구 배경 및 목적

- 흉부 X-ray(CXR)는 폐 질환 진단에 널리 사용되지만, 갈비뼈와 쇄골 같은 뼈 구조가 폐의 미세한 이상 소견을 가려 진단 오류를 유발할 수 있음[1][2].
- 특히 결핵(TB) 진단에서 뼈 구조가 폐 상부(apical) 영역의 병변을 가려 놓치기 쉬움[1][2].
- 본 연구는 딥러닝 기반 뼈 억제(bone suppression) 모델을 개발하여 X-ray에서 뼈 구조를 제거함으로써 결핵 일치 소견의 분류 및 검출 성능을 향상시키는 것을 목표로 함[1][2].

## 연구 방법

### 데이터셋

- JSRT, NIH–CC–DES, Shenzhen TB, Montgomery TB, RSNA, Pediatric pneumonia 등 다양한 공개 X-ray 데이터셋을 활용[1][2].
- JSRT 데이터셋에서 뼈 억제 이미지를 생성하고, NIH–CC–DES 데이터로 모델 성능을 교차 검증함[1][2].
- 결핵 분류 실험에는 Shenzhen(중국)과 Montgomery(미국) 결핵 데이터셋을 사용[1][2].

### 뼈 억제 모델

- 네 가지 딥러닝 구조(Autoencoder, Sequential ConvNet, Residual Learning, Residual Network(ResNet))의 뼈 억제 모델을 설계 및 비교함[1][2].
- ResNet 기반 모델(ResNet–BS)이 가장 뛰어난 성능을 보여, 이후 실험에 사용됨[1][2].

### 분류 모델

- VGG-16 기반 모델을 대규모 X-ray 데이터로 사전학습(pretrain) 후, 결핵 데이터셋에 미세조정(fine-tuning)하여 정상/결핵 분류를 수행함[1][2].
- 뼈 억제 이미지를 적용한 경우와 원본 이미지를 각각 학습시켜 성능을 비교함[1][2].

## 주요 결과

### 뼈 억제 성능

- ResNet–BS 모델이 가장 낮은 오차(Combined Loss, MAE)와 가장 높은 구조적 유사도(SSIM, MS–SSIM), PSNR을 기록함[1][2].
- 뼈 억제 이미지는 원본 대비 뼈 구조가 효과적으로 제거되면서도 폐 조직의 정보는 잘 보존됨[1][2].

### 결핵 분류 성능

- 뼈 억제 이미지를 사용한 모델이 원본 이미지를 사용한 모델보다 모든 지표(정확도, AUC, 민감도, 특이도, 정밀도, F1-score, MCC)에서 유의하게 우수함[1][2].
- Shenzhen 데이터셋에서 AUC가 0.9535, Montgomery 데이터셋에서 0.9635로, 각각 원본 대비 약 6~11% 향상됨[1][2].
- 통계적으로도 뼈 억제 모델이 분류 성능에서 유의미하게 우수함(p < 0.05)[1][2].

| 데이터셋     | 모델          | 정확도(ACC) | AUC    | 민감도 | 특이도 | F1-score | MCC    |
|--------------|--------------|-------------|--------|--------|--------|----------|--------|
| Shenzhen     | 뼈 억제      | 0.8879      | 0.9535 | 0.8805 | 0.8954 | 0.8873   | 0.7765 |
| Shenzhen     | 원본         | 0.8304      | 0.8991 | 0.8068 | 0.8537 | 0.8259   | 0.6620 |
| Montgomery   | 뼈 억제      | 0.9230      | 0.9635 | 0.8772 | 0.9687 | 0.9188   | 0.8539 |
| Montgomery   | 원본         | 0.7701      | 0.8567 | 0.7991 | 0.7411 | 0.7682   | 0.5537 |

## 결론 및 의의

- 딥러닝 기반 뼈 억제 모델은 X-ray에서 뼈 구조를 효과적으로 제거해 미세 병변의 가시성을 높이고, 결핵 분류 및 검출 성능을 크게 향상시킴[1][2].
- 뼈 억제 기술은 결핵뿐 아니라 다양한 폐 질환의 자동 진단 보조 시스템에서 활용 가능성이 높음[1][2].
- 특히 의료 인프라가 부족한 지역에서 X-ray 기반 결핵 스크리닝의 정확도를 높이는 데 중요한 역할을 할 수 있음[1][2].

---

**참고:** 본 논문은 실제 코드와 모델을 오픈소스로 제공하고 있어, 연구 재현 및 확장 연구에 활용할 수 있음[1][2].

[1][2]

[1] https://www.mdpi.com/2075-4418/11/5/840
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC8151767/
[3] https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-018-0544-y
[4] https://arxiv.org/abs/2104.04518
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC11640000/
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC8906182/
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC4531444/
[8] https://pmc.ncbi.nlm.nih.gov/articles/PMC8870832/
[9] https://www.mdpi.com/resolver?pii=diagnostics11050840
[10] https://pubmed.ncbi.nlm.nih.gov/34067034/
[11] https://www.mdpi.com/2075-4418/11/5/840/review_report
[12] https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression/blob/main/README.md
[13] https://pubmed.ncbi.nlm.nih.gov/29860192/
[14] https://scholar.kyobobook.co.kr/article/detail/4010025912536
[15] https://link.springer.com/10.1007/978-3-030-87602-9_6
[16] https://www.semanticscholar.org/paper/a0b3c32cb96214d6ff6792777dd48e1bb444d02d
[17] https://www.semanticscholar.org/paper/Improved-TB-classification-using-bone-suppressed-Rajaraman-Zamzmi/a0b3c32cb96214d6ff6792777dd48e1bb444d02d
[18] https://pmc.ncbi.nlm.nih.gov/articles/PMC9568934/
[19] https://www.sciencedirect.com/science/article/abs/pii/S0895611123000046


이 연구에서는 단계별 체계적 방법론을 제안합니다.  
첫째, ImageNet으로 훈련된 VGG-16 모델을 공개적으로 사용 가능한 CXR의 대규모, 다양한 결합 선택에서 재훈련하여 CXR 모달리티별 기능을 학습하도록 돕습니다. 학습된 지식은 공개적으로 사용 가능한 심천 및 몽고메리 TB 컬렉션에서 정상 폐 또는 폐결핵 증상을 보이는 CXR을 분류하는 관련 대상 분류 작업에서 성능을 개선하는 데 전달됩니다. 다음으로, 일본 방사선 기술 협회(JSRT) CXR 데이터 세트와 뼈 억제된 대응 데이터 세트에서 다양한 아키텍처를 가진 여러 뼈 억제 모델을 훈련합니다. 훈련된 모델의 성능은 기관 간 국립 보건원(NIH) 임상 센터(CC) **dual-energy subtraction (DES)** CXR 데이터 세트를 사용하여 테스트합니다. 성능이 가장 좋은 모델은 심천 및 몽고메리 TB 컬렉션에서 뼈를 억제하는 데 사용됩니다. 그런 다음 우리는 여러 성능 지표를 사용하여 뼈 억제되지 않은 몽고메리 TB 데이터 세트와 뼈 억제된 몽고메리 TB 데이터 세트로 훈련된 CXR 재훈련된 VGG-16 모델의 성능을 비교하고 통계적으로 유의미한 차이를 분석했습니다. 뼈 억제되지 않은 모델과 뼈 억제된 모델의 예측은 클래스 선택적 관련성 맵(CRM)을 통해 해석됩니다.

# Reference
https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression
