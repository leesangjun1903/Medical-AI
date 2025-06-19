# Explanatory classification of CXR images into COVID-19, Pneumonia and Tuberculosis using deep learning and XAI

## 딥러닝과 XAI를 활용한 흉부 X선 이미지의 COVID-19, 폐렴, 결핵 분류 논문 설명

이 연구는 흉부 X선(CXR) 이미지를 활용해 COVID-19, 폐렴, 결핵을 자동으로 분류하는 동시에 모델의 결정 과정을 설명하는 통합 프레임워크를 제안합니다. 의료진이 AI 모델의 진단 근거를 이해할 수 있도록 설계된 이 접근법은 높은 정확도와 해석 가능성을 동시에 달성합니다[2][6].

### 연구 배경 및 목적
- **문제 인식**: 흉부 X선은 폐 질환 진단의 핵심 도구이나, 숙련된 영상의학과 전문의 부족으로 신속한 진단이 어려움[1].
- **해결 목표**: 딥러닝 기반 자동 분류 시스템 구축과 동시에 "블랙박스" 모델의 결정을 설명하는 XAI(eXplainable AI) 통합[2][6].

### 제안 방법론
1. **딥러닝 모델 구조**:
   - 경량화된 단일 CNN(Convolutional Neural Network) 아키텍처 채택
   - 7,132장의 공개 CXR 데이터셋으로 학습(COVID-19/폐렴/결핵/정상 클래스)[2][6]
   - 10겹 교차 검증 적용으로 과적합 방지[2]

2. **XAI 기법 통합**:
   - **Grad-CAM**: 결정에 중요한 이미지 영역을 열지도(heatmap)로 시각화
   - **LIME**: 개별 예측에 영향을 미치는 지역적 특징 설명
   - **SHAP**: 피처 기여도를 샤플리 값으로 정량화[2][6]

### 실험 결과
- **분류 정확도**:
  | 평가 지표 | 값 |
  |---|---|
  | 학습 정확도 | 95.76% ± 1.15% |
  | 검증 정확도 | 94.54% ± 1.33% |
  | 테스트 정확도 | 94.31% ± 1.01% |
- **의료 전문가 검증**: 생성된 설명(열지도 등)을 영상의학과 전문가가 평가하여 임상적 타당성 확인[2][6]
- **기존 연구 대비 장점**: 복잡한 앙상블 모델 대비 경량 아키텍처로 높은 정확도 유지[6]

### 임상적 의의
- **신속한 스크리닝**: 리소스가 제한된 환경에서 1차 진단 도구로 활용 가능[4]
- **의사 결정 지원**: 의료진이 AI의 판단 근거를 확인하며 진단 신뢰도 향상[2][3]
- **다중 질환 대응**: 단일 모델로 COVID-19/폐렴/결핵 동시 감별 가능[5]

### 한계 및 향후 과제
- **데이터 다양성 부족**: 소규모 데이터셋으로 학습된 모델의 일반화 성능 검증 필요[6]
- **추가 임상 정보 통합**: 환자 병력, 증상 등 비영상 데이터 결합 미흡[2]
- **고급 증강 기법**: GAN(Generative Adversarial Network) 활용 데이터 다양성 확대 방안 제시[6]

이 연구는 AI의 설명 가능성을 높여 의료진과의 협진을 용이하게 하는 동시에, 경량 모델로 실용적인 배포 가능성을 입증했습니다[2][6]. 향후 대규모 다기관 데이터 검증과 환자 임상 정보 통합이 추가 과제입니다.

[1] https://arxiv.org/abs/2303.16754
[2] https://acquire.cqu.edu.au/articles/journal_contribution/Explanatory_classification_of_CXR_images_into_COVID-19_Pneumonia_and_Tuberculosis_using_deep_learning_and_XAI/25522738
[3] https://ieeexplore.ieee.org/document/10586202/
[4] https://ccsenet.org/journal/index.php/ijsp/article/view/0/51046
[5] https://onlinelibrary.wiley.com/doi/10.1002/ima.23014
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC9549800/
[7] https://linkinghub.elsevier.com/retrieve/pii/S0010482522008642
[8] https://link.springer.com/10.1007/s40745-025-00601-3
[9] https://www.sciencedirect.com/science/article/pii/S2405844024028329
[10] https://www.medrxiv.org/content/10.1101/2020.11.24.20235887v1.full
[11] https://www.semanticscholar.org/paper/3326768646178d756b72badeb66e28d61d62643c
[12] https://ieeexplore.ieee.org/document/10441082/
[13] https://www.ssrn.com/abstract=4379837
[14] https://link.springer.com/10.1007/s11517-022-02746-2
[15] https://ph02.tci-thaijo.org/index.php/tsujournal/article/view/253662
[16] https://www.sciencedirect.com/science/article/abs/pii/S0010482522008642
[17] https://pubmed.ncbi.nlm.nih.gov/36228463/
[18] https://pmc.ncbi.nlm.nih.gov/articles/PMC9914163/
[19] https://www.nature.com/articles/s41598-021-95680-6
[20] https://arxiv.org/html/2404.11428v1
[21] https://onlinelibrary.wiley.com/doi/10.1155/2022/4254631
[22] https://ieeexplore.ieee.org/document/9944138/
[23] https://www.mdpi.com/2227-7080/11/5/134
[24] https://link.springer.com/10.1007/s11042-021-11748-5
[25] https://pmc.ncbi.nlm.nih.gov/articles/PMC9281341/
[26] https://arxiv.org/pdf/2303.16754.pdf
[27] https://pmc.ncbi.nlm.nih.gov/articles/PMC10668574/
[28] https://pmc.ncbi.nlm.nih.gov/articles/PMC9485026/
[29] https://dl.acm.org/doi/10.1016/j.compbiomed.2022.106156
[30] https://www.medrxiv.org/content/10.1101/2020.06.21.20136598v1.full
[31] https://go.gale.com/ps/i.do?aty=open-web-entry&id=GALE%7CA772533666&it=r&p=AONE&sid=sitemap&sw=w&userGroupName=anon~7f509489&v=2.1
[32] https://www.scirp.org/(S(vtj3fa45qm1ean45%20vvffcz55))/journal/paperinformation?paperid=118603
