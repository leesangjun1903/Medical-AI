# "Improving the Interpretability of GradCAMs in Deep Classification Networks" 논문 상세 분석

## 개요

본 논문은 Alfred Schöttl이 2022년 Procedia Computer Science에 발표한 연구로, 딥러닝 분류 네트워크에서 GradCAM의 해석성을 향상시키는 방법론을 제시합니다[1][2]. 이 연구는 산업용 AI 응용 분야에서 비용과 안전이 중요한 상황에서 AI 결과의 설명 가능성이 높은 수요를 받고 있다는 배경에서 출발합니다[1].

## 연구 배경 및 동기

### 산업용 AI의 설명 가능성 필요성

딥러닝 분류 네트워크는 산업용 AI 응용의 백본 네트워크로서 중요한 역할을 합니다[1]. 이러한 응용 분야는 종종 비용이나 안전에 중요한 영향을 미치기 때문에, AI 결과의 설명 가능성은 매우 요구되는 특성입니다[1]. 특히 의료 분야와 같은 고위험 도메인에서는 AI 시스템의 투명성과 신뢰성 확보가 필수적입니다[3].

### 기존 GradCAM의 한계

기존의 GradCAM과 관련 방법들(CAM, GradCAM++ 등)은 활성화 영역을 지역화하여 분류가 어떻게 작동하는지 설명하도록 설계되었습니다[4]. 그러나 이러한 방법들은 모델에 구애받지 않는 민감도 기반 방법으로 특성화되지만, 약한 특징에 대한 해석성 개선에는 한계가 있었습니다[4].

## 핵심 방법론: CAM Fostering

### CAM Fostering의 개념

연구진이 제안한 CAM Fostering은 합성곱 층이나 풀링 층과 같은 로컬 층을 기반으로 한 분류 네트워크의 설명 가능성을 향상시키는 방법입니다[1][5]. 이 방법은 훈련 과정에서 (Grad)CAM 맵을 고려하여 훈련 손실을 수정하며, 추가적인 구조적 요소가 필요하지 않습니다[4].

### 핵심 기술적 특징

CAM Fostering 방법의 주요 특징은 다음과 같습니다[4]:

- **경량화**: 임베디드 시스템과 표준 깊은 아키텍처에 적용 가능
- **2차 도함수 활용**: 훈련 중 2차 도함수를 이용하여 추가 모델 층 불필요
- **지역화된 GradCAM 활성화 맵 촉진**: 약한 특징에 대해서도 CAM의 해석성 향상

## 해석성 측정 지표

### 3가지 CAM 해석성 측정치

연구진은 로컬 CAM 해석성을 정량화하기 위해 세 가지 측정치를 제안했습니다[4]:

1. **CAM 엔트로피 (ce)**: 활성화된 영역의 양을 측정
   - 수식: $$ce(M) = -∑ \bar M_{ij} ln \bar M_{ij}$$
   - 더 낮은 값이 더 선명한 CAM을 의미

2. **CAM 타원형 영역 (ca)**: 활성화된 영역의 집중도를 측정
   - 수식: $$ca(M) = \sqrt {λ1λ2} $$
   - 공분산 행렬의 고유값을 활용

3. **CAM 분산 (cd)**: 결정의 고유성을 측정
   - 수식: $$cd(M) = \frac{σ²M}{μ²M + \epsilon}$$
   - 더 높은 값이 더 고유한 결정을 의미

### 측정치의 의미

이 세 측정치는 근본적으로 다른 특성들을 다룹니다[4]. 엔트로피는 (이웃과 무관하게) 활성화된 영역의 양을 측정하고, 타원형 영역은 활성화된 영역의 집중도를 측정하며, 분산은 결정의 고유성을 측정합니다[4].

## 네트워크 손실 및 훈련 방법

### 손실 함수 수정

연구진은 기존의 교차 엔트로피 손실 함수에 엔트로피 손실 항을 추가하여 수정했습니다[4]:

```
l = ltrain + β ce(LGradCAM_c)
```

여기서 β는 해석성과 성능 간의 균형을 조절하는 하이퍼파라미터입니다[4].

### 2차 최적화 방법

흥미롭게도, 일반적인 그래디언트 디센트나 Adam과 같은 최적화 알고리즘을 적용하면 2차 방법으로 이어진다는 것이 주목할 만합니다[4]. 이는 CAM 엔트로피가 활성화 맵에서 활성 영역의 크기를 줄이고 맵을 선명하게 만들기 때문입니다[4].

## 실험 설계 및 결과

### 실험 설정

연구진은 다음과 같은 실험 설정을 사용했습니다[4]:

- **모델**: ImageNet에서 사전 훈련된 ResNet50 (50층, 2,350만 가중치)
- **데이터셋**: PASCAL VOC 2012 (192x192 픽셀로 크기 조정)
- **클래스**: 20개 객체 클래스
- **훈련**: 30 에포크, Adam 최적화기, 학습률 0.001

### 주요 실험 결과

실험 결과는 β=0 (기존 방법)과 β=100 (제안 방법) 간의 명확한 차이를 보여줍니다[4]:

| 지표 | β=0 | β=100 | 개선률 |
|------|-----|-------|--------|
| 평균 타원형 활성화 영역 | 기준값 | 20% 감소 | -20% |
| 분산 | 기준값 | 6배 증가 | +500% |
| 상대적 차이 | 33% | 114% | +246% |

### 시각적 결과 분석

실험의 한 예시에서 "개" 클래스에 대한 GradCAM 시각화를 비교한 결과[4]:

- **β=0**: 활성화와 비활성화 영역 간 상대적 차이 32%
- **β=100**: 활성화와 비활성화 영역 간 상대적 차이 114%

이는 제안된 방법이 더 의미 있는 부분(이 경우 주둥이와 눈)에 집중한다는 것을 보여줍니다[4].

## 계산 복잡도 및 성능

### 훈련 시간 분석

nVidia 1080Ti 그래픽 카드에서 Tensorflow2를 사용한 계산 결과[4]:

- **기존 손실 함수**: 에포크당 4분 35초
- **확장된 손실 함수**: 에포크당 8분 22초

이는 약 83%의 훈련 시간 증가를 의미하지만, 해석성 향상을 고려하면 합리적인 수준입니다[4].

### 성능 트레이드오프

연구 결과에 따르면, β=0일 때 최상의 정확도를 제공하지만, β 값이 클수록 더 지역화되고 설명 가능한 모델을 생성합니다[4]. 실제로 적당한 β 값에서의 성능 저하는 작다고 보고되었습니다[4].

## 관련 연구와의 비교

### Attention Branch Network (ABN)과의 관계

CAM Fostering은 Attention Branch Network (ABN)와 밀접한 관련이 있습니다[5]. ABN은 네트워크에 어텐션 브랜치를 추가하여 CAM 특징을 재생산하고 최적화하는 방법인 반면[4], CAM Fostering은 추가 층 없이 손실 함수 수정만으로 해석성을 향상시킵니다[4].

### 다른 설명 가능성 방법들과의 차이

기존의 SHAP 계열 방법들과 달리[4], CAM Fostering은 훈련 과정에서 직접적으로 해석성을 개선하는 접근법을 취합니다[4]. 이는 사후 설명 방법이 아닌 내재적 해석성 향상 방법으로 분류할 수 있습니다[6].

## 실제 응용 및 의의

### 의료 영상 분야 적용

CAM Fostering 기법은 특히 의료 영상 분야에서 중요한 응용 가능성을 보입니다[5]. 조직학적 이미지 분류에서 제안된 모델이 ResNet-50을 사용하여 평균 ADCC 지표에서 4.16% 개선과 일관성 지표에서 3.88% 개선을 달성했습니다[5].

### 산업용 AI 시스템에서의 활용

산업용 AI 응용에서 이 방법의 중요성은 비용과 안전이 중요한 환경에서 AI 결과의 설명 가능성이 높은 수요를 받고 있다는 점에 있습니다[1]. 특히 임베디드 시스템에서도 적용 가능한 경량화된 특성이 실용적 가치를 높입니다[4].

## 한계점 및 향후 연구 방향

### 현재 방법의 한계

1. **성능 저하**: 해석성 향상과 분류 성능 간의 트레이드오프가 존재합니다[4]
2. **하이퍼파라미터 조정**: β 값의 적절한 설정이 모델 성능에 중요한 영향을 미칩니다[4]
3. **단일 측정치 최적화**: 현재는 주로 엔트로피 기반 최적화에 집중되어 있습니다[4]

### 향후 개선 방향

연구진은 ca와 cd와 같은 더 복잡한 해석성 측정치들도 사용할 수 있지만, 계산 비용이 상당히 증가한다고 언급했습니다[4]. 향후 연구에서는 이러한 측정치들을 효율적으로 활용하는 방법을 개발할 필요가 있습니다[4].

## 결론

"Improving the Interpretability of GradCAMs in Deep Classification Networks" 논문은 딥러닝 모델의 해석성 향상을 위한 실용적이고 효과적인 방법을 제시합니다[1][2]. CAM Fostering 기법은 추가적인 모델 구조 변경 없이 손실 함수 수정만으로 GradCAM의 해석성을 크게 향상시킬 수 있음을 보여주었습니다[4].

이 연구의 주요 기여는 다음과 같습니다[4]:

1. **새로운 GradCAM 기반 해석성 측정치** 도입
2. **경량화된 해석성 향상 방법** 제안
3. **산업용 AI 시스템**에서의 실용적 적용 가능성 입증

특히 의료 영상[5]과 같은 고위험 도메인에서 AI 시스템의 투명성과 신뢰성을 높이는 데 중요한 기여를 할 것으로 기대됩니다[3]. 비록 성능과 해석성 간의 트레이드오프가 존재하지만, 적절한 하이퍼파라미터 조정을 통해 실용적인 수준에서 두 목표를 모두 달성할 수 있다는 점이 이 연구의 핵심 가치입니다[4].

[1] https://www.sciencedirect.com/science/article/pii/S1877050922002691
[2] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART117467290
[3] https://discuss.pytorch.org/t/second-order-derivatives-of-loss-function/71797
[4] https://arxiv.org/pdf/2009.12546.pdf
[5] https://www.scitepress.org/Papers/2024/125957/125957.pdf
[6] https://www.schott.com/en-gb/solutions-magazine/edition-3-2024/unlocking-the-full-potential-of-ai-in-healthcare
[7] https://ko.wikipedia.org/wiki/%EA%B4%84%ED%98%B8
[8] https://bsscco.github.io/posts/2019-02-21-how-read-the-brackets/
[9] https://en.wikipedia.org/wiki/Bracket
[10] https://www.youtube.com/watch?v=_BXIG5H4X0w
[11] https://www.britannica.com/topic/I-letter
[12] https://www.abbreviations.com/M
[13] https://www.abbreviations.com/P
[14] https://www.r-project.org/about.html
[15] https://ko.wikipedia.org/wiki/%C3%93
[16] https://www.youtube.com/watch?v=hWiT0UnOMn4
[17] https://ouci.dntb.gov.ua/en/works/7XqLnjq4/
[18] https://arxiv.org/html/2405.12175v1
[19] https://deepai.org/publication/grad-cam-visual-explanations-from-deep-networks-via-gradient-based-localization
[20] https://arxiv.org/pdf/1710.11063.pdf
[21] https://paperswithcode.com/paper/grad-cam-visual-explanations-from-deep
[22] https://arxiv.org/abs/2009.12546
[23] https://publikationen.reutlingen-university.de/files/3858/3858.pdf
[24] https://cs231n.github.io/convolutional-networks/
[25] https://www.sciencedirect.com/science/article/abs/pii/S0165168422002249
[26] https://ola2022.sciencesconf.org/data/pages/Proceedings_1.pdf
[27] https://arxiv.org/abs/2303.13166
[28] https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720455.pdf
[29] https://christophm.github.io/interpretable-ml-book/overview.html
[30] https://www.campana-schott.com/de/en/data-ai
[31] https://press.siemens.com/global/en/pressrelease/siemens-unveils-breakthrough-innovations-industrial-ai-and-digital-twin-technology-ces
[32] https://www.schott.com/en-gb/news-and-media/media-releases/2024/schott-reaches-a-strategic-milestone-with-the-founding-of-a-specialized-division
[33] https://ejsit-journal.com/index.php/ejsit/article/view/663
[34] https://arxiv.org/pdf/2109.00520.pdf
[35] https://press.siemens.com/global/en/pressrelease/siemens-drives-ai-adoption-industrial-operations-x-and-nvidia-accelerated-industrial
[36] https://www.scitepress.org/Papers/2025/131807/131807.pdf
[37] https://dictionary.cambridge.org/ko/%EC%82%AC%EC%A0%84/%EC%98%81%EC%96%B4/square-bracket
[38] https://arxiv.org/abs/1710.11063
[39] https://www.sciencedirect.com/science/article/abs/pii/S0278612523001024
[40] https://dblp.org/pid/117/0138
[41] https://www.insticc.org/node/TechnicalProgram/ICEIS/2024/presentationDetails/125957
[42] https://orcid.org/0000-0003-4921-1485
