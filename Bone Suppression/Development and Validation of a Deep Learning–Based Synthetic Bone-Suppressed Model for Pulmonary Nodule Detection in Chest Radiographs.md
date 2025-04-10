흉부 방사선 촬영은 작은 폐 결절의 경우 낮은 발견률로 폐암 검진을 위한 흉부 방사선 촬영의 한계를 지적해 왔습니다.  
흉부 방사선 촬영은 중요한 임상 소견에 대해 상당한 판독자 간 변동성과 최적의 민감도(sensitivity) 저하의 영향을 받습니다.  

따라서 작은 결절은 갈비뼈와 견갑골에 의해 차폐(숨겨지게)되어 엑스레이를 해석하는 동안 놓칠 수 있습니다. 이 문제를 해결하기 위해 뼈와 연조직을 구별하는 이중 에너지 차분 기술(dual-energy subtraction techniques)이 개발되었습니다.  
Dual-energy chest radiography, DECR은 폐 결절을 감지하고 특성화하는 능력을 향상시키는 것으로 입증되었지만, DECR의 단점으로는 전문 장비가 필요하고 방사선량이 약간 증가할 수 있다는 점이 있습니다.

```
Dual-energy chest radiography, DECR 에 관한 글.
https://www.upstate.edu/radiology/education/rsna/radiography/dual.php
```

딥러닝 기술은 이상있는 부분을 자동으로 감지하거나 방사선 전문의가 흉부 방사선 사진을 읽는 데 도움을 줄 수 있는 잠재력을 가지고 있습니다. 

```
Sensitivity and specificity
여기서 민감도는 어떤 조건의 만족 여부를 보고하는 검사를 설명하기 위한 수학적인 지표이다.
민감도(敏感度, sensitivity, true positive rate)는 실제로 양성인 사람이 검사에서 양성으로 판정될 확률이다.
특이도(特異度, specificity, true negative rate)는 실제로 음성인 사람이 검사에서 음성으로 판정될 확률이다.
진단 검사에서 민감도는 검사가 얼마나 잘 실제 양성을 판별할 수 있는지를 나타내고, 특이도는 검사가 실제 음성을 얼마나 잘 판별하는지를 의미한다.

검사의 목표가 조건을 만족하는 모든 사람을 찾아내는 것이라면 위음성의 수를 줄여야 하므로 검사의 민감도가 높아야 한다. 민감도가 높다는 것은 조건을 만족하는 사람이 검사에서 양성으로 나올 가능성이 높다는 것이다.
https://ko.wikipedia.org/wiki/%EB%AF%BC%EA%B0%90%EB%8F%84%EC%99%80_%ED%8A%B9%EC%9D%B4%EB%8F%84
```

# Development and Validation of a Deep Learning–Based Synthetic Bone-Suppressed Model for Pulmonary Nodule Detection in Chest Radiographs, 인용수 : 10

To develop and validate a deep learning–based synthetic bone-suppressed (DLBS) nodule-detection algorithm for pulmonary nodule detection on chest radiographs.  
흉부 방사선 사진에서 폐 결절 검출을 위한 딥러닝 기반 합성 뼈 억제(DLBS) 결절 검출 알고리즘을 개발하고 검증합니다.

This decision analytical modeling study used data from 3 centers between November 2015 and July 2019 from 1449 patients. The DLBS nodule-detection algorithm was trained using single-center data (institute 1) of 998 chest radiographs. The DLBS algorithm was validated using 2 external data sets (institute 2, 246 patients; and institute 3, 205 patients). Statistical analysis was performed from March to December 2021.

모델링 연구는 2015년 11월부터 2019년 7월까지 1449명의 환자로부터 수집된 3개 센터의 데이터를 사용  
DLBS 결절 검출 알고리즘은 998개의 흉부 방사선 사진에 대한 단일 센터 데이터(기관 1)를 사용하여 학습

The nodule-detection performance of DLBS model was compared with the convolution neural network nodule-detection algorithm (original model).  
Reader performance testing was conducted by 3 thoracic radiologists assisted by the DLBS algorithm or not. Sensitivity and false-positive markings per image (FPPI) were compared.

DLBS 모델의 결절 검출 성능을 CNN(원본 모델)과 비교했습니다.  
판독기의 성능 테스트는 3명의 흉부 방사선 전문의가 DLBS 알고리즘의 도움을 받아 수행했습니다.  

외부 검증 데이터 세트를 사용한 결과, 뼈 억제 모델은 결절 검출을 위한 원래 모델에 비해 더 높은 민감도를 보였습니다(91.5% [109 of 119] vs 79.8% [95 of 119]; P < .001).  
뼈 억제 모델을 사용한 FPPI의 전체 평균은 원래 모델에 비해 감소했습니다(0.07 [17 of 246] vs 0.09 [23 of 246]; P < .001).  

## What is FPPI?
```
False Positive Per Image (FPPI), FPPI는 이미지 한 장당 검출된 False Positive 샘플의 개수를 말한다.  
이때 FP란 즉, 객체가 없는데 객체가 있다고 잘못 예측한 bounding box를 말한다.

FPPI는 이미지 한 장 단위로 계산되기 때문에, 데이터에 따라 scale의 변동폭이 매우 넓다.  
예를 들어, 어떤 이미지에는 이미지 전체적으로 특별한 요소가 없어 FPPI가 0이겠지만, 어떤 이미지에는 객체, 혹은 객체로 인식할 만한 사물이 많아 FPPI가 큰 값이 될 수 있다.
주로 object detection 지표로 사용된다.  
- https://skyil.tistory.com/204
```

연구소 3의 데이터를 사용한 관찰자 성능 테스트에서 3명의 방사선 전문의의 평균 민감도는 77.5%였습니다.  
반면, DLBS 모델링에 의해 도움을 받은 방사선 전문의는 92.1% 였습니다.  
세 명의 방사선 전문의는 DLBS 모델에 의해 도움을 받았을 때 FPPI 수가 감소했습니다.

We developed a deep learning–based synthetic bone-suppressed (DLBS) pulmonary nodule-detection algorithm by modifying a conventional U-net to take advantage of the high frequency-dominant information that propagates from the encoding part to the decoding part.  
우리는 인코딩 부분에서 디코딩 부분으로 전파되는 고주파-지배 정보를 활용하기 위해 기존의 U-net을 수정하여 딥러닝 기반 합성 뼈 억제(DLBS) 및 폐 결절 검출 알고리즘을 개발했습니다.

## Dominant frequency of image
```
The dominant frequency of an image is the frequency with the highest magnitude of energy in the image's spectrum. You can analyze an image's frequency content using the Discrete Fourier Transform (DFT).
이미지의 지배적인 주파수는 이미지 스펙트럼에서 에너지의 크기가 가장 큰 주파수입니다. 이산 푸리에 변환(DFT)을 사용하여 이미지의 주파수 콘텐츠를 분석할 수 있습니다.
이미지를 서로 다른 지점의 픽셀 값이 느리게 변하는 low frequency component와 서로 다른 지점의 픽셀 값이 빠르게 변하는 high frequency component로 분해가 가능하다.
https://velog.io/@claude_ssim/%EA%B3%84%EC%82%B0%EC%82%AC%EC%A7%84%ED%95%99-Frequency-Domain-1
```

The main idea of the developed model is that when a feature is propagated from encoding to decoding, only the high frequency components are extracted and propagated.  
The proposed model also dramatically reduces the number of parameters by adding features of the encoding that propagate to the decoding part instead of the feature concatenation of U-net.  

개발된 모델의 주요 아이디어는 특징이 인코딩에서 디코딩으로 전파될 때 고주파 성분만 추출되어 전파된다는 것입니다. 제안된 모델은 또한 U-net의 특징 연결(concatenation) 대신 디코딩 부분으로 전파되는 인코딩의 특징을 추가하여 매개변수 수를 획기적으로 줄입니다.  

## U-Net 의 알고리즘 : Skip Architecture
![image](https://github.com/user-attachments/assets/039cdb71-ef47-41dc-a91c-70adccf6fef3)

```
Skip Architecture
동일한 Level에서 나온 Feature map을 더한다는 점이 FCN의 Skip architecture와 다른 점이라고 할 수 있습니다.
Contracting Path의 Feature map의 테두리 부분을 자른 후 크기를 동일하게 맞추어 두 feature map을 합쳐 줍니다.

https://wikidocs.net/148870
```

DLBS 결절 검출 알고리즘 개발 및 검증을 위해 3개의 3차 병원에서 획득한 전방 투영 흉부 방사선 사진을 수집했습니다.  

The developed deep learning–based model consisted of 2 subsystems responsible for (1) generating bone-and soft tissue–only images from single-energy chest radiography and (2) detecting suspicious pulmonary nodules, respectively.  
개발된 딥러닝 기반 모델은 (1) 단일 에너지 흉부 방사선 촬영에서 뼈와 연조직만 있는 이미지를 생성하고 (2) 의심스러운 폐 결절을 감지하는 두 개의 하위 시스템으로 구성되었습니다.  

For the first step, we previously developed a deep convolutional neural network (DCNN)-based synthetic bone-suppressed algorithm based on U-net which is a deep convolutional neural network architecture with multiresolution analysis performed by repeated convolution and feature-dimension changes. The main idea of the developed model was that, when a feature is propagated from encoding to decoding, only the high-frequency components are extracted and propagated. 
The model selectively projected the bone- and soft tissue–only chest radiography images from a single energy chest radiography image. The bone-suppressed chest radiographs were automatically synthesized through a DCNN-based encoder-decoder model. 

첫 번째 단계로, 우리는 이전에 U-net을 기반으로 한 심층 합성곱 신경망(DCNN) 기반 합성 뼈 억제 알고리즘을 개발했습니다.  
이는 반복적인 합성곱 및 특징의 차원 변화에 의해 수행되는 다중 해상도 분석을 포함하는 심층 합성곱 신경망 아키텍처입니다.  
개발된 모델의 주요 아이디어는 특징이 인코딩에서 디코딩으로 전파될 때 고주파 성분만 추출되어 전파된다는 것이었습니다.  
이 모델은 단일 에너지 흉부 방사선 촬영 이미지에서 뼈와 연조직만 있는 흉부 방사선 촬영 이미지를 선택적으로 투영했습니다.  
뼈가 억제된 흉부 방사선 촬영은 DCNN 기반 인코더-디코더 모델을 통해 자동으로 합성되었습니다.

## How to use U-Net for Bone suppression?
![image](https://github.com/user-attachments/assets/7070eb43-ea21-41ce-ae52-70a0c9d73b4e)

```
Development of a deep neural network for generating synthetic dual-energy chest x-ray images with single x-ray exposure, 인용수 : 19 논문에서는 고주파수로 분리한 feature map 을 디코더 부분으로 연결시키기 위해 Skip Architecture 대신 frequency feature map 을 다른 해상도로 변환한 feature map에 concat시킨 알고리즘 적용.
다른 해상도는 average pooling과 upsampling method 사용.

```

For the second step, we developed a pulmonary nodule–detection algorithm based on a convolution neural network (CNN) algorithm known as “you only look once” (YOLO).17 The developed algorithm customized a YOLO version 3 CNN algorithm for the detection of pulmonary nodules. In general, the network consists of 2 major components: (1) a feature extractor that screens nodule presence among the input data, and (2) a bounding box generator that determines nodule location.

두 번째 단계로, 우리는 (YOLO)로 알려진 합성곱 신경망(CNN) 알고리즘을 기반으로 한 폐 결절 탐지 알고리즘을 개발했습니다.  
개발된 알고리즘은 폐 결절 탐지를 위해 YOLO 버전 3 CNN 알고리즘을 맞춤화했습니다.  
일반적으로 네트워크는 두 가지 주요 구성 요소로 구성됩니다: (1) 입력 데이터 중 결절 유무를 선별하는 특징 추출기와 (2) 결절 위치를 결정하는 바운딩 박스 생성기입니다.

DLBS 모델은 합성 뼈 억제 영상이 포함된 훈련 데이터 세트를 사용하여 폐 결절을 감지하도록 훈련되었으며, CNN 모델은 원본 흉부 방사선 사진을 사용하여 폐 결절을 감지하도록 별도로 훈련되었습니다(그림 1 및 부록의 e그림 2).  
결절 감지 성능을 극대화하기 위해 5중 교차 검증(5-fold cross-validation)을 통해 앙상블 모델을 개발하고, 하드 음성 샘플링 방법(hard-negative sampling method)을 사용했습니다(앙상블 모델).

### hard-negative sampling method
```
Object Detection Network 에서 자주 나오는 hard negative sk hard negative mining라는 말이 자주 나온다.  
hard negative 는 실제로는 negative 인데 positive 라고 잘못 예측하기 쉬운 데이터이다.  
hard negative sample 을 직석하자만 hard : 어렵다, negative sample: 네거티브 샘플라고, 즉 네거티브 샘플라고 말하기 어려운 샘플이라는 뜻이다.  
모델입장에서 보면 해당 샘플에 대해 negative(아니다) 라고 해야 하는데 confidence 는 높게 나오는 상황을 말한다.
hard negative 는 원래 negative 인데 positive 라고 잘못 예측한 데이터를 말한다.  
hard negative는 마치 positive처럼 생겨서 예측하기 어렵다.  
hard negative mining(sampling) 는 hard negative 데이터를 (학습데이터로 사용하기 위해) 모으는(mining) 것이다.  
hard negative mining 으로 얻은 데이터를 원래의 데이터에 추가해서 재학습하면 false positive 오류에 강해진다.  

[[출처] hard negative mining|작성자 sogangori](https://blog.naver.com/sogangori/221073537958)
```

![image](https://github.com/user-attachments/assets/5200ede7-c7e8-4890-b437-334b84597759)
