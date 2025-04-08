# Chest X‐Ray Bone Suppression for Improving Classification of  Tuberculosis‐Consistent Findings, 인용수 : 52
본 연구는 전방 CXR에서 이러한 가려진 뼈 구조를 식별하고 제거하는 딥러닝(DL) ‐ 기반 뼈 억제 모델을 구축하여 결핵(TB)과 일치하는 증상을 감지하는 것과 관련된 DL 워크플로우를 포함한 방사선학적 해석의 오류를 줄이는 데 도움을 주는 것을 목표로 합니다.  
제안된 결합 손실 함수를 사용하여 다양한 심층 아키텍처를 가진 여러 뼈 억제 모델을 훈련하고 최적화했으며, 이들의 성능은 평균 절대 오차(MAE), 피크 신호 ‐ 대 ‐ 잡음비(PSNR), 구조적 유사성 지수 측정(SSIM), 다중 규모 구조적 유사성 측정(MS-SSIM) 등 여러 지표를 사용하여 ‐ 기관 간 테스트 설정에서 평가되었습니다.  
가장 우수한 성능을 보이는 모델(ResNet–BS)(PSNR = 34.0678; MS–SSIM = 0.9828)은 공개된 심천 및 몽고메리 TB CXR 컬렉션에서 뼈를 억제하는 데 사용되었습니다.  
VGG ‐16 모델은 공개된 대규모 CXR 컬렉션에서 사전 학습되었습니다.

이러한 모델의 성능은 정확도, 곡선 아래 면적(AUC), 민감도, 특이도, 정밀도, F ‐ 점수, 매튜스 상관 계수(MCC)와 같은 여러 성능 지표를 사용하여 통계적 유의성을 분석했으며, 이 예측은 class‐selective  relevance  maps(CRMs)을 통해 정성적으로 해석되었습니다.  
뼈 ‐ 억제 CXR로 훈련된 모델은 결핵 ‐ 일관된 소견의 검출을 개선했으며, 특징 공간의 데이터 포인트를 압축적으로 클러스터링하여 뼈 억제가 결핵 분류에 대한 모델 민감도를 향상시켰음을 나타냅니다.  

The researchers from the Budapest University  of  Technology  and  Economics used  their in‐house clavicle and rib shadow removal algorithms to suppress the bones in the247 JSRT CXRs and made the bone‐suppressed soft‐tissue images publicly available [28].   
Affine  transformations  including  rotations  (−10  to  10  degrees),  horizontal  and  vertical  shifting  (−5  to  5  pixels),  horizontal  mirroring,  zooming,  median,  maximum,  and  minimum, and unsharp masking are used to generate 4500 image pairs from this initial set of  CXRs and their bone‐suppressed counterparts.

부다페스트 공과대학교의 연구진은 247개의 JSRT CXR에서 뼈를 억제하기 위해 쇄골 및 갈비뼈 그림자 제거 알고리즘을 사용하여 뼈를 억제한 연질 조직 이미지를 공개했습니다.  
이 초기 CXR 세트와 뼈를 억제한 이미지로부터 데이터 증강을 이용해 4500개의 이미지 쌍을 생성했습니다.  

The augmented images are resized to 256  × 256 spatial resolution.  
The image contrast is enhanced by saturating the bottom and top 1% of all image pixel values.  
The grayscale pixel values are then normalized.

증강된 이미지는 256 × 256 공간 해상도로 크기가 조정되었습니다.  
이미지 대비는 모든 이미지 픽셀 값의 바닥과 상단 1%를 포화시켜 향상됩니다.  
그런 다음 그레이스케일 픽셀 값을 정규화합니다. 

다양한 아키텍처를 가진 여러 ConvNet 기반 뼈 억제 모델을 이 증강된 데이터셋에서 학습시켰습니다. 우리는 교차 기관 NIH-CC-DES 테스트 세트를 사용하여 성능을 평가했습니다.  

Four different model architectures are proposed toward the  task of bone suppression in CXRs as follows: (a) Autoencoder (AE) model (AE–BS) where  BS denotes bone suppression; (b) Sequential ConvNet model (ConvNet–BS); (c) Residual  learning model (RL–BS); and (d) Residual network model (ResNet–BS).

CXR에서 뼈 억제 작업을 위해 다음과 같은 네 가지 모델 아키텍처가 제안되었습니다:  
(a) 자동 인코더(AE) 모델(BS) 여기서 BS는 뼈 억제를 나타냅니다;  
(b) 순차적 ConvNet 모델(ConvNet–BS);  
(c) 잔차 학습 모델(RL–BS);  
(d) 잔차 네트워크 모델(ResNet–BS).

(i) AE–BS Model: The AE–BS model is a convolutional denoising AE with symmetrical encoder and decoder layers.  
The encoder consists of three convolutional layers with  16, 32, and 64 filters, respectively.  
The size of the input is decreased twice at the encoder  layers  and  increased  correspondingly  in  the  decoder  layers.   
As  opposed  to  the  conventional denoising AEs, the noise in the proposed AE–BS model represents the bony structures.  
The model trains on the original CXRs and their bone‐suppressed counterparts to  predict  a  bone‐suppressed  soft‐tissue  image.  

(i) AE-BS 모델: AE-BS 모델은 대칭 인코더와 디코더 레이어가 있는 합성곱 노이즈 제거 AE입니다.  
인코더는 각각 16개, 32개, 64개의 필터가 있는 세 개의 합성곱 레이어로 구성됩니다.  
입력 크기는 인코더 레이어에서 두 번 감소하고 디코더 레이어에서 그에 따라 증가합니다.  
기존의 노이즈 제거 AE와 달리, 제안된 AE-BS 모델의 노이즈는 뼈 구조를 나타냅니다.  
이 모델은 원래의 CXR과 그 뼈 ,억제된 연질 조직 이미지를 예측하기 위해 훈련됩니다. 

(ii) ConvNet–BS model: The ConvNet–BS model is a sequential model consisting of  seven convolutional layers having 16, 32, 64, 128, 256, 512, and 1 filter, respectively.  
Zero  paddings are used to preserve the dimensions of the input image at all convolutional layers.  
Lasso regularization (L1) penalties are used at each convolutional layer to induce penalty  on  weights  that  seldom  contribute  to  learning  meaningful  feature  representations.   
This helps in improving model sparsity and generalizing to unseen data.  
The deepest convolutional layer with the sigmoidal activation produces the bone‐suppressed soft‐tissue  image. 

(ii) ConvNet–BS 모델: ConvNet–BS 모델은 각각 16, 32, 64, 128, 256, 512, 1개의 필터를 가진 7개의 컨볼루션 레이어로 구성된 순차적 모델입니다.  
모든 컨볼루션 레이어에서 입력 이미지의 차원을 보존하기 위해 제로 패딩이 사용됩니다.  
각 컨볼루션 레이어에서 의미 있는 특징 표현을 학습하는 데 거의 기여하지 않는 가중치에 대한 페널티를 유도하기 위해 Lasso 정규화(L1) 페널티가 사용됩니다. 이는 모델 희소성을 개선하고 보이지 않는 데이터로 일반화하는 데 도움이 됩니다.  
시그모이드 활성화가 있는 가장 깊은 컨볼루션 레이어는 뼈 ‐ 억제 연조직 이미지를 생성합니다. 그림 2는 제안된 ConvNet–BS 모델의 아키텍처를 보여줍니다.

(iii)  RL–BS  model:  The architecture  of  the  RL–BS  model  consists  of  eight  convolutional layers having 8, 16, 32, 64, 128, 256, 512, and 1 filter, respectively.  
Zero paddings are  used at all convolutional layers to preserve the dimensions of the input image.  
The RLBS model learns the residual error between the predicted bone‐suppressed image and its  corresponding ground truth.  
The deepest convolutional slayer produces bone‐suppressed  images.  

(iii) RL-BS 모델: RL-BS 모델의 아키텍처는 각각 8개, 16개, 32개, 64개, 128개, 256개, 512개, 1개의 필터를 가진 8개의 컨볼루션 레이어로 구성됩니다.  
입력 이미지의 차원을 보존하기 위해 모든 컨볼루션 레이어에서 제로 패딩이 사용됩니다.  
RLBS 모델은 예측된 뼈 ‐ 억제 이미지와 해당하는 실측값 사이의 잔여 오차를 학습합니다.  
가장 깊은 컨볼루션 슬레이어는 뼈 ‐ 억제 이미지를 생성합니다.

(iv)  ResNet–BS  model:  The residual design utilizes shortcuts to skip over layers thereby eliminating  learning  convergence  issues  due  to  vanishing  gradients.   
This  facilitates  reusing  previous  layer  activations  until  the  weights  are  updated  in  the  adjacent  layer.   
These shortcuts  lead  to  improved  convergence  and  optimization  and  help  to  construct  deeper  models.

(iv) ResNet–BS 모델: 잔차 설계는 레이어를 건너뛸 수 있는 단축키를 활용하여 그래디언트가 사라짐에 따른 학습 수렴 문제를 제거합니다.  
이를 통해 인접 레이어에서 가중치가 업데이트될 때까지 이전 레이어 활성화를 재사용할 수 있습니다.  
이러한 단축키는 수렴 및 최적화를 개선하고 더 깊은 모델을 구성하는 데 도움이 됩니다.

EDSR 에서 영감을 받아 ReLU 활성화 레이어는 잔여 블록 외부에서 사용되지 않습니다.  
이 문헌은 배치 정규화가 정보 손실을 초래하고 활성화의 범위 처리 가능성을 감소시킨다는 것을 보여줍니다.  
따라서 배치 정규화 레이어와 최종 ReLU 활성화는 각 ResNet 블록에서 제거됩니다. 

## EDSR, Enhanced Deep Residual Networks for Single Image Super-Resolution, 2017
![image](https://github.com/user-attachments/assets/ea1d920a-67c5-4ff0-956c-db209f76eff9)

```
기존에 Residual block을 사용했던 SRResnet에서 Batch Normalization을 삭제했습니다.  
Batch Normalization은 feature를 normalize하기 때문에 유동성을 해칩니다.  
그리고 convolution layer와 동일한 메모리를 사용하기 때문에, Batch Normalization을 사용하지 않았을 때 훈련 동안 메모리 사용량이 40% 줄었습니다. 이는 BN을 줄임으로써 더 큰 모델을 구축 할 수 있음을 의미합니다.  

```
In this study, the performance of the proposed bone suppression models is evaluated  through constructing a loss function that benefits from the combination of mean absolute  error (MAE) and multiscale structural similarity index measure (MS–SSIM) losses, herein  referred  to  as  combined  loss.  

이 연구에서는 제안된 골 억제 모델의 성능을 평균 절대 오차(MAE)와 다중 스케일 구조적 유사성 지수(MS-SSIM) 손실(여기서 결합 손실이라고 함)을 결합하여 이점을 얻는 손실 함수를 구성하여 평가합니다.

In this study, an ImageNet‐pretrained VGG‐16 model [25] is retrained on a large collection of CXRs combined using RSNA CXR and pediatric pneumonia CXR data collections  producing  sufficient  diversity  in  terms  of  image  acquisition  and  patient  demographics to learn the characteristics of abnormal and normal lungs. This VGG‐16 model  is truncated at its deepest convolutional layer and appended with a global average pooling (GAP) layer, a dropout layer with an empirically determined dropout ratio (0.5), and  an  output  layer  with  two  nodes  to  predict  probabilities  of  the  input  CXRs  as  showing  normal  lungs or  other pulmonary  abnormalities.

The bone‐suppressed datasets are constructed by  using the best‐performing bone suppression model among the proposed models.  
For the  fine-tuning  task,  fourfold  cross‐validation  is  performed  in  which  the  baseline  and  bonesuppressed CXRs in the Shenzhen and Montgomery TB collections are split at the patient  level into four equal folds. 

이 연구에서는 이미지넷 ‐ 사전 훈련된 VGG ‐16 모델 [25]을 RSNA CXR 및 소아 폐렴 CXR 데이터 수집을 사용하여 결합된 대규모 CXR 컬렉션에 대해 재훈련하여 비정상 및 정상 폐의 특성을 학습하는 데 충분한 다양성을 제공합니다.  
이 VGG ‐16 모델은 가장 깊은 합성곱 층에서 절단되어 글로벌 평균 풀링(GAP) 층, 경험적으로 결정된 드롭아웃 비율(0.5)을 가진 드롭아웃 층, 입력된 CXR이 정상 폐 또는 기타 폐 이상을 보일 확률을 예측하기 위해 두 개의 노드를 가진 출력 층과 함께 추가됩니다.

이 CXR-VGGG ‐16 모델은 원래 심천 및 몽고메리 TB CXR 컬렉션(기준 모델)과 그 뼈가 억제된 모델(뼈 ‐ 억제 모델)에 대해 정밀 ‐ 조정을 통해 정상 폐 또는 폐 결핵 증상을 보이는 것으로 분류합니다.  
뼈 ‐ 억제 데이터셋은 제안된 모델 중 최고의 ‐ 성능을 사용하여 구성됩니다.  
미세 조정 작업을 위해 심천 및 몽고메리 콜에서 기준 CXR과 뼈가 억제된 네 배의 교차 검증을 수행합니다.
추가로 데이터 증강 기법을 사용하여 이미지를 생성합니다.  

## Results
![image](https://github.com/user-attachments/assets/df3a6121-26ce-4fd5-a298-ec56df08ea7a)

![image](https://github.com/user-attachments/assets/d7374e80-1ae7-45ac-a47e-222c4c0b7b23)

![image](https://github.com/user-attachments/assets/9b33e6ae-ed7c-44e8-a2da-30aa4a67f48c)

또한 CRM을 사용하여 심천 및 몽고메리 TB CXR 컬렉션을 사용하여 최고의 ‐ 성능 기준 모델과 골 ‐ 억제 모델의 예측을 해석하여 결핵 ‐ 일관된 결과를 현지화했습니다.  
그림 10a와 그림 10d는 각각 심천 및 몽고메리 TB CXR 컬렉션에서 얻은 원본 CXR의 사례를 보여줍니다. 전문가의 실제 주석은 빨간색 바운딩 박스와 함께 표시됩니다.

![image](https://github.com/user-attachments/assets/d205224f-e3ba-4c95-a976-c0183194ebaa)

## Class-selective Relevance Mapping
```
Visual Interpretation of Convolutional Neural Network Predictions in Classifying Medical Image Modalities
의료 이미지 내에서 차별적인 관심 영역(ROI)을 로컬라이제이션하고 시각화하기 위한 "클래스 선택적 관련성 매핑"(CRM)이라는 새로운 방법을 제안합니다.
이러한 시각화는 합성곱 신경망(CNN) 기반 DL 모델 예측에 대한 향상된 설명을 제공합니다.
CRM은 CNN 모델의 출력 계층에서 계산된 증분 평균 제곱 오차(MSE)의 선형 합을 기반으로 합니다.
마지막 합성곱 계층에서 생성된 피처 맵에서 각 공간 요소의 긍정적 및 부정적 기여도를 모두 측정하여 입력 이미지의 올바른 분류로 이어집니다.
```
