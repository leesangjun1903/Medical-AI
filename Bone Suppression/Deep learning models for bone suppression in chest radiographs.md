# Deep learning models for bone suppression in chest radiographs, 인용수 :  59회 인용
Bone suppression in lung radiographs is an important task, as it improves the results on other related tasks, such as nodule detection or pathologies classification.  
In this paper, we propose two architectures that suppress bones in radiographs by treating them as noise.  
In the proposed methods, we create end-to-end learning frameworks that minimize noise in the images while maintaining sharpness and detail in them.  
Our results show that our proposed noise-cancellation scheme is robust and does not introduce artifacts into the images.

본 논문에서는 방사선 사진에서 뼈를 노이즈로 취급하여 뼈를 억제하는 두 가지 아키텍처를 제안합니다.  
제안된 방법에서는 이미지의 선명도와 세부 사항을 유지하면서 노이즈를 최소화하는 종단 간 학습 프레임워크를 만듭니다.  
우리의 결과는 제안된 노이즈 제거 방식이 견고하며 이미지에 아티팩트가 도입되지 않는다는 것을 보여줍니다.

In this paper, we propose two architectures to perform bone suppression from CXRs.  
The architectures are supposed to denoise bones from images instead of subtraction of bone shadows or feature extraction analysis.  
Thus, they are completely different from what was already shown for bone suppression purposes.  
The first model is a family of convolutional autoencoders, while the second one is a family of a simple convolutional neural network (CNN).

이 논문에서는 CXR에서 뼈 억제를 수행하기 위한 두 가지 아키텍처를 제안합니다.  
이 아키텍처는 뼈 그림자를 제거하거나 특징 추출 분석을 수행하는 대신 이미지에서 뼈를 노이즈 제거하도록 되어 있습니다.  
따라서 이들은 뼈 억제 목적으로 이미 보여진 것과는 완전히 다릅니다.  
첫 번째 모델은 합성곱 오토인코더 계열이고, 두 번째 모델은 간단한 합성곱 신경망(CNN) 계열입니다.

Our learning framework for both models is driven by a denoising reconstruction function, instead of the, commonly used, identity.  
In our proposal, we explore a loss function that exploits the structural similarities in the image while maintaining the reconstruction as the main goal.  
Experimentally, we find the best configuration for our proposed models.  
Moreover, we found that our models are capable of transforming standard CXR images into soft-tissue images.  
Our final trained models will have a better lung disease detection rate.

두 모델의 학습 프레임워크는 일반적으로 사용되는 일반적 함수 대신 노이즈 제거 재구성 함수에 의해 구동됩니다.  
제안에서는 재구성을 주요 목표로 유지하면서 이미지의 구조적 유사성을 활용하는 손실 함수를 탐구합니다.  
실험적으로 제안된 모델에 가장 적합한 구성을 찾았습니다. 또한, 우리의 모델이 표준 CXR 이미지를 연조직 이미지로 변환할 수 있다는 것을 발견했습니다.  
최종 학습된 모델은 폐 질환 탐지율이 더 높아질 것입니다.

The goal of our framework is to feed a CXR image to the network and produce a bone suppressed image.  
Moreover, we propose a loss function that teaches the network (in an end-to-end fashion) to perform custom denoising on the input images.

우리 프레임워크의 목표는 네트워크에 CXR 이미지를 공급하고 뼈가 억제된 이미지를 생성하는 것입니다.  
또한, 입력된 이미지에 대해 사용자 지정 노이즈 제거를 수행하도록 네트워크에 end-to-end 방식으로 가르치는 손실 함수를 제안합니다.  

A. Autoencoder-like Convolutional Model 

Our first model is a stacked denoising autoencoder (AE).  
Model output lacks noise defined by user, and, in case of CXRs, the model produces the original image without bones.  
Overall, the model is a stack of convolutional autoencoders with encoder and decoder sharing the same but mirrored weights.  
Unlike usual denoising autoencoders, the noise is not normally distributed, and is represented by bony structures.  
We define our architecture as AE-like, because we are doing a reconstruction not of an original image, but of a bone suppressed, or denoised, one. The model consists of 3 autoencoders, each encoding an image into 16, 32, and 64 neurons, respectively. Figure 1(a) of hourglass shape shows that image size is decreased twice at every encoding layer and increased twice at corresponding decoding layers. Firstly, we optimize the parameters of our model by minimizing mean squared error between the model output and a corresponding soft-tissue image alone, and, secondly, we minimize mean squared error (MSE) along with maximizing multi-scale structural similarity (MS-SSIM) [22] of produced image and a soft-tissue image (the ground truth). We detail the loss functions in Section III-C.

A. 오토인코더와 유사한 컨볼루션 모델 

첫 번째 모델은 스택형 노이즈 제거 오토인코더(AE)입니다.  
모델 출력에는 사용자가 정의한 노이즈(뼈) 가 없으며, CXR의 경우 모델이 뼈 없이 원본 이미지를 생성합니다.  
전체적으로 이 모델은 인코더와 디코더가 동일하지만 미러링된 가중치를 공유하는 컨볼루션 오토인코더 스택입니다.  
일반적인 노이즈 제거 오토인코더와 달리 노이즈는 정규 분포가 아니며 뼈 구조로 표현됩니다.  
우리는 원래 이미지가 아닌 뼈가 억제된 이미지 또는 노이즈 제거된 이미지를 재구성하기 때문에 우리의 아키텍처를 AE와 유사하다고 정의합니다.  
이 모델은 각각 이미지를 16, 32, 64 뉴런으로 인코딩하는 3개의 오토인코더로 구성됩니다.  
모래시계 모양의 그림 1(a)은 이미지 크기가 각 인코딩 레이어에서 두 배 감소하고 해당 디코딩 레이어에서 두 배 증가함을 보여줍니다.  
첫째, 모델 출력과 해당 소프트 조직 이미지 간의 평균 제곱 오차를 최소화하여 모델의 매개변수를 최적화하고, 둘째, 생성된 이미지와 소프트 조직 이미지의 다중 스케일 구조적 유사성(MS-SSIM) [22]을 최대화하여 평균 제곱 오차(MSE)를 최소화합니다.  
우리는 섹션 III-C에서 손실 함수에 대해 자세히 설명합니다.

B. 다층 합성곱 신경망 모델 
두 번째 모델은 실제로 3, 4, 5, 6개의 레이어로 구성된 CNN 계열입니다.  
이 모델은 가중치가 공유되지 않으며, 특징 학습 과정은 특정 양의 학습 공간(레이어와 뉴런의 수)이 제공될 때 가장 효과적이라는 가설이 있습니다.  
즉, 4개의 CNN 모델 계열의 도움을 받아 얼마나 많은 공간이 필요한지 알아내야 합니다. 첫 번째와 두 번째 레이어에는 각각 16개와 32개의 숨겨진 뉴런이 있습니다.  
그러면 각 모델의 마지막 레이어에는 이전 모델보다 두 배 더 많은 뉴런이 있지만 하나의 레이어가 있습니다. 마지막 레이어는 뼈 억제 이미지 세트를 생성하는 출력 레이어입니다.  
4계층 네트워크는 3계층에 64개의 뉴런이 있고, 5계층 네트워크는 4계층에 128개의 뉴런이 있으며, 6계층 네트워크는 5계층에 256개의 뉴런이 있습니다. 6계층 네트워크는 그림 1(b)에 나와 있습니다.  
이미지 크기는 CNN의 모든 레이어에 걸쳐 유지됩니다. 우리는 모델 출력과 실제 데이터의 MS-SSIM을 최대화하고 MSE를 최소화하여 모델의 매개변수를 최적화합니다.

우리의 목표는 인공물을 도입하지 않고 CXR에서 뼈 구조를 제거하는 방법을 학습하는 네트워크를 훈련시키는 것입니다. 
이 목표를 달성하기 위해, 우리의 접근 방식은 노이즈가 제거된 이미지에서 원래 이미지로의 재구성 오류를 줄이고 구조 지수를 최대화하여 선명도를 유지하는 것입니다.  
우리는 손실 함수를 올바르게 선택하는 것이 재구성 결과에 강한 영향을 미친다는 것을 발견했습니다.  
따라서, 우리는 노이즈 제거 프레임워크에 가장 적합한 손실 함수를 찾기 위해 다양한 구성을 탐구합니다.  

MSE(또는 L2)는 재구성을 위한 사실상의 표준 오류 함수입니다.  
MSE의 한계로 인해 또 다른 인기 있는 참조 기반 지수는 구조적 유사성 지수(SSIM)입니다.  
이 지수는 로컬 구조 변화에 대한 HVS(human visual system) 의 민감도를 고려하여 이미지를 평가합니다 [23]. 픽셀 i에 대한 SSIM은 다음과 같이 정의됩니다

## HVS(human visual system)
```
HVS(human visual system)
인간의 시각 시스템은 시각 정보를 처리하기 위해 함께 작동하는 눈과 뇌로 구성됨 그것은 우리가 주변 세계를 보고, 인지하고, 상호작용할 수 있게 해줍니다 . 
https://www.seevividly.com/info/Physiology_of_Vision/The_Brain/Visual_System
```

Wang 등 [22]은 HVS의 민감도에 따라 다양한 척도로 계산된 SSIM의 무게를 측정하는 다중 척도 버전인 MS-SSIM을 제안했으며, 실험 결과 SSIM 기반 인덱스가 'l2'보다 우수하다는 것이 입증되었습니다.  
Zhao 등[24]은 SSIM과 MS-SSIM이 평균 절대 오차('1'이라고도 불리는)만큼 성능이 좋지 않다는 것을 보여주었습니다.  
그들은 또한 MS-SSIM과 'l1'의 조합이 최상의 절충안을 제공한다는 결론을 내렸습니다.  
그러나 MS-SSIM은 실험한 다른 손실 함수보다 고주파 영역에서 대비를 더 잘 보존합니다. 반면에 'l1'은 색상과 휘도를 보존하지만(오차는 국소 구조에 관계없이 동일하게 가중치가 부여됨) MS-SSIM과 완전히 동일한 대비를 생성하지는 않습니다.  

결론적으로 손실 함수로 MS-SSIM과 'l2'의 조합을 선택했습니다:

Images were cropped to square form and resized to the size of 440 × 440 pixels. We apply the contrast limited adaptive histogram equalization (CLAHE) for local contrast enhancement, so local details can therefore be enhanced even in regions that are darker or lighter than most of the image.   

이미지는 정사각형 형태로 자르고 440 × 440 픽셀 크기로 크기를 조정했습니다. 따라서 대부분의 이미지보다 어둡거나 가벼운 영역에서도 로컬 디테일을 향상시킬 수 있도록 대비 제한 적응 히스토그램 균등화(CLAHE)를 적용했습니다. 

