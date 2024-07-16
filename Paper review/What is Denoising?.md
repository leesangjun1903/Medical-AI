# Denoising

노이즈 제거는 노이즈에 의해 손상된 reference signal을 복구하는 과정입니다.   
컴퓨터 비전에서 reference signal는 일반적으로 물체나 장면의 왜곡되지 않은 이미지로 가정되며, 이미징 과정의 결과로 노이즈가 도입됩니다.   
노이즈의 양과 유형은 응용마다 달라집니다. 일반적인 노이즈 이미지의 예와 이에 대한 이미지 노이즈 제거를 수행한 결과는 그림 1과 같습니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-07-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.55.46.png)
왼쪽 : original 에서 noise 적용한 MRI 사진, 오른쪽 : 디노이징한 사진

## 배경
이미지화 과정에서 필연적으로 발생하는 결과는 노이즈입니다.   
노이즈의 원인으로는 신호 획득 및 처리 과정에서 발생하는 측정 및 quantization errors, 디지털 이미징 시스템의 센서 및 전자 장치에서 발생하는 thermal noise, 필름의 경우 사진 입자, 빛 자체의 물리적 특성 등이 있습니다.

denoising의 중요성은 입력 이미지의 노이즈가 모든 all subsequent visual processing에 악영향을 미칠 수 있다는 사실에서 비롯됩니다.   
brightness gradients와 같은 value of image-dependent quantities, accuracy in the localization of image features, object boundaries 실재 유무, 심지어 subjective perceptual properties of the image 모두 노이즈에 의해 어느 정도 영향을 받습니다.   
의료 영상, 천문학, low-light 또는 고속 촬영, synthetic aperture radar(SAR) 이미징과 같은 응용 분야는 일반적으로 기준 신호에 비해 더 많은 양의 노이즈가 특징입니다.  

용도에 따라 input을 denoising algorithm즘으로 preprocessing하면 시각적 처리의 추가 단계에서 얻은 결과를 개선할 수 있습니다.   
그러나 denoising process는 불완전하고 reference signal에 포함된 정보의 일부를 항상 파괴하기 때문에 노이즈 제거 방법은 신중하게 선택해야 합니다.  

## 이론
그레이스케일 이미지의 경우 인식되는 이미지 밝기는 [1]에 의해 근사화될 수 있습니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-07-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.26.39.png)

여기서 I는 관측된 brightness value, f(·)는 sensor response function, L은 image irradiance, ns는 noise contribution from brightness-dependent sources, nc는 constant noise factor, nq는 quantization noise입니다.  
이 모델의 모든 양을 추정하는 것은 자신만 아는 경우에는 너무 복잡한 문제이므로 실제로는 단순화된 관계 I = Ir + N을 대신하여 관찰된 I : observed image brightness, Ir : reference image that the denoising process must estimate, N을 noise component로 사용합니다.   
알려진 값(I, the observed brightness)이 하나뿐이고 알 수 없는 값이 두 개이므로 문제는 제한되지 않습니다.  
해결책을 계산할 수 있도록 Ir과 N의 속성에 추가적인 external constraints을 설정해야 합니다.   
일반적으로 N은 i.i.d. (independent and identically distributed) noise로 가정하며, 대부분의 노이즈 제거 방법은 고정된 표준 편차를 가진 zero mean Gaussian distribution로 가정합니다.   
Ir의 속성에 대한 제약 조건에 따라 handful of major classes로 denoising methods들을 그룹화할 수 있습니다.  

## Classes of Denoising Algorithms
Denoising Algorithms의 첫 번째 class는 brightness away from image edges가 uniform해야 한다는 가정 하에 the value of pixels within small image neighborhoods부분을 averaging하는 것을 기반으로 합니다.  
이러한 알고리즘은 strong brightness discontinuities을 유지하면서 이미지에서 균일해 보이는 영역을 smoothing하는 경향이 강하기 때문에 anisotropic smoothing를 수행합니다[2].  
original anisotropic smoothing algorithm[3], bilateral filter[4], minimizing total variation에 기반한 방법[5] 및 stochastic denoising algorithm[6]은 모두 homogeneous regions에 걸쳐 maximizing brightness uniformity하는 것을 기반으로 하는 노이즈 제거 방법의 예입니다.   

두 번째 class는 analysis of image statistics을 기반으로 합니다. 
기본 원리는 natural images의 statistical properties을 모델링할 수 있고 모델이 주어지면 examining the statistics of a noisy image하고 통계가 학습된 모델의 통계와 일치하도록 이미지를 변환하여 노이즈 제거를 수행할 수 있다는 것입니다[7]. 
일반적인 모델에는 distributions of filter responses와 pooled statistics for collections of image patches가 포함됩니다. 
Gaussian scale mixtures[8], fields of experts[9], nonlocal means method[10] 및 block matching algorithm[11]과 같은 알고리즘은 이미지 또는 작은 이미지 패치의 통계적 규칙성을 활용합니다.

salt and pepper noise을 제거하기 위해 특별히 고안된 세 번째 class의 이미지 노이즈 제거 방법은 outlier detection를 기반으로 합니다.  
이 프로세스는 이미지 주변의 distribution of brightness values를 추정하고 이 분포를 사용하여 이상치를 식별하고 제거하는 것을 포함합니다.   
median filter[12]는 이러한 등급의 노이즈 제거 알고리즘의 예입니다.

Denoising의 목표는 reference signal에 포함된 정보를 보존하면서 가능한 한 많은 노이즈를 제거하는 것입니다.   
이러한 이유로 noise-free reference signal이 알려진 이미지에서 PSNR(peak signal to noise ratio) 또는 SSIM(structured image similarity index)[13]을 사용하여 노이즈 제거 알고리즘을 평가하는 경우가 많습니다.

# 응용
노이즈 제거는 의료 이미징, 천문학 또는 synthetic aperture radar와 같은 영역의 이미지에 유용한 preprocessing 단계일 수 있습니다. 
low illumination conditions에서 디지털 사진 촬영과 archival footage 및 사진 복원에도 적용할 수 있습니다.  
이미지 편집 및 이미지 처리 프로그램에는 일반적으로 노이즈 제거 모듈이 포함됩니다.

# 실험 결과
서로 다른 알고리즘에 의해 생성된 결과의 품질은 이미지에서 이미지로, 서로 다른 도메인에 걸쳐 달라집니다.   
그러나 최근의 벤치마크[14]는 block matching algorithm이 서로 다른 Gaussian noise 양을 가진 natural images에서 전반적으로 더 나은 성능을 달성한다는 것을 나타냅니다.



