# CT Image Denoising and Deblurring With Deep Learning: Current Status and Perspectives

# Abstract
이 글에서는 컴퓨터 단층 촬영 이미지 노이즈 제거 및 디블러링을 위한 딥 러닝 방법을 개별적으로 동시에 검토합니다.  
그런 다음 대규모의 사전 학습 모델 및 LLM 모델과의 조합과 같은 이 분야의 유망한 방향에 대해 논의합니다.  
현재 딥 러닝은 데이터 기반 방식으로 의료 영상에 혁명을 일으키고 있습니다.  
빠르게 진화하는 학습 패러다임과 함께 관련 알고리즘과 모델은 임상 응용 분야로 빠르게 발전하고 있습니다.  

# Introduction
딥 러닝, 특히 심층 컨볼루션 신경망(CNN)은 10년 이상 컴퓨터 비전 분야를 지배해 왔습니다.  
End-to-end 방식의 공간 특징 추출 기능(spatial feature extraction)을 통해 고성능의 인식 모델을 보다 효율적으로 구축할 수 있습니다[1] (ResNet), [2] (VGG16), [3] (EfficientNet), [4] (DenseNet), [5] (WRN).  
Denoising, deblurring/super-resolution (SR) 및 이미지 생성을 포함한 심층 image-to-image 변환 작업은 주로 CNN 기반 인코더-디코더 아키텍처를 기반으로 하며, 평균 제곱 오차(MSE) 기반 reconstruction loss는 주로 네트워크[6] (VAE), [7](Denoising AE)를 훈련하는 데 채택됩니다.  
기존의 노이즈 제거 또는 디블러링 방법, 예를 들어 총 total variation-based method[8], [9], 특정 인스턴스별 하이퍼파라미터, 반복되는 계산, 느린 추론 속도로 인해 견고성과 확장성이 부족합니다.  
반면, 딥 러닝은 많은 수의 이미지에 대해 모델을 훈련하고 더 빠른 추론을 수행할 수 있는 데이터 기반 기술입니다.  
CNN은 의미 구조를 포함한 처리된 이미지에서 더 나은 모양과 텍스처 정보를 보존하여 재구성된 이미지의 더 높은 fidelity를 보장합니다.  
이미지 노이즈 제거 및 디블러링은 컴퓨터 비전 커뮤니티에서 고전적인 작업이며, 정량적 및 정성적 측정 측면에서 심층 컨볼루션 신경망에 의해 향상되었습니다[10], [11], [12] (DnCNN).

컴퓨터 단층 촬영(CT)은 비침습적으로 인체 내부 구조의 2D 또는 3D 이미지를 생성하는 최초의 기술입니다[13].  
CT 영상은 1970년대에 개발된 이래로 진단을 용이하게 하지만[14], 건강에 대한 잠재적인 X선 방사선에 대한 대중의 우려가 있습니다.  
가능한 한 방사선량을 낮추면서 임상적으로 허용 가능한 화질을 달성하는 것이 바람직합니다.  
화질은 검출기 특성, 환자 움직임, artifact, 계산 오류, 방사선량 등 재구성 과정의 많은 요인과 관련이 있습니다[15], [16].  
안타깝게도 원래 신호를 재구성하기 위해 전체 degradation process을 완벽하게 반전시키는 것은 다루기 어렵습니다.  
따라서 고품질 CT 이미지를 얻기 위해 노이즈 제거와 디블러링은 열화 과정을 최적으로 되돌리는 것을 목표로 하는 역 문제로 공식화됩니다.  
노이즈 제거와 디블러링의 연관성은 노이즈 제거 프로세스가 가장자리가 흐려지거나 세부 사항이 왜곡되어 이 두 가지 작업이 동시에 이상적으로 해결되어야 할 수 있다는 점에 있습니다.  
또한 고품질 CT 이미지는 딥 러닝 기반 이미지 분석에도 유리합니다.  
예를 들어, 노이즈가 많고 흐릿한 이미지는 작은 구조나 텍스처가 평활화(smooth)되고 잘못 표현되어 잘못된 양성 및 잘못된 음성으로 이어지는 경향이 있기 때문에 진단을 방해할 수 있습니다.  
따라서 유해한 요인에 대한 노이즈 제거 및 디블러링은 진단 성능을 최적화하는 것이 필수적입니다.  
인기 있는 딥 러닝 아키텍처에는 U-Net[17], V-Net[18], nU-Net[19]과 같은 CNN 기반 모델과 SIST[20], CTformer [21], Ted-Net[22], LIT-Former [23],Eformer[24] 등의 트랜스포머 기반 모델이 포함됩니다.

딥 러닝 기반 CT 노이즈 제거 방법은 크게 지도 학습과 자기/비지도 학습으로 분류할 수 있습니다.  
지도 학습 작업은 노이즈가 많은 이미지[즉, low-dose CT(LDCT)]와 깨끗한 이미지[즉, normal-dose CT(NDCT)] 간의 reconstruction loss을 최소화합니다[16].  

그러나 지도 학습에는 대량의 쌍을 이루는 데이터[25], [26], [27], [28]가 필요하며, 예를 들어 LDCT-NDCT 이미지 쌍은 동일한 환자가 쌍을 이루는 CT 스캔을 생성하기 위해 두 번 CT 스캔을 받도록 하는 것은 윤리적이지 않기 때문에 어려운 일입니다.  
이러한 한계를 피하기 위해 연구자들은 이상적이지 않은 수치 시뮬레이션을 활용하고 훈련 중에 깨끗한 이미지가 필요 없는 자기 지도 학습 또는 비지도 학습에 집중하기 시작합니다.  
이러한 학습 패러다임은 자기 지도 학습 작업[29](Noise2Noise), [30](Noise2Void), [31](Noise2Self), [32](Self2Self with dropout), [33], [34](Noise2Inverse), [35], [36], [37], [38](Deformed2Self), [40], [41], [42](Noise2Contrast), [43](SACNN).  
자기 지도 학습은 감독 대상에 비해 경쟁 성능을 유지하면서 쌍을 이루는 샘플의 필요성을 완화하는 데 효과적이지만 생성된 이미지의 이미지 품질은 여전히 추가 개선이 필요합니다.

Deep deblurring은 해당 저해상도(LR) 대응물의 고해상도(HR) 이미지를 복구하는 것을 목표로 합니다.  
이는 일반적인 역문제(inverse problem)입니다.  
Degradation 작업은 blurring, noising, downsampling 등이 될 수 있습니다.  
Image deblurring은 이미지 SR로 공식화하여 상세한 구조와 텍스처를 충실히 추정하면서 LR 또는 블러링된 이미지에서 HR 이미지를 복원할 수 있습니다[16], [44].  
Deep denoising와 동일한 supervised learning 설정에서 deblurring은 대량의 LR 및 HR 이미지 쌍도 필요합니다.  
Denoising와 달리 입력 이미지 크기가 SR 출력 이미지의 크기와 반드시 동일한 것은 아닙니다.  
또한 일부 연구자들은 Meta-learning 을 사용하여 임의의 스케일 SR을 탐색했습니다[45](MetaSR), [46](MetaUSR).  
Self-supervised deblurring은 인접한 CT 슬라이스 이미지 간의 보간과 같은 self-supervision 작업도 구성하는 데 중점을 둡니다[47].  
이 글에서는 주로 CT 이미지에 대한 딥 러닝 기반 디블러링 및 SR 방법을 다룹니다.

CT 이미지의 노이즈 제거 및 디블러링에 대한 최근의 고급 연구는 image inpainting 및 image editing과 같은 이미지 생성 작업에서 생성적 적대 신경망(GAN) 기반 방법[49]을 능가하는 생성 모델인 노이즈 제거 확산 확률 모델(DDPM)[48]을 기반으로 수행됩니다.  
또한 점진적 노이즈 추가 및 제거 프로세스를 학습하여 CT 이미지 노이즈 제거에도 적용되었습니다[50], [51](CoCoDiff).  
DDPM과 그 변형은 더 높은 fidelity로 CT 이미지 노이즈 제거에 힘을 실어주며, 새로 속도를 높인 대응 방식을 통해 확산 모델을 보다 실용적으로 만들 수 있습니다.  

모델 아키텍처의 경우 CNN 관련 방법이 의료 이미지 복원(IR) 문제에서 우수한 성능을 발휘함에도 불구하고 self-attention(SA) 기반 transformer[52] 아키텍처는 노이즈 제거, 디블러링, 분류, 분할 등 다양한 작업에서 인기를 얻고 있습니다.  
자연어 처리 분야에서 트랜스포머의 높은 성능에서 영감을 받은 vision transformer(ViT)[53]는 노이즈 제거 및 디블러링을 포함한 비전 작업의 성능을 더욱 질적으로 도약시킵니다.  
따라서 트랜스포머 기반 CT 이미지 노이즈 제거 및 디블러링의 최근 발전을 검토하고 CNN 방법과 비교합니다.

요약하자면, 먼저 섹션 II(INVERSE PROBLEM AND COMMON METRICS)에서 역 문제와 일반적으로 사용되는 메트릭을 소개하고 의료 이미지 노이즈 제거 및 디블러링에 대한 기존 검토[15], [44]와 달리, 이 글에서는 섹션 III(DEEP DENOISING METHODS) 및 IV(DEEP DEBLURRING METHODS) 의 Supervised Learning, Self-Supervised Learning, 메타 학습 및 고급 Diffusion 기반 방법과 같은 학습 패러다임에 따라 딥 러닝 기반 노이즈 제거 및 디블러링 방법에 대한 최근 연구를 분류하는 경향이 있습니다.  
특히 인기 있는 네트워크 아키텍처, 학습 알고리즘, 기본 모듈 및 일부 plug-and-play 트릭을 소개합니다.  
또한 이미지 재구성 과정에서 3D 시나리오에서 높은 효율성을 달성하면서 노이즈 제거와 디블러링을 함께 수행해야 하며, 이는 이 두 가지 작업만 연결할 수 있는 기존 방법으로는 어렵습니다.  
이러한 유형의 시너지를 위해 섹션 V(SIMULTANEOUS DENOISING AND DEBLURRING) 에서는 CT 이미지 노이즈 제거와 디블러링에 대한 몇 가지 결과를 설명합니다.  
그런 다음 대규모 비전 언어 모델과 visual prompt 학습에 비추어 섹션 VI(PROMISING DIRECTIONS) 에서는 이러한 임상 응용 기술을 개발하기 위한 인사이트와 방향을 추가로 제공합니다.  
이 글에서 검토한 논문을 수집하기 위해 2000년부터 2023년까지 기간을 두고 "딥 러닝 CT 이미지 노이즈 제거", "CT 이미지 디블러링/SR", "자기 지도 CT 노이즈 제거/디블러링/SR", "메타 학습 CT 디블러링/SR"이라는 검색어를 과학 웹에 입력했습니다.  
설문조사를 통해 우리의 동기는 가능한 한 많은 중요한 논문을 다루는 것입니다. 따라서 Google Scholar, PubMed, IEEE Xplore 등을 사용하여 관련 기사도 검색했습니다.

# INVERSE PROBLEM AND COMMON METRICS
## Inverse Problem
역 문제(Inverse problem)는 손상된 관찰 이미지에서 목표 신호(예: 1-D 신호, 2-D 이미지 및 고차원 텐서)를 재구성하는 것을 목표로 합니다.  
일반적으로 순방향 모델(forward model)은 다음과 같이 정의됩니다 :

$y = A_(\hat x) + \epsilon $

여기서 y는 손상된 측정값이고, $\hat x$ 는 알려지지 않은 손상된 신호이며, $A$ 는 손상된 프로세스이며 $\epsilon$ 은 가우시안 및 푸아송과 같은 추가 노이즈를 나타냅니다.  
그런 다음 목표는 $y$의 $\hat x$를 복구하는 것입니다.  
CT 이미지의 노이즈 제거와 디블러링은 역 문제의 두 가지 주요 사례입니다.  
위 식의 역 문제를 해결하기 위한 기존 접근 방식은 이미지 노이즈 제거와 마찬가지로 Tikohonov, total variation [8], [9]와 같은 다양한 regularization term으로 least square 기반 fidelity 목표를 최소화하는 데 중점을 두어 더 나은 가장자리 및 텍스처 보존을 제공합니다.  
이러한 방법은 Split Bregman 및 ADMM[55], [56], [57]과 같은 반복적인 최적화에 의존합니다.  
또 다른 고전적인 방법인 nonlocal mean(NLM)은 참조한 픽셀과 유사한 픽셀을 비국소적 방식으로 집계하여 이미지 노이즈를 통계적으로 억제하는 것입니다[58], [59], [60], [61].  
또한 dictionary learning-based, wavelet-based 및 BM3D 기반 방법도 이미지 노이즈 제거에 널리 적용되었습니다[62], [63], [64], [65], [66].  
이러한 전통적인 방법은 실제로 유용하지만 계산 비용이 많이 들고 일반화하기 어려운 경우가 많습니다.  
위에서 언급한 전통적인 방법은 수작업이 필요한 필터, 사양 정규화 및 민감한 임계값 선택을 적용하며, 이론적 해석 가능성이 높지만 많은 실제 임상 시나리오에서 결과 성능이 요구 사항을 충족하지 못할 수 있습니다.

딥 러닝은 지난 10년 동안 의료 영상 영역을 지배해 왔으며, end-to-end 딥 러닝 아키텍처는 정량적 및 정성적 결과 모두에서 위에서 언급한 기존 방법을 능가했습니다.  
기존 딥 러닝 방법의 대부분은 CNN과 트랜스포머를 기반으로 하며 supervised or self-/un-supervised loss function로 훈련됩니다.  
즉, 관찰된 이미지에서 깨끗한 신호로의 직접 매핑을 학습했습니다.  
일반적인 노이즈 제거 모델은 단축 연결의 이점을 누리는 U-Net 관련 [17], [19](V-Net) 아키텍처[25](RED-CNN), [26](CPCE-3D), [27], [28](WGAN_VGG)을 기반으로 구축됩니다.  
최근 Diffusion Probabilistic 모델링[50], 예를 들어 DDPM[48]은 이미지 생성, 이미지 인페인팅 등을 포함한 생성 모델링에서 좋은 성과를 거두었습니다.  
기존 딥 러닝 방법과는 다른 점진적인 degradation 절차를 시뮬레이션합니다.  
Diffusion 프로세스는 입력에 반복적으로 노이즈를 추가하는 동시에 깨끗한 신호가 복구될 때까지 훈련 프로세스가 역방향으로 노이즈를 제거합니다.  
Inverse problem의 경우 Diffusion 모델은 관측값을 입력으로 받아 VAE 및 GAN에 비해 더 높은 품질의 결과를 생성하는 노이즈 제거 프로세스를 시뮬레이션합니다.  
Song 등은 DDPM 외에도 점수 기반 확산 모델[67](NCSN), [68](SBGM), [69](Score-based generative model) 및 확률적 미분 방정식 기반 방법[70](Score_SDE), [71](Score-Based Diffusion ODEs)에 대한 멋진 작업을 수행했습니다.

이 글에서는 최신 접근 방식을 포함하여 섹션 III-V에서 이러한 딥 러닝 방법을 조사할 것입니다.

## Evaluation Metrics
먼저, CT 이미지 노이즈 제거 및 디블러링의 맥락에서 재구성된 이미지의 품질을 평가하기 위한 주요 지표를 소개합니다.  
이러한 측정은 종종 주어진 참조 이미지와 처리된 이미지 사이에서 계산됩니다.  
MSE와 Root MSE(RMSE)는 Euclidean distance를 기반으로 한 두 가지 픽셀 단위 차이 측정으로, 다음과 같이 정의됩니다:

![image](https://github.com/user-attachments/assets/9f4994a7-4415-452a-8837-61f3ac7f044e)

여기서 $I_{ND}$ 는 NDCT 이미지를 나타내며 $\hat I_{LD}$ 는 노이즈가 제거된 LDCT 이미지에 해당합니다.  
MSE와 RMSE는 두 이미지 간의 픽셀 단위 차이를 평가하는 데 유용하지만 인간의 시각 시스템(HVS)과 일치하는 데는 열등합니다.  
대표적인 예는 유사한 MSE 또는 RMSE 값을 가진 일부 재구성된 이미지가 있지만 시각적 품질이 심각하게 다르다는 것입니다.  
생체 모방 관점에서 structural similarity(SSIM)은 구조 유사성 평가에 널리 사용되며[73] HVS와 상관관계가 있습니다.  
$I_{ND}$ 및 $\hat I_{LD}$가 주어지면 SSIM은 다음과 같이 계산됩니다:

![image](https://github.com/user-attachments/assets/c2f0c80e-7305-4f3a-b7fd-ebc5ced07879)

여기서 $x = \hat I_{LD}$, $y = I_{ND}$, $μ$ 및 $σ$은 각각 평균과 분산을 나타내며 C1과 C2는 두 상수입니다.  
SSIM은 밝기, 대비 및 구조를 체계적으로 비교합니다.  

노이즈가 있는 이미지의 경우 노이즈 제거 프로세스 후 품질을 설명하는 메트릭이 필요합니다.  
Peak signal-to-noise ratio(PSNR)는 일반적으로 사용되는 메트릭입니다[74]. PSNR은 다음과 같이 정의됩니다 :

![image](https://github.com/user-attachments/assets/9382c3be-8c30-456d-8fcb-b5495c4c2c07)

여기서 $MAX(I_{ND})$ 는 $I_{ND}$의 최대로 측정 가능한 픽셀 값을 나타냅니다.  
분명히 PSNR은 MSE 값과 상관관계가 있으며, MSE가 0에 가까워지면 PSNR 값은 무한대, 즉 최상의 화질로 이동합니다.  
즉, PSNR은 구조 보존을 고려하지 않고 재구성된 LDCT와 NDCT 간의 수치적 차이를 측정합니다.

# DEEP DENOISING METHODS
이 섹션에서는 supervised learning, self/un-supervised learning 및 최근 Diffusion 기반 생성 모델과 같은 다양한 학습 패러다임에 따른 심층 노이즈 제거 방법을 종합적으로 검토합니다.  
그림 1에는 최근 몇 년 동안 발표된 대표적인 방법이 요약되어 있습니다.  

![image](https://github.com/user-attachments/assets/eda039d3-df57-4ecf-9391-ec1e99cc9e51)

인기 있는 노이즈 제거 데이터 세트는 표 I에 요약되어 있으며 관련 논문은 표 II에 요약되어 있습니다.

![image](https://github.com/user-attachments/assets/3ad735ef-91d9-4395-9475-c7d11fb67da1)

![image](https://github.com/user-attachments/assets/a4dba159-14fa-4e0d-b4d3-3260f2c17634)

## Supervised Denoising 
### Problem Formulation: 
지도 학습 설정에서 노이즈 제거의 학습 목표는 다음과 같이 공식화할 수 있는 페어링된 ground-truth 레이블로 신경망 기반 노이즈 제거 모델을 훈련하는 것입니다:

![image](https://github.com/user-attachments/assets/cba52d0c-e07b-483a-b666-13fc9cfcd483)

여기서 $f_θ$은 $θ$로 매개변수화된 심층 신경망이고, $L$ 은 MSE와 같은 image-to-image reconstruction loss을 나타냅니다.  
$I_{LD}$와 $I_{ND}$는 각각 LDCT 입력과 NDCT ground-truth 이미지입니다.  
의료 이미지 분할을 위해 제안된 U-Net 모델에서 영감을 받은 skip-connection이 있는 encoder–decoder 아키텍처는 의료 이미지 재구성 작업에서 잘 탐구되어 왔습니다[17].  
지도 학습 설정에서 딥 러닝 모델은 훈련을 위해 많은 양의 페어링된 NDCT 샘플이 필요합니다.

### CNN-Based Denoising: 
CNN 기반 CT 이미지 재구성은 [82]에서 처음 논의된 후 Chen 등[83]은 CT 이미지 노이즈 제거에 CNN을 적용하는 것의 효과를 검증했습니다.  
Wavelet transform은 딥 러닝 아키텍처에도 도입되었습니다[84].  
CNN-based denoising autoencoder(CNN-DAE)는 LDCT 노이즈 제거를 위한 autoencoder에 컨볼루션 연산을 처음 적용했습니다[7].  
U-Net의 다양한 스케일의 feature map 간에 자세한 정보를 학습하기 위해 Chen 등[25]은 LDCT 노이즈 제거를 위한 residual encoder–decoder CNN(RED-CNN)을 제안했습니다.  
RED-CNN은 구조적 세부 사항을 유지하면서 통계적 특성을 모델링할 필요가 없습니다.  
그림 2(a)는 RED-CNN의 아키텍처를 보여주며, U-Net과 유사한 개념의 encoder–decoder를 활용하지만 RED-CNN에 사용되는 skip-connection은 U-Net의 channel-wise concatenation과 element-wise addition로 구현됩니다.  

![image](https://github.com/user-attachments/assets/4b9ce13b-98c7-4c3a-a681-c5470b0de918)


이러한 encoder–decoder 아키텍처의 피할 수 없는 문제는 지속적인 downsampling을 위한 information loss(정보 손실)이 발생한다는 것입니다.
이러한 장애물을 극복하기 위해 Huang 등[85]은 보다 상세한 구조를 보존하는 LDCT를 위한 deep cascaded residual network(DCRN)를 제안했습니다.  
Detail 보존을 더욱 강화하기 위해 WGAN-VGG는 생성적 적대적 손실과 지각적 손실을 적용하여 MSE 기반 손실로 인한 중요한 구조적 세부 사항의 낮은 품질을 개선했습니다.  
GAN[49]은 판별기와 생성기가 서로 경쟁한 다음 더 나은 판별기와 생성기를 얻을 수 있는 zero-sum game의 개념을 아이디어로 가져온 생성 모델입니다.  
노이즈 제거 이미지 생성을 위한 CNN 생성기, perceptual loss을 평가하기 위한 VGG 네트워크, NDCT 이미지에서 생성된 이미지를 식별하기 위한 판별기 네트워크로 구성된 WGAN-VG의 프레임워크는 그림 2(b)에 나와 있습니다. 

![image](https://github.com/user-attachments/assets/43bd7857-2448-4c30-af96-1334813a4757)

생성된 이미지와 NDCT 이미지의 컨볼루션 특징 간의 유사성을 측정하기 위해 WGAN-VGG는 perceptual loss[86]을 활용했습니다.  
또한 WGAN-VGG는 LDCT와 NDCT 이미지 간의 불일치를 측정하기 위해 Wasserstein distance를 사용했습니다.  
이러한 손실은 노이즈와 아티팩트 문제를 완화하는 데 도움이 될 수 있습니다.  
Skip-connection은 널리 사용되는 기술인 image-to-image 변환 작업의 성능을 크게 향상시킵니다.

3D CT 스캔과 관련하여 Shan et al. [26]은 잘 훈련된 2D 네트워크를 활용하여 3D CT 노이즈를 개선할 것을 제안했습니다.  
그들은 adversarial loss과 perceptual loss도 적용한 그림 2(c)에 표시된 conveying path-based convolutional encoder–decoder(CPCE)를 제안했습니다.  

![image](https://github.com/user-attachments/assets/969a9f6a-4ca5-4236-ab89-b54e53d70117)

RED-CNN 및 WGAN-VGG와 달리 CPCE는 사전 훈련된 2D 커널에서 3D 노이즈 제거 네트워크를 구축하는 데 더 중점을 두었습니다[26].  
한 걸음 더 나아가, Shan et al. [27]은 modularized adaptive processing neural network(MAP-NN)이라고 불리는 end-to-process 학습 프레임워크에서 CPCE를 모듈로 공식화했습니다.  
MAP-NN은 점진적 훈련 루프에 전문가의 지식을 통합하여 성능을 향상시킵니다. 특히, 이는 방사선 전문의의 판단을 테스트 단계에 통합했습니다.  
이를 통해 결과의 신뢰성을 보장했습니다.  
놀랍게도 MAP-NN은 노이즈 억제, structural fidelity 및 처리 속도 면에서 상용 반복 재구성(IR) 방법을 능가합니다.  
MSE loss에도 불구하고 MAP-NN은 LDCT 및 NDCT 이미지에서 Sobel filtration에 대한 adversarial loss 및 MSE loss로 훈련되기도 했습니다.  
요약하면, CNN 아키텍처를 기반으로 MSE는 이미지 콘텐츠의 대략적인 구조를 보장합니다.  
Regularization를 사용하거나 adversarial, perceptual loss와 같은 신중하게 설계된 손실 함수를 사용하여 세부적인 구조를 보존하는 것이 더 중요합니다.

모델 구조의 경우, 그림 2는 세부 정보 보존을 위해 element-wise addition를 도입하는 RED-CNN, CT 노이즈 제거에서 적대적 학습 개념을 활용하는 선행 작업인 WGAN-VGG, conveying-path 및 적대적 학습을 적용하여 3D CT 데이터 노이즈 제거를 위한 향상된 프레임워크를 개발하는 등 서로 다른 관점의 세 가지 일반적인 아키텍처를 보여줍니다.  

![image](https://github.com/user-attachments/assets/3851ab11-0068-4dd1-a42c-41fdcc82d15c)

Loss function의 경우 RED-CNN은 주로 MSE loss을 사용했으며, CPCE는 MSE와 adversarial loss의 조합을 추가로 사용했습니다.  
WGAN-VGG는 단순한 MSE loss을 포기하고 CNN이 정확한 intermediate feature을 보존할 수 있는 perceptual loss을 사용했으며, 이는 output feature과 NDCT feature 간에 계산됩니다.  
그림 3에서 RED-CNN은 일부 작은 세부 구조가 누락된 비교적 매끄러운(smoothed) 결과를 얻을 수 있음을 알 수 있습니다.  

![image](https://github.com/user-attachments/assets/90053f34-46d6-4d82-943b-ee7f90a74bb2)

적대적 학습을 활용하여 WGAN-VGG와 CPCE는 적절한 텍스처로 보다 사실적인 이미지를 달성했습니다.  
또한 conveying-path를 통해 CPCE는 WGAN-VG보다 더 나은 성능을 발휘할 수 있습니다.

기존의 컨볼루션 아키텍처와 달리 Fan et al. [88]은 LDCT 노이즈 제거를 위한 quadratic neuron과 해당 quadratic autoencoder (Q-AE)를 제안했습니다.  
validation loss을 통해 Q-AE는 RED-CNN 및 CPCE-2D보다 더 나은 수렴 동작을 보여주었습니다.  
정성적 평가를 위해 방사선 전문의는 Q-AE가 얻은 노이즈 제거 및 세부 정보 보존 측면에서 약간 우수한 성능을 확인했습니다.  
강 외 [87]은 framelet-based 노이즈 제거 방법과 심층 신경망을 결합하여 제안된 convolutional framelet에 의한 성능을 보장하는 LDCT 노이즈 제거를 위한 wavelet residual network(WavResNet)를 제안했습니다.  
Sparse-view CT에서의 노이즈 제거를 위해 Chen 외 [100]은 희소하게 수집된 데이터 또는 언더샘플링된 측정값으로부터 단층 촬영 재구성(CT)을 위한 compressive sensing method을 제안했습니다.  
제안된 방법은 학습된 전문가의 평가 기반 재구성 네트워크를 기반으로 하며, 이는 Mayo 데이터 세트에서 우수한 성능을 생성합니다.  
Zhang 외 [101]은 강력한 희소 시점 CT 재구성을 위한 REDAEP를 제안했으며, 이는 $l^2$ loss을 더 나은 텍스처 보존을 위한 더 강력한 $l^p$ 손실로 대체했습니다.

CT 스캔의 여러 슬라이스들로 인해 multichannel 또는 multicolumn model과 트릭들은 더 많은 정보를 활용하는 데 집중되어 왔습니다.  
슬라이스들 간의 관계를 고려하기 위해 Li 등[43]은 SA 기법을 활용하여 평면 주의력과 깊이 주의력을 동시에 융합하는 3D CNN 생성기인 SACNN을 구축했습니다.  
그런 다음 CPCE[26] 및 WGAN-VGG[28]와 유사하게 SACNN도 adversarial loss, perceptual loss로 최적화되었습니다.  
Artifact and detail attention GAN(ADAGAN)[76]은 서로 다른 스케일의 입력에 따라 점진적으로 cascaded(여러가지 모델을 추가하여 늘려나가는 과정) 되는 multiscale network로 구성됩니다.  
또한 노이즈가 제거된 LDCT와 NDCT를 식별하기 위한 판별자로 multiscale Res2Net이 제안되었습니다.  
Zhang 등[89]은 comprehensive learning-enabled adversarial reconstruction(CLEAR)이라는 이미지 domain과 projection domain 재구성을 모두 포함하는 3D ResUNet 기반 재구성 프레임워크를 제안했습니다.

CLEAR에 사용된 판별기는 이미지와 projection domain 의 출력을 추가로 결합한 3D CNN이기도 합니다.  
최근 Li 등[102]은 CycleGAN, IdentityGAN, GAN-CIRCLE을 사용하여 LDCT 노이즈 제거를 위한 페어링되지 않은 딥 러닝 기술을 조사했습니다.  
추가 노이즈의 정의를 기반으로 Geng 등[90]은 노이즈가 많은 이미지를 해당 콘텐츠와 노이즈 부분으로 분해했습니다.  
특히 고전적인 U-Net과 DnCNN[12]으로 구현된 콘텐츠 predictor와 노이즈 predictor를 설계했습니다.  
그런 다음 PatchGAN을 적용하여 실제 이미지와 가짜 이미지에 대해 각각 예측된 콘텐츠와 노이즈를 구별했습니다[103].  
임상 환경에는 충분한 NDCT 이미지가 없기 때문에 연구자들은 페어링되지 않은 이미지에서 학습 쌍을 생성하려고 노력합니다.  
Physics-based noise model에서 영감을 받은 [80]에서 저자들은 페어링되지 않은 데이터에서 페어링된 원본 CT 이미지와 노이즈가 많은 CT 이미지를 생성하는 weakly supervised denoising framework를 설계했습니다.  
이 프레임워크에는 작은 노이즈의 차이를 점진적으로 보정하여 LDCT에서 NDCT 이미지로 직접 매핑하는 문제를 우회하는 progressive denoising module이 포함되어 있습니다.  
Noise power spectrum과 signal detection accuracy는 진단 이미지 품질을 정량적으로 평가하는 데 사용됩니다.  
실험 결과는 제안된 방법이 감독된 노이즈 제거 방법보다 우수한 놀라운 노이즈 제거 성능을 달성한다는 것을 보여줍니다.  
이 프레임워크는 데이터 수집에 유연하며 모든 선량 수준에서 페어링되지 않은 데이터를 활용할 수 있습니다.

### Transformer-Based Denoising:
Parvaiz et al. [104]는 이중 향상 기능을 갖춘 transformer를 사용하여 LDCT 노이즈를 제거하기 위한 DEformer를 제안했습니다.  
제안된 방법은 이미지 노이즈 제거를 수행하기 위해 겹치지 않고 window-based SA transformer block을 사용하는 주요 부분과 double enhancement module을 통해 edge, texture, and context의 정보를 향상시키기 위해 맞춤화된 두 개의 작은 부분의 세 부분으로 구성됩니다[94].  
Luthra et al. [24]는 의료 이미지 노이즈 제거를 위한 transformer-based encoder–decoder network이기도 한 Eformer를 제시했습니다.  
Eformer는 학습 가능한 Sobel–Feldman operator를 edge information 향상을 위한 훈련에 통합합니다.  
동시에 계산 효율성은 transformer block의 겹치지 않는 window-based SA에 의해 가져옵니다.  
Wang et al. [95]은 masked autoencoder를 사용하여 LDCT 노이즈 제거를 위한 SwinIR+MAE를 제안했습니다.  
제안된 방법은 1) 노이즈가 많은 이미지와 깨끗한 이미지 간의 매핑을 학습하는 denoising autoencoder와 2) 노이즈 제거 프로세스를 안내하는 마스크를 생성하는 mask generator의 두 부분으로 구성됩니다.  
Zhang et al. [96]은 dual-path transformer를 사용하여 LDCT 노이즈 제거를 위한 TransCT를 제안했습니다.  
제안된 방법은 1) SA transformer block을 사용하여 이미지 노이즈를 제거하는 주요 경로와 2) 입력 이미지의 표현을 향상시키는 multiscale feature fusion module을 사용하는 보조 경로의 두 가지 경로로 구성됩니다.
Li et al. [97]은 컨볼루션 및 Swin-transformer 기반의 multidomain sparse-view CT reconstruction 네트워크를 사용하여 sparse-view CT reconstruction을 위한 MDST를 제안했습니다.  
제안된 방법은 1) 컨볼루션 신경망과 Swin-transformer를 사용하여 sparse-view 프로젝션으로부터 CT 이미지를 재구성하는 multidomain sparse-view CT reconstruction network와 2) 재구성된 이미지와 실제 이미지 간의 일관성을 강화하는 multidomain consistency loss의 두 부분으로 구성됩니다.

Kim et al. [98]은 task-agnostic ViT를 사용하여 이미지 처리의 distributed learning을 위한 TAViT를 제안했습니다.  
이는 1) 입력 이미지의 표현을 학습하는 task-agnostic ViT와 2) 이미지 처리 작업을 수행하는 task-specific module의 두 부분으로 구성됩니다.  
Li 등은 parallel transformer network가 있는 이중 도메인을 사용하여 희소 뷰 CT 이미지 재구성을 위해 맞춤화된 DDPTransformer를 제안했습니다.  
DDPT 트랜스포머는 1) 이미지 도메인과 projection 도메인 모두에서 입력 이미지의 표현을 학습하는 dual-domain transformer와 2) 이미지 재구성 작업을 수행하는 parallel transformer network의 두 부분으로 구성됩니다.  
DDPTransformer에서 사이노그램이 도메인인 subnet과 이미지 도메인 subnet은 모두 여러 개의 DDPTransformer Block[99]으로 구성됩니다.  

위의 supervised denoising method 외에도 Noise2Noise[29]는 깨끗한 이미지를 지도학습할 필요 없이 이미지 노이즈 제거를 할 수 있는 선구적인 작업입니다.  
이는 point-wise $L_2$ loss라는 trivial한 속성에서 영감을 받았고, 이는 target pixel들이 대상 픽셀의 Expectation와 일치하는 무작위 값으로 대체되면 $L_2$ loss에 대한 Expectation가 변경되지 않음을 의미합니다 : 

![image](https://github.com/user-attachments/assets/e88337c6-9d10-4a8a-b379-e8bb53302d14)

즉, $p(I_{ND}|I_{LD})$를 conditional expectation이 동일한 임의의 분포로 대체해도 최적의 모델 $f_θ$은 변경되지 않습니다.  
그러면 다음과 같은 empirical risk를 최소화할 수 있습니다:

![image](https://github.com/user-attachments/assets/fe99e638-aea3-41cb-850d-2d0220f3eff5)

여기서 $\hat I_{LD}$ 와 $\hat I_{ND}$는 $E(\hat I_{LD}|\hat I_{ND}) = I_{ND}$가 되도록 $I_{ND}$를 조건으로 한 손상된 분포에서 추출됩니다.  
실제 구현에서 Noise2Noise는 노이즈가 없는 두 손상된 이미지 간의 $L_2$ loss을 최소화합니다.  
애플리케이션의 경우 Noise2Noise는 Gaussian additive noise, Poisson noise, and multiplicative Bernoulli noise을 더 잘 줄일 수 있습니다.  
또한 텍스트 제거에서도 잘 작동합니다.  
한 걸음 더 나아가 Hasan et al(105)는 세 개의 generator로 구성된 LDCT에 맞게 조정된 hybrid-collaborative Noise2Noise framework를 훈련했습니다.  
이 새롭고 효과적인 학습 패러다임은 다음과 같이 논의될 많은 self-supervised denoising method을 가공하게 합니다.

## Self/Un-Supervised Denoising
지도 학습은 높은 fidelity와 인상적인 세부 정보 보존으로 CT 이미지 노이즈 제거에 힘을 실어주었지만, 많은 양의 쌍을 이루는 NDCT 이미지가 필요하기 때문에 임상 시나리오에서 사용하기는 여전히 어렵습니다.  
최근 몇 년 동안 self-supervised learning은 특히 쌍을 이루는 NDCT 지도학습이 없는 LDCT의 경우 이미지 노이즈 제거에서 큰 관심을 받고 있습니다.  
이 섹션에서는 서로 다른 모델 아키텍처보다는 NDCT 실제 이미지가 없는 학습 패러다임에 더 중점을 둡니다.  
관련 방법은 대략 self-supervised and unsupervised의 두 가지 클래스로 분류할 수 있습니다.  
두 클래스는 개념적으로 다릅니다.  
Self-supervised 방법은 LDCT 이미지로만 노이즈 제거 모델을 학습하는 데 초점을 맞추고, unsupervised 방법은 쌍을 이루지 않은 NDCT 이미지로 노이즈 제거 모델을 학습하는 것을 목표로 합니다.  
### Self-Supervised Denoising: 
Self-Supervised Denoising 설정에서 중요한 단계는 LDCT의 이미지 또는 feature에서 pseudo-supervision을 탐색하는 것이며, 노이즈 제거 문제는 다음과 같이 공식화됩니다[33]:

![image](https://github.com/user-attachments/assets/7915caa2-5305-42e2-b869-6f2081b400a4)

여기서 σ(·)는 온라인 또는 오프라인 방식으로 설계할 수 있는 의사 감독 생성 프로세스를 나타냅니다.  
널리 사용되는 자체 감독 노이즈 제거 방법은 blind-spot method입니다.  
Noise2void(N2V)[30]는 그림 4에 나와 있는 첫 번째 blind-spot method입니다.  

![image](https://github.com/user-attachments/assets/cb6ac050-002e-48f6-8d1d-67b52f6adb96)

그림 4(d)의 기존 개별 픽셀 예측은 receptive field(원뿔 그림) 내의 모든 픽셀에 따라 달라집니다.  
그러나 노이즈 입력을 학습 목표로 삼는 Noise2Noise의 아이디어를 채택하면 모델은 노이즈 제거 능력을 잃고 이미지 자체를 학습할 수 있습니다.  
N2V의 핵심 아이디어는 픽셀의 receptive field를 학습하여 딥 네트워크가 이미지 자체를 학습하는 것을 방지한다는 것입니다.  
그림 4(b)와 (c)에서 중앙 픽셀을 제외한 모든 픽셀이 receptive field로 간주된다는 것을 알 수 있습니다.  
실제로 N2V는 먼저 계층화된 설정에서 무작위로 선택된 픽셀을 마스킹합니다.  
그런 다음 원래 노이즈 입력의 해당 픽셀을 학습 대상으로 취급합니다.  
Blind-spot 개념을 일반화하기 위해 noise2self(N2S)는 입력 자체에서 깨끗한 신호를 학습하기 위한 J-invariant theory을 제안했습니다[31].

#### J-invariant theory : $J$를 차원 ${1,...,m}$과 $j ∈ J$의 분할이라고 가정합니다. 함수 $f : R^m → R^m$은 $f_j(x)$가 $x_j$의 값에 의존하지 않으면 J-invariant이며, 각 $j ∈ J$에 대해 j-invariant이면 J-invariant입니다[31].

객관적인 self-supervised loss는 다음과 같이 정의됩니다:

![image](https://github.com/user-attachments/assets/f92dc241-bc3b-4d1a-8725-6eb5ea494a80)

$E[x|y] = y$ 일 때, $〈f (x) − y, x − y〉=0$ 라고 합니다. 

#### Proposition 1 [31]: $x$를 $y$, 즉 $E[x|y] = y$의 unbiased estimator라고 하고, 각 부분 집합 $j ∈ J$의 노이즈는 $y$에 조건부로 하는 complement $J^c$와 독립적이라고 가정합니다.  
그런 다음 f를 J-invariant이라고 가정하면 : 

![image](https://github.com/user-attachments/assets/33941762-ac45-416d-a43e-484239444402)

따라서 self-supervised loss은 다음과 같이 쓸 수 있습니다 : 

![image](https://github.com/user-attachments/assets/fbc386fc-cf95-47ff-b17a-e7a85d40979c)

입력 이미지 특정 픽셀과 마스킹을 통해 그 픽셀을 제외한 부분으로 예측한 그 부분의 픽셀값의 차이 : Loss

J-invariant theory을 기반으로 Lei 등[33]은 self-supervised learning 목표를 구성하는 데 다른 관점을 취하는 strided noise2neighbors(SN2N)을 제안했습니다.  
N2V와 N2S는 CNN의 입력 구조에 영향을 미치는 입력 이미지에서 대상 픽셀을 마스킹하는 데 중점을 둡니다.  
반면, SN2N은 N2V와 N2S에 사용된 원래 입력이 아닌 학습 대상을 수정할 것을 제안했습니다.  
SN2N의 학습 목표는 다음과 같습니다:

![image](https://github.com/user-attachments/assets/c2206987-02fa-4b2e-92d0-5ef7f20cf03b)

중심 부분을 제외한 원래 이미지 픽셀들로 예측한 중심 픽셀과 주변 픽셀들을 이용하여 Bilinear Intepolation 으로 변환한 중심 픽셀값의 차이

여기서 $A(·)$는 중심 픽셀의 이웃한 픽셀들에 대한 변환을 나타내며, 예를 들어 SN2N에서 사용되는 bilinear interpolation은 근사시킬 대상을 구성하기 위해 맞춤화됩니다.  
SN2N은 대상에 대한 계층화된 마스킹을 활성화하면서 입력 이미지를 변경하지 않습니다.  
그림 5에서 K-SVD[107]와 N2N은 부드러운(smooth) 결과를 나타내는 반면, N2V와 N2S는 이 문제를 해결한다는 것을 알 수 있습니다.  

![image](https://github.com/user-attachments/assets/8f070069-8540-4c2f-8e64-80c357a8bc49)

그러나 N2V와 N2S는 손상된 입력으로 인한 명백한 artifact를 보존합니다.

독립적인 노이즈에 대한 제한된 가정을 깨기 위해 Noise2Sim은 mild 조건에서 독립적인 노이즈와 상관관계에 있는 노이즈를 동시에 처리하도록 제안되었습니다[40].  
그림 6에서 Noise2Sim은 정성적 및 정량적(PSNR 및 SSIM) 측정 측면에서 일부 지도 학습 방법을 능가하기도 하며, 이는 CT 재구성을 위한 self-supervised learning의 잠재력을 보여줍니다.  

![image](https://github.com/user-attachments/assets/67fe6b85-cc81-4b3d-994e-0a74f933923e)

또한 그림 6에서 Noise2Sim은 다른 것보다 작은 구조와 상세한 텍스처를 더 잘 보존할 수 있습니다.  
N2Void가 상대적으로 더 나쁜 결과를 얻었음이 분명하며, 이는 blind spot의 무작위로 선정된 대상 때문이라고 추측합니다.  
또한 RED-CNN은 더 높은 RMSE를 보존하면서 더 높은 PSNR 값을 달성했다는 점에 주목합니다.

즉, RMSE는 어느 정도 oversmoothing 효과와 상관관계가 있을 수 있습니다.  
또한 MAP-NN과 N2Clean도 상대적으로 oversmoothing 효과를 나타내며, RMSE 값은 Noise2Sim보다 높습니다.  
Liu 등[106]은 변형 가능한 컨볼루션을 사용하여 코로나19 노이즈 제거를 수행하도록 DCDNet을 제안했으며, 그림 7에서 NLM이 코로나19 병변의 중요한 특성인 텍스처를 oversmooth한 것을 확인할 수 있습니다.  

![image](https://github.com/user-attachments/assets/2e6311ed-8249-4b56-b72f-93c564812f3e)

Neighbor2neighbor(NB2NB)는 학습 이미지 쌍을 생성하기 위한 random neighbor subsampler를 개발했으며, 이는 노이즈가 있는 이미지만으로 학습하는 기존 노이즈 제거기와 앞서 self-supervised denoising method을 능가합니다[108].

Corr2Self는 또한 self-supervised denoising method로, NDCT 이미지에 의존하지 않고 CNN을 훈련하기 위해 슬라이스 간 및 슬라이스 내 LDCT 데이터 간의 상관관계를 탐구했습니다.  
이 접근 방식은 참조한 데이터의 필요성을 제거하고 LDCT 이미지만으로 CNN 기반 노이즈 제거기를 훈련하려고 시도합니다.  
이 방법을 사용하면 실시간 미세 조정을 통해 성능을 더욱 향상시킬 수 있습니다[109].  
Meta-learning은 LDCT에 대한 의사 레이블을 생성하는 데 검증되었습니다.  
 
Zhu 등[110]은 LDCT 노이즈 제거 작업에서 고품질 CT 이미지를 얻기 위한 Meta-learning 전략인 SMU-Net을 제안했습니다.  
SMU-Net은 Teacher Network, pseudo-labels generation 및 Student Network를 포함한 세 가지 모듈로 구성됩니다.  
SMU-Net은 LDCT 이미지의 세부 구조를 보존하는 데 효과적입니다.  
위에서 언급한 모든 방법은 stratified pixel selection strategy을 적용했으며, 이 전략은 receptive field의 노이즈 픽셀들이 center target(N2V 및 N2S)을 예측하는 데 정확한지 여부, 노이즈 픽셀의 보간으로 인해 예상치 못한 smoothness 및 discontinuity(SN2N)이 발생하는지 여부를 고려하지 않았습니다.  
따라서 더 나은 마스킹 전략 또는 노이즈와 신호의 더 나은 분해는 이러한 방법을 더욱 향상시키는 데 도움이 될 것입니다.

### Unsupervised Denoising:
Wavelet transform은 LDCT 노이즈 제거를 지원하기 위해 적용되기도 합니다.  
Gu et al. [77]은 wavelet-assisted preprocessing를 사용하여 LDCT 이미지의 노이즈 성분을 분해하는 WAND를 제안했습니다.  
실험 결과는 기본 CycleGAN에 대해 제안된 방법의 성능을 크게 개선하는 데 WAND의 효과를 입증합니다.  
노이즈 분리를 위해 wavelet transform은 가장 많은 노이즈 정보를 포함하는 고주파 신호를 캡처하는 데 사용됩니다.  
심장병 환자 이미지의 일치된 LDCT와 NDCT는 실제로 얻을 수 없기 때문에 모델은 CycleGAN을 사용하여 unsupervised 방식으로 훈련되어 LDCT 및 NDCT 이미지에서 고주파 신호를 추출했습니다.  
그림 8에 표시된 것과 같이 WAND는 Noise2Noise 및 CycleGAN과 비슷한 성능을 발휘합니다.  

![image](https://github.com/user-attachments/assets/1a2e7241-425c-418c-8a7d-d9746c4fa895)

실제로 X선 CT 이미지의 모션 Artifact는 blurring, streaking, ghosting, 내부 구조의 변형 또는 거친 이미지 왜곡과 같은 다양한 형태로 나타납니다.  
Ko et al. [78]은 X선 CT 이미지의 고정되었거나 고정되지 않은 motion artifact를 줄이기 위해 attention module을 제안했습니다.  
이 attention module은 해당 feature importance에 따라 residual feature를 증폭하고 약화시켜 모델 용량을 개선하는 데 도움이 됩니다.  
모델은 1) rigid motion 또는 2) step-and-shoot fan-beam CT (FBCT) 하에서 고정되었거나 고정되지 않은 motion을 모두 사용하여 제안된 네 가지 벤치마크 데이터 세트에 대해 훈련 및 평가되었습니다.

전 외[93]는 volumetric LDCT denoising task을 위한 MM-Net이라는 unsupervised learning-based framework를 설명했습니다.  
MM-Net 훈련의 첫 번째 단계에서 초기 denoising network, 즉 multiscale attention U-Net(MSAU-Net)은 인접한 여러 개의 슬라이스 입력으로 노이즈가 억제된 중심 슬라이스를 예측하는 것을 목표로 self-supervised 방식으로 훈련됩니다.  
둘째, U-Net 기반 노이즈 제거기는 사전 훈련된 MSAU-Net을 기반으로 훈련되어 novel multipatch 및 multimask matching loss을 도입하여 이미지 품질을 개선합니다.  
MM-Net은 임상 및 동물 데이터의 다양한 영역에서 질적 및 정량적 측정 모두에서 기존의 unsupervised algorithm을 능가하는 성능을 보였습니다[93].  
또한 unsupervised algorithm은 NDCT 이미지로 훈련된 지도학습된 모델과 유사한 노이즈 제거 성능을 달성했습니다.

## Diffusion-Based Denoising
Diffusion model은 노이즈를 추가하여 데이터를 점진적으로 손상시킨 다음 이 과정을 반전시켜 샘플을 생성하는 일종의 확률적 생성 모델입니다[50].  
일반적인 DDPM [48], [112]은 1) 데이터를 무작위 노이즈로 추가하는 forward chain과 2) 무작위 노이즈를 다시 데이터로 변환하는 reverse chain의 두 가지 Markov chain을 사용합니다.  
Forward chain은 모든 데이터 분포를 간단한 이전 가우시안 분포로 변환하도록 설계된 반면, reverse Markov chain은 transition kernel(그림 9), 즉 심층 신경망의 매개변수를 학습하여 전자를 반전시킵니다. 

![image](https://github.com/user-attachments/assets/6f3658f9-d70c-48b2-897a-7067e9926eea)

새로운 데이터를 생성하기 위해 먼저 이전 분포에서 무작위 노이즈를 샘플링한 다음 reverse Markov chain을 통해 노이즈 샘플링을 수행합니다.  
구체적으로, 주어진 데이터 분포 $x_0 ~ q(x_0)$에 대해 $q(x_t|x_{t-1})$에 의해 구동되는 Markov chain의 forward process는 노이즈가 추가된 이미지 $x_1, x_2,...,x_T$를 생성하며, 그러면 분해된 joint probability distribution를 얻을 수 있습니다 :

![image](https://github.com/user-attachments/assets/480137bd-c720-4744-a80e-4b4829762406)

Transition kernel은 종종 Gaussian perturbation으로 선택됩니다 :

![image](https://github.com/user-attachments/assets/b503d0d3-8c9b-44fc-a120-d08b93e7bb21)

$β_t ∈ (0, 1)$ 는 확산 계수로, 하이퍼파라미터입니다.  

![image](https://github.com/user-attachments/assets/7b013ca5-1694-4a55-af18-ce8e0241b1e2)

![image](https://github.com/user-attachments/assets/6021f566-5ba9-41bb-ab56-e4c2176f41bb)

따라서 $x_0$이 주어지면 가우시안 변수 $\epsilon \approx N (0, I)$ 을 따르는 노이즈를 추가하여 $x_t$를 얻을 수 있습니다 :

![image](https://github.com/user-attachments/assets/72c463b8-3ed5-4113-ac94-4f5b118244b4)

Diffusion model로 얻은 이미지 생성 성능은 이미지 편집과 같은 많은 응용 분야에서 GAN 기반 방법의 성능을 능가했습니다[50].  
인기 있는 확산 모델에는 DDPM[48], score-based generative model(SGM)[67], [68](SBGM), 확률적 미분 방정식(ScoreSDE)[70], [71]이 포함됩니다.  
Score function에서 영감을 받은 Xie et al. [113]은 Poisson noise, Gaussian noise, Gamma noise, and Rayleigh noise를 처리할 수 있도록 노이즈 모델에 구애받지 않는 unsupervised image denoising 방법을 개발했습니다.  
또한 DDPM[114]을 기반으로 다양한 유형의 노이즈에 대한 생성 이미지 노이즈 제거 접근 방식을 제안했습니다.

[111]에서는 의료 영상의 선형 역 문제를 해결하기 위해 일반적인 SGM을 제안했습니다.  
이 방법은 순전히 생성적인 방법으로, 훈련 중에 물리적 측정 프로세스에 대한 사전 지식이 필요하지 않으며 모델 재교육 없이 추론 단계에서 다양한 영상 프로세스에 빠르게 적응할 수 있습니다.  
저자들은 최근 도입된 score-based generative model을 활용하여 역 문제를 해결하기 위해 fully unsupervised method을 제안했습니다.  
특히, 먼저 의료 이미지에 대해 SGM을 훈련하여 이전 분포를 캡처합니다.  
그런 다음 훈련된 모델을 사용하여 관찰된 데이터의 likelihood을 최적화하여 역 문제를 해결합니다.  
반면, Xie와 Lie[115]는 언더샘플링된 의료 이미지 재구성을 위한 새롭고 통합된 방법인 measurement-conditioned DDPM(MC-DDPM)을 제안했습니다.  
이전 연구와 달리 MC-DDPM은 MRI 재구성에서 k-space간으로 정의되고 undersampling mask에 조건화됩니다.

