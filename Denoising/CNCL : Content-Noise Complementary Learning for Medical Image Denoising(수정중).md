# Content-Noise Complementary Learning for Medical Image Denoising

# Two Deep predictors : content-noise complementary learning(CNCL) : U-Net, DnCNN, and SRDenseNet with GAN

# Abs
의료 영상 노이즈 제거는 큰 과제에 직면해 있지만 수요가 많습니다.  
이미지 영역의 의료 영상 노이즈 제거에는 혁신적인 딥 러닝 전략이 필요합니다.  
이 연구에서는 두 개의 딥 러닝 예측기를 사용하여 이미지 데이터 세트의 각 콘텐츠와 노이즈를 보완적으로 학습하는 간단하면서도 효과적인 전략인 content-noise complementary learning(CNCL) 전략을 제안합니다.  
CNCL 전략을 기반으로 한 의료 영상 노이즈 제거 파이프라인이 제시되며, 다양한 대표 네트워크(U-Net, DnCNN, and SRDenseNet)가 예측기로 조사되는 generative adversarial network(GAN)로 구현됩니다.  
이러한 구현된 모델의 성능은 CT, MR 및 PET를 포함한 의료 영상 데이터 세트에서 검증되었습니다.  
결과는 이 전략이 시각적 품질 및 정량적 메트릭 측면에서 state-of-the-art denoising algorithm을 능가하며, 이 전략은 강력한 일반화 기능을 입증한다는 것을 보여줍니다.  
이러한 결과는 이 간단하지만 효과적인 전략이 향후 임상적 영향을 미칠 수 있는 의료 영상 노이즈 제거 작업에 대한 유망한 잠재력을 보여준다는 것을 입증합니다.  

# Introduction
컴퓨터 비전의 관행에는 손상된 이미지에서 노이즈를 제거하고 실제 이미지를 복원하는 것을 목표로 하는 이미지 노이즈 제거가 자주 포함됩니다.  
의료 영상 처리에서 노이즈 제거는 노이즈가 질병 진단을 방해하고 후속 임상 의사 결정에 영향을 미칠 수 있기 때문에 특히 중요합니다[1].  
Computed tomography(CT), magnetic resonance imaging(MRI) 및 positron emission tomography(PET)은 임상 진단에서 일반적인 세 가지 의료 영상 양식입니다.  
CT와 MRI는 고해상도에서 구조적인 정보를 제공하는 반면 PET는 대사 및 기능 정보를 제공하는 분자 영상 양식입니다[2].  
이러한 양식에 대한 노이즈 제거 알고리즘은 스캔 시간, 방사선 강도 및 이미지 품질 간의 상호관계를 돌파하는 것을 목표로 합니다.  
CT의 경우 지난 10년 동안 일반적으로 튜브 전류를 줄이거나 X선 튜브의 노출 시간을 단축하여 이온화 방사선 노출을 줄이기 위한 저선량 검사가 진행되는 추세를 보였습니다[3].  
CT 및 PET와 달리 MRI 검사에는 방사선 위험이 없는 반면, 좁은 공간과 MRI의 긴 획득 시간은 특히 폐쇄공포증 환자의 불안을 유발할 수 있습니다[4].  
환자의 불편을 완화하기 위해 빠른 MRI 획득을 달성하기 위해 k-space를 under-sampling하려는 시도가 있습니다[5].  
PET 스캔에서도 적은 수의 일치화로 완벽한 이미지를 재구성하려는 욕구가 있습니다[6].  
위에서 언급한 이러한 시도에도 불구하고, 재구성된 이미지는 종종 손상되어 다양한 노이즈 또는 아티팩트를 유발합니다.  
CT, MR 및 PET의 이미지 품질을 향상시키기 위한 다양한 방법이 제안되었습니다.  
이러한 이미지 복원 방법은 재구성 전 원시 데이터 전처리, 재구성 중 또는 재구성 후 후 처리를 포함하여 다양한 단계에서 구현됩니다.  
전자의 두 가지는 종종 공급업체별로 다른 원시 데이터 수집에 의존하지만, 이미지 후처리는 손상된 이미지에서 직접 작동하고 기존 임상 절차에 쉽게 통합될 수 있습니다.  
따라서 많은 연구자들이 이미지 영역에서 의료 이미지 노이즈 제거 문제를 해결합니다.  
의료 이미지 노이즈 제거 분야에서 전통적인 이미지 처리 알고리즘에는 non-local means(NLM)[7], block-matching 3D(BM3D)[8], diffusion filters[9] 등이 포함됩니다.  
이러한 알고리즘은 노이즈를 다양한 정도로 줄일 수 있지만 노이즈 제거된 이미지에서 over-smoothness하고 residual error가 발생합니다.  

최근 딥 러닝은 의료 영상 작업에서 segmentation[10], material decomposition[11], [12] 및 노이즈 제거와 같은 우수한 기능을 보여주었습니다.  
low-dose CT denoising를 위해 Chen 등은 routine-dose CT values을 추정하기 위한 residual encoder-decoder convolutional neural network(RED-CNN)을 제안했습니다[13].  
Fan 등은 low-dose CT denoising를 위한 quadratic autoencoder(Q-AE) 네트워크를 설계했습니다[14].  
또한 일련의 generative adversarial network(GAN)가 저선량 CT 노이즈 제거 작업[15], [16]에 적용되었으며, 그 중 일부는 perceptual loss[17], [18]을 도입했습니다.  
MRI의 경우 [19]와 [20] 모두 왜곡된 MRI 이미지에서 aliasing artifacts와 streaking artifacts를 제거하기 위해 U-Net을 채택했습니다.  
현 등은 U-Net 노이즈 제거와 k-space correction을 공동으로 사용하여 under-sampled MR images를 복원했습니다[21];  
Jiang 등은 denoising convolutional neural network(DnCNN)을 활용하여 Rician noise에 의해 손상된 MR 이미지를 복구했습니다[22];  
Kidoh 등은 뇌 MR 이미지의 노이즈를 제거하기 위해 shrinkage convolutional neural network(SCNN)과 딥 러닝 기반 재구성(DLR) 네트워크를 설계했습니다[23].  
low-dose PET denoising 분야에서 Xiang 등은 auto-context CNN model을 사용하여 25% 선량 PET 및 MRI 이미지 쌍에서 전체 선량 PET 이미지를 추정했습니다[24];  
Sano 등은 저선량 PET 이미지에서 전체 선량 PET 이미지를 재구성하기 위해 U-Net을 수정했습니다[25];  
Wang 등은 저선량 PET 이미지에서 고품질 전체 선량 PET 이미지를 추정하기 위해 conditional generative adversarial 네트워크(cGAN)를 기반으로 한 접근 방식을 제시했습니다[26];  
Kim 등은 PET 이미지 노이즈 제거를 수행하기 위해 DnCNN을 최적화했습니다[27];  
Lu 등은 종양학 PET 데이터에서 convolutional autoencoder(CAE) 네트워크, U-Net 및 GAN의 노이즈 제거 성능을 종합적으로 조사했습니다[28];  
Zhou 등은 low-dose gated PET에 대한 노이즈 제거 및 모션 추정을 동시에 수행하기 위해 통합 모션 보정 및 denoising adversarial network(DPET)를 제안했습니다[29];  
[30] 및 [31] 모두 PET 노이즈 제거를 위해 이전에 심층 이미지에 초점을 맞춘 통합 모션 보정 및 노이즈 제거 작업이 있었습니다.  
또한 cycle consistent generative adversarial networks(Cycle GAN)[32]–[34] 및 Wasserstein generative adversarial network(WGAN)[35]를 기반으로 한 일부 저선량 PET 노이즈 제거 작업이 있었습니다.  
위에서 언급한 이러한 방법은 의료 이미지 노이즈 제거 작업에 대한 능력을 입증했으며, 대부분 네트워크 구조를 최적화하거나 case-dependent loss functions를 설계하는 데 중점을 두었습니다.  
또한 content images(즉, full-dose or full-sampled images)를 직접 학습하거나 [15]–[18], [26]–[29], [33], [35], [36] 노이즈 이미지를 학습하여 [13], [14], [19], [20], [22], [25], [28], [32], [34], [37] 차감하는 방식을 이용하여 콘텐츠 이미지를 역으로 얻을 수 있습니다.  
content learning과 noise learning 모두 고유한 장점이 있습니다:  
noise learning은 성능 저하를 방지하고 보다 구조적이고 대조적인 세부 정보를 보존할 수 있는 반면 content learning은 보다 안정적인 노이즈 제거 성능을 보여줄 수 있습니다[28].  

이전에 content and the noise의 encoder-decoder network인 WhiteNNer[38]를 사용한 초점 공유 레이저 내시경 이미지 노이즈 제거에 적용하려는 시도가 있었습니다.  
그러나 WhiteNNer는 잡음이 Gaussian distribution를 따른다는 가정 하에 이전에 노이즈 훈련을 위한 loss regularization로만 사용했습니다.  
유사하게, Liao 등은 CT metal artifact reduction problem를 해결하기 위해 artifact disentanglement network(ADN)를 제안했는데, 이는 네트워크 훈련에서 콘텐츠뿐만 아니라 metal artifact도 학습했습니다[39].  
ADN에서 학습된 metal artifact coding은 새로운 metal artifact로 손상된 이미지를 생성하는 데 사용되어 cycle-consistent loss을 후속적으로 계산할 수 있게 되었습니다.  
즉, 테스트 단계에서 WhiteNNer와 ADN은 모두 noise feature 또는 metal artifact coding을 폐기했는데, 이는 WhiteNNer의 학습된 노이즈와 ADN에서 학습된 metal artifact coding이 최종 예측 콘텐츠와 직접적인 관련이 없음을 의미합니다.  
따라서 의료 이미지 노이즈 제거 작업을 위해 최종 콘텐츠 예측을 위해 이전 콘텐츠와 이전 노이즈를 더 잘 활용하고 두 학습 패러다임의 강점을 통합할 수 있는 새로운 전략을 모색해야 할 필요성이 제기되고 있습니다.  

이 작업에서는 이미지 영역에서 의료 이미지 노이즈 제거를 위한 content-noise complementary learning(CNCL) 전략을 제안합니다.  
콘텐츠 또는 노이즈를 학습하는 기존 방법과 달리 제안된 노이즈 제거 전략은 두 predictor가 동시에 보완적인 방식으로 콘텐츠와 노이즈를 학습하고 콘텐츠 predictor와 노이즈 predictor 모두의 추출된 feature에서 최종 콘텐츠를 재구성합니다.  
제안된 CNCL 전략을 기반으로 predictor 가 평행하게 놓인 방식으로 구성된 심층 CNN인 범용 의료 이미지 노이즈 제거 파이프라인을 제시합니다.  
저희는 이 파이프라인을 GAN 프레임워크를 기반으로 구현하여 의료 이미지 노이즈 제거를 end-to-end 작업으로 취급합니다.  
저희는 제안된 전략을 3개의 의료 이미지 데이터 세트(CT, PET, MR)와 1개의 자연 이미지 데이터 세트(스마트폰 이미지 노이즈 제거 데이터 세트, SIDD)에 대해 검증합니다[40].  
결과는 제안된 전략이 질적으로나 정량적으로 최첨단 방법을 능가한다는 것을 보여줍니다.  

이 논문을 요약하면 다음과 같습니다: 
1) 우리는 콘텐츠와 노이즈를 보완적으로 학습하여 두 패러다임의 강점을 충분히 통합하는 의료 이미지 노이즈 제거를 위한 간단하면서도 효과적인 CNCL 전략을 제안합니다. 
2) CNCL 노이즈 제거 파이프라인이 제시되고 GAN 프레임워크를 기반으로 구현됩니다.  
또한 세 가지 대표적인 고전 CNN 모델(즉, U-Net[10], DnCNN[41], SRDenseNet[42])을 generator 부분의 predictor로 하고 노이즈 제거 성능을 종합적으로 조사합니다.  
3) 제안된 CNCL 전략은 여러 노이즈 제거 데이터 세트에서 검증됩니다.  결과는 우리 전략이 다양한 노이즈 유형을 처리할 수 있음을 보여줍니다.  
전략의 일반화 기능과 규제도 포괄적으로 조사됩니다.

# METHODS AND MATERIALS
## Content-Noise Complementary Learning
Icorrupted는 low-dose 또는 under-sampling으로 인한 노이즈 손상 이미지이고, Icontent는 해당 full-dose 또는 full-sampling image라고 가정합니다.  
딥러닝 기반 의료 이미지 노이즈 제거 분야에서 Icorrupted는 콘텐츠 Icontent와 노이즈 Inoise[13], [15]의 합으로 간주될 수 있습니다.  

- Icorrupted = Icontent + Inoise.

기존의 딥러닝 기반 이미지 노이즈 제거 방법은 두 가지로 나눌 수 있습니다.  
하나는 손상된 이미지를 콘텐츠 이미지에 직접 매핑하는 기능을 사용합니다:

- I'content = pc(Icorrupted)

I'content : predicted content  
pc : content predictor   

다른 하나는 먼저 노이즈를 예측하고 학습된 노이즈에서 입력된 손상된 이미지를 빼서 최종 콘텐츠 이미지를 획득합니다(즉, residual mapping):

- I'content = Icorrupted − I'noise = Icorrupted − pn(Icorrupted)

I'noise : predicted noise  
pn : noise predictor  

콘텐츠 학습과 노이즈 학습 모두 고유한 장점이 있습니다 : 콘텐츠 학습은 보다 안정적인 노이즈 제거 성능을 보여주고[28], 노이즈 학습은 성능 저하 방지 및 구조 보존에 도움이 됩니다[13].  

두 패러다임의 강점을 결합하기 위해 pc와 pn을 상호 보완적으로 공동 활용하는 CNCL 전략을 제안합니다.  
CNCL은 이미지 노이즈 제거를 입력과 출력이 각각 Icorrupted와 I'content인 end-to-end 작업으로 취급합니다.  
CNCL 노이즈 제거의 모든 과정은 딥러닝 함수 G로 표현할 수 있습니다:  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.14.51.png)

I'content1 : output of pc  
I'noise : output of pn  
I'content2 : difference between Icorrupted and I'noise  
f : fusion mechanism  
먼저 Icorrupted를 pc와 pn의 입력으로 사용하여 각각 예측한 콘텐츠 이미지 Icontent1과 예측한 노이즈 Inoise를 얻습니다.  
그런 다음 Inoise에서 Icorrupted를 빼서 다른 예측 콘텐츠 이미지 Icontent2를 계산합니다.    
그 후 융합 메커니즘 f를 사용하여 Icontent1과 Icontent2를 결합하여 최종 예측 콘텐츠 Icontent2를 얻습니다.  

CNCL의 학습 프로세스는 다음과 같은 최적화 문제로 공식화할 수 있습니다: 

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.22.11.png)

CNCL 전략을 기반으로 그림 1과 같은 의료 이미지 노이즈 제거 파이프라인을 제안합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.25.41.png)

그림에서 {Icorrupted, Icontent} 쌍이 주어지면 (1)에 따라 노이즈 이미지 Inoise를 계산할 수 있습니다.  
그 결과, 하나의 이미지 쌍 {Icorrupted, Icontent}을 {Icorrupted, Icontent} 및 {Icorrupted, Inoise}으로 확장할 수 있습니다.  
그런 다음 {Icorrupted, Icontent}을 pc 학습에 적용하고 {Icorrupted, Inoise}를 사용하여 pn을 학습합니다.

## Implementation Based on GAN
최적화 문제 (5)의 딥러닝 함수 G를 더 잘 최적화하기 위해 제안된 CNCL 전략을 GAN 프레임워크를 기반으로 구현했습니다.  
G는 GAN 프레임워크의 generator 부분에 해당합니다.  
L1 또는 L2와 같은 기존의 loss function만 사용하면 이미지가 지나치게 평활화되고 흐릿하여 일부 텍스처 정보가 손실될 수 있다고 여러 번 보고되었습니다[43].  
discriminator의 도움으로 노이즈가 제거된 이미지에 보존된 세부 정보는 이전 작업 [15], [17], [18]에서 눈에 띄게 개선되었습니다. 

1) 네트워크 구조:  
그림 2와 같이 대부분의 GAN과 마찬가지로 제안된 GAN 프레임워크는 생성기와 판별기의 두 부분으로 구성됩니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.41.08.png)

생성기에는 평행한 방식으로 구성된 두 개의 예측기가 있습니다.  
예측기는 동일한 네트워크 아키텍처를 가지며 CNN으로 구현할 수 있습니다.  
융합 메커니즘은 concatenation operation에 이어 1×1 컨볼루션 연산으로 구성됩니다.  
융합 메커니즘은 weighted averaging operation으로 기능하며 컨볼루션이 존재하기 때문에 weight를 학습할 수 있습니다.

유명한 판별기인 PatchGAN은 패치 규모의 구조 불일치에 패널티를 주어 texture or style loss의 한 형태로 기능합니다[43].  
따라서 제안된 GAN 프레임워크에 PatchGAN을 discriminator로 통합합니다.  
PatchGAN의 입력은 실제 이미지 쌍 {Icontent, Icorrupted, Inoise} 또는 가짜 이미지 쌍 {Icontent, Icorrupted, Inoise}입니다.  
생성기는 판별기를 속이기 위해 가짜 데이터를 생성하려고 하는 반면, PatchGAN은 가짜 데이터와 실제 데이터를 구별하기 위해 노력합니다.  

2) Loss Function:
[43]의 추천에 따라 PatchGAN loss와 L1 loss를 전체 Loss로 공동 사용합니다.  
loss LCNCL−GAN은 다음과 같이 표현할 수 있습니다:

- LCNCL−GAN = LGAN(G, D) + φ LL1(G)

LGAN(G, D) : GAN loss contributed by PatchGAN  
LL1(G) : L1 loss  
φ : loss weight of LL1(G)

- LGAN(G, D) = Ex,y[log(D(x, y))]+Ex,y[log(1−D(x, G(x)))]

E[·] : expectation operator
G : generator  
D : discriminator  
x : corrupted image Icorrupted  
G(x) : predicted noise and the predicted content (i.e., I'noise and I'content)  
y : real noise and the real content (i.e., Inoise and Icontent)  
D는 이 목표를 최대화하려고 노력하고, G는 이를 최소화하려고 노력합니다.  

L1 loss는 다음과 같이 표현됩니다.  

- LL1(G) = LL1−content + λLL1−noise

LL1−content : mean absolute error between the predicted content I'content and the real content Icontent  
LL1−noise : mean absolute error between the predicted noise I'noise and real noise Inoise  
λ : loss weight of LL1−noise  

## Experimental Setup

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.57.41.png)

1) Predictor: 제안된 CNCL 전략의 효과를 입증하기 위해 GAN 기반 프레임워크에서 세 가지 종류의 Predictor를 조사했습니다.  
이 모델은 U-Net[10], DnCNN[41], SRDenseNet[42] 등 세 가지 대표적이고 널리 사용되는 CNN 모델이었습니다.  
그림 3a는 이 작업에 사용된 U-Net의 구조를 보여줍니다.  
기존 버전 [10]과 비교하여 U-Net을 두 가지 측면에서 수정했습니다.  
컨볼루션 전후에 동일한 feature map size를 보장하기 위해 padding을 사용하고 regularization를 제공하기 위해 batch normalization [44]를 추가했습니다.  
이 작업에 사용된 DnCNN 아키텍처는 그림 3b에 나와 있습니다.  
[41]의 초기 DnCNN과 달리 이 작업에 사용된 DnCNN은 residual learning을 버리고 ReLU를 Leaky ReLU로 대체했습니다.  
그림 3c는 SRDenSeNet의 구조를 보여주며, 그림 3d는 dense block을 자세히 보여줍니다.  
[42]의 SRDenSeNet과 비교하여 deconvolution, bottleneck, and reconstruction layers를 4개의 컨볼루션 세트, batch normalization 및 Leaky ReLU layers로 교체하여 output image size가 input size와 동일한지 확인하고, dense block의 수와 컨볼루션 레이어의 수를 모두 8개에서 6개로 줄였습니다.

우리는 제안된 CNCL 전략의 GAN 기반 구현을 CNCL 기반 네트워크라고 이름 지었습니다.  
Predictor가 두 개의 U-Net 인 경우 CNCL 기반 네트워크는 CNCL-U-Net으로 약칭되었습니다.  
마찬가지로 CNCL-DnCNN과 CNCL-SRDenseNet은 각각 예측 변수가 두 개의 DnCNN과 두 개의 SRDenseNet인 CNCL 기반 네트워크를 나타냅니다.  

2) 데이터 준비: 제안된 노이즈 제거 전략을 더 잘 평가하기 위해 CT, PET 및 MRI의 세 가지 다른 영상 양식에 걸쳐 세 가지 의료 이미지 데이터 세트에 대해 CNCL-U-Net, CNCL-DnCNN 및 CNCL-SRDenseNet로 조사했습니다.  
이러한 양식은 임상 진단에 널리 사용됩니다.  
손상된 CT 및 PET 이미지는 저선량에서 획득된 반면 손상된 MR 이미지는 k-space을 under-sampling하여 생성되었습니다.  
그림 4는 세 가지 슬라이스 예제를 보여주며, 서로 다른 이미지 양식의 노이즈 유형이 크게 달라지는 것을 알 수 있습니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.00.41.png)

a) CT dataset: 제안된 노이즈 제거 전략은 먼저 Mayo Clinic에서 2016년 NIH-AAPM-Mayo Clinic 저선량 CT 그랜드 챌린지1에 승인된 공개적으로 사용 가능한 저선량 CT 데이터 세트에서 검증되었습니다.  
이 데이터 세트에는 10명의 익명 환자로부터 10개의 전선량 복부 CT 2D 슬라이스와 해당하는 시뮬레이션된 25% 선량 CT 2D 슬라이스가 있습니다.  
전선량 데이터는 120kV 및 200개의 유효 mAs에서 획득되었고, 25% 선량 데이터는 프로젝션 데이터에 Poisson noise를 추가하여 생성되었습니다.  
데이터 세트에는 2,378개의 2D 슬라이스 쌍(512 × 512 픽셀)이 포함되었습니다.  
이전 작업 [14], [17]의 데이터 분할을 참조하여 훈련을 위해 7명의 환자 스캔에서 1,709개의 CT 슬라이스 쌍을 선택했으며, 3명의 환자 스캔에서 669개의 CT 슬라이스 쌍을 테스트용으로 제외했습니다.  

b) PET datset: 통합 3-Tesla PET/MR 스캐너(uPMR 790, United Imaging Healthcare, China)를 사용하여 의심스러운 신경 내분비 종양 환자 12명을 대상으로 PET 스캔을 수행했습니다.  
프로토콜은 베이징 암 병원 기관 심사 위원회(2020KT15호)의 승인을 받았습니다.  
주입된 18F 라벨 추적자의 양은 3.7MBq/kg이었고, PET 스캔은 주입 후 30분 후에 수행되었습니다.  
획득한 각 PET 스캔에 대해 원시 데이터(리스트 모드)를 후처리한 후 두 가지 재구성이 수행되었습니다.  
하나는 획득한 모든 이벤트를 포함하고 다른 하나는 전체 이벤트의 10%를 활용합니다.  
두 재구성 모두에 대한 재구성 행렬은 144 × 144로 설정되었으며, 4,271개의 2D 슬라이스 쌍의 몸통 PET 슬라이스를 얻었습니다.  
이 쌍은 6배 검증에서 훈련 또는 테스트 데이터로 할당되었으며, 각 라운드에서 10명의 환자의 슬라이스 쌍은 훈련 세트에, 나머지 2명의 환자의 슬라이스 쌍은 테스트 세트에 할당되었습니다.

c) MR 데이터 세트: MR 데이터 세트에는 PET 데이터 세트와 동일한 임상 시험에서 10명의 환자의 복부 2D 슬라이스가 포함되어 있습니다.  
모든 MR 데이터는 2점 딕슨 시퀀스인 3D WFI(water-fat imaging) 기술을 사용하여 PET 데이터 수집과 동일한 PET/MR 스캐너에 의해 수행되었습니다.  
각 축 슬라이스에 대해 물, 지방, in-phase 및 opposed-phase 이미지를 포함하여 4가지 유형의 이미지가 생성되었습니다.  
쌍을 이룬 MR 슬라이스를 얻기 위해 먼저 스캐너에서 콘텐츠 이미지로 전체 샘플링된 MR 슬라이스를 수집했습니다.  
그런 다음 슬라이스는 3차원 푸리에 변환을 사용하여 k-공간으로 변환되었습니다.  
4배 의사 무작위 언더샘플링 방식이 적용되어 언더샘플링된 k-공간을 얻었습니다.  
마지막으로 3차원 역고속 푸리에 변환이 언더샘플링된 MR 슬라이스를 검색하기 위해 배포되었습니다.  
총 6,560개의 MR 2D 슬라이스 쌍(544 × 384 픽셀)이 있습니다. MR 데이터는 5배 교차 검증 전략에 의해 조사되었습니다.  
즉, 8명의 환자가 훈련 단계에 참여했을 때 다른 2명의 환자가 테스트에 사용되었습니다.  

3) 훈련 세부 정보: 훈련 단계에서 모든 네트워크는 NVIDIA GeForce RTX 2080Ti GPU를 사용하여 PyTorch 프레임워크[45]에 구현되었습니다.  
Adam은 모든 네트워크의 훈련을 위한 최적화기 역할을 했습니다.  
Momentum 1 and momentum 2는 각각 0.5와 0.999로 설정되었습니다.  
Learning rate은 0.0002였으며 batch size는 2였습니다.  
L1 loss(8)과 LCNCL−GAN(6)의 가중치 λ과 φ은 각각 1과 100으로 설정되었습니다.  
φ 설정은 PatchGAN 논문[43]에서 도출되었으며, λ은 최고의 노이즈 제거 성능을 달성하기 위해 다양한 실험을 통해 결정되었습니다.  
훈련 전 컨볼루션과 batch normalization의 초기 가중치는 각각 N(0, 0.02)과 N(1.0, 0.02)의 정규 분포를 따르는 난수였으며, 초기 편향은 모두 0으로 설정되었습니다.  
모든 네트워크의 훈련 epoch는 300이었고, 훈련 시간은 표 I에 나열되어 있습니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.18.38.png)

참조 방법을 포함한 다양한 알고리즘에 대한 자세한 파라미터 설정은 첨부 자료의 표 SI에 나와 있습니다.  

4) 정량적 평가: CNCL 기반 네트워크의 성능을 정량적으로 평가하기 위해 구조적 유사성 지수(SSIM)[46], 평균 제곱근 오차(RMSE)[47], 잡음비 대비 피크 신호(PSNR)[48]와 같은 지표가 채택되었습니다:

# Results
## Ablation Studies
CNCL 전략을 통해 콘텐츠와 노이즈를 상호 보완적으로 학습하는 효과를 검증하기 위해 NIH-AAPM-Mayo CT 데이터 세트에 대한 절제 연구를 수행했습니다.  
저희는 CNCL 기반 네트워크를 하나의 예측기만 있는 해당 기준 네트워크와 비교했습니다.  
구체적으로 CNCL-U-Net을 콘텐츠만 학습한 단일 U-Net 예측기와 비교하고 CNCL-U-Net을 노이즈만 학습한 단일 U-Net 예측기와 비교했습니다.  
유사하게 CNCL-DnCNN은 콘텐츠만 학습한 단일 DnCNN 예측기와 노이즈만 학습한 단일 DnCNN 예측기와 각각 비교했습니다.  
CNCL-SRDenSeNet은 콘텐츠만 학습한 단일 SRDenSeNet 예측기와 노이즈만 학습한 단일 SRDenSeNet 예측기와 각각 비교했습니다.  
공정한 비교를 위해 위에서 언급한 네트워크의 모든 교육은 생성기 부분을 제외하고 그림 2에서 동일한 GAN 프레임워크 하에 있었습니다. 

표 II는 절제 연구의 정량적 결과를 열거합니다.  
결과는 CNCL의 효과를 보여줍니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.30.52.png)

일반적으로 U-Net은 DnCNN 및 SRDenseNet보다 제안된 CNCL 전략을 통해 더 큰 개선을 경험했습니다.  
Fig. 5는 절제 연구의 대표적인 슬라이스를 제시합니다. 

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.37.54.png)
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.38.25.png)
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.38.47.png)

내용 또는 노이즈를 학습하기 위해 단일 predictor를 사용했을 때 노이즈가 제거된 출력 이미지가 CNCL 기반 네트워크보다 더 많은 노이즈를 포함했음을 알 수 있습니다.  

CNCL은 필연적으로 네트워크 매개 변수를 증가시켰습니다.  
CNCL을 통해 두 개의 predictor를 활용했는데, 이는 매개 변수가 두 배로 증가했음을 의미합니다.  
CNCL 전략의 효과를 추가로 검증하기 위해 CNCL-U-Net을 단일 wide-U-Net predictor와 비교했습니다.  
wide-U-Net의 컨볼루션 레이어는 더 많은 feature channel을 추가하여 넓혔습니다.  
그 결과 wide-U-Net과 CNCL-U-Net의 매개 변수 번호는 동일했습니다.  
공정한 비교를 위해 wide-U-Net의 훈련은 그림 2의 GAN 프레임워크와 동일하게 했습니다.  
표 II에 제시된 바와 같이 wide-U-Net은 콘텐츠 학습과 노이즈 학습 모두에서 CNCL-U-Net보다 크게 좋지 않았었습니다.  

네트워크 구조에 대한 절제 연구 외에도 손실 함수에 대한 절제 연구도 수행했습니다.  
PatchGAN[43]을 기반으로 L1 loss값(8)과 같이 전체 손실 함수에 노이즈의 새로운 L1 regularization 항을 추가했습니다.  
추가된 노이즈의 L1 regularization 항의 효과를 검증하기 위해 서로 다른 손실 함수를 가진 두 개의 CNCL-U-Net을 비교했습니다.  
하나는 GAN loss, 콘텐츠의 L1 손실 및 노이즈의 L1 손실을 전체 loss로 결합한 것이고, 다른 하나는 GAN 손실과 콘텐츠의 L1 손실만 채택했습니다.  
NIH-AAPM-Mayo CT 데이터 세트에 대한 두 개의 CNCL-U-Net의 정량적 결과는 첨부 자료의 표 SII에 나와 있습니다.  
결과는 추가된 노이즈 정규화 항의 효과를 입증했습니다.  
GAN 손실, 콘텐츠의 L1 손실, 노이즈의 L1 손실을 전체 손실로 결합하면 콘텐츠 예측기와 노이즈 예측기를 더 잘 감독하고 훈련하여 노이즈 제거 성능이 향상될 수 있습니다.  

## Comparison With Reference Methods


## Comparison Among CNCL-Based Networks
위의 세 가지 노이즈 제거 작업에서 제안된 모든 CNCL 기반 네트워크는 우수한 노이즈 제거 기능을 검증했습니다.  
또한 CNCL 기반 네트워크에서 생성된 이미지가 세부적으로 약간 다르다는 점도 눈에 띕니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.43.40.png)

저선량 CT 노이즈 제거의 경우 CNCL-DnCNN과 CNCL-SRDenseNet이 더 나은 RMSE와 PSNR을 얻었지만(표 III), 그림 6에서 빨간색 화살표로 지적된 혈관과 같은 일부 구조적 객체를 흐리게 만들었습니다. 

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-18%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.44.07.png)

CNCL-U-Net은 손상된 이미지에서 더 많은 구조적 정보를 복원하여 혈관과 병변의 경계를 더 명확하게 만들 수 있습니다.  
저선량 PET 노이즈 제거 작업에서 작은 병변의 감지 가능성은 중요한 과제 중 하나입니다.  
모든 CNCL 기반 네트워크는 작은 병변을 다양한 정도로 보존할 수 있었고, CNCL-DnCNN과 CNCL-SRDenseNet은 병변 영역에서 더 높은 강도를 유지했습니다(부 자료의 그림 S2).  
표 V에 따르면 under-sampling MR 노이즈 제거의 경우 CNCL-DnCNN이 최고의 정량적 결과를 달성했습니다.  
이미지 비전 측면에서 모두 MR 노이즈 제거의 만족스러운 시각적 품질을 생성했습니다.  
일반적으로 predictor 선택은 작업 기반이며 원하는 애플리케이션에 따라 다릅니다.

# Discussion

# CONCLUSION
이 연구에서는 두 개의 predictor를 사용하여 이미지 데이터 세트의 내용과 노이즈를 보완적으로 학습하는 의료 이미지 노이즈 제거를 위한 간단하면서도 효과적인 CNCL 전략을 제안했습니다.  
구현된 CNCL 기반 네트워크는 세 가지 의료 이미지 데이터 세트(CT, MRI, PET)에 걸쳐 검증되었습니다.  
그 결과 CNCL 기반 네트워크가 최첨단 방법보다 질적, 양적으로 우수한 성능을 발휘하는 것으로 나타났습니다.  
또한 CNCL 전략의 보편성과 일반화 능력도 조사되었습니다.  
요약하면, 제안된 CNCL 전략은 의료 이미지 노이즈 제거 능력을 입증하여 임상 응용 분야에 대한 특정 잠재력을 보여주었습니다.

