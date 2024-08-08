# ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks

# ESRGAN

# Abs
초해상도 생성 적대적 네트워크(SR-GAN)[1]는 단일 이미지 초해상도를 얻기 위해 현실적인 텍스처를 생성할 수 있는 중요한 작업입니다.  
그러나 환각에 빠진 세부 사항은 종종 불쾌한 결함(artifact)를 동반합니다.  
시각적 품질을 더욱 향상시키기 위해 저희는 SRGAN의 세 가지 주요 구성 요소인 네트워크 아키텍처, 적대적 손실 및 지각 손실을 철저히 연구하고 각각을 개선하여 향상된 SRGAN(ESRGAN)을 도출합니다.  
특히, 저희는 기본 네트워크 구축 단위로 배치 정규화 없이 Residual-in-Residual Dense Block(RRDB)을 소개합니다.  
또한, 저희는 Relativistic GAN의 아이디어로 판별자가 절대값 대신 상대값을 예측하도록 하였다.  
마지막으로, 활성화 전 기능을 사용하여 지각 손실을 개선하여 밝기 일관성과 텍스처 복구에 대한 더 강력한 감독을 제공할 수 있습니다.  
이러한 개선의 혜택을 받아 제안된 ESRGAN은 SRGAN보다 더 현실적이고 자연스러운 텍스처로 일관되게 더 나은 시각적 품질을 달성했습니다.

# Proposed methods
- SRGAN과의 차이
SRGAN의 성능향상을 위해 논문에서는 크게 세가지를 바꾼다.
1. Network Architecture
2. elativistic Discriminator
3. Perceptual Loss


## Network Architecture

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F81527999-ad80-48cc-94d7-767698293599%2Fimage.png)

SRGAN에서는 Batch Normalization(BN)을 사용하지만, train 데이터셋과 test 데이터셋의 통계치가 달라 BN을 사용하면 artifact 현상이 생기게 되며, 일반화 성능이 저하된다.  
따라서 저자들은 안정적인 학습과 일관된 성능을 위해 BN을 제거하였다.  
BN을 제거함으로 계산 복잡도, 메모리 사용량에서 이점이 생긴다.  

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F4640cc7b-9842-46e1-b128-9e812a4f139c%2Fimage.png)
기존의 SRGAN의 구조는 그대로 사용하며 Block만 교체한 모습이다.  
RRDB는 기존 SRGAN의 Residual Block 보다 더 깊고 복잡한 구조로, 주 경로에서 dense block을 사용하는데, 이로 인해 네트워크 용량은 커진다.

이것들 외에 Residual scaling : 불안정성을 방지하기 위해 주 경로에 추가하기 전에 0과 1 사이의 상수를 곱해 residuals 스케일을 축소한다.  
Smaller initialization : residual 구조는 초기 매개변수 분산이 작을 수록 더욱 쉽게 학습시킬 수 있다.  
등의 기술도 이용하였다.

## Relativistic Discriminator
기존의 SRGAN의 판별자는 하나의 input 이미지(x)가 진짜이고 자연스러운 것일 확률을 추정했다.  
relativistic discriminator는 실제 이미지(Xr)가 가짜 이미지(Xf) 보다 상대적으로 더 현실적일 확률을 예측한다.  

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F5a978bc2-f9d3-4601-8c2a-397b74f8856b%2Fimage.png)

## Perceptual Loss
기존에는 activation 이후에 feature map을 사용했지만, activation 전에 feature map을 사용함으로써 SRGAN보다 더 효과적인 perceptual loss(Lpercep)를 개발하였고, 이를 통해 기존에 있던 2가지의 문제점을 해결하였다.  
매우 깊은 네트워크 activation 이후에 활성화된 features들은 매우 sparse함으로, 낮은 성능으로 이어진다.

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F50658d2f-d28b-4a04-8d98-1349c1f64005%2Fimage.png)

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F04da0e1b-6eef-4c41-9fab-27602258da24%2Fimage.png)

활성화 후 feature들을 사용하는 것은 ground-truth 이미지와 비교했을 때 일관성이 없는 복원된 밝기를 유발한다.  
(왼쪽 그래프 빨간색 gt, 파란색 after activation, 초록색 before activation)

## Total Loss
최종적으로 loss는 perceptual loss, RaGan loss, L1 loss가 사용된다.

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F382f8dc8-abe4-4484-9b68-e9e23c6b3719%2Fimage.png)

## Network Interpolation
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2Fbfaf8cd4-ac28-46b9-a6bf-a8ce01c949cf%2Fimage.png)

PSNR-oriented network(Gpsnr)을 학습한 후, 미세 조정을 통해 GAN-based network(Ggan)를 얻었다.  
이것으로 기존 GAN 방식의 학습이 진행되면서 perceptual quality가 좋아져도 artifact가 생기는 문제를 어느정도 해결한다.  
또 모델을 재학습시킬 필요없이 지속적으로 지각 품질과 정확도의 균형을 유지할 수 있게 해준다.

# Experiments
## The PIRM-SR Challenge
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2Ff0c0349a-d9e2-430d-823a-c8a0a5eedea7%2Fimage.png)

## Qualitative Results
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F45980c30-2efd-4af5-87a2-e2ac398cd562%2Fimage.png)

## Benchmark
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2Fd51374fa-b979-46e8-8900-84fa21121b98%2Fimage.png)

# Conclusion
저희는 이전 SR 방법보다 지속적으로 더 나은 지각별 품질을 달성하는 ESRGAN 모델을 제시했습니다.  
이 방법은 지각 지수 측면에서 PIRM-SR 챌린지에서 1위를 차지했습니다.  
저희는 BN 레이어가 없는 여러 RDDB 블록을 포함하는 새로운 아키텍처를 공식화했습니다.  
또한 제안된 심층 모델의 훈련을 용이하게 하기 위해 잔여 스케일링 및 더 작은 초기화를 포함한 유용한 기술이 사용됩니다.  
또한 한 이미지가 다른 이미지보다 더 현실적인지 여부를 판단하는 방법을 배우는 상대론적 GAN을 판별기로 사용하여 생성기가 더 자세한 텍스처를 복구하도록 안내했습니다.  
또한 활성화 전 기능을 사용하여 지각 손실을 개선하여 더 강력한 감독을 제공하고 더 정확한 밝기와 현실적인 텍스처를 복원했습니다.

# Reference
https://velog.io/@kanghoon12/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-ESRGAN
https://velog.io/@joon6093/%EC%9D%BC%EB%8B%A8-%EB%B0%95%EC%A3%A0-ESRGAN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%95%B4%EC%83%81%EB%8F%84-%ED%96%A5%EC%83%81
https://velog.io/@danielseo/Computer-Vision-ESRGAN






