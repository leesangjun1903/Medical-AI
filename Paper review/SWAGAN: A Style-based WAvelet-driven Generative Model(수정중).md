# SWAGAN: A Style-based WAvelet-driven Generative Model

# Abs
최근 몇 년간 GAN(Generative Adversarial Networks)의 시각적 품질에 상당한 진전이 있었다.  
그럼에도 불구하고, 이러한 네트워크들은 여전히 눈에 띄게 편향된 구조와 좋지 않은 손실 함수로 인한 고주파 콘텐츠(high frequency content)의 품질 저하로 어려움을 겪고 있다.  
이 문제를 해결하기 위해 주파수 영역(frequency domain)에서 점진적 생성을 구현하는 새로운 범용 스타일과 wavelet 기반의 GAN(SWAGAN)을 제시한다.  
SWAGAN은 Generator와 Discriminator 구조 전체에 wavelet을 통합하여 모든 단계에서 주파수 인식 잠재 표현(frequency-aware latent representation)을 적용한다.  
이 접근 방식은 생성된 이미지의 시각적 품질을 향상시키고 계산 성능을 크게 향상시킨다.  
SyleGAN2 프레임워크에 통합하고 wavelet 영역에서 content 생성이 더욱 사실적인 고주파 콘텐츠의 고품질 이미지로 이어진다는 것을 확인함으로써 이 방법의 장점을 입증한다.  
또한, 저자들은 모델의 잠재 공간에서 StyleGAN이 다수 editing tasks의 기반이 될 수 있는 품질을 유지하고 있는지 확인하고, 주파수 인식 접근 방식이 개선된 downstream visual quality를 유도한다는 것을 보여준다.

# Introduction

![](https://velog.velcdn.com/images%2Fdanielseo%2Fpost%2Fa64fd118-48af-41d4-8aff-f2f9f967170f%2F%EC%BA%A1%EC%B2%98.PNG)

(Figure 1. 이미지나 특징 공간이 아닌 wavelet 영역에서 직접 작업함으로써 신경망의 스펙트럼 편향을 완화하고 다른 모델이 실패하는 고주파 데이터를 성공적으로 생성할 수 있다.  
저자들의 모델은 StyleGAN2와 같은 SOTA 모델을 벗어난 패턴을 만들 수 있다.  
심지어 Training set이 단일 이미지만 포함하는 over-fitting setup에서도 사용할 수 있다.  
저자들은 (왼쪽에서 오른쪽으로) 원본 이미지와 StyleGAN2의 output, SWAGAN의 output을 보여준다.  
각 StyleGAN2와 SWAGAN의 output은 해당 이미지에 대해 24시간 동안 학습한 것이다.)

![](https://velog.velcdn.com/images%2Fdanielseo%2Fpost%2F444e76ee-53d4-4e5c-8c96-a3672c404781%2F%EC%BA%A1%EC%B2%98.PNG)

(Figure 2. 저자들의 style 기반 Generator 구조는 증가하는 해상도 스케일에서 wavelet coefficients를 예측한다.)

# Background and related work

# Method
## Wavelet Transform
저자들의 방법의 핵심은 이미지를 일련의 채널로 분해하는 wavelet transform으로, 각각은 서로 다른 범위의 주파수 content를 나타낸다.  
저자들은 Haar Wavelets를 변환의 기본 함수로 사용했는데, 이는 다중 주파수 정보를 잘 나타내는 문서화된 능력과 결합된 단순한 특성 때문이다.  
이 모델은 1단계 wavelet 분해(decomposition)로 작동하며, 여기서 각 이미지는 LL, LH, HL, HH의 4개 하위 밴드(sub-bands)로 분할된다.  
이러한 밴드(band)는 일련의 low-pass와 high-pass wavelet filters를 통해 이미지를 전달함으로써 얻어지며, wavelet coefficients 역할을 한다.  
첫 번째 하위 대역인 LL은 저주파 정보에 해당하며, 실제로 입력 이미지의 blurred 버전과 시각적으로 유사하다.  
나머지 하위 대역인 LH, HL, HH는 각각 수평, 수직, 대각선 방향의 고주파 content에 해당한다.

## Network Architecture
저자들의 wavelet-aware 구조는 StyleGAN2의 original implementation을 기반으로 한다.  
original implementation은 이미지 공간에서 직접 content를 생성하지만, 제안된 구조는 주파수 영역에서 작동한다.  
마찬가지로, 저자들의 Discriminator는 이미지의 RGB 공간뿐만 아니라 전체 wavelet 분해까지 고려하도록 설계되었다.

Generator가 wavelet 영역에서 직접 content를 생성하게 함으로써 다음 두 가지 측면에서 이득을 얻을 수 있다.  
첫째, 신경망은 저주파 영역에서 학습을 우선시한다.  
표현(representation)을 주파수 기반 표현으로 변환함으로써 표현에 대한 저주파수 수정(modifications)을 학습하여 네트워크가 이미지 영역의 고주파수 변화에 영향을 미칠 수 있도록 한다.  
이것은 나중에 네트워크가 고주파수를 학습하도록 동기를 제공할 수 있지만, 학습 작업을 더 쉽게 만들지는 않기 때문에 단순한 loss 기반 수정과는 다르다.  
둘째, wavelet 분해는 공간적으로 더 촘촘하다. 1단계 wavelet 분해에서 각 2N x 2N 영상은 각각 N x N 계수의 4개 채널로 완전히 표시된다.  
이를 통해 추가 filters가 필요한 대신 전체 생성 프로세스 전반에 걸쳐 저해상도 표현에 대한 convolution을 사용할 수 있다.  
그러나, 이 trade-off는 인기 있는 딥러닝 프레임워크를 사용할 때 유리할 수 있다.

마찬가지로, Discriminator에 주파수 정보를 제공함으로써 네트워크는 생성된 이미지에서 종종 누락되는 고주파 content를 더 잘 식별할 수 있다.  
그 결과, Generator는 그럴듯한 고주파 데이터를 다시 생성하도록 동기를 부여받는다.

