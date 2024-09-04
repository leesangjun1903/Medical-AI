# Denoising Diffusion Probabilistic Models

# DDPM

# Abs
저희는 비평형 열역학의 고려 사항에서 영감을 얻은 잠재 변수 모델 클래스인 확산 확률 모델을 사용하여 고품질 이미지 합성 결과를 제시합니다.  
저희의 최상의 결과는 Langevin dynamics을 이용한 diffusion probabilistic model과 denoising score matching 사이의 연결에 따라 설계된 가중치 변동 경계에 대한 훈련을 통해 얻어지며, 저희 모델은 자연스럽게 autoregressive decoding의 일반화로 해석될 수 있는 점진적 손실 압축 방식을 선보입니다.  
unconditional CIFAR10 데이터 세트에서 저희는 9.46의 Inception 점수와 3.17의 최첨단 FID 점수를 얻었습니다.  
256x256 LSUN에서 ProgressiveGAN과 유사한 샘플 품질을 얻었습니다.  

# Introduction
간단히 결론부터 설명하면 DDPM은 주어진 이미지에 time에 따른 상수의 파라미터를 갖는 작은 가우시안 노이즈를 time에 대해 더해나가는데, image가 destroy하게 되면 결국 noise의 형태로 남을것이다. (normal distribution을 따른다.)  
이런 상황에서 normal distribution 에 대한 noise가 주어졌을때 어떻게 복원할 것인가에 대한 문제이다.  
그래서 주어진 Noise를 통해서 완전히 이미지를 복구가 된다면 image generation하는 것이 된다.  
이 논문에서는 diffusion probabilistic models의 과정을 보여준다.  
diffusion model은 유한한 시간 뒤에 이미지를 생성하는 variational inference을 통해 훈련된 Markov chain을 parameterized한 형태이다.  
Markov Chain은 이전의 샘플링이 현재 샘플링에 영향을 미치는 p(x_t-1|x_t) 형식을 의미한다.  
그래서 이 diffusion model에서의 한 방향에 대해서는 주어진 이미지에 작은 gaussian noise를 점진적으로 계속 더해서 완전히 image가 destroy 되게하는 과정을 의미한다.

# Background

# Diffusion models and denoising autoencoders

## Forward Process and LT

## Reverse Process and L1:T−1

## Data scaling, reverse process decoder, and L0

## Simplified training objective

# Experiments

# Conclusion

