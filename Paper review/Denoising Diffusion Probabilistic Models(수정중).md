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

이 논문은 diffusion model(리뷰)을 발전시킨 논문이다. Diffusion model은 parameterized Markov chain을 학습시켜 유한 시간 후에 원하는 데이터에 맞는 샘플을 만드는 모델이다.  
Forward process에서는 Markov chain이 점진적으로 noise를 추가하여 최종적으로 가우시안 noise로 만든다.  
반대로 reverse process는 가우시안 noise로부터 점진적으로 noise를 제거하여 최종적으로 원하는 데이터에 맞는 샘플을 만든다.  
Diffusion이 작은 양의 가우시안 noise로 구성되어 있기 때문에 샘플링 chain을 조건부 가우시안으로 설정하는 것으로 충분하고,  
간단한 신경먕으로 parameterize할 수 있다.

기존 diffusion model은 정의하기 쉽고 학습시키기 효율적이지만 고품질의 샘플을 만들지 못하였다.  
반면, DDPM은 고품질의 샘플을 만들 수 있을 뿐만 아니라 다른 생성 모델 (ex. GAN)보다 더 우수한 결과를 보였다.  
또한, diffusion model의 특정 parameterization이 학습 중 여러 noise 레벨에서의 denoising score matching과 비슷하며, 샘플링 중 Langevin dynamics 문제를 푸는 것과 동등하다는 것을 보였다.

그 밖의 특징으로는

1. DDPM은 고품질의 샘플을 생성하지만 다른 likelihood 기반의 모델보다 경쟁력 있는 log likelihood가 없다.
2. DDPM의 lossless codelength가 대부분 인지할 수 없는 이미지 세부 정보를 설명하는 데 사용되었다.
3. Diffusion model의 샘플링이 autoregressive model의 디코딩과 유사한 점진적 디코딩이라는 것을 보였다.

# Background

![](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0e46190c-ca06-48f9-b201-7377c8b31f18%2F%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-02-20_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_4.28.54.png?table=block&id=b6e28b39-c04c-4041-a44c-c201e417f5f3&cache=v2)

Diffusion model : $p_\theta (x_0) := \int p_\theta (x_{0:T}) dx_{1:T}$  
$x_{0:T}$ size = $x_0 \sim q(x_0 )$ size  
$p_\theta (x_{0:T})$ : Reverse Process  
$p(x_T ) = \mathcal{N} (x_T ; 0, I )$ 에서 시작하는 Gaussian transition으로 이루어진 Markov chain으로 정의된다.

$p_\theta (x_{0:T}) := p(x_T) \prod_{t=1}^T p_\theta (x_{t-1}|x_{t})$  
$p_\theta (x_{t-1}|x_{t}) := \mathcal{N} (x_{t-1} ; \mu_\theta (x_t , t), \Sigma_\theta (x_t , t))$

Diffusion model이 다른 latent variable model과 다른 점은 forward process 혹은 diffusion process라 불리는 approximate posterior $q(x_{1:T}|x_0)$ 가 $\beta_1, \cdots, \beta_T$ 에 따라 가우시안 noise를 점진적으로 추가하는 Markov chain이라는 것이다.  

$q (x_{1:T}|x_0) := \prod_{t=1}^T q (x_{t}|x_{t-1})$    
$q (x_{t}|x_{t-1}) := \mathcal{N} (x_{t} ; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$

학습은 negative log likelihood에 대한 일반적인 variational bound을 최적화하는 것으로 진행된다.

$L:= \mathbb{E} [-\log p_\theta (x_0)] \le \mathbb{E_q} \bigg[ -\log \frac{p_\theta (x_{0:T})}{q(x_{1:T}|x_0)} \bigg] \le \mathbb{E_q} \bigg[ -\log p(x_T) - \sum_{t \ge 1} \log \frac{p_\theta (x_{t-1}|x_t)}{q(x_t|x_{t-1})} \bigg]$

$beta_t$ 는 reparameterization으로 학습하거나 hyper-parameter로 상수로 둘 수 있다.  
또한 $beta_t$가 충분히 작으면 forward process와 reverse process가 같은 함수 형태이므로 reverse process의 표현력은 $p_\theta (x_{t-1}|x_t)$에서 가우시안 conditional의 선택에 따라 부분적으로 보장된다.

Forward process에서 주목할만한 것은 closed form으로 임의의 시간 t에서 샘플링 $xt$가 가능하다는 것이다.

$\alpha_t := 1-\beta_t, \quad \bar{\alpha_t} := \prod_{s=1}^t \alpha_s$  
$q(x_t | x_0) = \mathcal{N} (x_t ; \sqrt{ \bar{\alpha_t}} x_0 , (1-\bar{\alpha_t})I)$

# Diffusion models and denoising autoencoders

## Forward Process and LT

## Reverse Process and L1:T−1

## Data scaling, reverse process decoder, and L0

## Simplified training objective

# Experiments

# Conclusion

