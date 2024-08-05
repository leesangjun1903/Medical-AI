# Medical image denoising using convolutional denoising autoencoders

# denoising autoencoder

# Abs
이미지 노이즈 제거는 의료 이미지 분석에서 중요한 전처리 단계입니다.  
지난 30년 동안 다양한 노이즈 제거 성능을 가진 다양한 알고리즘이 제안되었습니다.  
보다 최근에는 모든 기존 방법을 능가하는 딥 러닝 기반 모델이 큰 가능성을 보여주었습니다.  
그러나 이러한 방법은 큰 훈련 샘플 크기와 높은 계산 비용이 필요한 경우에 제한적입니다.  
이 논문에서는 작은 샘플 크기를 사용하여 컨볼루션 레이어를 사용하여 구성한 노이즈 제거 Autoencoder를 사용하여 의료 이미지의 효율적인 노이즈 제거에 사용할 수 있음을 보여줍니다.  
이질적인 이미지를 결합하여 샘플 크기를 높여 노이즈 제거 성능을 높일 수 있습니다.  
가장 간단한 네트워크는 노이즈와 신호가 사람의 눈에 구분되지 않을 정도로 손상 수준이 높은 이미지를 재구성할 수 있습니다.

# Introduction
X선, 자기공명영상(MRI), 컴퓨터 단층촬영(CT), 초음파 등을 포함한 의료 영상은 노이즈에 취약합니다[21].  
다른 영상 획득 기술을 사용하는 것부터 방사선에 노출되는 환자를 줄이려는 시도까지 다양한 이유가 있습니다.  
방사선의 양이 감소함에 따라 노이즈는 증가합니다[1].  
적절한 영상 분석을 위해 사람과 기계 모두 노이즈 제거가 종종 필요합니다.  
컴퓨터 비전의 고전적인 문제인 이미지 노이즈 제거는 자세히 연구되어 왔습니다.  
편미분 방정식(PDE)[18], [20], [22], Wavelet[6], DCT[29], BLS-GSM[19] 등과 같은 도메인 변환, NL-means[30], [3]을 포함한 비국소적 기법, BM3D[7]와 같은 비국소적 수단(non local)과 도메인 변환(domain transformations)의 조합 및 희소 코딩 기법(sparse coding techniques)을 이용하는 모델 제품군에 이르기까지 다양한 방법이 존재합니다.

z = x + y
여기서 z는 원본 이미지 x와 일부 noise, y의 합으로 생성된 잡음 이미지입니다.  
대부분의 방법은 가능한 한 가까운 z를 사용하여 x를 근사하려고 합니다. 대부분의 경우 y는 잘 정의된 프로세스에서 생성된 것으로 가정됩니다.

최근 딥 러닝[14], [11], [23], [2], [10]의 발전으로 딥 아키텍처 기반 모델의 결과가 유망해졌습니다.  
Autoencoders은 이미지 노이즈 제거[24], [25], [28], [5]에 사용되었습니다.  
- 이들은 기존의 노이즈 제거 방법을 쉽게 능가하고 노이즈 생성 프로세스의 사양에 덜 제한적입니다.  
- 컨볼루션 레이어를 사용하여 구성된 노이즈 제거 autoencoder은 강력한 공간 상관 관계를 활용할 수 있는 능력에 비해 이미지 노이즈 제거 성능이 더 우수합니다.
- 이 논문에서는 컨볼루션 레이어를 사용하여 구축된 스택형 노이즈 제거 자동 인코더가 의료 이미지 데이터베이스의 전형적인 작은 샘플 크기에서 잘 작동한다는 증거를 제시합니다.

이는 최적의 성능을 위해서는 심층 아키텍처 기반 모델에 매우 큰 훈련 데이터 세트가 필요하다는 생각과 반대입니다.  
또한 이러한 방법은 대부분의 다른 노이즈 제거 방법이 실패하는 지점에서 노이즈 수준이 매우 높을 때도 신호를 복구할 수 있음을 보여줍니다.

# Pre
Autoencoder는 역전파를 사용하여 항등 함수에 대한 근사치를 학습하려고 하는 신경망의 한 종류입니다.  
Autoencoder는 먼저 입력값 x을 넣고 결정론적 매핑(deterministic mapping)을 사용하여 hidden representation y 에 매핑(encode)합니다.  
여기서 s는 임의의 비선형 함수가 될 수 있습니다. 그런 다음 Latent representation y는 유사한 매핑을 사용하여 x와 동일한 모양의 재구성 z로 다시 매핑(decode)됩니다.  
모델 매개 변수(W,W',b,b')는 재구성 오류를 최소화하도록 최적화되어 있으며, 이는 제곱 오차 또는 Cross-entropy와 같은 다양한 손실 함수(Loss funciton)를 사용하여 평가할 수 있습니다.  

1) Denoising Autoencoder: Denoising Autoencoder는 일반적인 Autoencoder[24]의 확률적 확장입니다.
즉, 노이즈가 많은 버전이 주어지면 모델이 입력 재구성을 학습하도록 강제합니다.
확률적 손상 프로세스는 입력 중 일부를 무작위로 0으로 설정하여 Denoising Autoencoder가 무작위로 선택된 누락된(corrupted) 값을 예측하도록 강요합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.49.52.png)

Denoising autoencoder를 쌓아 깊은 네트워크(stacked denoising autoencoder)를 만들 수 있습니다.  
하위 레이어의 출력은 현재 레이어로 공급되며 레이어 단위별로 학습이 완료됩니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.50.58.png)

2) Convolutional autoencoder: Convolutional autoencoders [16]은 컨볼루션 인코딩 및 디코딩 계층이 있는 표준 autoencoder architecture을 기반으로 합니다.
convolutional autoencoders은 고전 autoencoders에 비해 이미지 구조를 활용하기 위해 convolutional neural networks의 완전한 기능을 활용하기 때문에 이미지 처리에 더 적합합니다.
convolutional autoencoders에서 가중치는 모든 입력 위치에서 공유되므로 q지역적 공간성을 유지하는 데 도움이 됩니다.

i번째 feature map 이 weights와 bias의 activation 선형 결합식으로 표현되고, latent feature maps을 이루어 가중치 dimension에 대한 flip operation과 convolution 연산합니다.  
그리고 bias 와의 activation 선형 결합으로 reconstruction 식을 표현합니다.  
역전파는 매개변수에 대한 오류 함수의 기울기 계산에 사용됩니다.

# Evaluation

한 번에 하나의 이미지를 손상시키는 대신 이미지를 나타내는 각 행이 있는 flatten된 데이터 세트가 손상되어 모든 이미지가 동시에 교란되었습니다. 그런 다음 손상된 데이터 세트가 모델링에 사용되었습니다.  
이미지의 일관성과 정확성을 위해 PSNR(peak signal to noise ratio) 대신 SSIM(structural similarity index measure)을 사용하여 이미지를 비교했습니다[27].  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%202.17.11.png)

# Conclusion
저희는 컨볼루션 레이어를 사용하여 구성된 denoising autoencoder가 의료 이미지의 효율적인 노이즈 제거에 사용될 수 있음을 보여주었습니다.  
저희는 작은 훈련 데이터 세트를 사용하면 좋은 노이즈 제거 성능을 달성할 수 있으며, 300개 정도의 훈련 샘플만 있으면 좋은 성능에 충분하다는 것을 보여주었습니다.  
향후 작업은 소규모 샘플 노이즈 제거를 위한 최적의 아키텍처를 찾는 데 초점을 맞출 것입니다. 


