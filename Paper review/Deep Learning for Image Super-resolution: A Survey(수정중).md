# Deep Learning for Image Super-resolution: A Survey

# Abs
이미지 초해상도(SR)는 컴퓨터 비전에서 이미지와 비디오의 해상도를 향상시키기 위한 이미지 처리 기술의 중요한 클래스입니다.  
최근 몇 년 동안 딥 러닝 기술을 사용한 이미지 초해상도의 놀라운 발전을 목격했습니다.  
이 기사는 딥 러닝 접근 방식을 사용한 이미지 초해상도의 최근 발전에 대한 포괄적인 조사를 제공하는 것을 목표로 합니다.  
일반적으로 SR 기술에 대한 기존 연구는 supervised SR, unsupervised SR, domain-specific SR의 세 가지 주요 범주로 그룹화할 수 있습니다.  
또한 공개적으로 사용 가능한 벤치마크 데이터 세트 및 성능 평가 metric과 같은 몇 가지 다른 중요한 문제도 다룹니다.  
마지막으로, 향후 커뮤니티에서 추가로 해결해야 할 몇 가지 미래 방향과 해결되지 않은 문제를 강조하는 것으로 이 조사를 마무리합니다.  

# Introduction
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.37.43.png)

# PROBLEM SETTING AND TERMINOLOGY
## Problem Definitions
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.20.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.34.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.43.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.54.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.40.03.png)

## Datasets for Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.17.32.png)

## Image Quality Assessment
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.17.50.png)

### Peak Signal-to-Noise Ratio
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.17.59.png)

### Structural Similarity
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.18.22.png)

### Mean Opinion Score
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.18.42.png)

### Learning-based Perceptual Quality
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.18.52.png)

### Task-based Evaluation

### Other IQA Methods

## Operating Channels
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.19.06.png)

## Super-resolution Challenges
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.19.21.png)

# SUPERVISED SUPER-RESOLUTION
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.19.35.png)

## Super-resolution Frameworks

### Pre-upsampling Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.20.17.png)

### Post-upsampling Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.20.26.png)

### Progressive Upsampling Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.21.05.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.21.20.png)

### Iterative Up-and-down Sampling Super-resolution

## Upsampling Methods
모델의 업샘플링 위치 외에도 업샘플링을 수행하는 방법이 매우 중요합니다.  
다양한 전통적인 업샘플링 방법이 있었지만 [20], [21], [88], [89]를 사용하여 엔드투엔드 업샘플링을 학습하는 것이 점차 추세가 되었습니다.  
이 섹션에서는 몇 가지 전통적인 보간 기반 알고리즘과 딥러닝 기반 업샘플링 레이어를 소개합니다.  

### Interpolation-based Upsampling
이미지 보간, 일명 이미지 스케일링은 디지털 이미지의 크기를 조정하는 것을 말하며 이미지 관련 응용 분야에서 널리 사용됩니다.  
전통적인 보간 방법에는 nearest neighbor  보간, bilinear 및 bicubic 보간, Sinc and Lanczos 리샘플링 등이 있습니다.  
이러한 방법은 해석이 가능하고 구현하기 쉽기 때문에 CNN 기반 SR 모델에서 여전히 널리 사용되는 방법도 있습니다.

#### Nearest-neighbor Interpolation.

#### Bilinear Interpolation.

#### Bicubic Interpolation.

### Learning-based Upsampling
보간 기반 방법의 단점을 극복하고 엔드 투 엔드 방식으로 업샘플링을 학습하기 위해 SR 필드에 transposed convolution layer와 sub-pixel layer가 도입됩니다.  

#### Transposed Convolution Layer.

#### Sub-pixel Layer.

#### Meta Upscale Module.

오늘날 이러한 학습 기반 계층은 가장 널리 사용되는 업샘플링 방법이 되었습니다.  
특히 업샘플링 후 프레임워크(Post-upsampling Super-resolution)에서 이러한 계층은 일반적으로 저차원 공간에서 추출된 고수준 표현을 기반으로 HR 이미지를 재구성하는 최종 업샘플링 단계에서 사용되므로 고차원 공간에서 압도적인 작업을 피하면서 엔드투엔드 SR을 달성합니다.  

## Network Design
오늘날 네트워크 설계는 딥 러닝의 가장 중요한 부분 중 하나였습니다.  
초해상도 분야에서 연구자들은 최종 네트워크를 구성하기 위해 4개의 SR 프레임워크(Super-resolution Frameworks) 위에 모든 종류의 네트워크 설계 전략을 적용합니다.
이 절에서는 이러한 네트워크를 네트워크 설계를 위한 필수 원칙 또는 전략으로 네트워크를 분해하고 소개하며 장점과 한계를 하나씩 분석합니다.

### Residual Learning
He 등[96]이 철저한 매핑 방법 대신 잔차 학습을 위한 ResNet을 제안하기 전에 residual learning은 그림 7a에서 볼 수 있듯이 SR 모델 [48], [88], [97]에서 널리 사용되었습니다.  
이 중 잔차 학습 전략은 크게 global, local residual learning으로 나눌 수 있습니다.  

![]()

#### Global Residual Learning
이미지 SR은 입력 이미지가 대상 이미지와 높은 상관관계가 있는 이미지 대 이미지 변환 작업이기 때문에 연구자들은 그들 사이의 잔차, 즉 전역 잔차 학습만 학습하려고 합니다.  
이 경우 완전한 이미지에서 다른 이미지로의 복잡한 변환을 학습하는 것을 피하며, 대신 누락된 고주파 세부 정보를 복원하기 위해 residual map만 학습하면 됩니다.  
대부분의 영역의 잔차가 0에 가깝기 때문에 모델 복잡성과 학습 난이도가 크게 낮아집니다.  
따라서 SR 모델[26], [55], [56], [98]에서 널리 사용됩니다.

#### Local Residual Learning
로컬 잔차 학습은 ResNet[96]의 잔차 학습과 유사하며, 계속 증가하는 네트워크 깊이로 인한 degradation 문제[96]를 완화하고 훈련 난이도를 줄이며 학습 능력을 향상시키는 데 사용됩니다.  
SR[70], [78], [85], [99]에도 널리 사용됩니다. 

실제로 위의 방법들은 모두 shortcut connection(종종 작은 상수에 의해 확장됨)과 element-wise addition로 구현되는 반면, 전자는 입력 이미지와 출력 이미지를 직접 연결하는 반면, 후자는 일반적으로 네트워크 내부에서 깊이가 다른 레이어 사이에 여러 개의 shortcut을 추가한다는 차이점이 있습니다.

### Recursive Learning
많은 파라미터를 도입하지 않고 더 높은 수준의 특징을 학습하기 위해 그림 7b와 같이 동일한 모듈을 여러 번 재귀적으로 적용하는 것을 의미하는 재귀적 학습이 SR 필드에 도입됩니다.  

![]()

그 중 16-recursive DRCN[82]은 재귀적 단위로 단일 컨볼루션 계층을 사용하여 많은 파라미터 없이 SRCNN[22]의 13 × 13보다 훨씬 큰 41 × 41의 receptive field에 도달합니다.  
DRRN[56]은 25번의 재귀적 단위로 ResBlock[96]을 사용하여 17-ResBlock(기준선)보다 훨씬 더 나은 성능을 얻습니다.  
나중에 Tai et al. [55]는 모든 재귀의 출력이 연결되고 암기와 망각을 위해 추가 1 × 1 컨볼루션을 거치는 6번의 재귀적 ResBlock으로 구성된 메모리 블록을 기반으로 MemNet을 제안합니다.  
cascading residual network(CARN)[28]도 여러 ResBlock을 포함하는 유사한 재귀적 단위를 채택합니다.  
최근 Li et al. [86]은 반복적인 up and-down sampling SR 프레임워크를 사용하고 전체 네트워크의 가중치가 모든 재귀에 걸쳐 공유되는 재귀적 학습 기반의 피드백 네트워크를 제안합니다.  

또한 연구자들은 서로 다른 부분에 서로 다른 재귀 모듈을 사용합니다.  
특히 Han et al. [85]는 LR 상태와 HR 상태 간의 신호를 교환하기 위해 이중 상태 순환 네트워크(DSRN)를 제안합니다.  
각 time step(즉, 재귀)에서 각 세부 항목들의 표현이 업데이트되고 LR-HR 관계를 더 잘 탐색하기 위해 교환됩니다.  
유사하게 Lai et al. [65]는 임베딩 및 업샘플링 모듈을 재귀적 단위로 사용하므로 성능 손실이 거의 없는 대신 모델 크기를 훨씬 줄입니다.  

일반적으로 재귀적 학습은 과도한 파라미터를 도입하지 않고도 더 진보적인 표현을 학습할 수 있지만 여전히 높은 계산 비용을 피할 수 없습니다.  
그리고 본질적으로 vanishing or exploding gradient problems 때문에 잔차 학습(섹 3.3.1) 및 multi-supervision(섹 3.4.4)과 같은 일부 기술은 종종 이러한 문제를 완화하기 위해 재귀적 학습과 통합됩니다[55], [56], [82], [85].  

### Multi-path Learning
다중 경로 학습은 다양한 작업을 수행하는 여러 경로를 통해 기능을 전달하고 더 나은 모델링 기능을 제공하기 위해 이를 다시 융합하는 것을 말합니다.  
구체적으로 다음과 같이 global, local and scale-specific multi-path learning으로 나눌 수 있습니다.

#### Global Multi-path Learning

#### Local Multi-path Learning

#### Scale-specific Multi-path Learning

### Dense Connections
Huang 등[102]이 dense block을 기반으로 DenseNet을 제안한 이후, 조밀한 연결은 비전 작업에서 점점 더 인기를 얻고 있습니다.  
Dense block의 각 layer에 대해 이전 모든 레이어의 feature map이 입력으로 사용되고 자체 feature map이 모든 후속 레이어의 입력으로 사용되어 l-ayer dense block(l ≥ 2)에서 l · (l - 1)/2 연결로 이어집니다.  
조밀한 연결은 gradient vanishing을 완화하고 신호 전파를 향상시키며 feature 재사용을 장려하는 데 도움이 될 뿐만 아니라 작은 growth rate(dense blocks의 채널 수)을 사용하고 모든 input feature map을 연결한 후 채널을 squeezing해서 모델 크기를 크게 줄일 수 있습니다.  

low-level and high-level feature을 융합하여 고품질의 세부 정보를 재구성하기 위한 풍부한 정보를 제공하기 위해 그림 7d에서 볼 수 있듯이 dense connection이 SR 필드에 도입됩니다.  
Tong et al. [79]는 dense blocks을 채택하여 69-SRDenseNet을 구성할 뿐만 아니라 서로 다른 dense block 사이에 dense connection을 삽입합니다.  
즉, 모든 dense block에 대해 이전 블록의 feature map은 input으로 사용되고 자체 feature map은 모든 후속 블록에 input으로 사용됩니다.  
이러한 계층 수준 및 블록 수준의 dense connection은 MemNet [55], CARN [28], RDN [93] 및 ESRGAN [103]에서도 채택됩니다.  
DBPN [57]도 dense connection을 광범위하게 채택하지만 다운 샘플링 유닛과 마찬가지로 dense connection은 모든 업샘플링 유닛 사이에 있습니다.

### Attention Mechanism
#### Channel Attention
서로 다른 채널 간의 feature 표현의 독립성과 상호 작용을 고려하여 Hu et al. [104]는 그림 7c에서 볼 수 있듯이 채널 독립성을 명시적으로 모델링하여 학습 능력을 향상시키는 "squeeze-and-excitation" 블록을 제안합니다.  
이 블록에서 각 입력 채널은 global average pooling(GAP)을 사용하여 channel descriptor(즉, 상수)로 압축된 다음, 이러한 설명자를 두 개의 dense layer로 공급하여 입력 채널에 대한 채널별 scaling factor를 생성합니다.  
최근 Zhang et al. [70]은 channel attention mechanism을 SR과 통합하고 모델의 표현 능력과 SR 성능을 현저하게 향상시키는 RCAN을 제안합니다.  
feature correlation를 더 잘 학습하기 위해 Dai et al. [105]는 second-order channel attention(SOCA) 모듈을 추가로 제안합니다.  
SOCA는 GAP 대신 second-order feature를 사용하여 channel-wise features를 적응적으로 rescale하고 더 많은 정보와 판별 표현을 추출할 수 있도록 합니다.

#### Non-local Attention

### Advanced Convolution

### Region-recursive Learning

### Pyramid Pooling

### Wavelet Transformation

### Desubpixel

### xUnit

## Learning Strategies

### Loss Functions

### Batch Normalization

### Curriculum Learning

### Multi-supervision

## Other Improvements

### Context-wise Network Fusion

### Data Augmentation

### Multi-task Learning

### Network Interpolation

### Self-Ensemble

## State-of-the-art Super-resolution Models

# UNSUPERVISED SUPER-RESOLUTION

## Zero-shot Super-resolution

## Weakly-supervised Super-resolution

## Deep Image Prior

# DOMAIN-SPECIFIC APPLICATIONS

## Depth Map Super-resolution

## Face Image Super-resolution

## Hyperspectral Image Super-resolution

## Real-world Image Super-resolution

## Video Super-resolution

## Other Applications

# CONCLUSION AND FUTURE DIRECTIONS

## Network Design

## Learning Strategies

## Evaluation Metrics

## Unsupervised Super-resolution

## Towards Real-world Scenarios


