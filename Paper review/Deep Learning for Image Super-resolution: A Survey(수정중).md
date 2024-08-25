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

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.21.05.png)

### Iterative Up-and-down Sampling Super-resolution

## Upsampling Methods

### Interpolation-based Upsampling

### Learning-based Upsampling In order to overcome the shortcomi

## Network Design

### Residual Learning

### Recursive Learning

### Multi-path Learning

### Dense Connections

### Attention Mechanism

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


