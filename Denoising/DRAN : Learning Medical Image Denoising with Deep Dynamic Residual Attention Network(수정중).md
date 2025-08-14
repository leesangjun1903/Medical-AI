# Learning Medical Image Denoising with Deep Dynamic Residual Attention Network

## 1. 핵심 주장 및 주요 기여
“Learning Medical Image Denoising with Deep Dynamic Residual Attention Network” 논문은 **의료 영상의 다양한 잡음 환경**을 단일 네트워크로 효과적으로 제거할 수 있는 새로운 딥러닝 기반 모델인 **Dynamic Residual Attention Network (DRAN)** 을 제안한다.[1]
주요 기여는 다음과 같다:  
- **대규모 다중 모달 의료영상 학습**: X-선, MRI, CT, 초음파, 현미경, 피부과 영상 등 이질적 데이터 70만여 장을 학습에 활용해 **다양한 잡음 분포에 강건한** 모델 구현.[1]
- **동적 커널+잔차 주의 블록 (DRAB)**: 입력 특성에 따라 가중치를 조정하는 **Dynamic Convolution** 과, 불필요 저수준 특성 흐름을 제어하는 **Noise Gate** 결합으로 잡음 제거 성능 및 학습 안정성 개선.[1]
- **정량·정성적 성능 우위**: PSNR, SSIM 기준 기존 BM3D, DnCNN, Residual MID 대비 최대 **11dB 이상** 향상, 실제 의료영상에서도 **눈에 띄는 화질 개선** 입증.[1]

## 2. 문제 정의 및 제안 기법

### 2.1 해결하고자 하는 문제
- 기존 의료영상 잡음 제거 기법들은 **특정 모달리티나 제한적 잡음 수준**에서만 작동하며,  
- 다양한 병원 환경·장비에서 발생하는 **실제 잡음**에 대해 일반화 성능이 부족함.  

### 2.2 제안하는 방법
- 잡음 제거를 $$c = v - n$$ 형태로 모델링하고, 입력 의료영상 $$v$$로부터 잔차 잡음 $$n$$을 예측하는 맵핑 함수 $$F$$ 학습:  

$$
    c = v - F(v)
  $$

- **목표 손실**: 픽셀 단위 $$L_1$$ 노름  

$$
    L = \|n_\text{ref} - n_\text{pred}\|_1
  $$

### 2.3 모델 구조
DRAN은 크게 세 개의 **Dynamic Residual Attention Block (DRAB)** 으로 구성되며, 각 블록은 다음 단계로 구성된다:[1]
1. **Dynamic Convolution Layer** $$\sum_{k=1}^K \pi_k(x)(W_k x + b_k)$$  
2. **Batch Normalization & ReLU**  
3. **Noise Gate**: 공간적 게이팅 $$\phi(G)\odot\sigma(F)$$ 으로 불필요 저수준 특징 선택적 차단  
4. **Residual Skip Connection**  

전체 네트워크는 입력→초기 컨볼루션→3×DRAB→최종 컨볼루션→출력 형태를 따른다.

### 2.4 성능 향상
- **정량평가**: PSNR 최대 +11.17dB, SSIM 최대 +0.3065 향상.[1]
- **실제영상**: Kaggle 실제 MRI·CT·초음파 데이터에서 잡음 제거 후 진단 보조(CAD), 분할(U-Net), 객체 탐지(Mask R-CNN) 정확도 개선.[1]

### 2.5 한계
- **합성 잡음 기반 학습**: Gaussian 노이즈 합성에 의존, 실제 잡음 분포 차이로 일반화 한계.  
- **RGB 채널 제한**: 다채널 의료기법(단일 채널 CT/MRI) 적용 시 추가 튜닝 필요.  

## 3. 모델의 일반화 성능 향상 관점
- **대규모·다중 모달 학습**: 다양한 영상 포맷과 장비 노이즈를 포함한 학습 데이터로 일반화 강화.  
- **동적 커널**: 입력별 적응적 필터링으로 잡음 분포 변화에 유연 대응.  
- **Noise Gate**: 공간적 중요도를 학습해 과도한 정보 전달 억제, 실제 환자 데이터에도 노이즈 제거 효과 유지.  

이러한 설계는 **새로운 모달리티나 장비**에도 추가 학습 없이도 빠른 적응 가능성을 제시한다.

## 4. 향후 연구 영향 및 고려 사항
- **비지도·자기지도 학습**: 실제 노이즈-무노이즈 쌍 데이터 확보가 어려운 한계를 극복하기 위해, 노이즈 통계 모델링 기반 **비지도 학습 프레임워크** 개발 필요.  
- **단일 채널 적용**: CT/MRI 등 그레이스케일 영상 특화 DRAN 경량화 및 채널 수 조정 연구.  
- **임상 적용 검증**: 병원 협력으로 **실제 환자 영상** 대규모 검증과, 진단 정확도·처리 속도·안정성 평가 필수.  
- **모델 경량화**: 엣지 디바이스 실시간 적용을 위한 **파라미터 축소**, 연산 최적화 연구가 뒤따라야 함.  

***

참고문헌  
 Sharif, S. M. A.; Naqvi, R. A.; Biswas, M. Learning Medical Image Denoising with Deep Dynamic Residual Attention Network. *Mathematics* **2020**, *8*, 2192.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e0f5c891-4f34-400c-885f-b668aa727c5a/mathematics-08-02192-v2.pdf

# Learning Medical Image Denoising with Deep Dynamic Residual Attention Network

# dynamic residual attention network
# dynamic convolution
# noise gate
# residual learning

# Abs
이미지 노이즈 제거는 의료 이미지 분석에서 중요한 역할을 합니다.  
여러가지 방법으로 노이즈가 많은 이미지 샘플의 지각 품질을 향상시켜 진단 프로세스를 크게 가속화할 수 있습니다.  
그러나 의료 이미지 노이즈 제거를 광범위하게 실행할 수 있음에도 불구하고 기존 노이즈 제거 방법은 여러 전문 분야에 걸친 의료 이미지에 나타나는 다양한 범위의 노이즈를 해결하는 데 부족한 점을 보여줍니다.  
이 연구는 상당한 양의 데이터 샘플에서 residual noise를 학습하여 이러한 어려운 노이즈 제거 작업을 완화합니다.  
또한 제안된 방법은 네트워크 아키텍처가 attention mechanism으로 알려진 feature의 상관 관계를 활용하고 공간적으로 정제된 residual feature들과 결합하는 새로운 심층 네트워크를 도입하여 학습 프로세스를 가속화합니다.  
실험 결과는 제안된 방법이 정량적 및 정성적 비교 모두에서 기존 작업을 크게 능가할 수 있음을 보여줍니다.  
또한 제안된 방법은 실제 이미지 노이즈를 처리할 수 있으며 시각적으로 방해되는 결함들을 생성하지 않고도 다양한 의료 이미지 분석 작업의 성능을 향상시킬 수 있습니다.  

# Introduction
