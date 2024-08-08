# CAMixerSR: Only Details Need More “Attention”

# CAMixerSR

# Abs
CAMixerSR은 이미지 초해상도를 위한 콘텐츠 인식 믹서로, 다양한 SR 작업에서 우수한 성능을 보입니다.  
CAMixer는 복잡한 영역에는 자기 주의를, 간단한 영역에는 합성곱을 할당하여 모델 가속 및 토큰 믹싱 전략을 통합합니다.  
CAMixerSR은 가벼운 SR, 대형 입력 SR 및 전방향 이미지 SR에 대해 우수한 성능을 보이며 SOTA 품질-계산 교환을 달성합니다.  

대규모 이미지(2K-8K) 초해상도(SR)에 대한 수요가 빠르게 증가하고 있는 상황을 충족시키기 위해 기존 방법은 두 가지 독립적인 트랙을 따릅니다:  
1) content-aware routing을 통해 기존 네트워크를 가속화하고  
2) token mixer refining를 통해 더 나은 초해상도 네트워크를 설계합니다.  

이럼에도 불구하고 품질과 복잡성 균형의 추가적인 개선을 제한하는 피할 수 없는 결함(예: 유연하지 않은 경로 또는 비차별적 처리)에 직면합니다.
이러한 단점을 지우기 위해 간단한 컨텍스트에 대해서는 컨볼루션을 할당하고 희소한 텍스처에 대해서는 추가적인 변형 가능한 window-attention를 할당하는 content-aware mixer(CAMIXer)를 제안하여 이러한 체계를 통합합니다.
특히, CAMixer는 학습 가능한 예측기를 사용하여 windows warping을 위한 오프셋, 윈도우를 분류하기 위한 마스크, dynamic property를 가진 convolutional attentions를 포함한 여러 bootstraps을 생성하며, 이는 보다 유용한 텍스처를 스스로 포함하도록 attention를 조절하고 컨볼루션의 표현 기능을 향상시킵니다.
이 모델은 예측기의 정확도를 향상시키기 위해 global classification loss을 도입합니다. 단순히 CAMixer를 적층하여 대규모 이미지 SR, lightweight SR 및 모든 방향을 가지는 이미지 SR에서 우수한 성능을 달성하는 CAMixerSR을 얻습니다.



# Reference
- https://linnk.ai/insight/%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC/camixersr-content-aware-mixer-for-image-super-resolution-3DtqCD_s/
