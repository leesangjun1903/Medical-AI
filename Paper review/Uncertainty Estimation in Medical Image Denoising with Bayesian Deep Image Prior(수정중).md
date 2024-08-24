# Uncertainty Estimation in Medical Image Denoising with Bayesian Deep Image Prior

# MCDIP
# Bayesian DIP

# Abs

딥 러닝을 사용한 inverse medical imaging의 불확실성 정량화는 거의 주목을 받지 못했습니다.  
그러나 대규모 데이터 세트로 훈련된 심층 모델은 해부학적으로 존재하지 않는 재구성된 이미지에서 환각을 일으키고 결함을 생성하는 경향이 있습니다.  
저희는 무작위로 초기화된 컨볼루션 네트워크를 재구성된 이미지의 매개변수로 사용하고 관찰한 이미지와 일치하도록 경사 하강을 수행하며, 이를 deep image prior라고 합니다.  
이 경우 사전 훈련이 수행되지 않기 때문에 재구성할 때 환각을 겪지 않습니다.  
저희는 이를 Monte Carlo dropout을 사용한 Bayesian approach 방식으로 확장하여 우연에 의한 확실성 및 인식적 불확실성을 모두 정량화합니다.  
제시된 방법은 다양한 의료 영상 양식의 노이즈 제거 작업에 대해 평가됩니다.  
실험 결과는 저희의 접근 방식이 잘 보정된 불확실성을 가져온다는 것을 보여줍니다.  
즉, 예측 불확실성은 예측 오류와 상관관계가 있습니다.  
이를 통해 신뢰할 수 있는 불확실성 추정이 가능하며 inverse medical imaging 작업에서 환각 및 결함 문제를 해결할 수 있습니다. 


# Ref
https://aistudy9314.tistory.com/47
