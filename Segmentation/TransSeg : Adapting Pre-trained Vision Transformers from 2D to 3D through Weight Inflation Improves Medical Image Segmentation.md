# TransSeg

기존 솔루션은 3D 영상을 2D 슬라이스로 분할하고 각 슬라이스를 독립적으로 예측하여 중요한 깊이 정보를 잃거나 Transformer 아키텍처를 수정하여 사전 학습된 가중치를 활용하지 않고 3D 입력을 지원합니다.  
이 작업에서는 간단하면서도 효과적인 가중치 인플레이션 전략을 사용하여 사전 학습된 Transformer를 2D에서 3D로 조정하여 전이 학습과 depth 정보의 이점을 유지합니다.

# Reference
- https://github.com/yuhui-zh15/TransSeg/tree/main?tab=readme-ov-file
