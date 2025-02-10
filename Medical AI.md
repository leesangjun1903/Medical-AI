# 의료 인공지능의 모든 것 | Medical AI
https://standing-o.github.io/posts/medical-ai-survey/

## Machine Learning, Deep Learning

## Learning Problem, Hybrid Learning Problem

## Learning Techniques


# Task Definition in Medical Imaging

## 분류 Classification
주로 질병/암 진단과 같은 이진 분류에 대한 것이며, 의료 데이터의 경우 보통 일반 컴퓨터 비전 데이터 셋의 양보다 적기 때문에 (~수백/수천), Transfer learning 기법을 활용해 데이터 셋이 부족하다는 문제를 해결할 수 있습니다.

## Detection
Detection Task는 주로 장기, 병변 및 랜드마크 위치를 지정해주는데 쓰이며, 특히 3D 데이터 파싱을 위해 다양한 접근 방식이 제안되고 있습니다.  
ex. 3D 영역을 2D 직교 평면의 조합으로 처리함, 3개의 독립적인 2D MRI 슬라이스로 원거리 대퇴 표면의 랜드마크 식별, 2D CT 볼륨 파싱 후 3D 경계 상자를 식별하여 심장, 대동맥 궁, 하행 대동맥 근처의 ROI 식별.

### False Positive Detection3
정상이지만 비정상으로 간주되는 픽셀로, CAD 시스템의 민감도를 줄이면서 잘못된 의료적 개입을 야기합니다.

## Segmentation
장기 및 하위 구조를 분할하여 부피 및 형태와 관련된 임상 파라미터의 정량적 분석을 가능하게 합니다.

## Registration
한 의료영상을 다른 의료영상에 공간적으로 맞추는 과정이며, 일반적으로 두 영상을 비교하여 유사도를 계산하고 이를 최적화하여 정합을 수행합니다.

## Localization
2D 및 3D 공간, 그리고 시간(4D)에서 장기나 다른 기관의 위치를 인식합니다.

## Content-based Image Retrieval
유사한 사례 기록 식별, 희귀 장애 이해, 환자 치료 개선을 위해 수행됩니다.  
주로 픽셀 수준 정보에서 효과적인 특징 표현 추출하거나, 이를 의미 있는 개념과 연관시키기 위해 활용됩니다.

## Image Generation and Enhancement
이미지 생성 및 향상 기법은 장애 요소 제거, 이미지 정규화, 이미지 품질 향상, 데이터 완성, 패턴 발견 등의 사례에 활용될 수 있습니다.

## Text Report
주로 리포트를 텍스트 라벨로 사용하여 텍스트 설명과 이미지를 같이 학습하여, 테스트 시에는 의미있는 클래스 라벨 예측을 가능케 합니다.

## Application Areas

![image](https://github.com/user-attachments/assets/7f47f99c-d779-4e6f-84f4-5d195824cbe2)

# Challenges
대규모 데이터셋 부족, Labeling(Annotation) 어려움, 라벨 노이즈 문제, 이진 분류의 한계, 클래스 불균형, 추가 정보 통합, ...

3D 흑백이나 다중 채널 이미지와 같은 대부분의 의료 영상에서는 사전 훈련된 네트워크나 아키텍처가 존재하지 않아 새롭게 개발된 네트워크가 필요합니다.  
의료 영상에는 비등방성 복셀 크기, 다양한 채널 간의 작은 등록 오류, 다양한 강도 범위와 같은 고유한 문제가 있습니다.  
분류 문제로 제기할 수 없는 작업들도 종종 존재하며, 이 경우 비딥러닝 방법(Counting, Segmentation, Regression)으로 후처리가 필요합니다.
