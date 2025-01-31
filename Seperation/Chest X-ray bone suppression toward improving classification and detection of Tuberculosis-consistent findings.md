# Chest X-ray bone suppression toward improving classification and detection of Tuberculosis-consistent findings
이 연구에서는 단계별 체계적 방법론을 제안합니다.  
첫째, ImageNet으로 훈련된 VGG-16 모델을 공개적으로 사용 가능한 CXR의 대규모, 다양한 결합 선택에서 재훈련하여 CXR 모달리티별 기능을 학습하도록 돕습니다. 학습된 지식은 공개적으로 사용 가능한 심천 및 몽고메리 TB 컬렉션에서 정상 폐 또는 폐결핵 증상을 보이는 CXR을 분류하는 관련 대상 분류 작업에서 성능을 개선하는 데 전달됩니다. 다음으로, 일본 방사선 기술 협회(JSRT) CXR 데이터 세트와 뼈 억제된 대응 데이터 세트에서 다양한 아키텍처를 가진 여러 뼈 억제 모델을 훈련합니다. 훈련된 모델의 성능은 기관 간 국립 보건원(NIH) 임상 센터(CC) **이중 에너지 감산(DES)** CXR 데이터 세트를 사용하여 테스트합니다. 성능이 가장 좋은 모델은 심천 및 몽고메리 TB 컬렉션에서 뼈를 억제하는 데 사용됩니다. 그런 다음 우리는 여러 성능 지표를 사용하여 뼈 억제되지 않은 몽고메리 TB 데이터 세트와 뼈 억제된 몽고메리 TB 데이터 세트로 훈련된 CXR 재훈련된 VGG-16 모델의 성능을 비교하고 통계적으로 유의미한 차이를 분석했습니다. 뼈 억제되지 않은 모델과 뼈 억제된 모델의 예측은 클래스 선택적 관련성 맵(CRM)을 통해 해석됩니다.

# Reference
https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression
