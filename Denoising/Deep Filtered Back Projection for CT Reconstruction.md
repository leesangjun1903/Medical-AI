# Deep Filtered Back Projection for CT Reconstruction

## 1. 핵심 주장 및 주요 기여 요약  
이 논문은 **기존의 분석적 CT 재구성 기법인 Filtered Back Projection(FBP)의 효율성**은 유지하면서, **딥러닝 기반으로 최적화된 필터와 비선형 보간 연산자를 학습**하여 저선량(low-dose) 및 희소 투영(sparse-view) 환경에서의 재구성 품질을 크게 향상시킨 **DeepFBP** 프레임워크를 제안한다.  
- **학습 가능한 윈도우 함수**: 전통적인 램-락(Ram–Lak) 필터에 윈도우 함수를 곱한 필터를, 네트워크로 최적화하여 잡음 억제와 해상도 보존의 균형을 자동 학습.  
- **학습 가능한 비선형 보간 연산자**: 백프로젝션 시 선형 보간의 한계를 극복하기 위해, 주변 투영 값을 비선형 결합하는 1D 컨볼루션 기반 네트워크로 구현.  
- **경량 구조**: 전체 학습 파라미터 수가 0.61M 미만으로, RED-CNN(1.1M) 대비 절반 수준이며, 계산 속도는 기존 FBP 대비 거의 동일하게 유지.  

## 2. 문제 정의, 제안 기법, 모델 구조, 성능 및 한계  
### 2.1 해결하고자 하는 문제  
- 저선량 CT에서 FBP는 속도는 빠르나, 잡음과 링 아티팩트가 심해 진단 품질 저하.  
- 통계적 반복 재구성(statistical iterative reconstruction)은 품질은 높으나 계산 비용이 매우 큼.  
- 딥러닝 후처리(post-processing) 방식은 물리 모델을 이용하지 않아 한계.  

### 2.2 제안 방법  
DeepFBP는 다음 세 모듈로 구성되어 end-to-end 학습된다.  
1. **학습 가능한 필터 $$h(\omega)$$**  
   - 전통적 필터:  

$$
       \text{filter}_0(\omega) = |\omega|.
     $$  

   - 윈도우 추가:

$$
       \text{filter}(\omega) = h(\omega)\cdot|\omega|,
     $$  
  
  여기서 $$h(\omega)$$는 네트워크가 직접 최적화하는 벡터 파라미터.  
   - 전략 I: 모든 투영 각도에 동일 필터 적용 (Filter I)  
   - 전략 II: 각각의 투영 각도별로 개별 필터 적용 (Filter II)  

2. **학습 가능한 비선형 보간 $$T$$**  
   - 입력 크기: $$(L\times360)$$ sinogram  
   - 구조: 3개의 1D depth-wise convolution + BN + PReLU residual block  
   - 보간:  

$$
       p_{x_i,y_i,\theta} = (1-z)\,T(p)\_a + z\,T(p)_{a+1},
     $$  

$$\ a = \lfloor x_i\cos\theta + y_i\sin\theta\rfloor$$.  

3. **CNN 기반 후처리**  
   - Lim et al.의 Residual Super-Resolution 네트워크 간소화 버전 (3 residual blocks + 3 convolution blocks)  

### 2.3 모델 구조  
```
Sinogram → 1D FFT → Learned Filter → 1D IFFT → Nonlinear Interpolation Network → Backprojection → Post-processing CNN → Output CT
```
- 학습 단계: L2 손실로 정상선량 CT와 직접 비교하여 end-to-end 업데이트(필터, 보간, 후처리 동시 최적화).  
- 파라미터: DeepFBP I ≈ 0.26M, DeepFBP II ≈ 0.61M.  

### 2.4 성능 향상  
AAPM 저선량 CT 챌린지 데이터셋(12명)에서 세 가지 시나리오 실험:  
1. **저선량 투영 (360각)**  
   - DeepFBP II가 전통 FBP 대비 PSNR +2.9 dB, FBPConvNet 대비 +1.8 dB 향상.  
2. **잡음 추가 정상선량 투영**  
   - DeepFBP II가 기존 방법들 중 최고 PSNR 달성.  
3. **희소 투영 저선량 (90각)**  
   - DeepFBP II가 TV 기반 반복 재구성 대비 +5 dB, FBPConvNet/RED-CNN 대비 +1 dB 상승.  
- **계산 속도**: GPU 환경에서 FBP 대비 +1.3 ms 오버헤드, TV 반복보다는 100배 이상 빠름.  

### 2.5 한계  
- **노이즈 레벨 민감도**: 필터는 훈련 시 설정한 잡음 조건에 특화되어, 새로운 노이즈 조건에서 재학습 필요.  
- **투영 각도·크기 의존성**: 희소·비균등 투영 스캔 시 일반화 성능 확인 필요.  

## 3. 모델 일반화 성능 향상 가능성  
- **Self-adaptive Filtering**: 노이즈 분포 예측 모듈 추가로 필터 파라미터를 투영별·환자별로 동적으로 조정하여 다양한 잡음 레벨에 대응.  
- **Multi-task Learning**: 정상선량·저선량·희소 투영 데이터를 함께 학습해 보간 네트워크의 표현력을 강화.  
- **데이터 증강 및 도메인 적응**: 시뮬레이션 기반 다양한 스캔 조건(관전압, 튜브 전류)으로 훈련하여 실제 임상 스캐너 차이를 극복.  
- **Uncertainty Estimation**: Bayesian 네트워크로 예측 신뢰도를 출력, 이상 케이스 검출 및 재학습 가이드 제공.  

## 4. 향후 연구 영향 및 고려사항  
- **물리·데이터 융합 재구성**: DeepFBP는 분석적 알고리즘과 딥러닝 융합의 성공 사례로, 다른 영상 모달리티(MRI, PET)에도 적용 가능성을 제시.  
- **임상 적용 검증**: 실제 스캐너 데이터, 다양한 신체 부위, 병변 유형에서 정밀 평가 필요.  
- **경량화 및 실시간 처리**: GPU 없는 임상 환경을 고려한 모델 압축·양자화 및 하드웨어 가속 연구.  
- **해석 가능성(interpretable AI)**: 학습된 필터·보간 함수의 주파수/공간 도메인 특성을 분석하여 영상 의학 전문가에게 신뢰 제공.  
- **규제 승인**: 의료기기 인증 기준 만족을 위해 학습 데이터 관리, 알고리즘 버전 관리, 안전성·효용성 문서화 필수.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f4b2b435-0e24-4c28-9d8a-46a2ed99d19b/Deep_Filtered_Back_Projection_for_CT_Reconstruction.pdf
