# MR Image Denoising and Super-Resolution Using Regularized Reverse Diffusion

## 1. 핵심 주장 및 주요 기여
이 논문은 기존 MMSE 기반 딥러닝 denoiser가 가지는 *블러링* 문제와 *분포 편향* 민감성을 극복하기 위해, **Score-based Diffusion Model**의 후방 확산(reverse diffusion) 과정을 활용한 새로운 MRI 노이즈 제거 및 초해상화 프레임워크(R2D2+)를 제안한다.  
주요 기여:
- **유연한 노이즈 제거 제어**: reverse diffusion 진행 단계 수를 조정하는 하이퍼파라미터 α로 임상의가 자율적으로 노이즈 제거 정도를 선택 가능.  
- **저주파수 정규화**: k-공간 저주파 성분을 고주파 교정 과정에 지속 주입하여 구조 보존 및 과한 왜곡 방지.  
- **사후(super-)해상화**: 동일 score 네트워크로 denoising 후 *데이터 일관성(data consistency)* 을 적용, 고해상도 영상 복원.  
- **불확실성 정량화**: 여러 posterior 샘플링을 통해 픽셀별 분산 맵을 생성, 임상적 의사결정 지원.  

## 2. 문제 정의 및 제안 방법
### 해결하고자 하는 문제
- 실제 MRI 스캔의 노이즈 분포는 단순 Gaussian 가정에서 벗어나며, MMSE 기반 denoiser는 *블러링*과 *OOD(Out-of-Distribution)* 데이터에 취약함.

### 제안 방법 개요
1. **Noise Level Estimation**  
   복원 전 노이즈 분산 σ_est를 비매개적(eigenvalue) 방법으로 추정.  
2. **Reverse Diffusion Denoiser**  
   초기 노이즈 영상 $$x_{N'}$$ 에서 후방 확산 과정을 $$N'=\alpha\,t'\,N$$ 단계만 수행:  
   
$$
   x_i \leftarrow x_{i+1} + (\sigma_{i+1}^2-\sigma_{i}^2)s_\theta(x_{i+1},\sigma_{i+1})
   +\sqrt{\sigma_{i+1}^2-\sigma_i^2}\,z,\ z\sim\mathcal N(0,I)
   $$  

3. **Low-frequency Regularizer**  
   각 iteration마다  
   
$$
   x_i \leftarrow \lambda\,\mathcal F^{-1}P_\Omega\mathcal F\,x_{N'}
   +(1-\lambda)\,x_i
   $$  
   
$$\lambda\in$$는 저주파 주입 강도, $$P_\Omega$$는 저주파 영역 마스크[1]

4. **Post-hoc Super-Resolution**  
   denoising 결과 $$x_0$$ 기반으로 forward diffusion 후 소수의 reverse diffusion과  
   
$$
   \hat x_i \leftarrow (I-P)\hat x'_i + x_0
   $$  
   
과정을 반복하여 고해상도 영상 복원.

### 모델 구조
- **Score Network**: VE-SDE 기반 ncsnpp 아키텍처, Fourier feature로 시간 조건부 학습  
- **학습 세부**: σ_min=0.01, σ_max=378, Adam 최적화, linear warm-up, EMA(0.9999), batch=2  

## 3. 성능 향상 및 제한점
### 성능 향상
- **SNR/CNR**: 제안 기법 R2D2+는 BM3D, Noise2Noise, Noise2Score 대비 SNR·CNR 모두 유의미하게 향상.  
- **구조 보존**: 저주파 정규화로 혈관 구조 변경 및 인공물 최소화.  
- **유연성**: α 조절로 약한/강한 denoising 선택 가능.  
- **불확실성 시각화**: posterior 샘플링 기반 분산 지도 제공.

### 한계 및 고려사항
- **계산 비용**: 수십 회의 네트워크 순전파 필요, 실시간 임상 적용 시 최적화 필요.  
- **노이즈 추정 정확도**: 매우 비정형 노이즈 분포에는 σ_est 오차로 인한 과소/과다 denoising 발생 가능.  
- **해상도 복원 시 SNR 소폭 감소**: SR 단계에서 SNR이 약간 낮아짐(Table III).  

## 4. 일반화 성능 향상 관점
- *분포 이동*에 강건: 훈련 데이터(무릎 MRI)와 완전히 다른 실제 간 MRI에서도 우수한 성능 입증.  
- *비파라메트릭* 노이즈 추정과 posterior 샘플링 결합으로, Gaussian 가정 위배에도 안정적 동작.  
- 저주파 정규화와 non-expansive mapping 이론 적용으로 OOD 데이터에 구조 손상 없이 수렴을 보장.

## 5. 향후 연구 영향 및 고려사항
- **임상 적용**: 계산 속도 최적화 및 GPU 가속화로 실시간 denoising 시스템 구현  
- **노이즈 모델 확장**: 복합·비정규 노이즈 추정 기법 개선, 이상치 노이즈 대응 연구  
- **다양한 영상 모달리티**: CT·초음파 등 다른 의료 영상을 위한 diffusion 기반 inverse problem 확장  
- **추가 불확실성 정량화**: 분포 기반 리스크 평가 및 Bayesian 적응형 α 선택 전략 개발

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/efca1e09-1dd7-444a-8fa2-bf09882f2354/2203.12621v1.pdf
