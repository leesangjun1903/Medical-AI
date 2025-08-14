# Fast Enhanced CT Metal Artifact Reduction Using Data Domain Deep Learning

## 1. 핵심 주장 및 주요 기여  
이 논문은 고밀도 금속 물체가 존재할 때 발생하는 CT 영상의 심각한 금속 아티팩트(스트리킹 현상)를 해결하기 위해, 투영 데이터(domain)에 직접 딥러닝 기반의 보간(completion) 기법을 적용한 **Deep-MAR**(Deep Metal Artifact Reduction) 프레임워크를 제안한다. 주요 기여는 다음과 같다.  
1. **Deep-MAR 프레임워크**: 금속 오브젝트가 오염시킨 투영선(sinogram) 구간을 마스킹하고, 그 결측 데이터를 CGAN(Conditional GAN)으로 보완하여 잔여 아티팩트를 최소화.  
2. **CGAN 기반 보간**: U-Net 유사 구조의 생성기(Generator)와 전체 sinogram을 판별하는 판별기(Discriminator)를 결합한 CGAN으로, ℓ₂ 손실과 적대적 손실(adversarial loss)을 조합한 목적함수(1)로 학습.  
3. **시뮬레이션+전이학습**: 실제 보안 스캐너 데이터를 한정적으로만 확보할 수 있는 제약을 극복하기 위해, 물리 기반 X선 시뮬레이션으로 대규모 합성훈련세트를 생성한 뒤 소량의 실제 데이터로 전이학습(transfer learning) 수행.  
4. **효율성 및 실용성**: Deep-MAR는 GPU에서 sinogram 보간을 58ms에 처리하고, CPU 기반 FBP 재구성과 결합 시 실시간 워크플로우에 적합.

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- 금속 물체가 강하게 X선을 감쇠 또는 차단하면서 투영선에 결손이 발생 → FBP(Filtered Back Projection) 재구성 시 심각한 스트리킹 아티팩트 유발  
- 기존 보간(LI-MAR)·가중 보간(WNN-MAR)·이미지 후처리 기법 한계: 대규모 금속 영역·다양한 구성에서 성능저하

### 2.2 제안 방법  
1) 금속 검출 및 마스킹  
   - 원 sinogram으로 FBP 재구성 → 화소값 임계치(thresholding) + 형태학적 연산으로 금속 분할 → 투영도(domain)에서 금속영역 M 마스크 획득  
2) Sinogram 보완 via CGAN  
   - 입력 $$x$$: 결손 영역 삭제된 sinogram, 정답 $$y$$: 금속 없는 완전 sinogram  
   - 목적함수  

$$
       G^* = \arg\min_G ; \max_D \mathcal{L}\_{cGAN}(G,D) + \lambda \, \mathbb{E}_{x,y}\bigl\|y - G(x)\bigr\|_2^2
     $$  

$$
       \mathcal{L}\_{cGAN}(G,D) = \mathbb{E}_{x,y}\bigl[\log D(x,y)\bigr] + \mathbb{E}_x\bigl[\log(1 - D(x, G(x)))\bigr]
     $$  
     
  - $$\lambda=10$$로 ℓ₂ 손실과 판별기 손실 균형  
   - Generator: 5×5 커널, 스트라이드 2 컨볼루션 6단계·업샘플링 6단계, U-Net 스타일 스킵 커넥션  
   - Discriminator: 전체 sinogram 입력을 받아 진위 판별 (패치 기반이 아님)

3) FBP 재구성 및 금속 재삽입  
   - 보완된 sinogram + 원래 금속 트레이스로 FBP → 최종 아티팩트 저감 영상

### 2.3 성능 향상  
- **시뮬레이션 데이터** (510 케이스)  
  - Sinogram 보완 MSE: LI-MAR 대비 89%↓, WNN-MAR 대비 87%↓  
  - 재구성 MSE: LI-MAR 대비 74%↓, WNN-MAR 대비 78%↓  
  - SSIM·PSNR 대폭 개선  
- **실제 데이터** (159 케이스)  
  - Sinogram 보완 MSE: LI-MAR 대비 80%↓, WNN-MAR 대비 75%↓  
  - 재구성 MSE: LI-MAR 대비 59%↓, WNN-MAR 대비 68%↓  
- **전이학습 효과**: 실제 데이터만으로 학습한 모델 대비 MSE 23–31% 추가 절감  
- **계산비용**: 보완 58ms (GPU), FBP 1.55s (CPU)

### 2.4 한계  
- 2D 평행빔 시뮬레이션에 집중했으며, 실제 3D 원뿔빔 CT 확장 필요  
- 산란(scatter) 모델 미포함(계산 비용 고려)  
- 금속 분할 단계 임계치·모폴로지 기법 의존 → 분할 오류는 보완 성능에 영향  
- 학습 데이터: 시뮬레이션·제한적 실제샘플 위주 → 매우 이질적 물체·조건 일반화 불확실

***

## 3. 모델 일반화 성능 향상 관점  
- **전이학습**: 대규모 합성 데이터에서 습득한 표현을 소량 실제 데이터로 빠르게 적응시켜, 도메인 간 편차 감소  
- **Attention Map 분석**: sinogram 내 근접·광역 정보를 적응적으로 활용 → 고정 창 기반 보간보다 다양한 금속 구성에 대응  
- **Latent Space 시각화**: t-SNE 상에서 유사 장면이 군집화되어, CGAN 인코더가 의미적 특징을 잘 학습함을 확인  
- 향후 **도메인 적응(unsupervised domain adaptation)**, **Semi-/Self-supervised 학습** 기법 도입으로 실제 환경 일반화 강화 가능  

***

## 4. 향후 연구 영향 및 고려사항  
- **보안·의료 CT** 전 영역에 Deep-MAR 파이프라인 적용: 3D/원뿔빔·다중에너지 CT 확장  
- **금속 분할**: DL 기반 세그멘테이션으로 분할 성능 고도화  
- **산란·잡음 모델링**: Monte Carlo 기반 산란 시뮬레이션 결합으로 실제성 강화  
- **자율 학습**: 실제 금속 미포함 장면에서 자가 생성 마스킹 활용한 준지도 학습으로 레이블링 부담 완화  
- **안정성·검증**: 임상·보안 현장 데이터로 대규모 검증, 극단적 케이스에 대한 안정성 확보  

이를 통해 금속 아티팩트 저감 분야의 **딥러닝 기반 데이터 도메인 보간** 연구가 활발해지며, 실용적 CT 워크플로우에 통합 가능한 새로운 표준으로 발전할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/54f26265-2b8c-4311-9cd8-0e9d2b18891d/Fast_Enhanced_CT_Metal_Artifact_Reduction_Using_Data_Domain_Deep_Learning.pdf
