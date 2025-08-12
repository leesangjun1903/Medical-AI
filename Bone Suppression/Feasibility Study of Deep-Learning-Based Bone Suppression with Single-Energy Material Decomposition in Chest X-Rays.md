# Feasibility Study of Deep-Learning-Based Bone Suppression with Single-Energy Material Decomposition in Chest X-Rays

## 핵심 주장 및 주요 기여 요약
이 논문은 **단일 에너지 물질 분해(SEMD) 기법**을 활용해 임상적으로 손쉽게 확보 가능한 CT 데이터를 통해 **대량의 뼈·연부 조직 분리 데이터(DES 유사 영상)**를 생성하고, 이를 이용해 Residual U-Net 기반의 딥러닝 모델을 학습시켜 흉부 X선 영상에서 뼈 구조를 효과적으로 억제함으로써 폐 병변 검출 민감도를 향상시킬 수 있음을 증명한다.

- SEMD로 생성된 합성 투영 영상과 연부 조직 선택 영상 간의 차이를 학습하여 실제 흉부 X선에서도 뼈 억제가 가능한 딥러닝 모델을 제안
- 임상 CT 70건(AAPM Lung CT Challenge)으로부터 7,200장 투영 영상→소프트 티슈 레이블 영상을 얻어 모델 학습
- 실제 흉부 촬영 2예에 적용 시 폐 결절 가시성 및 이상 소견 검출 신뢰도 향상을 확인

## 해결하고자 하는 문제
흉부 X선 검사에서 갈비뼈·흉골 등 뼈 구조가 폐 병변을 가려 **진단 오류**(특히 조기 폐암 미검출)를 유발하는 문제를 해결하고자 한다.  
- 기존 Dual-Energy Subtraction(DES) 장비는 이중 노출·교정 문제, 높은 피폭량·특수 하드웨어 요구 등 실용성 제한  
- Deep learning 기반 bone suppression은 충분한 레이블 데이터 확보가 어려움  

## 제안 방법
1. **단일 에너지 물질 분해(SEMD)**  
   - CT 볼륨 데이터를 임계값·레이 트레이싱 기법으로 선 투영하여 전체 감쇠 길이 $$L=L_1+L_2$$ 산출  
   - 단일 에너지 투영 $$P_E\approx \mu_1(E_{\mathrm{eff}})L_1 + \mu_2(E_{\mathrm{eff}})L_2$$를 가정해  
  
$$
       L_1 \approx \frac{P_E - \mu_2(E_{\mathrm{eff}})L}{\mu_1(E_{\mathrm{eff}})-\mu_2(E_{\mathrm{eff}})}, 
       \quad
       L_2 \approx -\frac{P_E - \mu_1(E_{\mathrm{eff}})L}{\mu_1(E_{\mathrm{eff}})-\mu_2(E_{\mathrm{eff}})}
     $$  
   
- Soft-tissue(L₁) 및 Bone(L₂) 선택 영상 추출  

2. **Residual U-Net 모델 구조**  
   - Encoder–Decoder 방식의 U-Net에 **Residual learning** 추가  
   - Contraction 단계: 3×3 conv ×2 → 2×2 max-pool  
   - Expansion 단계: 2×2 up-conv → 3×3 conv ×2  
   - 입력 $$x$$: SEMD 합성 투영 영상, 레이블 $$y$$: SEMD 소프트 티슈 영상  
   - 손실 함수: $$\min_w \|y - N(w\mid x)\|_2^2 $$  

3. **학습 및 데이터셋**  
   - AAPM Lung CT Challenge 70건: 50건(7,200 projections) 학습, 15건 검증, 5건 테스트  
   - 학습률 1e-2→1e-3, 에포크 500, 패치 크기 512×512, Adam optimizer, GTX 1070Ti에서 17.6h 소요  

4. **성능 평가**  
   - **PSNR ≈17.9 dB**, **SSIM ≈0.90** (합성 투영 vs. 레이블)  
   - 실제 흉부 X선 2예(AP/LAT): 폐 결절 관찰 신뢰도(Test 4→Output 7점 개선)  

## 모델 구조와 성능 향상
- **Residual U-Net**으로 수렴 속도 및 억제 효과 개선  
- SEMD 기반 대량 레이블 확보로 **오버피팅 최소화**  
- 실제 데이터 적용 시 폐 병변 시각적 대비 향상 및 관찰 신뢰도 상승  

## 한계 및 일반화 성능 향상 가능성
1. **입력 각도 다양성**  
   - 합성 투영 0–360° 전 방향 활용으로 AP 위치 특화 데이터 부족  
   - Near-AP 각도 중심의 정규화된 데이터셋 구축 또는 paired Cycle-GAN 활용 시 일반화 가능성  
2. **네트워크 심층화**  
   - Residual U-Net 구조는 비교적 얕아 복잡한 뼈 억제에 한계  
   - 더 깊거나 attention 메커니즘 도입 모델로 일반화 성능 향상  
3. **CT 장비 종속성**  
   - 서로 다른 CT 스캐너 특성 혼합 시 노이즈·감쇠 특성 편차 발생  
   - 스캐너별 특화 SEMD 파이프라인 개발로 데이터 일관성 확보  

## 향후 연구 영향 및 고려 사항
- **임상 적용 확대**: 대규모 다기관 임상 영상으로 검증, 환자군 다변화  
- **모델 고도화**: GAN 기반 adversarial 학습, self-supervised 학습으로 레이블 의존도 감소  
- **피폭량 최적화**: SEMD 활용 시 CT 추가 촬영 없이 뼈 억제 영상 확보 가능성 탐색  
- **실시간 처리**: 의료 현장 실시간 bone suppression 워크플로우 구축 및 검증  

이 연구는 **의료 영상 처리 분야**에서 **뼈 억제 기반 폐 병변 검출** 가능성을 확대한 선구적 작업으로, 후속 연구에서 **데이터 정규화**, **모델 아키텍처 혁신**, **임상 검증 확대**를 통해 실제 진단 보조 시스템으로 발전할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c8dd48b8-a65c-4e53-91a6-b92ad7bcf207/bjr.20211182.pdf
