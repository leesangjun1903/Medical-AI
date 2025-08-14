# Deep Sinogram Completion with Image Prior for Metal Artifact Reduction in CT Images  

## 1. 핵심 주장 및 주요 기여  
“Deep Sinogram Completion with Image Prior” 논문은  
- **금속 삽입체로 인해 손상된 CT 투영(신오그램)을 딥러닝으로 복원**  
- **이미지 도메인(사전 영상)과 신오그램 도메인을 공동 학습**  
- 두 개의 네트워크(PriorNet, SinoNet)를 **종단간(end-to-end)** 으로 훈련하여 상호 보완  
- 금속 아티팩트를 효과적으로 제거하면서 해부학적 구조 유지  

주요 기여  
1. 이미지 도메인과 신오그램 도메인을 결합한 **공동 학습 프레임워크** 제안  
2. 사전 영상(PriorNet)과 잔차 신오그램 학습(SinoNet residual learning) 전략 도입  
3. 시뮬레이션 및 실제 CT 데이터에서 기존 기법 대비 **RMSE 최대 6.85 HU 개선**, SSIM 향상  

***

## 2. 문제 정의 및 제안 방법  

### 2.1 해결하려는 문제  
- 금속 임플란트(치과 필링, 인공관절 등)는 CT 투영에서 심한 왜곡(별형·흐림 아티팩트) 유발  
- 선형 보간(LI), 물리 모델링, 반복 재구성 기법은 제한적 성능  
- 순수 이미지 도메인 네트워크는 학습한 패턴에 과도 적합(overfitting) 시 일반화 저하  

### 2.2 프레임워크 개요  
1. 입력  
   - 원본 금속 손상 신오그램 $$S_{ma}$$  
   - 금속 위치 마스크 $$\mathrm{Tr}$$  
2. 선형 보간(LI)으로 1차 보정 신오그램 $$S_{LI}$$ 획득  
3. **PriorNet**:  
   - 입력: 금속 손상 영상 $$X_{ma}$$와 LI 보정 영상 $$X_{LI}$$  
   - 출력: 사전 영상  

$$
     X_{prior} = X_{LI} + f_P\bigl([X_{ma},\,X_{LI}]\bigr)
   $$  
   
   - 손실: $$L_{prior} = \|X_{prior}-X_{gt}\|_1$$  

4. 사전 영상의 순방향 투영:  

$$
     S_{prior} = P(X_{prior})
   $$  

5. **SinoNet** (잔차 신오그램 보정):  
   - 입력: $$[S_{prior}-S_{LI},\,\mathrm{Tr}]$$  
   - 출력: 잔차 보정값  
   - 보정 전후 합성:  

```math
       S'\_{corr} = f_S\bigl([S_{prior}-S_{LI},\,\mathrm{Tr}]\bigr) + S_{LI},\quad
       S_{corr} = S'_{corr}\odot \mathrm{Tr} + S_{LI}\odot(1-\mathrm{Tr})
```  
   
   - 손실:  

$$
       L_{sino} = \|S_{gt}-S_{corr}\|\_1 + \beta\,\|S_{gt}-S'_{corr}\|_1
     $$  

6. **FBP 손실** (재구성 일관성):  

$$
     L_{FBP} = \|(P^{-1}(S_{corr}) - X_{gt})\odot(1-M)\|_1
   $$  

7. **총 손실**:  

$$
     L_{total} = L_{prior} + \alpha_1 L_{sino} + \alpha_2 L_{FBP}
   $$  

### 2.3 모델 구조  
- PriorNet: 채널 절반 U-Net 형태, 입력 2채널, 출력 1채널 영상  
- SinoNet: Mask Pyramid U-Net 기반, 잔차 학습(residual learning)으로 경계 연속성 강화  
- Differential forward/back-projection 모듈(ODL 라이브러리) 포함하여 종단간 학습  

***

## 3. 성능 향상 및 한계  

### 3.1 성능 비교  
- DeepLesion 데이터셋(시뮬레이션) 기준  
  - LI: RMSE 50.31 HU, SSIM 0.9455  
  - NMAR: 47.03 HU, 0.9594  
  - CNNMAR: 43.27 HU, 0.9706  
  - cGANMAR: 39.01 HU, 0.9754  
  - DuDoNet: 38.00 HU, 0.9766  
  - **본 논문**: **31.15 HU**, **0.9784**  
- 실제 CT 데이터에서도 시각적 artifact 감소 우수  

### 3.2 일반화 성능  
- 다양한 부위(복부→두부) 전이 실험에서 **추가 학습 없이** 우수한 artifact 저감  
- 잔차 신오그램 학습과 물리적 제약(FBP 손실)이 **오버피팅 억제** 및 **일반화** 도움  

### 3.3 한계  
- **시뮬레이션 데이터 의존**: 실제 프로젝션 데이터 확보 어려움  
- 금속 마스크 오분할(segmentation)에 민감, 자동화 단계 필요  
- 대형 금속 삽입체나 복잡한 임플란트 형상 일반화 추가 연구  

***

## 4. 향후 연구 방향 및 고려 사항  

- 실제 임상 투영 데이터 확보 및 평가  
- 딥러닝 기반 금속 분할/위치 검출 모듈 통합하여 완전 자동화  
- **비지도 학습** 또는 **도메인 적응(domain adaptation)** 기법으로 레이블 없는 실제 임상 데이터 활용  
- 다양한 금속 재질·크기·형상에 대한 **강건성(robustness)** 추가 검증  
- 다중 에너지(dual-energy) CT, 스펙트럼 CT 등 차세대 기법과의 융합 연구  

***

**요약**: 본 논문은 사전 영상과 잔차 기반 신오그램 보정을 결합한 딥러닝 프레임워크를 제안하여 금속 아티팩트를 효과적으로 저감하고, FBP 손실을 통해 물리적 일관성을 보장함으로써 **일반화 성능**을 크게 향상시켰다. 앞으로 실제 임상 데이터 및 자동화 모듈 통합을 통한 전임상 적용성이 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cc97bf15-975e-4629-9ad9-594fe51be520/2009.07469v1.pdf
