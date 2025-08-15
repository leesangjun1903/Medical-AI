# MR-self Noise2Noise: self-supervised deep learning–based image quality improvement of submillimeter resolution 3D MR images

**핵심 주장**  
스스로 감독(self-supervised) 학습만으로 부가적인 ‘깨끗한’ 레이블 없이 임상에서 흔히 얻어지는 잡음이 포함된 3D MP-RAGE 및 VWI(혈관 벽 영상) 데이터를 활용해, 딥러닝 기반으로 초미세(0.7 mm³) 해상도 MR 영상의 노이즈를 효과적으로 저감하고 구조적 디테일을 향상시킬 수 있다.

**주요 기여**  
1. Routine clinical scan만으로 ‘Noise2Noise’ 학습 쌍 생성  
   -  Raw k-space를 kz 축 방향으로 교차 샘플링해 잡음 독립 이미지 2세트 생성  
   -  ACS 라인 기반 GRAPPA 보간으로 언더샘플링 패턴 차이 보정[1]
2. 2.5D U-Net 아키텍처 적용  
   -  입력으로 연속 7 슬라이스 병합, 출력은 중앙 슬라이스 재구성  
   -  최종 출력은 원본 이미지와 네트워크 출력의 가중합:

$$ \mathrm{MRI}\_{\mathrm{sN2N}} = \alpha\,\mathrm{MRI}_{\mathrm{orig}} + (1-\alpha)\,\mathrm{NetOutput} $$  
   
   -  검증 데이터에서 PSNR 최적화로 α=0.3 선정  
3. 다양한 영상(sequence)에서의 범용성 입증  
   -  MP-RAGE 외 VWI(혈관 벽 영상)에도 동일 모델 적용 시 유의미한 화질 개선 달성  
4. 임상·정량적 평가  
   -  Qualitative Likert 점수: 전반적 화질 3.7→4.9, 세부 구조 점수 2.8→4.4 (p<0.001)  
   -  Lesion conspicuity 4.1→5.0 (p<0.03)  
   -  Quantitative: PSNR +0.5–2.3 dB, SSIM +0.002–0.047, NRMSE –0.2–5.6% (p<0.001)  
   -  Volumetry(DSC): FSL-FAST/FIRST 유사 또는 개선 (p<0.02)  

***

## 문제 정의  
임상용 3D MP-RAGE·VWI에서 submillimeter급 해상도를 얻으려면 장시간 스캔이 필수적이나, 시간 제약·모션 아티팩트로 인해 SNR 개선 한계 존재. 기존 DNN 기반 노이즈 저감 방법은 “clean” 레이블 필요하거나 별도 장시간 스캔을 요구해 임상 적용이 어려움.

***

## 제안 방법 상세  

### 1. Self-supervised 학습 데이터 생성  
-  k-space raw data를 kz 축 방향으로 alternate sampling하여 두 개의 언더샘플링 k-space 생성.[1]
-  ACS 라인으로 GRAPPA(3×2×2 kernel) 보간 수행 → IFFT + Coil combination → 두 개의 잡음 독립 이미지 획득.  

### 2. 네트워크 구조  
-  2.5D U-Net 기반:  
  – 입력: 7연속 슬라이스 (채널 dimension concat)  
  – 인코더-디코더 형태, 3×3 Conv + InstanceNorm + ReLU 블록, Max-pool 및 up-sampling  
  – 출력: 입력 시퀀스의 중앙 슬라이스 복원  
-  학습 손실: MSE between two noisy images  
-  테스트 시: conventional MR 입력 → network output → weighted sum (α=0.3)  

***

## 성능 향상  

| 평가 항목             | MP-RAGEoriginal | MRIsN2N       | p-value    |
|-----------------------|-----------------|---------------|------------|
| Overall Likert score  | 3.7 ± 0.5       | 4.9 ± 0.3     | <0.001     |
| PSNR (dB, no noise)   | 41.2 ± 0.8      | 41.7 ± 0.9    | <0.001     |
| SSIM                  | 0.984 ± 0.003   | 0.986 ± 0.003 | <0.001     |
| NRMSE (%)             | 4.4 ± 0.6       | 4.2 ± 0.6     | <0.001     |
| FSL-FAST DSC          | 0.955 ± 0.006   | 0.956 ± 0.006 | 0.10       |
| Lesion conspicuity     | 4.1 ± 0.7       | 5.0 ± 0.0     | <0.03      |

***

## 모델의 일반화 성능 향상 가능성  
- **Convolution 기반 로컬 노이즈 학습**: 필터가 국소적 노이즈 특성을 학습하므로, 다른 sequence·파라미터 데이터에도 적용 가능  
- **Fine-tuning**: 소량의 새로운 modality(VWI 등) 데이터를 이용한 전이학습으로 추가 성능 개선 여지  
- **Data augmentation**: 다양한 가우시안 노이즈 레벨·패턴 추가 시, 폭넓은 SNR 환경에서 더 안정적 성능 확보 가능  

***

## 향후 연구 영향 및 고려사항  
1. **Supervised 대비 성능 비교**: ‘clean’ 레이블 기반 모델과 직접 비교 연구 필요  
2. **다양한 장비·시퀀스 검증**: 다른 제조사·coil·field strength 임상 데이터에서 재현성 평가  
3. **저 SNR 경계조건 탐색**: Noise2Noise의 zero-mean noise 가정이 깨지는 저 SNR 환경에서 한계 규명  
4. **Phase 정보 적용**: SWI·QSM 등 위상 기반 영상으로 확장  
5. **대규모 임상 평가**: 다양한 병변·질환 데이터로 효과성과 안정성 검증  

이 논문은 임상에서 추가 스캔 불필요한 self-supervised MR 노이즈 저감 프레임워크를 제시함으로써, 고해상도·고SNR 영상을 보다 보편적이고 효율적으로 구현할 수 있는 가능성을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/457ad376-5f3a-494c-b99d-56f787510952/s00330-022-09243-y.pdf
