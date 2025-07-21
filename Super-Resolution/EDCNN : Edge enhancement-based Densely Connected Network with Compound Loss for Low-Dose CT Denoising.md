# EDCNN : Edge enhancement-based Densely Connected Network with Compound Loss for Low-Dose CT Denoising

**핵심 주장 및 주요 기여**  
본 논문은 저선량 CT(LDCT) 이미지의 과도한 노이즈 및 과도한 평활화 문제를 동시에 해결하기 위해,  
1. **Trainable Sobel 기반의 엣지 강화 모듈**(Edge Enhancement Module)  
2. **Dense Connection을 활용한 완전 합성곱 네트워크 구조**(EDCNN)  
3. **MSE 손실과 다중 스케일 지각 손실(Perceptual Loss)의 결합**(Compound Loss)  
를 제안한다. 이를 통해 노이즈 억제와 세부 구조 보존을 균형 있게 개선하였다.

## 1. 문제 정의  
- **저선량 CT 이미지**는 방사선량을 줄이는 대신 노이즈가 증가하여 진단 품질이 저하된다.  
- 기존 CNN 기반 복원 기법들은 MSE 기반 학습 시 *과도한 평활화*(over-smoothing), 지각 손실 기반 학습 시 *잡음 무늬*(texture-like artifacts)가 발생하는 한계가 있다.

## 2. 제안 방법  
### 2.1. Edge Enhancement Module  
- 전통적 Sobel 필터를 **학습 가능한 Sobel factor** α로 확장하여 수직·수평·대각선 방향의 경계 강조 필터(4종)를 **그룹**으로 구성한다.  
- 입력 LDCT 이미지에 다수(32개) 그룹의 trainable Sobel convolution을 적용하여 경계(feature map)를 추출한 뒤, 이를 원본 채널에 concatenate하여 후속 블록에 공급한다.  

### 2.2. EDCNN 네트워크 구조  
- **완전 합성곱 FCN 구조**로,  
  -  엣지 모듈 출력을 각 합성곱 블록에 **dense skip-connection**으로 연결  
  -  8개의 중첩된 `(1×1 conv → 3×3 conv)` 블록 (각 필터 수 32)  
  -  마지막 3×3 필터는 채널 1 출력  
  -  **Residual 연결**로 원본 LDCT와 네트워크 출력 합산  
- DenseNet의 아이디어를 차용하여, 엣지 정보와 원본 정보를 깊은 네트워크 전반에 전달해 세부 보존 강화  

### 2.3. Compound Loss  
- Per-pixel MSE 손실:  
  $$L_{mse}=\frac1N\sum_{i=1}^N\|F(x_i;\theta)-y_i\|^2$$  
- 다중 스케일 지각 손실 (ResNet-50 기반):  
  $$L_{multi-p}=\frac1{N\,S}\sum_{i=1}^N\sum_{s=1}^S\|\phi_s(F(x_i;\theta))-\phi_s(y_i)\|^2$$  
  -  φ_s: ResNet-50의 s번째 스테이지(feature map)  
- 두 손실의 가중 결합:  
  $$L_{compound}=L_{mse}+w_p\,L_{multi-p},\quad w_p=0.01$$  

## 3. 성능 향상  
| 모델        | PSNR (dB)        | SSIM           | RMSE            | VGG-P (perceptual) |
|-------------|------------------|----------------|-----------------|--------------------|
| LDCT        | 36.76 ± 0.97     | 0.9465 ± 0.011 | 0.0146 ± 0.0016 | 0.0377 ± 0.0055    |
| REDCNN   | **42.39** ± 0.76 | 0.9856 ± 0.0029| **0.0076**±0.0007| 0.0218±0.0048      |
| WGAN    | 38.60 ± 0.95     | 0.9647 ± 0.0078| 0.0108 ± 0.0013 | 0.0072 ± 0.0019    |
| CPCE    | 40.82 ± 0.79     | 0.9740 ± 0.0050| 0.0093 ± 0.0009 | 0.0043 ± 0.0011    |
| **EDCNN**   | 42.08 ± 0.81     | **0.9866**±0.0031| 0.0079±0.0007   | **0.0061**±0.0014  |

- **PSNR/RMSE**: MSE 기반 모델(REDCNN)이 높으나,  
- **SSIM/VGG-P**: EDCNN이 최고 성능 달성 → *시각적 세부 구조 보존과 잡음 억제의 균형* 실현[Table I].  
- **주관적 평가**(20명 판독, 5점 척도)에서도 EDCNN이 종합 품질 최고점 획득(Table II).

## 4. 모델의 일반화 성능 향상 잠재력  
1. **Trainable 엣지 모듈**은 CT 스캔 설정·환자 체격·촬영 부위별 경계 특성이 달라도 α 파라미터 학습을 통해 *동적 적응* 가능.  
2. **Dense Skip-connection**은 정보 소실을 최소화하여 작은 병변·미세 구조에서도 과적합 없이 세부 보존.  
3. **Compound Loss**의 가중치 조정(w_p)으로 MSE⇆Perceptual emphasis 비율을 데이터셋에 맞춰 재조정할 수 있어 타 기관·장비 데이터셋에서도 *손쉬운 튜닝* 기대.  

## 5. 한계 및 향후 연구 고려사항  
- **합성 데이터 의존성**: 실험은 합성 쿼터 선량 데이터(AAPM-Mayo 챌린지)로 진행. 실제 초저선량·임상 CT에서의 성능 검증 필요.  
- **추가 손실 함수**: GAN 기반 adversarial loss나 edge-coherence loss 통합을 통한 현실감 증대 연구 여지.  
- **실시간 적용**: 8블록 구조로 연산량 다소 높음. 경량화 또는 지연시간 평가 필요.  
- **다양한 해부학적 부위**: 폐·심장·뇌 등 조직별 noise 특성 차이에 따른 전이학습·fine-tuning 전략 연구 권장.

## 6. 향후 연구에 미치는 영향  
- **적응형 엣지 강조**: 다양한 의료 영상 복원·초해상도에도 trainable Sobel 모듈 적용 가능성.  
- **다중 스케일 Perceptual Loss**: ResNet-50 기반 다중 스케일 지각 손실은 비의료 영상 변환에도 일반화 잠재력.  
- **구조적 정보 보존**: Dense 연결+Residual 구조의 결합은 *의료 세부 표현 강화* 분야의 표준 설계로 자리매김 기대.  

앞으로 본 연구의 **적응적 엣지 학습 기법**과 **Compound Loss** 프레임워크를 바탕으로, 실제 임상 환경에서의 정량·정성 평가, 경량화 모델 개발, 다양한 횡단면·다중 모달리티 적용 연구가 더욱 활발해질 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3dbabe9d-884c-4512-b82c-20cfd18a948a/2011.00139v1.pdf
