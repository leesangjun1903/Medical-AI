# 2-Step Sparse-View CT Reconstruction with a Domain-Specific Perceptual Network

## 1. 핵심 주장과 주요 기여

### **핵심 주장**
본 논문은 **sparse-view CT 재구성에서 두 단계 접근법이 단일 단계 네트워크보다 우수하다**는 것을 주장합니다. 특히 **sinogram domain과 reconstruction domain 각각의 특성을 활용**하여 최대한의 세부사항을 보존하면서 streak artifact를 효과적으로 제거할 수 있다고 제시합니다.

### **주요 기여**
- **2단계 재구성 프레임워크**: 먼저 SIN(Sinogram Inpainting Network)으로 sparse sinogram을 super-resolution하고, 이후 PRN(Postprocessing Refining Network)으로 잔여 artifact를 제거
- **Domain-Specific Perceptual Loss (DP)**: VGG 기반 perceptual loss보다 정확도, 메모리, 시간 효율성이 우수한 경량 perceptual loss 제안
- **CT 도메인 특화 모듈**: Two-Ends sinogram flipping, High-Frequency loss 등 tomography 특성에 맞춘 손실함수 개발
- **4dB 이상의 성능 향상**: 기존 state-of-the-art 대비 현저한 PSNR 개선 달성

## 2. 해결하고자 하는 문제와 제안 방법

### **문제 정의**
Sparse-view CT는 각도 undersampling으로 인해 **streak artifact**가 발생하여 진단 품질이 급격히 저하됩니다. 기존 방법들은 도메인 특화 정보를 충분히 활용하지 못해 고도로 undersampled된 데이터에서 신뢰할 수 있는 재구성을 제공하지 못합니다.

### **제안 방법**

#### **2단계 프레임워크**

**Step 1: Sinogram Inpainting Network (SIN)**
- 입력: 23-angle sparse sinogram → 출력: 180-angle full sinogram
- Two-Ends preprocessing으로 경계 artifact 방지
- 1D super-resolution으로 누락된 projection 정보 복원

**Step 2: Postprocessing Refining Network (PRN)**  
- Multi-resolution cascade input: 4채널 (23, 45, 90, 180-angle FBP 재구성)
- 잔여 localized artifact 제거

#### **주요 수식**

**Discriminator Perceptual Loss:**

$$ L_{DP}(\hat{y}, y) = \frac{1}{N} \sum_{j=1}^{N} \frac{1}{C_j H_j W_j} \|\phi_j(\hat{y}) - \phi_j(y)\|_2^2 $$

**High-Frequency Loss (CT 도메인 특화):**

$$ L_{HF}(\hat{y}, y) = \|\hat{y} * h - y * h\|_1 $$

**SIN 총 목적함수:**

$$ G_{SIN}^* = \arg \min_G [\lambda_1 L_{adv}(G) + \lambda_2 L_c + \lambda_3 L_{DP} + \lambda_4 L_{HF}] $$

## 3. 모델 구조

### **네트워크 아키텍처**
- **Generator**: 적응형 U-Net (average pooling + bilinear upsampling)
- **Discriminator**: 4-layer patch discriminator (global/local)
- **Two-Ends flipping**: Orthographic projection 특성 활용한 데이터 augmentation
- **Multi-resolution cascade**: PRN에 다양한 해상도의 재구성 입력 제공

### **핵심 설계 원칙**
- Max-pooling → Average-pooling: 미분 가능성 보장
- Strided convolution 대신 bilinear upsampling: Checkerboard artifact 방지
- Skip connection: 저수준 특성 일관성 유지

## 4. 성능 향상 및 한계

### **성능 향상**
- **정량적 성과**: 기존 방법 대비 **4dB 이상 PSNR 향상**, **5% SSIM 개선**
- **State-of-the-art 비교**: FISTA-TV, cGAN, Neumann Network 대비 우수한 성능
- **Ablation study**: 각 모듈의 효과성 체계적 검증

| 방법 | PSNR (σ) | SSIM (σ) |
|------|----------|----------|
| FISTA-PD-TV | 30.61 (2.67) | 0.839 (0.036) |
| cGAN | 30.86 (1.92) | 0.762 (0.043) |
| Neumann Network | 28.72 (2.09) | 0.697 (0.069) |
| **Ours** | **34.90 (2.15)** | **0.877 (0.029)** |

### **한계점**
- **계산 복잡성**: 2단계 네트워크로 인한 훈련/추론 시간 증가
- **데이터 의존성**: 특정 도메인(의료 영상)에 특화된 설계로 일반화 제한
- **하이퍼파라미터 민감성**: 다중 손실함수의 가중치 조정 필요
- **메모리 요구사항**: Multi-resolution cascade input으로 인한 메모리 사용량 증가

## 5. 일반화 성능 향상 가능성

### **도메인 적응성**
본 논문의 접근법은 **측정 도메인과 재구성 도메인의 이중 활용**이라는 핵심 아이디어를 통해 다양한 tomographic imaging 문제에 적용 가능합니다:

- **Metal Artifact Removal (MAR)**: Sinogram inpainting 전략 활용
- **Limited Angle Tomography (LAT)**: 1D super-resolution 기법 적용  
- **다양한 CT 스캔 프로토콜**: 압축비와 각도 설정에 따른 적응성

### **Transfer Learning 잠재력**
- **Domain-Specific Perceptual Loss**: 다른 의료 영상 modality로의 확장 가능
- **Two-Ends preprocessing**: 순환 대칭성을 가진 다른 imaging system 적용
- **Multi-resolution framework**: 다양한 resolution requirement를 가진 응용 분야

## 6. 향후 연구 영향과 고려사항

### **연구 영향**
- **Dual-domain learning paradigm**: 측정-재구성 도메인 결합 접근법의 새로운 표준 제시
- **Domain-specific loss design**: 특정 물리적 모델에 특화된 손실함수 개발 방향 제시
- **Explainable reconstruction**: 물리적 모델과 학습 기반 방법의 해석 가능한 결합

### **향후 연구 고려사항**

#### **기술적 개선**
- **실시간 처리**: 2단계 네트워크의 계산 효율성 개선 필요
- **Few-shot learning**: 제한된 훈련 데이터로도 robust한 성능 달성
- **Self-supervised learning**: Ground truth 의존성 완화

#### **일반화 확장**
- **Multi-modal integration**: MRI, PET 등 다른 modality와의 결합
- **Cross-domain validation**: 다양한 해부학적 구조와 병리학적 조건에서의 검증
- **Uncertainty quantification**: 재구성 결과의 신뢰도 정량화

#### **임상 적용**
- **Regulatory validation**: FDA 등 규제 승인을 위한 임상 검증 필요
- **Radiologist workflow integration**: 실제 임상 환경에서의 사용성 평가
- **Patient safety**: 저선량 CT의 진단 정확도와 환자 안전성 balance

본 연구는 **물리적 모델링과 딥러닝의 효과적 결합**을 통해 computational imaging 분야에서 중요한 이정표를 제시하며, 향후 의료 영상 재구성 기술 발전의 핵심 방향을 제시합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/02ff2cad-f99f-4186-8d85-8df6a8f17564/2012.04743v1.pdf
