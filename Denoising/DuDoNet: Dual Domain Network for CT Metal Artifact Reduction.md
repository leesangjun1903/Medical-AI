# DuDoNet: Dual Domain Network for CT Metal Artifact Reduction

## 핵심 주장 및 주요 기여  
DuDoNet은 CT 영상에서 금속 임플란트로 인한 스트리킹(streaking) 및 섀도우(shadow) 아티팩트를 효과적으로 제거하기 위해 “투 도메인(사이노그램 + 이미지 도메인)”을 동시에 학습하는 최초의 **end-to-end** 딥러닝 프레임워크다.  
1. **Dual-Domain Refinement**: 금속 영역 보간(sinogram inpainting)과 이미지 도메인 보정을 결합해 상호 보완  
2. **Radon Inversion Layer (RIL)**: 필터드 백프로젝션(FBP)을 미분 가능하게 구현, 사이노그램–이미지 간 그라디언트 전파 지원  
3. **Radon Consistency Loss**: 재구성된 CT 이미지의 1차 손실을 통해 사이노그램 일관성 유지  
4. **Mask Pyramid U-Net**: 작은 금속 임플란트 정보 손실 방지를 위한 멀티스케일 마스크 피라미드 구조

***

## 문제 정의  
- CT 재구성 모델 $$P^{\dagger}$$에 금속 임플란트 $$\mathrm{IM}(E)$$가 존재하면 측정된 사이노그램  

$$
    Y = -\log\int \eta(E) \, e^{-P(X + \mathrm{IM}(E))}\,dE
  $$  
  
  내부 금속 영역 $$\mathrm{PIM}$$로 인해 재구성 영상  
  
$$
    P^{\dagger}Y = \hat X - P^{\dagger}\Bigl[\log\int \eta(E)e^{-P\mathrm{IM}(E)}\,dE\Bigr]
  $$  
  
  에 구조화된 금속 아티팩트가 추가됨.  
- 기존 단일 도메인(사이노그램 또는 이미지) 기반 MAR 방법은  
  - 비지역적·구조적 아티팩트 완전 제거 불가  
  - 사이노그램 보간 시 재구성 영상에 2차 아티팩트 생성  

***

## 제안 방법  

### 1. Sinogram Enhancement Network (SE-Net)  
- 입력: 선형 보간된 사이노그램 $$Y_{LI}$$ 및 금속 마스크 $$M_t$$  
- 출력: 보정된 사이노그램  

$$
    Y_{\mathrm{out}} = M_t \odot G_s(Y_{LI}, M_t) + (1 - M_t)\odot Y_{LI}
  $$  

- 손실: $$L_1$$ 복원 손실  

$$
    L_{Gs} = \|Y_{\mathrm{out}} - Y_{\mathrm{gt}}\|_1
  $$  

- **Mask Pyramid**: 다운샘플된 다양한 스케일의 마스크를 U-Net 각 계층에 결합하여 작은 금속 흔적 유지

### 2. Radon Inversion Layer (RIL)  
- **Parallel-Beam Conversion**: 팬빔 $$(\gamma,\beta)$$→병렬빔 $$(t,\theta)$$  

$$
    t = D\sin\gamma,\quad \theta = \beta + \gamma
  $$  

- **Ram-Lak 필터링**:  

```math
    Q(t,\theta) = \mathcal{F}_t^{-1}\bigl\{|\omega|\cdot\mathcal{F}_t\{Y_{\mathrm{para}}\}\bigr\}
```  

- **Back-Projection**:  

$$
    \hat X(u,v) = \int_0^{\pi} Q(u\cos\theta + v\sin\theta,\theta)\,d\theta
  $$  

- **Radon Consistency Loss**:  

$$
    L_{RC} = \|f_R(Y_{\mathrm{out}}) - X_{\mathrm{gt}}\|_1
  $$  
  
  (이미지 도메인 손실을 통해 사이노그램 재구성 일관성 강화)

### 3. Image Enhancement Network (IE-Net)  
- 입력: $$\hat X = f_R(Y_{\mathrm{out}})$$ 및 $$X_{LI}=f_R(Y_{LI})$$  
- 출력: 잔차 학습 방식으로 정제된 CT 영상  

$$
    X_{\mathrm{out}} = X_{LI} + G_i(\hat X,\,X_{LI})
  $$  

- 손실: $$L_1$$ 영상 복원 손실  

$$
    L_{Gi} = \|X_{\mathrm{out}} - X_{\mathrm{gt}}\|_1
  $$

### 4. 전체 손실 함수  

$$
  L = L_{Gs} + L_{RC} + L_{Gi}
$$

***

## 모델 구조  
- **SE-Net**: 마스크 피라미드 U-Net (다운샘플·업샘플 시 마스크 결합)  
- **RIL**: CUDA 최적화 팬빔→병렬→FBP 모듈  
- **IE-Net**: 표준 U-Net 구조  
- 총 파라미터 수 억 단위, 2×1080Ti GPU로 380 에폭 학습  

***

## 성능 향상 및 한계  

| 메소드          | 평균 PSNR (dB) | 평균 SSIM |
|-----------------|---------------:|----------:|
| LI              |       25.47    |   0.8917  |
| NMAR            |       27.51    |   0.9001  |
| cGAN-CT         |       28.07    |   0.8733  |
| RDN-CT          |       31.74    |   0.9156  |
| CNNMAR          |       29.52    |   0.9243  |
| **DuDoNet (Ours)** | **33.51**    | **0.9379**|

- **일관성 손실(RC) 도입** 시 PSNR +0.3 dB 이상 향상  
- **Mask Pyramid** 덕분에 작은 금속 임플란트 주변 복원력 +0.2 dB  
- **연산 효율**: 단일 FBP 구현 대비 4× 빠른 후처리 속도  
- **한계**:  
  - 실제 임상 사이노그램 확보 어려움  
  - 극소형 금속(픽셀 <32)이나 복합 재질 임플란트 일반화 검증 필요  
  - 3D CT 볼륨 전체 적용 시 메모리·연산 부담

***

## 일반화 성능 향상 관점  
- **도메인 제약 완화**: 두 도메인 간 상호 그라디언트 전파로 다양한 아티팩트 패턴 학습  
- **마스크 피라미드**: 스케일별 정보 유지로 다양한 크기 금속 대응력 강화  
- **Radon Consistency**: 비지도적 유사 영상(fine-tuning)에도 안정적 적응 가능  
- 잠재적 확장:  
  - **다중 모달** CT/MRI 금속 간섭 제어  
  - **Sparse‐View CT** 또는 **노이즈 높은 데이터** 복원  

***

## 향후 연구 영향 및 고려 사항  
DuDoNet은 **딥러닝 기반 물리 모델 통합**의 표준을 제시하며,  
- **다른 역문제(inverse problems)** (초해상도, 노이즈 제거, 희소 투영 CT)에도 Dual-Domain 학습 틀 확장 촉진  
- **실제 임상 데이터** 적용을 위해 사이노그램 접근성·프라이버시 문제 해결 필요  
- **3D 확장**을 위한 메모리 효율화 연구와 **경량화 모듈** 설계  
- **Self-supervised 학습** 기법과 결합해 레이블 없이도 마스크 일관성 학습 가능성 모색  
- **임상 검증**: 다양한 금속 재질·형태·위치 데이터셋 구축 및 성능 비교 필요  

이 논문은 CT 금속 아티팩트 제거 분야에 딥러닝과 전통 물리 모델을 효과적으로 결합하는 새로운 패러다임을 제시하며, 향후 의료 영상 복원 연구에 중요한 이정표가 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fb20253f-0def-43b1-9d1f-92f17a02cc7c/1907.00273v1.pdf
