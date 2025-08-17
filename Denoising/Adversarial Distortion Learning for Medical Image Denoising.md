# Adversarial Distortion Learning for Medical Image Denoising

## 1. 핵심 주장 및 주요 기여  
**Adversarial Distortion Learning (ADL)**은 2D/3D 의료 영상의 노이즈 제거 시 텍스처, 대비, 전역·국부 구조를 효과적으로 보존하면서 재구성 성능을 극대화하는 적대적 학습 프레임워크를 제안한다.[1]
- 두 개의 Efficient-UNet 기반 오토인코더(디노이저·디스크리미네이터)를 상호 최적화하여, 디노이저가 생성한 영상과 노이즈-프리 레퍼런스를 판별기가 구분하지 못할 때까지 반복 학습  
- 멀티스케일 피라미달(pyramidal) 구조와 손실 함수로 고·저해상도에서 텍스처·에지 정보 보존  
- 노이즈 통계나 영상 분포에 대한 사전 지식 없이도 다양한 의료 영상에 *재학습 없이* 바로 적용 가능한 **일반화 능력** 보유  

***

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제  
의료 영상 denoising은  

$$
y = x + n
$$

형태의 관측 노이즈 $$n$$을 제거하여 원본 $$x$$를 복원하는 비정준 역문제이다.  
기존 딥러닝 기반 기법은 텍스처·대비 손실, 과적합, 다양한 영상모달리티 일반화 부족 문제를 갖는다.

### 2.2 제안 네트워크 구조: Efficient-UNet  
- **Encoder–Bridge–Decoder** 계층으로 구성된 U-Net 변형  
- **Content Enhancer**: 노이즈 레벨이 낮을 때 엣지·텍스처 활성화  
- **Transformer Blocks**: 각 디코더 스케일별 출력을 영상 도메인으로 매핑  
- 디스크리미네이터는 경량화된 동일 구조에 픽셀 단위 분류용 mapper 추가  

```
Input y → Encoder → Bridge → Decoder → T₃ → T₂ → T₁ → Output \hat{x}
                              ⇑
                       Content Enhancer
```

### 2.3 손실 함수  
디노이저 손실 $$\mathcal{L}_G$$:  

```math
\mathcal{L}_G = \lambda_{\ell_1} \|G(y)-x\|_1
+\lambda_p\sum_{j=1}^J\|\Delta_j G(y)-\Delta_j x\|_1
+\lambda_H\,\mathbb{E}\bigl[\log\cosh\bigl(H[G(y)]-H[x]\bigr)\bigr]
```

- $$\ell_1$$ fidelity  
- ATW 기반 피라미달 텍스처 보존 ($$\Delta_j$$는 $$j$$단계 차분)  
- 전역 구조 유지용 히스토그램 손실 $$H[\cdot]$$  

각 스케일 $$s$$에 대해 동일 손실을 적용 후 평균화하여 멀티스케일 학습  

$$
\mathcal{L}\_G^{\text{multi}} = \frac{1}{3}\sum_{s=1}^3 \mathcal{L}_G^{(s)}
$$

디스크리미네이터 손실 $$\mathcal{L}_D$$:  
브리지 출력과 각 디코더 스케일 출력에 픽셀 단위 마진 손실을 결합하여 학습  

***

## 3. 성능 향상 및 한계

| 데이터셋 | 비교 기법 | PSNR↑ | SSIM↑ |
|:---------:|:---------:|:------:|:------:|
| Dermatoscopy (σ=25) | SwinIR | 35.5 dB | 0.90 |
|  “[본 논문]” | **36.1 dB** | **0.93** |

- 다양한 2D·3D 의료 영상 실험에서 기존 BM3D/BM4D, 최신 CNN·트랜스포머 대비 PSNR·SSIM 평균 1~2% 향상[1]
- CPU 대비 GPU 추론 속도 4배 이상 빨라 (2D 143 ms vs 539 ms)  
- **한계**: 복잡한 멀티스케일 손실 연산으로 메모리 사용량 증가, 실제 임상 잡음(비가우시안)에는 추가 검증 필요  

***

## 4. 일반화 성능 향상 관점  
- **Domain-agnostic 학습**: 2D 모델은 ImageNet, 3D 모델은 IXI MRI만으로 학습 후 다양한 의료 모달리티에 *재학습 없이* 적용  
- **피라미달 학습**: 고·저해상도 모두에서 구조·텍스처 학습 → 새로운 노이즈·분포에도 일관된 복원  
- **Bias-free 네트워크**: 편향 파라미터 제거로 과적합 저감 및 선형성 강화  

이로써 적은 의료 데이터만으로도 타 도메인 영상에 강건하게 작동하며, 실제 환경 배포 시 **데이터 수집 부담**을 크게 낮춤.[1]

***

## 5. 미래 연구에 미치는 영향 및 고려사항  
**영향**:  
- 멀티스케일·히스토그램 기반 손실 설계가 다양한 영상 복원·합성 문제에 확장 적용 가능  
- 도메인 적응 없이도 일반화 가능한 네트워크 구조 설계 방향 제시  

**고려사항**:  
- 실제 임상 데이터의 복잡한 잡음·아티팩트 대응력 추가 평가  
- 경량 모델화 및 메모리 최적화를 통한 임베디드 환경 적용  
- 비감독·약지도 학습과 결합한 적대적 학습 프레임워크 연구  

***

 Ghahremani et al., “Adversarial Distortion Learning for Medical Image Denoising,” arXiv:2204.14100v2.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7c79c23e-0b5e-4e90-b9a3-a4738c53f240/2204.14100v2.pdf
