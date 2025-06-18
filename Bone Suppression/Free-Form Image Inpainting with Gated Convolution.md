# Free-Form Image Inpainting with Gated Convolution

## 개요

이 논문은 자유로운 형태의 마스크(free-form mask)와 사용자 가이드(guidance)를 활용해 이미지를 자연스럽게 복원하는 새로운 이미지 인페인팅(이미지 구멍 채우기) 시스템을 제안합니다. 기존의 단순한 사각형 마스크에만 국한되지 않고, 임의의 모양과 위치의 손실 영역을 효과적으로 복원하는 것이 특징입니다[1][2][3].

---

## 기존 방법의 한계

- **Vanilla Convolution**: 모든 입력 픽셀을 동일하게 취급해 마스킹된 영역과 그렇지 않은 영역을 구분하지 못해 색상 불일치, 경계 흐림 등의 문제가 발생합니다[1][2].
- **Partial Convolution**: 마스킹된 영역과 유효한 영역을 구분하지만, 각 레이어마다 마스크가 단순히 0 또는 1로만 업데이트되어 정보 손실이 발생하고, 깊은 레이어에서는 마스크 정보가 사라집니다. 또한, 사용자 입력(스케치 등)과의 호환성이 떨어집니다[2].

---

## 주요 기여 및 기술

### 1. Gated Convolution

- **핵심 아이디어**: 각 채널과 각 공간 위치(spatial location)마다 동적으로 feature를 선택하는 gating 메커니즘을 학습합니다. 즉, 네트워크가 데이터로부터 소프트 마스크를 자동으로 학습하여, 마스킹 여부와 상관없이 더 유연하게 정보를 처리할 수 있게 합니다[1][2][3].
- **수식**: 입력 feature에서 gating 값을 시그모이드 함수로 계산하고, 이를 활성화 함수(activation function)를 거친 feature와 곱해 최종 출력을 만듭니다. 이때 gating과 feature를 위한 필터는 서로 다릅니다[2].
- **장점**: 임의의 모양 마스크, 희미한 마스크, 사용자 스케치 등 다양한 조건에서 뛰어난 성능을 보입니다[1][2].

### 2. SN-PatchGAN

- **문제점**: 기존의 글로벌/로컬 GAN은 사각형 마스크에만 적합해 free-form 마스크에는 한계가 있습니다[1][2].
- **해결책**: Spectral Normalization을 적용한 PatchGAN(SN-PatchGAN)을 제안하여, 이미지의 다양한 위치와 의미(채널별)를 고려한 판별을 수행합니다. 이로써 학습이 빠르고 안정적이며, 높은 품질의 인페인팅 결과를 얻을 수 있습니다[1][2][3].
- **Loss**: 최종 손실 함수는 픽셀 단위의 L1 재구성 손실과 SN-PatchGAN 손실로 구성됩니다[1][2].

### 3. 네트워크 구조

- **구성**: Gated Convolution을 쌓아 encoder-decoder 네트워크를 구성합니다. U-Net의 skip connection 대신 간단한 encoder-decoder 구조를 사용해 마스킹 영역의 경계에서도 자연스러운 결과를 얻습니다[1][2].
- **Contextual Attention**: 상황별 attention 모듈을 통합해 긴 거리의 의존성(long-range dependency)도 잘 포착합니다[2].

---

## Free-Form Mask 생성 및 User-Guided Inpainting

- **Mask 생성**: 학습 중 임의의 free-form mask를 자동으로 생성하는 알고리즘을 제안해 실제 사용 환경과 유사한 다양한 마스크를 제공합니다[1].
- **사용자 가이드**: 사용자 입력(스케치 등)을 추가 채널로 받아, 사용자가 원하는 형태로 이미지를 복원할 수 있습니다[1][2].

---

## 실험 및 결과

- Places2, Celeb-HQ 등 다양한 데이터셋에서 실험을 진행했으며, 기존 Partial Convolution 등과 비교해 더 자연스럽고 이질감 없는 결과를 보여주었습니다[2].
- Gated Convolution 방식은 마스크 부분의 색상 일관성, 경계 처리, 세밀한 디테일에서 뛰어난 성능을 입증했습니다[1][2].

---

## 결론

- **Gated Convolution**은 free-form image inpainting에서 기존 방법의 한계를 극복하고, 다양한 마스크와 사용자 입력 조건에서 우수한 성능을 보입니다[1][2][3].
- **SN-PatchGAN**은 간단하면서도 효과적이고, 빠르고 안정적인 학습을 가능하게 합니다[1][2][3].
- 이 시스템은 실제 사진 편집(객체 제거, 워터마크 삭제, 얼굴 수정 등)에서 실용적으로 활용될 수 있습니다[3].

---

### 참고: 논문 원문 및 코드  
- [논문 PDF 및 코드 링크](https://github.com/JiahuiYu/generative_inpainting)[3]

[1][2][3]

[1] https://hhhhhsk.tistory.com/30
[2] https://2bdbest-ds.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Free-Form-Image-Inpainting-with-Gated-ConvolutionICCV-2019
[3] https://paperswithcode.com/paper/free-form-image-inpainting-with-gated
[4] https://velog.io/@yooniverseis/paper-reviewFree-Form-Image-Inpainting-with-Gated-Convolution
[5] https://wdprogrammer.tistory.com/71
[6] https://ar5iv.labs.arxiv.org/html/1806.03589
[7] https://arxiv.org/abs/1806.03589
[8] https://scispace.com/papers/free-form-image-inpainting-with-gated-convolution-1bgpqx14nj?citations_page=30
[9] https://github.com/sirius-image-inpainting/Free-Form-Image-Inpainting-With-Gated-Convolution
[10] https://www.sciencedirect.com/science/article/abs/pii/S0141938222001391
