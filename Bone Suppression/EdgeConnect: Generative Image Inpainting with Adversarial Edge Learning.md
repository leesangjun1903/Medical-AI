# EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning

## 1. 개요

EdgeConnect는 딥러닝을 활용한 이미지 인페인팅(Image Inpainting) 분야의 대표적인 논문으로, 기존 방식들이 복원 영역의 디테일이 부족하거나 흐릿하게 복원되는 문제를 개선하기 위해 제안된 모델입니다[1][2][3]. 이 논문은 이미지의 **에지(Edge)** 정보를 먼저 복원한 후, 이를 바탕으로 실제 이미지를 복원하는 **2단계(two-stage) 구조**를 채택합니다[1][2].

---

## 2. 기존 인페인팅 방식과 한계

- **Diffusion-based**: 주변 픽셀 값을 점진적으로 채워가는 방식. 큰 영역 복원에는 한계가 있음.
- **Patch-based**: 이미지 내 유사 영역을 찾아 복사하는 방식. 계산량이 많고, 복잡한 구조 복원에 한계.
- **Learning-based**: 딥러닝 기반 생성 모델(GAN 등)로 복원. 하지만 종종 부드럽고 비현실적인 결과를 냄[2].

---

## 3. EdgeConnect의 핵심 아이디어

- **Line First, Color Next**: 실제 그림을 그릴 때처럼, 먼저 선(Edge)을 그리고 그 위에 색과 질감을 입히는 방식에서 착안[2].
- **2단계 구조**:
  1. **Edge Generator**: 결손 영역의 에지(선)를 생성.
  2. **Image Completion Network**: 생성된 에지를 바탕으로 실제 이미지를 복원[1][2].

---

## 4. 모델 구조 및 학습 방식

### 4.1. Edge Generator

- 입력: 마스킹된 그레이스케일 이미지, 해당 영역의 에지 맵, 마스크.
- 출력: 결손 영역의 새로운 에지 맵.
- 학습: **Adversarial Loss**(GAN 손실)와 **Feature-Matching Loss**를 조합하여 학습. Feature-Matching Loss는 생성된 에지와 실제 에지의 중간 레이어 특성 차이를 최소화함[2].

### 4.2. Image Completion Network

- 입력: 마스킹된 컬러 이미지와, 결손 영역을 채운 에지 맵.
- 출력: 최종 복원 이미지.
- 학습: 네 가지 손실 함수(L1 Loss, Adversarial Loss, Perceptual Loss, Style Loss)를 조합한 **Joint Loss** 사용[2].
  - **Perceptual Loss**: VGG-19 등 사전학습된 네트워크의 중간 레이어를 활용해, 생성 이미지와 실제 이미지의 특징 차이를 최소화.
  - **Style Loss**: 이미지의 질감(스타일) 유사성을 측정하여, 블러 현상이나 체커보드 아티팩트를 줄임.

### 4.3. 기타 특징

- 에지 맵 생성에는 **Canny Edge Detector** 사용, Gaussian Smoothing의 표준편차는 실험적으로 최적값(2) 사용[2].
- 마스크는 사각형(정형)과 불규칙(비정형) 두 가지 모두 실험[2].
- 학습 데이터: CelebA, Places2, Paris StreetView 등 공개 데이터셋 활용[1][2].

---

## 5. 성능 및 실험 결과

- **정성적 평가**: 기존 방법보다 결손 영역의 디테일이 뛰어나고, 블러/체커보드 아티팩트가 적은 결과를 보임[2].
- **정량적 평가**: L1, SSIM, PSNR, FID 등 다양한 지표에서 기존 SOTA(State-of-the-art) 모델 대비 우수한 성능 확인[2].
- **Ablation Study**: Edge Generator의 존재가 복원 품질에 미치는 영향 실험. 에지 생성 단계가 있을 때 성능이 확실히 향상됨을 확인[2].

---

## 6. 결론 및 의의

- EdgeConnect는 **에지 정보를 활용한 2단계 복원** 전략을 통해, 기존 인페인팅 기법의 한계를 극복하고 세밀한 디테일까지 복원할 수 있음을 실험적으로 증명했습니다[1][2].
- Restoration(복원), Removal(제거), Synthesis(합성) 등 다양한 이미지 편집 분야에 응용 가능성이 높습니다[2].

---

## 7. 요약 표

| 단계               | 입력                                         | 출력                | 주요 손실 함수                             |
|--------------------|----------------------------------------------|---------------------|--------------------------------------------|
| Edge Generator     | 마스킹된 흑백 이미지, 에지 맵, 마스크         | 복원된 에지 맵      | Adversarial, Feature-Matching              |
| Image Completion   | 마스킹된 컬러 이미지, 복원된 에지 맵         | 최종 복원 이미지    | L1, Adversarial, Perceptual, Style         |

---

EdgeConnect 논문은 인페인팅 분야에서 **에지 기반의 2단계 접근법**이 실제 복원 품질을 크게 높일 수 있음을 보여준 대표적인 연구입니다[1][2][3].

[1] https://subinium.github.io/LR007/
[2] https://big-dream-world.tistory.com/80
[3] https://arxiv.org/abs/1901.00212
[4] https://dsba.snu.ac.kr/seminar/?mod=document
[5] https://ettrends.etri.re.kr/ettrends/194/0905194005/
[6] https://blog.everdu.com/397
[7] https://ettrends.etri.re.kr/ettrends/184/0905184009/0905184009.html
[8] https://ostin.tistory.com/240
[9] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NPAP13263461
[10] https://www.fsp-group.com/kr/knowledge-app-42.html
