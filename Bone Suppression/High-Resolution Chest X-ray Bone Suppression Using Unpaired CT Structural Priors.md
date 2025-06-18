# High-Resolution Chest X-ray Bone Suppression Using Unpaired CT Structural Priors 

논문은 흉부 X-ray(Chest X-ray, CXR) 영상에서 뼈 구조(늑골, 쇄골 등)를 효과적으로 억제(bone suppression)하여 폐 질환 진단의 정확도를 높이는 새로운 방법을 제안합니다[1]. 이 논문은 뼈가 없는 X-ray 영상을 얻기 어려운 현실적 한계를 극복하기 위해, CT(Computed Tomography)에서 얻은 구조적 정보를 활용하며, CT와 X-ray가 짝지어지지 않은(unpaired) 데이터만으로 학습이 가능한 점이 특징입니다[1][2].

---

## 왜 뼈 억제가 필요한가?  
- 흉부 X-ray에서 늑골, 쇄골 등 뼈 구조가 폐 조직을 가려 미세 병변(폐암, 결핵 등) 발견에 방해가 됩니다[1].
- 뼈 억제 이미지는 방사선 전문의와 인공지능 진단 시스템 모두의 진단 성능을 높여줍니다[1].

---

## 기존 방법의 한계  
- 기존에는 Dual Energy Subtraction(DES) 방식으로 뼈 억제 영상을 얻었으나, 추가 장비가 필요하고 환자 움직임 등으로 인한 한계가 있습니다[2].
- 딥러닝 기반 방법도 있었으나, 대부분 뼈 억제 정답(ground truth) 영상이 필요해 대규모 데이터 확보가 어렵습니다[2].

---

## 제안 방법의 핵심  
### 1. Unpaired CT 데이터 활용  
- CT에서 얻은 3차원 정보를 2차원 X-ray 형태로 변환한 **디지털 재구성 방사선 영상(DRR, Digitally Reconstructed Radiograph)**을 생성합니다[1].
- CT와 X-ray 데이터가 짝지어져 있지 않아도, DRR을 중간다리로 활용해 학습이 가능합니다[1].

### 2. Coarse-to-Fine(저해상도→고해상도) 전략  
- 먼저 저해상도에서 DRR과 X-ray 간의 도메인 차이를 줄이고, 뼈 구조 분해(bone decomposition)를 학습합니다[1].
- 이 과정은 Laplacian of Gaussian(LoG) 도메인에서 진행되어, 학습 수렴 속도가 빨라지고 도메인 차이가 완화됩니다[1].
- 이후 결과를 고해상도로 업샘플링하여, 원본 고해상도 X-ray에서 뼈 영역만 정밀하게 억제합니다[1].

### 3. 최종 뼈 억제  
- 최종적으로 생성된 뼈 이미지를 원본 X-ray에서 빼줌으로써 뼈 억제 영상을 얻습니다[1].

---

## 실험 및 성능  
- 두 개의 공개 CXR 데이터셋에서 실험을 진행했습니다[1].
- 기존 비지도(unsupervised) 뼈 억제 방법들 대비 우수한 성능을 보였습니다(PSNR, MAE 등 성능지표 개선)[2].
- 임상 평가에서, 뼈 억제 영상을 활용하면 방사선 전문의의 폐 질환 오진율(false-negative rate)이 15%에서 8%로 감소했습니다[1].
- 뼈 억제 영상과 원본 X-ray를 함께 입력하면, 폐 질환 분류(딥러닝) 성능도 향상됩니다[1].

---

## 임상적 의의  
- 뼈 억제 영상은 실제 진단에서 폐 결절, 미세 병변 탐지에 큰 도움을 주며, 인공지능 기반 폐 질환 자동 분류에도 효과적입니다[1][2].
- 추가 장비나 짝지어진 데이터 없이, 기존 CT와 X-ray 데이터만으로 적용할 수 있어 현실적 활용성이 높습니다[1].

---

## 요약  
이 논문은 기존의 한계를 극복하여, **짝지어지지 않은 CT와 X-ray 데이터만으로 고해상도 뼈 억제 X-ray 이미지를 생성하는 혁신적 방법**을 제시합니다. 임상적으로도 진단 정확도를 높여주는 효과가 입증되었습니다[1][2].

---

| 주요 내용              | 설명                                                                                  |
|----------------------|-------------------------------------------------------------------------------------|
| 목적                 | 뼈가 없는 X-ray 영상을 생성하여 진단 정확도 향상                                         |
| 핵심 기술             | Unpaired CT 기반 DRR 생성, LoG 도메인, Coarse-to-Fine 전략                              |
| 성능                  | 기존 방법 대비 우수, 오진율 감소, 딥러닝 분류 성능 향상                                   |
| 임상적 장점           | 추가 장비 불필요, 실제 진단 및 AI에 모두 효과적                                          |

[1][2]

[1] https://pubmed.ncbi.nlm.nih.gov/32275586/
[2] https://mednexus.org/doi/pdf/10.1016/j.radmp.2024.12.003?download=true
[3] https://github.com/MIRACLE-Center/High-Resolution-Chest-X-ray-Bone-Suppression
[4] https://www.mdpi.com/resolver?pii=diagnostics11050840
[5] https://cdn.amegroups.cn/journals/amepc/files/journals/4/articles/67433/public/67433-PB2-9455-R2.pdf
[6] https://pubmed.ncbi.nlm.nih.gov/35993343/
[7] https://suzukilab.first.iir.titech.ac.jp/ja/wp-content/uploads/2020/01/ChenSEtAl_VDE-ICU_PhysMedBiol2016-1.pdf
[8] https://www.themoonlight.io/ko/review/bs-ldm-effective-bone-suppression-in-high-resolution-chest-x-ray-images-with-conditional-latent-diffusion-models
[9] https://pubmed.ncbi.nlm.nih.gov/40293914/
[10] https://scholar.kyobobook.co.kr/article/detail/4010025912460
[11] https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression
[12] https://www.sciencedirect.com/science/article/abs/pii/S0895611123000046
[13] https://mednexus.org/doi/10.1016/j.radmp.2024.12.003
[14] https://pubmed.ncbi.nlm.nih.gov/34888191/
[15] https://scholar.kyobobook.co.kr/article/detail/4010025912536

# 논문 방법(Method) 쉽게 설명

## 전체 흐름 요약

이 논문은 **짝지어진 데이터 없이(CT, X-ray 각각 따로 있음)** CT에서 얻은 구조 정보를 활용해, 고해상도 흉부 X-ray에서 뼈 구조(늑골, 쇄골 등)만 효과적으로 억제하는 방법을 제안합니다.  
핵심은 **CT에서 DRR(디지털 재구성 방사선 영상)을 만들고, 이를 X-ray와 연결해 뼈 구조만 분리**하는 것입니다[1][2].

---

## 단계별 상세 설명

### 1. CT로부터 DRR(Digitally Reconstructed Radiograph) 생성

- **CT 데이터**는 3차원(3D) 구조 정보를 갖고 있습니다.
- 이 CT를 바탕으로, 실제 X-ray처럼 보이는 **2차원(2D) DRR 이미지를 생성**합니다.
- DRR은 CT의 뼈 구조를 X-ray 관점에서 투영한 이미지로, 뼈 구조 정보가 명확하게 나타납니다[1].

### 2. DRR과 X-ray의 도메인 차이 줄이기 (LoG 도메인)

- CT에서 만든 DRR과 실제 X-ray는 촬영 방식, 화질, 노이즈 등 여러 차이가 있습니다.
- 이 차이를 줄이기 위해 **Laplacian of Gaussian(LoG) 도메인**에서 두 이미지를 변환·비교합니다.
  - LoG는 이미지의 윤곽, 경계, 뼈 구조를 더 잘 드러내는 변환 방식입니다.
- 이렇게 하면 DRR과 X-ray의 구조적 차이를 효과적으로 줄일 수 있습니다[1].

### 3. Coarse-to-Fine(저해상도→고해상도) 전략

- **1단계:** 저해상도에서 DRR과 X-ray를 비교·학습해 뼈 구조만 분리하는 모델을 만듭니다.
- **2단계:** 저해상도에서 얻은 뼈 구조 분리 결과를 **고해상도**로 업샘플링합니다.
- **3단계:** 고해상도 X-ray의 뼈 영역만 정밀하게 보정·분리합니다.
- 이렇게 하면 **고해상도에서도 뼈 구조만 정확히 분리**할 수 있습니다[1].

### 4. 최종 Bone Suppression(뼈 억제) 이미지 생성

- 위 과정에서 얻은 **뼈 이미지(분리된 뼈 구조)**를 원본 X-ray에서 **빼줍니다(뺄셈 연산)**.
- 그 결과, **뼈가 억제된(거의 없는) X-ray 이미지**가 만들어집니다.
- 이 이미지는 폐 조직, 미세 병변 등을 더 명확히 볼 수 있게 해줍니다[1][2].

---

## 실제 구현 예시 (코드 흐름)

1. **CycleGAN 등 도메인 변환 네트워크**로 DRR과 X-ray의 구조적 차이 극복
2. **UNet 기반 분리 네트워크**로 뼈 구조만 추출
3. **히스토그램 매칭** 등 후처리로 X-ray와 뼈 억제 이미지의 밝기/톤 맞춤
4. **최종적으로 원본 X-ray에서 뼈 이미지를 빼서** 뼈 억제 영상 생성[2]

---

## 표로 정리

| 단계                | 설명                                                                 |
|-------------------|--------------------------------------------------------------------|
| DRR 생성           | CT 데이터로 X-ray와 유사한 2D DRR 이미지 생성                         |
| 도메인 차이 완화    | LoG 변환으로 DRR과 X-ray의 구조적 차이 줄임                            |
| Coarse-to-Fine     | 저해상도에서 뼈 분리 → 고해상도 업샘플링 → 정밀 뼈 분리                 |
| 뼈 억제 영상 생성   | 분리된 뼈 이미지를 원본 X-ray에서 빼서 뼈 억제 영상 완성                  |

---

## 한 줄 요약

**CT에서 얻은 DRR(뼈 구조)과 X-ray를 LoG 도메인에서 비교·학습하여, 저해상도→고해상도 순으로 뼈만 분리하고, 이를 원본 X-ray에서 빼주는 방식**입니다[1][2].

---

- 이 방법은 **짝지어진 데이터가 필요 없고, 추가 장비 없이 기존 CT/X-ray 데이터로 적용**할 수 있어 현실적으로 매우 유용합니다[1].

[1][2]

[1] https://pubmed.ncbi.nlm.nih.gov/32275586/
[2] https://github.com/MIRACLE-Center/High-Resolution-Chest-X-ray-Bone-Suppression
[3] https://www.sciencedirect.com/science/article/abs/pii/S0895611123000046
[4] https://rctd.uic.cu/rctd/user/setLocale/en_US
[5] https://cdn.amegroups.cn/journals/amepc/files/journals/4/articles/67433/public/67433-PB2-9455-R2.pdf
[6] https://pubmed.ncbi.nlm.nih.gov/34888191/
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC9540269/
[8] https://arxiv.org/html/2412.15670v2
[9] https://www.mdpi.com/resolver?pii=diagnostics11050840
[10] https://pubmed.ncbi.nlm.nih.gov/40293914/
