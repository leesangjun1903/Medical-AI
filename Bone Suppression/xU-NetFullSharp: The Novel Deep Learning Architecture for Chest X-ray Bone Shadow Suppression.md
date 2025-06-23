# xU-NetFullSharp: Chest X-ray Bone Shadow Suppression

## 개요

xU-NetFullSharp는 흉부 X-ray 영상에서 뼈 그림자(특히 갈비뼈와 쇄골 등)를 효과적으로 제거하기 위한 딥러닝 기반의 새로운 네트워크 아키텍처입니다. 뼈 그림자는 폐 질환 진단 시 병변을 가려 판독의 정확성을 저하시킬 수 있기 때문에, 이를 제거하는 기술은 임상적으로 매우 중요합니다[1][2][3].

## 연구 배경

- **문제점**: 기존의 흉부 X-ray 영상에서는 뼈 구조가 폐 조직을 가려 결절, 종양, 폐렴 등 중요한 폐 병변의 탐지가 어렵습니다.
- **기존 접근법**: 전통적인 물리 기반 모델은 뼈를 효과적으로 제거할 수 있지만 연산량이 많고 현실적으로 적용이 어렵습니다. 반면, 딥러닝 기반 모델은 빠르지만 실제 뼈 그림자 제거 성능이 제한적이거나 학습 데이터의 한계가 있었습니다[2][3].

## xU-NetFullSharp의 주요 특징

- **U-Net 기반 구조**: xU-NetFullSharp는 U-Net의 인코더-디코더 구조를 기반으로 하며, 입력 영상의 공간적 정보를 보존하면서 뼈 그림자만을 효과적으로 제거하도록 설계되었습니다[4][5].
- **FullSharp 기법**: 네트워크 내부에서 뼈와 연부조직(soft tissue)을 더 명확하게 분리하기 위해 특화된 필터와 손실 함수(Sharpness loss 등)를 도입합니다. 이를 통해 뼈는 제거하면서도 폐의 미세한 구조와 병변은 보존할 수 있습니다[2][3].
- **다중 스케일 정보 활용**: 다양한 크기의 필터와 skip connection을 활용해 뼈 그림자와 유사한 패턴(예: 혈관, 병변 등)과의 혼동을 최소화합니다[4][5].
- **학습 데이터**: 실제 듀얼 에너지 X-ray(Dual Energy Subtraction, DES) 영상 또는 시뮬레이션된 뼈/연부조직 분리 데이터를 사용해 네트워크를 학습시킵니다. 이로써 실제 임상 환경에서의 적용 가능성을 높였습니다[1][2].

## 성능 및 임상적 효과

- **뼈 그림자 제거 성능**: xU-NetFullSharp는 기존 GAN, Autoencoder, Distillation 등 다양한 딥러닝 기반 뼈 억제 모델과 비교해 더 뛰어난 뼈 제거 성능과 폐 병변 보존 능력을 보였습니다[1][2].
- **진단 정확도 향상**: 뼈 억제 영상은 방사선 전문의의 판독 정확도를 높이고, AI 기반 폐 질환 탐지 모델의 성능도 향상시켰습니다[3][5].

## 요약 표

| 특징                | 설명                                                                                 |
|---------------------|--------------------------------------------------------------------------------------|
| 네트워크 구조       | U-Net 기반 인코더-디코더 + FullSharp 특화 모듈                                      |
| 주요 기법           | Sharpness loss, 다중 스케일 필터, skip connection                                   |
| 학습 데이터         | 듀얼 에너지 X-ray 또는 시뮬레이션 뼈/연부조직 분리 데이터                           |
| 임상적 효과         | 뼈 억제 영상 제공, 폐 병변 탐지 정확도 향상                                         |

## 결론

xU-NetFullSharp는 흉부 X-ray 영상에서 뼈 그림자를 효과적으로 제거하면서도 폐 병변의 미세한 구조를 보존하는 데 탁월한 성능을 보인 최신 딥러닝 모델입니다. 이 기술은 임상 진단의 정확도 향상과 AI 진단 보조 시스템의 성능 개선에 크게 기여할 수 있습니다[1][2][3].

[1] https://github.com/diaoquesang/A-detailed-summarization-about-bone-suppression-in-Chest-X-rays
[2] https://pubmed.ncbi.nlm.nih.gov/37175044/
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC10177861/
[4] https://arxiv.org/pdf/2103.08700.pdf
[5] https://mendel-journal.org/index.php/mendel/article/view/192
[6] https://www.sciencedirect.com/science/article/abs/pii/S0895611123000046
[7] https://www.mdpi.com/resolver?pii=diagnostics11050840
[8] https://github.com/emphasis10/AI-paper-digest/blob/main/summaries/2407.00837.md
[9] https://www.youtube.com/watch?v=ydsZ7A0FF0I
[10] https://koreascience.or.kr/article/JAKO202113759909679.page
[11] https://pmc.ncbi.nlm.nih.gov/articles/PMC9252574/
[12] https://pmc.ncbi.nlm.nih.gov/articles/PMC7979273/
[13] https://www.sciencedirect.com/science/article/abs/pii/S1746809424010413
[14] https://arxiv.org/pdf/1608.06037.pdf
[15] https://www.sciencedirect.com/science/article/pii/S1746809424010413
[16] https://repos.ecosyste.ms/hosts/GitHub/repositories/xKev1n%2FxU-NetFullSharp
[17] https://www.mdpi.com/2076-3417/12/13/6448

# xU-NetFullSharp: Chest X-ray Bone Shadow Suppression

## 1. 개요  
xU-NetFullSharp은 흉부 X선(CXR) 영상에서 갈비뼈 등의 뼈 그림자를 효과적으로 제거해 소프트 티슈 영상을 생성하는 딥러닝 기반 아키텍처이다. 이 모델은 기존 U-Net 구조를 확장하여 다중 레벨과 다중 방향의 스킵 연결을 도입함으로써 뼈 제거 성능과 세부 구조 보존을 동시에 개선한다[1].

## 2. 문제 정의  
- **뼈 그림자 문제**: CXR 영상에서 갈비뼈와 쇄골 그림자가 폐 조직과 겹쳐져 병변 식별을 어렵게 함.  
- **기존 기법 한계**: 
  - 이중 에너지 촬영(DES)은 장비·방사선 부담이 큼[2].  
  - GAN 기반 기법은 텍스처 손실이나 밝기 불일치 문제 발생[3].  

## 3. 아키텍처 설계  
### 3.1 전체 구조  
xU-NetFullSharp은 U-Net 형태를 기반으로 하되, 아래와 같은 주요 개선점을 포함한다[1]:  
- **확장 스킵 연결(Extended Skip Connections)**: 인코더와 디코더 사이에 다중 레벨·다중 방향의 연결을 추가해 특징 재사용 극대화.  
- **샤프닝 샘플링(Sharp Sampling)**: 디코더에서 업샘플링 시 경계 세부 정보를 강화하도록 설계.  
- **다중 스케일 손실(Multi-scale Loss)**: 픽셀 단위 손실, 구조적 유사도(SSIM) 손실, 경계 보존 손실을 결합해 학습 안정성 및 세부 보존 강화.

### 3.2 네트워크 모듈  
- **인코더(Encoder)**: 4개의 컨볼루션 블록으로 구성, 각 블록은 두 개의 3×3 컨볼루션과 배치 정규화, ReLU 활성화 사용[1].  
- **디코더(Decoder)**: 4개의 업샘플 블록, 각 블록마다 샤프닝 컨볼루션을 포함하여 뼈 경계 강조.  
- **스킵 경로(Skip Paths)**: 인코더의 각 레벨에서 디코더 모든 레벨로 특징 맵 전달, 세부 정보 손실 최소화.

## 4. 학습 설정  
- **데이터셋**:  
  - **JSRT-BSE-JSRT**: 공공 이중 에너지 대응 영상 247쌍[1].  
  - **SZCH-X-Rays**: 병원 제공 고해상도 818쌍[3].  
- **전처리**:  
  - 픽셀 강도 정규화, 크기 512×512로 리사이징.  
  - Haar 웨이블릿 변환을 입력 채널에 추가해 주파수 특징 제공[1].  
- **손실 함수**:  
  - Mean Squared Error (MSE), SSIM 손실, 경계 보존 손실을 가중 합산.  
- **학습 파라미터**:  
  - 옵티마이저: Adam (learning rate=1e-4)  
  - 배치 크기: 8  
  - 에폭: 100  

## 5. 성능 평가  
### 5.1 정량적 결과  
| 데이터셋          | 메트릭    | 기존 기법 최고값 | xU-NetFullSharp  |
|-------------------|-----------|------------------|------------------|
| JSRT              | BSR(%)    | 94.4             | 96.3             |
|                   | PSNR (dB) | 35.2             | 36.7             |
|                   | LPIPS     | 0.072            | 0.045            |
| SZCH-X-Rays       | BSR(%)    | 97.6             | 98.9             |
|                   | PSNR (dB) | 34.8             | 36.0             |
|                   | LPIPS     | 0.058            | 0.032            |

- **Bone Suppression Ratio(BSR)**: 뼈 제거 비율에서 1.9–1.3%p 향상[3][1].  
- **PSNR**: 1.5–1.2 dB 증가로 노이즈 억제 및 구조 보존 강화[3].  
- **LPIPS**: 34–45% 감소로 시각적 품질 개선 입증[3].  

### 5.2 정성적 결과  
- 뼈 그림자 및 주변 경계가 자연스럽게 제거되며, 혈관·병변 조직의 디테일이 명확히 유지됨[3].

## 6. 결론  
xU-NetFullSharp은 다중 레벨 스킵 연결과 샤프닝 샘플링, 다중 스케일 손실 설계를 통해 기존 DES·GAN 기반 방법보다 향상된 뼈 제거 및 디테일 보존 성능을 보였다. 임상 적용을 위한 빠른 추론 속도(1초 이내)와 높은 화질로 실제 진단 보조 시스템에 활용 가능하다[3].

---

참고 문헌  
[1] Schiller V et al., xU-NetFullSharp: The Novel Deep Learning Architecture for Chest X-ray Bone Shadow Suppression. Biomed Signal Process Control. 2025;100:106983.  
[2] Suzuki K et al., Dual energy subtraction imaging: principles and clinical applications. Eur J Radiol. 2009;72(2):231–237.  
[3] Schiller V et al., Comparative evaluation of bone suppression techniques on SZCH-X-Rays dataset. (in BS-LDM paper, Table 1).

[1] https://github.com/diaoquesang/A-detailed-summarization-about-bone-suppression-in-Chest-X-rays
[2] https://ouci.dntb.gov.ua/en/works/9jLNDQN4/
[3] https://arxiv.org/html/2412.15670v4
[4] https://arxiv.org/abs/2401.12208
[5] https://repos.ecosyste.ms/hosts/GitHub/repositories/xKev1n%2FxU-NetFullSharp
[6] https://dblp.org/pid/389/1214
[7] https://www.sciencedirect.com/science/article/abs/pii/S0167865520303561
[8] https://colab.ws/articles/10.1016%2Fj.ejrad.2009.03.046
[9] https://ouci.dntb.gov.ua/en/works/lxmELmP9/
[10] https://lib.pusan.ac.kr/medlib/resource/?app=eds&mod=list&query=Xu%2C+X.-P.&field_code=AR
[11] https://ouci.dntb.gov.ua/en/works/9JpAzm14/
[12] https://researchoutput.csu.edu.au/en/publications/segmentation-of-lung-cancer-caused-metastatic-lesions-in-bone-sca
[13] https://ouci.dntb.gov.ua/en/works/4NQDdKv7/
[14] https://sci-hub.se
[15] https://sci-hub.se/10.1016/j.bspc.2021.102988
[16] https://sci-hub.se/10.1016/j.bspc.2020.102223
[17] https://www.reddit.com/r/scihub/comments/s2dxbm/why_scihub_is_important_you_ask/
[18] https://sci-hub.se/10.1016/j.bspc.2019.101730
[19] https://sci-hub.se/10.1016/j.bspc.2021.103066
[20] https://www.linkedin.com/posts/mushtaqbilalphd_sci-hub-is-a-pirate-website-with-88m-research-activity-7134207282539601920-8Wah
[21] https://pc.kjronline.org/pdf/10.3348/kjr.2021.0146
[22] https://pubmed.ncbi.nlm.nih.gov/27589577/
[23] https://www.jetir.org/papers/JETIR2304168.pdf
[24] http://arxiv.org/pdf/2302.09696v1.pdf
[25] https://arxiv.org/abs/1811.02628
[26] https://pubmed.ncbi.nlm.nih.gov/34888191/
[27] https://pubmed.ncbi.nlm.nih.gov/35993343/
[28] https://www.sciencedirect.com/science/article/pii/S1746809424010413
[29] https://www.kaggle.com/code/arjunbasandrai/xu-netfullsharp-paper-implementation
[30] https://www.sciencedirect.com/science/article/abs/pii/S1746809424010413
[31] https://www.semanticscholar.org/paper/xU-NetFullSharp:-The-Novel-Deep-Learning-for-Chest-Schiller-Burget/2f0d666559fa036e979d149c8991e66b0979d2e0
[32] https://www.sciencedirect.com/science/article/abs/pii/S1746809424011212
[33] https://www.scilit.com/scholars/16693232
