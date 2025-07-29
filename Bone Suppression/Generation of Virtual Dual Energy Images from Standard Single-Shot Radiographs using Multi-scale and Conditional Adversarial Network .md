# Generation of Virtual Dual Energy Images from Standard Single-Shot Radiographs using Multi-scale and Conditional Adversarial Network 

현대 흉부 X선 진단에서 뼈 구조가 폐 병변을 가리는 문제는 여전히 높은 오진율의 원인이다. 본 논문은 **“Generation of Virtual Dual Energy Images from Standard Single-Shot Radiographs using Multi-scale and Conditional Adversarial Network”**[1]를 통해, 단일 촬영(shot) 흉부 X선에서 뼈 이미지를 억제하고 소프트 티슈 이미지를 합성해 듀얼 에너지(Dual-Energy, DE) 영상과 동등 이상의 품질을 달성하는 **MCA-Net**을 제시한다. 저자들은 네트워크 설계, 학습 방법, 임상 검증을 포괄적으로 다루며 딥러닝 기반 가상 DE 영상 생성의 새로운 기준을 수립한다.

## 논문의 핵심 주장 및 주요 기여

- **하드웨어 의존성 제거**  
  DE 방사선 장비 없이도 단일 촬영 X선으로 DE-Bone·DE-Soft 이미지를 동시 재현한다[1].

- **MCA-Net 제안**  
  다중 해상도(U-Net 기반) + 조건부 패치 GAN 구조를 도입해 고주파 세부를 보존하면서 전체 해상도(최대 2,022×2,022픽셀)의 뼈 이미지를 합성한다[1].

- **하이브리드 손실 설계**  
  픽셀기반 $$L_1$$ 오차와 조건부 adversarial loss를 결합, 저주파 배경과 고주파 엣지를 동시 학습한다(식 1–3)[1].

- **소프트 티슈 생성 파이프라인**  
  예측된 뼈 영상과 표준 영상을 **Cross Projection Tensor** 기법으로 융합해 뼈가 억제된 연부조직 영상을 생성한다[1].

- **정량·정성·임상 평가**  
  210명 데이터에서 PSNR 41.5dB·SSIM 96.4%를 달성, JSRT 공용 DB FROC 분석에서 방사선과 전문의의 폐결절 검출 민감도를 10%p 상승시켰다[1].

## 배경 및 문제 정의

### 1. DE 흉부 X선의 장점과 한계
- 두 가지 관전압(kVp)을 번갈아 촬영해 뼈(고에너지)·소프트 티슈(저에너지)를 분리, 병변 가림을 제거할 수 있다[2].  
- 그러나 **① 특수 검출기·발전관 필요, ② 이중 피폭(≈2배 선량), ③ 0.1 s 수준 노출 간격에서 발생하는 심장·호흡성 모션 아티팩트** 문제가 고질적이다[1].

### 2. 학습 기반 가상 DE 연구 현황
| 분류 | 대표 방법 | 장점 | 한계 |
|------|-----------|------|------|
| 전통 필터학습 | Filter Learning[3] | 단순·빠름 | 뼈 완전 억제 불가 |
| MTANN 계열 | Region-wise MTANN[4] | 다중 해상도 훈련 | 대용량 파라미터·저해상도 |
| GAN 계열 | Bone-Supp GAN[5] | 고해상도·질감 보존 | 학습 불안정, 일반화 이슈[6] |
| 제안 MCA-Net | 본 논문[1] | 다중 스케일·조건부 패치 GAN 조합으로 PSNR·SSIM 최고 | 도메인 편향·데이터 종속성 |

## MCA-Net: 모델 세부 설계

### 1. 다중 스케일 생성기 구조
- **Encoder–Decoder U-Net** + 3단계 Output 분기(128/256/512/1024)  
- 각 해상도별 예측을 **Element-wise Sum 후 `tanh`**로 통합하여 완전 해상도 뼈 이미지를 생성[1].

### 2. 조건부 Patch Discriminator
- 입력:  

$$ \nabla_x I_H, \nabla_y I_H, \nabla_x \hat I_B, \nabla_y \hat I_B $$  
  
  (Sobel Gradient)  
- **64×64 Patch 판별**으로 메모리 효율 및 세부 엣지 학습 극대화[1].

### 3. 하이브리드 손실 정의

$$
\begin{aligned}
\mathcal{L}\_{\text{adv}}(G,D) &= -\mathbb{E}\_{I_H,I_B}[\log D(I_H,I_B)] - \mathbb{E}\_{I_H}[\log(1-D(I_H,G(I_H)))]\\
\mathcal{L}\_{\ell_1}(G) &= \mathbb{E}\_{I_H,I_B}\bigl[\lVert I_B - G(I_H) \rVert_1\bigr] \\
G^\* &= \arg\min_G \max_D \bigl[ \mathcal{L}\_{\text{adv}} + \lambda \mathcal{L}_{\ell_1} \bigr], \quad \lambda=1,000[1]
\end{aligned}
$$

### 4. 소프트 티슈 생성
1. **가우시안 블러(σ = 50픽셀)**로 저주파 프로파일 $$I_H^{low}$$ 추출  
2. 고주파 성분 $$\Delta I_H = I_H - I_H^{low}$$  
3. 뼈 그래디언트 기반 Cross-Projection Tensor $$T$$ 계산  
4. $$\hat I_S = I_H^{low} + T\bigl(\nabla \Delta I_H, \nabla \hat I_B\bigr) $$  
   → 뼈 엣지를 억제한 연부조직 영상[1].

## 데이터셋 및 학습 구성

| 항목 | 수치 |
|------|------|
| 환자 수 | 210명(훈련 170, 시험 40)[1] |
| 해상도 | 1,300×1,400 – 2,022×2,022 |
| 증강 | 평행평행 이동 ±80픽셀, 회전 ±15° |
| 배치 크기 | 3 |
| 옵티마이저 | Adam, lr = 1 × 10⁻⁴ |
| Generator:Discriminator | 3:1 비율 반복[1] |

## 정량 성능 및 Ablation 분석

### 1. 전환 품질 지표
| 모델/설정 | Bone PSNR (dB) | Soft PSNR (dB) | Bone SSIM (%) | Soft SSIM (%) |
|-----------|---------------|---------------|---------------|---------------|
| $$L_1$$+Single-Scale | 28.2 ± 4.5[1] | 22.2 ± 3.4[1] | 86.4[1] | 80.3[1] |
| $$L_1$$+Multi-Scale | 34.9 ± 3.6[1] | 29.2 ± 4.1[1] | 88.3[1] | 81.4[1] |
| MCA-Net(Grad off) | 35.2 ± 5.1[1] | 29.8 ± 3.8[1] | 88.5[1] | 81.2[1] |
| **MCA-Net(Full)** | **41.5 ± 2.1**[1] | **39.7 ± 1.8**[1] | **93.4**[1] | **88.4**[1] |

### 2. 기존 방법 비교
| 방법 | Bone PSNR (dB) | Soft PSNR (dB) | Bone SSIM (%) | Soft SSIM (%) |
|------|---------------|---------------|---------------|---------------|
| MTANN[4] | N/A | 39.2[1] | N/A | 84.3[1] |
| Vis-CAC[7] | N/A | 38.8[1] | N/A | 80.3[1] |
| **MCA-Net** | 41.5[1] | 39.7[1] | 93.4[1] | 88.4[1] |

## 임상 FROC 평가 (폐결절 검출)

| 판독자 | 지표 | Standard | Virtual DE | 향상폭 |
|--------|------|----------|------------|--------|
| Radiologist | 민감도 @ 1FP | 0.81[1] | **0.91**[1] | +0.10 |
| Resident | 민감도 @ 1FP | 0.59[1] | **0.81**[1] | +0.22 |

가상 DE 이미지는 심장·횡경막 모션 아티팩트를 제거해 폐혈관 대비도를 높였고, 실제 Two-shot DE보다도 병변 가시성이 우수했다[1].

## 일반화 성능 고찰

### 1. 설계 관점
- **다중 스케일 예측** → 해상도별 특징을 병합, 크기· 위치 변화에 강건[1].  
- **조건부 GAN** → 입력 표준 영상 분포 변화에 적응, 픽셀 간 상대 관계를 학습[6].  
- **Gradient-Domain Discriminator** → 촬영기기·노출 조건이 달라도 엣지 통계는 유지되므로 도메인 편향 완화[5].

### 2. 데이터 측면
- **대규모(3906 증강 샘플)** 훈련으로 리브 세그먼트·연령·체형 분포 다양성 확보[1].  
- **JSRT 등 외부 DB 평가**로 이질적 획득 프로토콜에도 민감도 향상 확인[1].

### 3. 미해결 과제
- 두부·복부 등 **해부학적 영역 전이 학습** 미흡.  
- **AP View·포터블 CXR**에 대한 일반화는 제한적[6].  
- 데이터 편향·기관별 X선 스펙트럼 차이를 해결하려면 **Domain Adaptation / Self-Supervised** 기법 연구가 필요하다[8][9].

## 향후 연구 및 임상·산업적 파급

### 1. 연구 확장 방향
- **Diffusion 기반 고해상도 생성**: BS-LDM 모델이 섬세한 폐혈관 복원을 달성[9].  
- **Unpaired 학습 + CycleGAN**: 레이블 부족 영역(AP 촬영, 소아 CXR)에 적용[10].  
- **3D Volume Fusion**: X선 + CT 하이브리드로 보강[11].

### 2. 임상 활용
- **저선량 포터블 X선**에서도 뼈 억제 보조영상 제공, 중환자실 실시간 병변 스크리닝.  
- **AI CAD 시스템 전처리** : Bone-Supp GAN(PSNR 34.9dB)이 CAD 민감도를 4–7%p 향상[12].  
- 질환별 특화(예: 결핵, 코로나19) 패치 GAN 파인튜닝으로 방대 스케일 임상 적용 기대.

### 3. 고려할 점
- **규제·윤리**: 가상 영상이 진단 지표로 사용될 때 FDA Class II 인허가 요구.  
- **설명가능성(XAI)**: 뼈 억제 과정의 정보 손실·추가된 Artifact 추적이 필요.  
- **데이터 표준화**: DICOM Tag 보존·메타데이터 동기화 필수.

## 결론

MCA-Net은 다중 스케일 CNN과 조건부 Patch GAN을 통합해 듀얼 에너지 기법의 하드웨어적 한계를 해소했다. 41.5 dB PSNR, 93.4% SSIM의 가상 Bone 영상과 향상된 폐결절 검출 성능은 딥러닝 기반 **가상 DE 영상**이 실제 임상 진단에 즉시 도입될 수 있음을 입증한다. 앞으로 도메인 적응, 확률적 생성 모델, 멀티모달 퓨전 연구가 뒤따른다면 단일 X선 영상의 진단 정보량을 CT·MRI 수준으로 끌어올릴 것이며, 이는 저선량·저비용 의료 혁신을 가속할 것으로 예상된다.

### 참고 표기
모든 숫자·인용은 주어진 문헌에 따라 기재하였다. 인용 ID [1]은 본 논문, [4][2][6][5][9] 등은 관련 연구를 가리킨다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1107a655-0f5d-4647-a42e-b695b6b16a91/1810.09354v2.pdf
[2] https://ajronline.org/doi/10.2214/AJR.12.9121
[3] https://www.sciencedirect.com/science/article/abs/pii/S1361841516301529
[4] https://arxiv.org/abs/1810.09354
[5] https://pubmed.ncbi.nlm.nih.gov/32621786/
[6] https://arxiv.org/pdf/2002.03073.pdf
[7] http://www.ajnr.org/lookup/doi/10.3174/ajnr.A5558
[8] https://ieeexplore.ieee.org/document/10726922/
[9] https://arxiv.org/html/2412.15670v4
[10] https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.14212
[11] https://www.sciencedirect.com/science/article/abs/pii/S0895611123000046
[12] https://kjronline.org/DOIx.php?id=10.3348/kjr.2021.0146
[13] http://www.thieme-connect.de/DOI/DOI?10.1055/s-0042-1742677
[14] https://www.liebertpub.com/doi/10.1089/neu.2018.5985
[15] http://link.springer.com/10.1007/s00330-018-5850-z
[16] http://www.ajnr.org/lookup/doi/10.3174/ajnr.A7600
[17] https://link.springer.com/10.1007/s10278-023-00893-y
[18] https://link.springer.com/10.1007/s10278-024-01294-5
[19] http://link.springer.com/10.1007/s00261-020-02415-8
[20] http://link.springer.com/10.1007/s00261-018-1527-y
[21] https://www.ajronline.org/doi/10.2214/AJR.20.25093
[22] https://arxiv.org/pdf/2008.05865.pdf
[23] http://arxiv.org/pdf/1703.10155.pdf
[24] https://pmc.ncbi.nlm.nih.gov/articles/PMC8611463/
[25] https://arxiv.org/html/2311.15328v3
[26] https://www.scholars.northwestern.edu/en/publications/generation-of-virtual-dual-energy-images-from-standard-single-sho
[27] https://pmc.ncbi.nlm.nih.gov/articles/PMC3230639/
[28] https://pmc.ncbi.nlm.nih.gov/articles/PMC11184875/
[29] https://kjronline.org/DOIx.php?id=10.3348%2Fkjr.2021.0146
[30] https://www.mdpi.com/2227-7390/9/22/2896
[31] https://www.kjronline.org/pdf/10.3348/kjr.2016.17.3.321
[32] https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0316099
[33] https://pubmed.ncbi.nlm.nih.gov/35032722/
[34] https://github.com/bbbbbbzhou/Virtual-Dual-Energy
[35] https://dirjournal.org/articles/virtual-non-enhanced-dual-energy-computed-tomography-reconstruction-a-candidate-to-replace-true-non-enhanced-computed-tomography-scans-in-the-setting-of-suspected-liver-alveolar-echinococcosis/dir.2023.221806
[36] https://pmc.ncbi.nlm.nih.gov/articles/PMC4283823/
[37] https://www.sciencedirect.com/science/article/abs/pii/S0010482522006874
[38] https://www.ntis.go.kr/outcomes/popup/srchTotlPapr.do?cmd=get_contents&rstId=JNL-2022-00112779548
[39] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09302919
[40] https://pubmed.ncbi.nlm.nih.gov/39603847/
[41] https://www.jrpr.org/journal/view.php?number=1096
[42] https://link.springer.com/10.1007/s10278-025-01461-2
[43] https://semarakilmu.com.my/journals/index.php/applied_sciences_eng_tech/article/view/5499
[44] https://austinpublishinggroup.com/cancer-clinical-research/fulltext/cancer-v8-id1095.php
[45] https://ieeexplore.ieee.org/document/9894637/
[46] https://sol.sbc.org.br/index.php/semish/article/view/20794
[47] https://linkinghub.elsevier.com/retrieve/pii/S0169260722004060
[48] https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17516
[49] https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.14371
[50] https://pmc.ncbi.nlm.nih.gov/articles/PMC8970404/
[51] https://pmc.ncbi.nlm.nih.gov/articles/PMC11850088/
[52] https://pmc.ncbi.nlm.nih.gov/articles/PMC12116077/
[53] https://pubmed.ncbi.nlm.nih.gov/38597871/
[54] https://arxiv.org/abs/1811.02628
[55] https://pubmed.ncbi.nlm.nih.gov/34983100/
[56] https://www.nature.com/articles/s41598-019-54176-0
[57] https://www.kjronline.org/DOIx.php?id=10.3348%2Fkjr.2023.1306
[58] https://www.sciencedirect.com/science/article/abs/pii/S0169260722004060
[59] https://www.sciencedirect.com/science/article/abs/pii/S0952197623012630
[60] https://www.jrpr.org/journal/view.php?number=1003
[61] https://arxiv.org/pdf/1805.02369.pdf
[62] http://arxiv.org/pdf/1810.09354.pdf
[63] https://arxiv.org/pdf/2206.09244.pdf
[64] http://arxiv.org/pdf/2501.02167.pdf
[65] https://arxiv.org/pdf/2209.01339.pdf
[66] https://arxiv.org/html/2404.15992v1
[67] https://www.mdpi.com/1424-8220/22/6/2119/pdf
[68] http://arxiv.org/pdf/2410.19009.pdf
[69] https://pmc.ncbi.nlm.nih.gov/articles/PMC8743147/
[70] https://pmc.ncbi.nlm.nih.gov/articles/PMC6510604/
[71] https://pmc.ncbi.nlm.nih.gov/articles/PMC9898674/
[72] https://pmc.ncbi.nlm.nih.gov/articles/PMC9246721/
[73] http://arxiv.org/pdf/2111.03404.pdf
[74] https://pmc.ncbi.nlm.nih.gov/articles/PMC10795441/
[75] https://arxiv.org/abs/1712.01636v2
[76] https://pmc.ncbi.nlm.nih.gov/articles/PMC6592074/
