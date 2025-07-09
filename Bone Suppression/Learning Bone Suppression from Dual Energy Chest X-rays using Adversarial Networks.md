# Learning Bone Suppression from Dual Energy Chest X-rays using Adversarial Networks 

## 핵심 주장과 주요 기여

### 해결하고자 하는 문제

**기존 Dual Energy Imaging의 한계:**
- 환자가 두 번의 방사선 노출을 받아야 함[1]
- 두 촬영 간격 사이의 심박동으로 인한 아티팩트 발생[1]
- 전문 장비가 필요하여 비용이 높음[1]
- 단일 에너지 흉부 X선에서 늑골과 쇄골 등의 골격 구조가 폐 병변 진단을 방해함[1]

### 제안하는 방법

**1. 조건부 생성적 적대 신경망 (Conditional GAN) 활용:**

생성자(Generator)의 목적 함수는 다음과 같이 정의됩니다[1]:

$$ J^{(G)} = -\frac{1}{2}E_{z \sim p_z}[\log D(G(z))] $$

판별자(Discriminator)의 목적 함수는[1]:

$$ J^{(D)}(\theta^{(D)}, \theta^{(G)}) = -\frac{1}{2}E_{x \sim p_{data}}[\log D(x)] - \frac{1}{2}E_{z \sim p_z}[\log(1-D(G(z)))] $$

조건부 GAN의 최종 목적 함수는[1]:

$$ G^* = \arg\min_G \max_D V(D,G) + \lambda L_1 $$

여기서 $$L_1 = E_{x,y \sim p_{data}, z \sim p_z}[||x - G(y,z)||_1] $$ 는 L1 거리 손실입니다[1].

**2. Haar 2D 웨이블릿 분해 (Haar 2D Wavelet Decomposition):**

입력 이미지를 네 개의 주파수 성분으로 분해합니다[1]:
- 저주파 성분 (LL): 서브샘플링된 원본 이미지
- 고주파 성분 (LH, HL, HH): 수직, 수평, 대각선 방향의 세부 정보

이를 통해 모델이 주파수 세부 정보를 인식적 가이드라인으로 활용하여 빠르고 효율적으로 수렴할 수 있도록 합니다[1].

## 모델 구조

### 생성자 (Generator) 아키텍처[1]:
- **입력**: 1024×1024 그레이스케일 이미지 → Haar 2D 웨이블릿 분해로 512×512×4 변환
- **기본 구조**: U-Net 기반의 합성곱 오토인코더 + skip connections
- **구성 요소**: 
  - 12개의 잔여 블록 (Residual Blocks)
  - 중앙에 attention block (squeeze and excitation block)
  - Skip connections으로 고주파 정보를 디코더로 전달

### 판별자 (Discriminator) 아키텍처[1]:
- **구성**: 7개의 합성곱 층 + 1개의 완전연결층
- **입력**: Haar 2D 웨이블릿 분해된 4개 성분
- **특징**: History buffer와 minibatch discrimination 추가

### 개선 기법

**1. History Buffer[1]:**
과거 생성된 이미지들을 버퍼에 저장하여 mode collapse 방지:

$$ \text{Buffer size} = 2k \text{ (배치 크기와 동일)} $$

**2. Minibatch Discrimination[1]:**
배치 내 이미지들 간의 거리를 측정하여 다양성 증진:

$$ o(x_i) = \sum_{j=1}^n -e^{(||M_{i,b} - M_{j,b}||_1)} \in \mathbb{R}^B $$

## 성능 향상 및 실험 결과

### 정량적 성능 지표[1]:

| 모델 | PSNR | PSNR (Lung) | SSIM (Lung) |
|------|------|-------------|-------------|
| CNN | 19.229 | 26.350 | 0.9031 |
| CNN + Haar Wavelets | 22.289 | 25.840 | 0.7906 |
| CNN + GAN | 21.477 | 26.343 | 0.8496 |
| **CNN + GAN + Haar Wavelets (제안 방법)** | **24.080** | **28.582** | **0.9304** |

### 데이터셋[1]:
- **총 환자 수**: 348명
- **훈련/검증/테스트**: 80%/10%/10% 분할
- **이미지 크기**: 2017×2017 → 1024×1024로 재조정
- **형식**: DICOM 형식에서 선형 윈도잉 적용

## 모델의 일반화 성능 향상 가능성

### 1. 웨이블릿 기반 주파수 분석의 효과[1]:
- 다양한 주파수 성분에서의 특징 추출로 다양한 영상 조건에 대한 강인성 향상
- 고주파 세부 정보 보존으로 미세한 병변 검출 능력 개선

### 2. 적대적 훈련의 장점[1]:
- 픽셀 단위 손실 함수의 한계(흐릿한 이미지 생성) 극복
- 더 사실적이고 선명한 bone suppression 이미지 생성

### 3. 한계점과 개선 방안[1]:
- **데이터 부족**: 348명의 상대적으로 작은 데이터셋
- **일반화 한계**: 다양한 X선 장비와 촬영 조건에 대한 검증 필요
- **계산 복잡도**: 실시간 처리를 위한 최적화 필요

## 한계점

### 1. 데이터셋 크기의 제약[1]:
- 상대적으로 작은 데이터셋으로 인한 일반화 성능 한계
- 다양한 병리학적 조건에 대한 검증 부족

### 2. 기술적 한계[1]:
- 작은 혈관의 윤곽 캐처에 실패하는 경우 존재
- Motion artifact 있는 이미지에서의 성능 검증 부족

### 3. 임상 검증의 부족[1]:
- 실제 진단 성능 개선에 대한 임상적 평가 부족
- 다양한 폐 질환에 대한 진단 정확도 향상 효과 미검증

## 향후 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

**1. 의료 영상 처리 분야의 새로운 접근법 제시:**
- GAN과 웨이블릿 변환의 결합이 의료 영상 품질 개선에 효과적임을 입증[2][3][4]
- Bone suppression 외에도 다양한 의료 영상 향상 기법에 응용 가능[5][6]

**2. 비용 효율적인 대안 제시:**
- 기존 dual energy 시스템의 고비용 문제 해결[7]
- 단일 촬영으로 bone suppression 구현하여 방사선 노출량 감소[1]

**3. 후속 연구들의 발전:**
- 최근 연구들에서 더 발전된 GAN 모델 적용[8][9][10]
- Diffusion model과의 결합[9], Transformer 기반 접근법[11] 등으로 확장

### 앞으로 연구 시 고려할 점

**1. 데이터셋 확장과 다양성:**
- 더 큰 규모의 다기관 데이터셋 구축 필요[12][13]
- 소아 환자 데이터 포함하여 연령대별 일반화 성능 검증[12]
- 다양한 X선 장비와 촬영 프로토콜에 대한 robustness 평가

**2. 모델 성능 개선:**
- 최신 GAN 아키텍처 활용 (StyleGAN, Progressive GAN 등)[8][10]
- Attention mechanism 강화[8][11]
- Multi-scale feature extraction 개선[8][14]

**3. 임상적 검증:**
- 실제 진단 정확도 개선 효과 정량적 측정[3][4]
- 방사선과 의사의 주관적 평가 포함[15][16]
- COVID-19, 폐렴 등 특정 질환에 대한 진단 성능 평가[17][18]

**4. 계산 효율성:**
- 실시간 처리를 위한 모델 경량화[18][19]
- Edge computing 환경에서의 배포 고려
- GPU 메모리 사용량 최적화[20]

**5. 새로운 기술과의 융합:**
- Vision Transformer와의 결합[11][21]
- Diffusion model 활용[9][22]
- Self-supervised learning 적용으로 라벨링 의존성 감소[13]

이 논문은 의료 영상 처리 분야에서 GAN과 웨이블릿 변환의 효과적인 결합을 보여준 초기 연구로서, 후속 연구들이 더 발전된 기술과 대규모 데이터셋을 활용하여 임상적으로 실용적인 솔루션을 개발하는 데 중요한 기반을 제공했습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3b0b5184-26ff-4ef4-835b-53dcbff84dca/1811.02628v1.pdf
[2] https://www.semanticscholar.org/paper/7a72defecffcbc377ee2ed781bf6315982a77bb8
[3] https://pubmed.ncbi.nlm.nih.gov/34983100/
[4] https://kjronline.org/DOIx.php?id=10.3348%2Fkjr.2021.0146
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC10055771/
[6] https://pubs.rsna.org/doi/full/10.1148/rg.2021200151
[7] https://kaimaging.com/bone-suppression-software-vs-spectraldr-modern-methods-of-bone-subtraction/
[8] https://ieeexplore.ieee.org/document/10726922/
[9] https://ieeexplore.ieee.org/document/10635371/
[10] https://pmc.ncbi.nlm.nih.gov/articles/PMC11850088/
[11] https://www.computer.org/csdl/journal/ai/2025/03/10726922/21frsCGmuwo
[12] https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17516
[13] https://www.sciencedirect.com/science/article/abs/pii/S0895611123000046
[14] https://ieeexplore.ieee.org/document/10295958/
[15] https://koreascience.kr/article/JAKO202218577243628.page
[16] https://pubmed.ncbi.nlm.nih.gov/36731328/
[17] https://sol.sbc.org.br/index.php/semish/article/view/20794
[18] https://austinpublishinggroup.com/cancer-clinical-research/fulltext/cancer-v8-id1095.php
[19] https://www.mdpi.com/2072-666X/12/11/1418
[20] https://arxiv.org/abs/2104.10268
[21] https://dl.acm.org/doi/10.1145/3731867.3731883
[22] https://arxiv.org/html/2412.15670v2
[23] https://www.sec.gov/Archives/edgar/data/1419554/000164117225017063/form424b4.htm
[24] https://www.sec.gov/Archives/edgar/data/1419554/000149315225008383/form10-k.htm
[25] https://www.sec.gov/Archives/edgar/data/1840425/000121390025034116/ea0237686-10k_osrhold.htm
[26] https://www.sec.gov/Archives/edgar/data/1614744/000121390025022192/ea0232483-20f_purple.htm
[27] https://www.sec.gov/Archives/edgar/data/1794546/000095012325006117/carl_-_s-1_-_june_2025.htm
[28] https://www.sec.gov/Archives/edgar/data/1664710/000166471025000018/kros-20241231.htm
[29] https://www.sec.gov/Archives/edgar/data/1835022/000095017025040782/coya-20241231.htm
[30] https://link.springer.com/10.1007/s10278-025-01508-4
[31] https://link.springer.com/10.1007/978-3-030-61609-0_20
[32] https://academic.oup.com/bjr/article/7451560
[33] https://www.semanticscholar.org/paper/88c2621c23cd092db8431ce5dc4b77d594970573
[34] https://arxiv.org/abs/1811.02628
[35] https://pure.korea.ac.kr/en/publications/generating-dual-energy-subtraction-soft-tissue-images-from-chest-
[36] https://arxiv.org/abs/2002.03073
[37] https://www.sciencedirect.com/science/article/abs/pii/S0169260722004060
[38] https://github.com/diaoquesang/A-detailed-summarization-about-bone-suppression-in-Chest-X-rays
[39] https://qims.amegroups.org/article/view/67433/html
[40] https://www.semanticscholar.org/paper/Learning-Bone-Suppression-from-Dual-Energy-Chest-Oh-Yun/7a72defecffcbc377ee2ed781bf6315982a77bb8
[41] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002797427
[42] https://www.sec.gov/Archives/edgar/data/1697532/000095017025070557/aplt-20250331.htm
[43] https://www.sec.gov/Archives/edgar/data/1828253/000117184325001916/evax20241231_20f.htm
[44] https://www.sec.gov/Archives/edgar/data/2032020/000119312525068914/d830581d10k.htm
[45] https://www.sec.gov/Archives/edgar/data/1227636/000155837025003833/stim-20241231x10k.htm
[46] https://www.sec.gov/Archives/edgar/data/1709626/000095017025042709/ncna-20241231.htm
[47] https://www.sec.gov/Archives/edgar/data/1863006/000121390025017221/ea0231157-20f_valenssemi.htm
[48] https://www.sec.gov/Archives/edgar/data/1745999/000095017025026076/beam-20241231.htm
[49] https://onlinelibrary.wiley.com/doi/10.1111/1759-7714.70014
[50] https://dl.acm.org/doi/10.1145/3503161.3547925
[51] https://www.mdpi.com/2076-3417/11/11/5237
[52] https://ieeexplore.ieee.org/document/10549172/
[53] http://thesai.org/Publications/ViewPaper?Volume=14&Issue=7&Code=IJACSA&SerialNo=104
[54] https://www.mdpi.com/2076-3417/11/11/5237/pdf?version=1622809584
[55] https://www.kjronline.org/DOIx.php?id=10.3348%2Fkjr.2023.1306
[56] https://arxiv.org/abs/2005.10687
[57] https://pubmed.ncbi.nlm.nih.gov/37985163/
[58] https://pubmed.ncbi.nlm.nih.gov/35032722/
[59] https://www.nature.com/articles/s41598-022-13658-4
[60] https://arxiv.org/abs/1809.07294
[61] https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.14212
[62] https://www.nature.com/articles/s41598-023-36785-y
[63] https://www.sciencedirect.com/science/article/pii/S1361841518308430
[64] https://www.sciencedirect.com/science/article/pii/S1078817424002670
[65] https://www.sec.gov/Archives/edgar/data/1419554/000164117225009746/form10-q.htm
[66] https://www.sec.gov/Archives/edgar/data/1419554/000164117225017165/form8-k.htm
[67] https://www.sec.gov/Archives/edgar/data/1419554/000164117225005775/formdef14a.htm
[68] https://www.sec.gov/Archives/edgar/data/1419554/000164117225013937/form8-k.htm
[69] https://www.sec.gov/Archives/edgar/data/1419554/000149315224007432/form10-k.htm
[70] https://ieeexplore.ieee.org/document/10959743/
[71] https://onlinelibrary.wiley.com/doi/10.1002/ima.22501
[72] https://ieeexplore.ieee.org/document/10047368/
[73] https://www.mdpi.com/2313-433X/9/2/32
[74] https://www.nature.com/articles/s41598-023-49534-y
[75] https://link.springer.com/10.1007/s11548-023-02958-3
[76] https://linkinghub.elsevier.com/retrieve/pii/S0010482525007437
[77] https://pmc.ncbi.nlm.nih.gov/articles/PMC8611463/
[78] https://openreview.net/forum?id=pXEnurdRAx&noteId=g7VTptcqsr
[79] https://arxiv.org/pdf/2111.03404.pdf
[80] https://pmc.ncbi.nlm.nih.gov/articles/PMC7931407/
[81] https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Fu_DW-GAN_A_Discrete_Wavelet_Transform_GAN_for_NonHomogeneous_Dehazing_CVPRW_2021_paper.pdf
[82] https://arxiv.org/abs/2410.17966
[83] https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0265691
[84] https://core.ac.uk/download/480696668.pdf
[85] https://www.themoonlight.io/ko/review/generative-adversarial-wavelet-neural-operator-application-to-fault-detection-and-isolation-of-multivariate-time-series-data
[86] https://www.sciencedirect.com/science/article/abs/pii/S0957417424004056
[87] https://www.nature.com/articles/s41598-025-87240-z
[88] https://www.sciencedirect.com/science/article/pii/S2667305322000850
[89] https://kjronline.org/DOIx.php?id=10.3348/kjr.2021.0146
[90] https://www.mdpi.com/2227-9059/10/9/2323
[91] https://arxiv.org/pdf/2002.03073.pdf
[92] https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/acm2.14212
[93] https://pmc.ncbi.nlm.nih.gov/articles/PMC10795441/
[94] https://pmc.ncbi.nlm.nih.gov/articles/PMC8743147/
[95] https://www.mdpi.com/2077-0383/14/6/2091
[96] https://pmc.ncbi.nlm.nih.gov/articles/PMC6510604/
[97] http://arxiv.org/pdf/1906.10089.pdf
[98] https://www.mdpi.com/2075-4418/13/1/159/pdf?version=1672736840
[99] https://pmc.ncbi.nlm.nih.gov/articles/PMC8970404/
[100] https://arxiv.org/abs/2310.17216
[101] https://pmc.ncbi.nlm.nih.gov/articles/PMC9992336/
[102] https://www.mdpi.com/1424-8220/25/5/1567
[103] https://www.mdpi.com/2075-4418/12/5/1121/pdf?version=1651307320
[104] https://www.mdpi.com/2077-0383/13/12/3556/pdf?version=1718696249
[105] https://pmc.ncbi.nlm.nih.gov/articles/PMC8204145/
[106] https://pmc.ncbi.nlm.nih.gov/articles/PMC11204848/
[107] https://pmc.ncbi.nlm.nih.gov/articles/PMC9235085/
[108] http://medrxiv.org/lookup/doi/10.1101/2021.11.17.21266472
[109] https://link.springer.com/10.1007/978-3-030-32226-7_31
[110] https://jurnal.polibatam.ac.id/index.php/JAGI/article/view/307
[111] http://downloads.hindawi.com/journals/acisc/2017/9571262.pdf
[112] https://www.mdpi.com/1099-4300/24/12/1754/pdf?version=1669806085
[113] https://pmc.ncbi.nlm.nih.gov/articles/PMC9380794/
[114] https://content.sciendo.com/downloadpdf/journals/amns/5/2/article-p435.pdf
[115] https://pmc.ncbi.nlm.nih.gov/articles/PMC8598340/
[116] https://pmc.ncbi.nlm.nih.gov/articles/PMC9105730/
[117] https://openmedicalimagingjournal.com/VOLUME/7/PAGE/9/PDF/
[118] https://pmc.ncbi.nlm.nih.gov/articles/PMC4329222/
[119] https://pmc.ncbi.nlm.nih.gov/articles/PMC7522455/
