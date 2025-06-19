## Comparison of Various GAN‐Based Bone Suppression Imaging for High‐Accurate Markerless Motion Tracking of Lung Tumors in CyberKnife Treatment 

이 연구는 **골격 구조의 간섭으로 인한 CyberKnife 시스템의 폐 종양 추적 오류 문제**를 해결하기 위해 다양한 GAN(Generative Adversarial Network) 모델을 활용한 골 억제 영상 기술을 비교 분석했습니다. 주요 내용은 다음과 같습니다:

### 1. 연구 배경 및 목적
- **문제점**: 폐암의 정위체부방사선치료(SBRT) 시 호흡으로 인한 종양 이동이 발생하며, CyberKnife의 무표지 추적(fiducial-free tracking) 방식은 **골격과 종양의 중첩 시 추적 정확도가 저하**됩니다[1][5].
- **해결 방안**: kV X선 영상에서 골 구조를 억제하면서 종양 가시성을 유지하는 **GAN 기반 영상 처리 기술**을 개발해 추적 정확도를 향상시키는 것이 목적입니다[1][4].

### 2. 연구 방법
- **데이터셋 구축**:
  - 4D XCAT 팬텀(가상 인체 모델)을 이용해 **56개 사례**에 대한 CT 영상 생성(골 구조 포함/제거 버전)[1][4].
  - 좌/우 45° 경사 방향으로 X선 투영 영상 캡처 → **1,120개 이미지** 분할[4].
- **GAN 모델 비교**:
  - 6가지 GAN 아키텍처(CycleGAN, DualGAN, CUT, FastCUT, DCLGAN, SimDCL)를 활용해 골 억제 영상(BSIphantom) 생성[1][4].
  - **성능 평가 지표**: 구조적 유사도 지수(SSIM), 피크 신호 대 잡음비(PSNR), 프레셰 시작 거리(FID)[1][4].
- **환자 데이터 검증**:
  - 실제 환자 1,000개 치료 영상으로 골 억제 영상(BSIpatient) 생성[3][4].
  - 템플릿 매칭을 통한 **상호상관계수(ZNCC)** 로 추적 정확도 비교[3][4].

### 3. 연구 결과
- **팬텀 데이터 평가**:
  - BSIphantom은 SSIM **0.96 ± 0.02**, PSNR **36.93 ± 3.93**으로 골 제거 영상과 높은 유사성[1][4].
  - **FID 점수 기준 SimDCL이 68.93로 최적 성능** (낮을수록 우수)[4].
- **환자 데이터 적용 성능**:
  - BSIpatient의 ZNCC는 **0.773 ± 0.143**으로 실제 치료 영상(**0.763 ± 0.136**)보다 우수[3].
  - 6개 모델 중 5개에서 ZNCC가 유의미하게 향상됨[3][4].

### 4. 결론 및 의의
- **기술적 성과**: GAN 기반 골 억제 영상은 **종양과 골 구조의 분리 정확도를 극대화**하여 CyberKnife의 추적 오류를 감소시킵니다[1][3][4].
- **임상적 기여**: 이 기술은 **방사선 조사 영역 정밀도 향상**을 통해 정상 조직 손상 위험을 줄이고, **무표지 추적의 신뢰성**을 제고합니다[1][4].
- **최적 모델**: **SimDCL**이 FID와 ZNCC 평가에서 전반적으로 우수한 성능을 보였습니다[1][4].

### 5. GAN 모델 성능 비교 표
다음은 주요 GAN 모델의 성능을 종합한 표입니다:

| **GAN 모델** | **SSIM** | **PSNR** | **FID** | **ZNCC 향상** |
|-------------|----------|----------|---------|--------------|
| **SimDCL**  | 0.96     | 36.93    | 68.93   | ✓            |
| CycleGAN    | 0.95     | 35.20    | 75.41   | ✓            |
| DCLGAN      | 0.94     | 34.80    | 72.15   | ✓            |
| FastCUT     | 0.93     | 33.10    | 81.30   | ✗            |

연구는 GAN 기반 영상 처리 기술이 **폐 종양의 동적 추적 정확도를 혁신적으로 개선**할 수 있음을 입증하며, 향후 임상 적용을 위한 중요한 기반을 마련했습니다[1][3][4].

[1] https://onlinelibrary.wiley.com/doi/10.1111/1759-7714.70014
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC11850088/
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC10795441/
[4] https://assets-eu.researchsquare.com/files/rs-3212146/v1/498fd58d-0529-4722-b67a-90450e02a4f3.pdf
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC3356163/
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC9879935/
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC12050046/
[8] https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.14212
[9] https://www.sciencedirect.com/science/article/abs/pii/S0167814024001002
[10] https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/acm2.14500
[11] https://journals.sagepub.com/doi/full/10.1177/15330338241232557
[12] https://pmc.ncbi.nlm.nih.gov/articles/PMC10880520/
[13] https://pmc.ncbi.nlm.nih.gov/articles/PMC8743147/
[14] https://pmc.ncbi.nlm.nih.gov/articles/PMC3386520/
[15] https://pmc.ncbi.nlm.nih.gov/articles/PMC10691617/
[16] https://pubmed.ncbi.nlm.nih.gov/39994000/
[17] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10795441/
[18] https://pubmed.ncbi.nlm.nih.gov/37985163/
[19] https://pubmed.ncbi.nlm.nih.gov/26153580/
