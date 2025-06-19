# Massive training artificial neural network MTANN for reduction of false positives in computerized detection of lung nodules in low-dose computed tomography

이 연구는 저선량 컴퓨터 단층촬영(CT) 영상에서 폐결절을 자동으로 검출하는 과정에서 발생하는 **위양성(false positives)을 줄이기 위한 MTANN(Massive Training Artificial Neural Network) 기법**을 제안합니다. 기존 컴퓨터 보조 검출(CAD) 시스템은 폐결절과 유사한 구조물(혈관, 기타 음영 등)을 잘못 탐지하는 문제가 있었습니다. MTANN은 이러한 문제를 해결하기 위해 개발된 인공신경망(ANN) 기반 패턴 인식 기술로, 특히 **다중 MTANN(Multi-MTANN)** 구조를 통해 다양한 비결절(non-nodule) 유형을 효과적으로 구분합니다[1][2].  

---

### 핵심 메커니즘  
1. **MTANN 구조**:  
   - **직접 영상 처리**: 기존 ANN과 달리, MTANN은 영상 데이터를 직접 처리합니다. 입력 영상에서 추출한 수많은 부분 영역(subregions)과 "결절 가능성" 분포를 나타내는 교사 영상(teacher images)으로 훈련됩니다[1][2].  
   - **출력 생성**: 훈련된 MTANN은 입력 CT 영상을 스캔하여 **결절 확률 맵(output image)** 을 생성합니다. 이 맵에서 계산된 점수(score)를 기반으로 결절과 비결절을 구분합니다[1].  

2. **다중 MTANN(Multi-MTANN) 확장**:  
   - **전문가형 MTANN 병렬 구성**: 단일 MTANN의 한계를 극복하기 위해 여러 MTANN을 병렬로 배치합니다. 각 MTANN은 **특정 비결절 유형(예: 다양한 크기의 혈관, 기타 음영)** 을 전문적으로 구분하도록 훈련됩니다[1][2].  
   - **논리적 AND 연산**: 개별 MTANN의 출력을 결합할 때, **모든 MTANN이 결절로 판정한 영역만 최종 검출**합니다. 이로 인해 특정 비결절 유형만 제거되고 결절은 보존됩니다[1][2].  

---

### 실험 및 결과  
- **훈련 데이터**: 10개의 전형적 결절과 9종류의 비결절(종류당 10개, 총 90개)을 사용해 9개의 MTANN으로 구성된 Multi-MTANN을 훈련[1][2].  
- **검증 데이터**: 63건의 저선량 CT 스캔(1,765개 단면)에서 58개 결절과 1,726개 비결절을 대상으로 테스트[1][2].  
- **성능**:  
  - **위양성 83% 감소**: 1,726개 비결절 중 1,424개 제거[1][2].  
  - **결절 검출 민감도 98.3%**: 58개 결절 중 57개 정확히 식별[1][2].  
  - **개선된 성능**: 기존 CAD 대비 단면당 위양성 비율이 0.98 → 0.18, 환자당 위양성은 27.4 → 4.8로 감소하며 전체 민감도 80.3% 유지[1][2].  

---

### 의의 및 한계  
- **의의**: MTANN은 **복잡한 폐 구조에서 결절을 정확히 식별**하는 데 탁월하며, CAD 시스템의 신뢰성을 크게 향상시킵니다[1][5]. 최근 연구에서는 3D MTANN이 악성/양성 결절 분류에서 AUC 0.91의 정확도를 보여 적용 가능성을 확대했습니다[3].  
- **한계**: 훈련 데이터의 다양성 부족이나 미세 결절(subtle nodules) 검출에서 한계가 있을 수 있으나, 연조직 기법 적용 시 미세 결절 검출 민감도가 30.95% → 33.33%로 개선된 사례도 보고되었습니다[5].  

이 연구는 인공지능 기반 의료 영상 분석의 발전에 기여하며, 특히 **폐암 조기 진단의 정확성 향상**에 실질적인 도구로 활용될 수 있습니다[1][3][2].

[1] https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.1580485
[2] https://pubmed.ncbi.nlm.nih.gov/12906178/
[3] https://dl.acm.org/doi/10.1145/3655755.3655784
[4] https://www.mdpi.com/2072-6694/15/18/4655
[5] https://www.researchsquare.com/article/rs-25307/v1
[6] http://proceedings.spiedigitallibrary.org/proceeding.aspx?doi=10.1117/12.480181
[7] http://ieeexplore.ieee.org/document/1501920/
[8] https://www.semanticscholar.org/paper/ced55891340c99b58044e10f1859e66fe0d2a6ff
[9] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/5032/0000/Effect-of-a-small-number-of-training-cases-on-the/10.1117/12.480181.full
[10] http://atm.amegroups.com/article/view/49169/html
[11] http://medrxiv.org/lookup/doi/10.1101/2025.01.13.25320295
[12] https://iopscience.iop.org/article/10.1088/1361-6560/acef8c
[13] https://www.aapm.org/meetings/02AM/VirtualPressRoom/Suzuki.pdf
[14] https://suzukilab.first.iir.titech.ac.jp/ja/wp-content/uploads/2020/01/SuzukiEtAl-MedPhy2003-7-MTANN-LDCT.pdf
[15] https://cir.nii.ac.jp/crid/1360011144678728576
[16] https://arxiv.org/pdf/1901.07858.pdf
[17] https://pmc.ncbi.nlm.nih.gov/articles/PMC11853243/
[18] https://arxiv.org/pdf/1711.02074.pdf
[19] https://arxiv.org/pdf/1812.11204.pdf
