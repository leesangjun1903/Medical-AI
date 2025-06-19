## Comparing two classes of end-to-end machine-learning models in lung nodule detection and classification: MTANNs vs. CNNs

### 핵심 실험 결과

- **제한된 데이터 환경에서 MTANNs의 우월성**:
    - 훈련 데이터가 적을 때(10개 노듈/90개 음성 샘플) MTANNs는 CNN보다 현저히 높은 성능을 보임[^1_1].
    - 폐결절 검출에서 MTANNs는 100% 민감도로 환자당 **2.7개 오탐률**을 기록한 반면, 최고 성능 CNN은 **22.7개 오탐률**을 보임(p<0.05)[^1_1].
    - 분류 작업에서 MTANNs의 AUC(0.8806)는 CNN(0.7755)보다 유의미하게 높음[^1_1][^1_5].
- **대용량 데이터 환경에서의 변화**:
    - 데이터 양이 증가하면 CNN 성능이 향상되나, MTANNs와의 차이는 여전히 유의미함(p<0.05)[^1_1][^1_2].
    - 예: 5-fold 교차 검증 시나리오에서 CNN의 오탐률이 감소했으나 MTANNs에 미치지 못함[^1_1].


### 아키텍처 차이점

| **특징** | **MTANNs** | **CNNs** |
| :-- | :-- | :-- |
| **컨볼루션** | 외부에서 수행 (필터 기반 특징 추출) | 내부 계층에서 수행 |
| **출력 형태** | 연속 값 맵 (이미지) | 클래스 라벨 |
| **훈련 데이터** | ±10개 이미지로 충분 | 수천~백만 개 필요 |
| **훈련 시간** | GPU 없이 몇 시간 | GPU 사용 시 며칠 소요 |

### 최근 동향 및 시사점

1. **현재 의료 영상 분야의 주류**:
    - CNN 기반 모델(ResNet, U-Net, 3D CNN)이 폐결절 분석의 표준으로 자리잡음[^1_3][^1_4].
    - 최신 연구들은 **95% 이상의 정확도**와 **90% 이상의 AUC**를 보고하며, 특히 대용량 데이터셋(LIDC-IDRI 등)에서 효과적[^1_4][^1_5].
2. **MTANNs의 잠재적 활용 분야**:
    - **데이터 부족 환경**(희귀 질환, 소규모 병원)에서 여전히 유용성 인정[^1_1][^1_2].
    - **실시간 진단 시스템**에 적합한 경량화 모델로 재조명될 가능성[^1_5].
3. **혼합 모델의 부상**:
    - 특징 추출(Texture/Shape features) + CNN 분류기의 조합이 **오탐률 감소**에 효과적[^1_3][^1_5].
    - 예: Fu et al.의 연구에서 3D CT 영상과 혈청 바이오마커를 결합한 모델은 90.6% 정확도 달성[^1_6].

### 한계점 및 향후 과제

- **MTANNs의 확장성 문제**: 깊은 네트워크 설계 어려움, 복잡한 병변 분류에 취약[^1_1].
- **CNN의 데이터 의존성**: 저품질/불균형 데이터 시 성능 급감[^1_4][^1_5].
- **해결 방향**:
    - **전이 학습**(Transfer Learning)을 통한 CNN의 데이터 효율성 개선[^1_5].
    - **MTANN-CNN 융합 모델** 탐구(예: MTANN으로 전처리 + CNN 분류)[^1_2][^1_5].

> 논문은 의료 영상 분석에서 **상황에 따른 최적 모델 선택**의 중요성을 강조합니다. 데이터 가용성, 진단 목적(검출 vs. 분류), 컴퓨팅 자원 등을 고려한 접근이 필요합니다[^1_1][^1_2][^1_5].

<div style="text-align: center">⁂</div>

[^1_1]: https://suzukilab.first.iir.titech.ac.jp/wp-content/uploads/2020/01/TajbakhshNSuzukiK_ComparingMTANNsVsCNNs_PR2017-1.pdf

[^1_2]: https://meritis.fr/deep-learning-in-medical-imaging-introducing-mtann/

[^1_3]: https://iopscience.iop.org/article/10.1088/2057-1976/ad9154

[^1_4]: https://pubmed.ncbi.nlm.nih.gov/37763314/

[^1_5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10532719/

[^1_6]: https://arxiv.org/html/2410.14769v1

[^1_7]: https://linkinghub.elsevier.com/retrieve/pii/S0031320316302795

[^1_8]: https://link.springer.com/10.1007/s12652-023-04711-9

[^1_9]: https://link.springer.com/10.1007/s10278-023-00809-w

[^1_10]: https://link.springer.com/10.1007/s10278-023-00822-z

[^1_11]: http://link.springer.com/10.1007/978-3-319-54526-4_7

[^1_12]: https://www.sciencedirect.com/science/article/pii/S0031320316302795

[^1_13]: https://ui.adsabs.harvard.edu/abs/2017PatRe..63..476T/abstract

[^1_14]: https://www.semanticscholar.org/paper/Comparing-two-classes-of-end-to-end-models-in-lung-Tajbakhsh-Suzuki/a5075a9a0d30ea5e5ef1816f25b64a109436a966

[^1_15]: https://ieeexplore.ieee.org/document/10452545/

[^1_16]: https://www.semanticscholar.org/paper/0aad16fbbe39a9d5703e2ed11b97b08f7285b513

[^1_17]: http://link.springer.com/10.1007/s10278-019-00221-3

[^1_18]: https://dl.acm.org/doi/10.1145/3655755.3655784

[^1_19]: https://ieeexplore.ieee.org/document/10499183/

[^1_20]: https://www.mdpi.com/1424-8220/19/17/3722

[^1_21]: https://ieeexplore.ieee.org/document/10540203/

[^1_22]: https://ieeexplore.ieee.org/document/9972595/

[^1_23]: https://www.mdpi.com/2306-5354/10/11/1245

[^1_24]: https://www.sciencedirect.com/science/article/abs/pii/S0031320316302795

[^1_25]: https://pubmed.ncbi.nlm.nih.gov/30440489/

[^1_26]: https://www.techscience.com/csse/v48n6/58686/html

[^1_27]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0188290

[^1_28]: https://www3.cs.stonybrook.edu/~qin/research/2018-pr-multi-view-cnns-for-lung-nodule.pdf

[^1_29]: https://ieeexplore.ieee.org/document/10274463/

[^1_30]: https://www.nature.com/articles/s41598-024-51833-x

[^1_31]: https://www.ijsrcseit.com/index.php/home/article/view/CSEIT25112820

[^1_32]: https://gvpress.com/journals/IJHIT/vol13_no2/vol13_no2_2020_04.html

[^1_33]: https://pubs.aip.org/aip/acp/article/3161/1/020093/3310499/Automatic-lung-nodule-classification-using-deep

[^1_34]: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304378

[^1_35]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6749467/

[^1_36]: http://arxiv.org/abs/1904.05956

