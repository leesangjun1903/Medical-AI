# An Efficient and Robust Method for Chest X-ray Rib Suppression That Improves Pulmonary Abnormality Diagnosis

## 연구 배경 및 목적

- 흉부 X-ray(CXR)는 폐 질환 진단에 널리 사용되지만, 갈비뼈와 같은 고대비 뼈 구조가 폐 조직의 이상 소견을 가려 진단 정확도를 저해할 수 있음[1][2].
- 기존 뼈 억제(bone suppression) 방법은 물리 기반 모델(정확하지만 매우 느림)과 딥러닝 기반 모델(빠르지만 학습 데이터 한계로 성능 저하)로 나뉨[1][2].
- 본 논문은 두 접근법의 장점을 결합해 효율적이고 강인한 갈비뼈 억제 방법을 제안하여 폐 질환 진단 성능을 향상시키는 것을 목표로 함[1][2].

## 연구 방법

### 전체 파이프라인

1. **물리 기반 ST-smoothing 알고리즘**을 활용해 실제 갈비뼈가 제거된 CXR-정답(GT) 이미지를 생성함[1][2].
2. 생성된 GT 이미지와 원본 CXR 이미지 쌍을 이용해 **딥러닝 이미지 디노이징 네트워크(SADXNet)**를 학습시킴[1][2].
   - SADXNet은 DenseNet 구조를 기반으로 하며, U자형 구조와 채널 유지로 폐 조직의 세밀한 정보를 보존함[1][2].
   - 손실 함수는 PSNR(신호대잡음비), MS-SSIM(다중 스케일 구조적 유사도) 등을 조합하여 뼈 구조를 효과적으로 억제함[1][2].

### 데이터셋

- VinDr-RibCXR(245장, 전문가가 갈비뼈 마스킹), NODE21(폐결절 검출용), ChestX-ray14(14종 폐 질환 분류/검출용) 등 공개 데이터셋을 활용함[1][2].

### 평가 방법

- 뼈 억제 성능은 RMSE, PSNR, MS-SSIM 등으로 평가[1][2].
- 실제 임상적 유용성은 폐결절 검출, 다중 폐 질환 분류/검출 등 다운스트림 작업에서 AUC, FN, FP, TP 등 지표로 평가함[1][2].

## 주요 결과

### 뼈 억제 성능

- SADXNet은 물리 기반 GT와 시각적으로 거의 구별이 안 될 정도로 갈비뼈 억제에 성공함[1][2].
- 기존 물리 기반 방법(ST-smoothing)은 한 장 처리에 40~70분 소요되지만, SADXNet은 1초 미만으로 빠르게 예측 가능함[1][2].

### 임상적 진단 성능 향상

#### 폐결절 검출(NODE21)

| 학습/평가 입력 | AUC   | FN  | FP   | TP  |
|:--------------:|:-----:|:---:|:----:|:---:|
| 원본/원본      | 94.76%[1] | 48[1] | 1273[1] | 372[1] |
| 억제/억제      | 95.32%[1] | 45[1] | 1193[1] | 375[1] |
| 혼합/원본      | 97.99%[1] | 32[1] | 1070[1] | 388[1] |
| 혼합/억제      | 97.31%[1] | 33[1] | 1082[1] | 387[1] |

- 혼합(원본+억제) 이미지로 학습 시 AUC가 약 3%p 상승, FP가 203건 감소함[1][2].

#### 다중 폐 질환 분류/검출(ChestX-ray14)

| 학습/평가 입력 | AUC   | FN  | FP   | TP  |
|:--------------:|:-----:|:---:|:----:|:---:|
| 원본/원본      | 80.54%[1] | 137[1] | 3029[1] | 245[1] |
| 억제/억제      | 81.55%[1] | 138[1] | 2909[1] | 244[1] |
| 혼합/원본      | 87.16%[1] | 116[1] | 2644[1] | 275[1] |
| 혼합/억제      | 86.89%[1] | 124[1] | 2701[1] | 267[1] |

- 혼합 학습 시 AUC가 약 6~7%p 상승, FP가 385건 감소함[1][2].

- 질환별로도 혼합 학습이 전반적으로 가장 높은 성능을 보임(예: 심비대 AUC 93.84%, 폐부종 95.32% 등)[1][2].

## 결론 및 의의

- 물리 기반+딥러닝 결합 방식으로 갈비뼈 억제의 정확성과 효율성을 모두 달성함[1][2].
- 뼈 억제 이미지를 활용하면 폐결절 및 다양한 폐 질환의 검출/분류 성능이 유의하게 향상됨[1][2].
- 논문에서 사용된 데이터셋과 SADXNet 모델은 오픈소스로 공개되어 연구 재현과 확장 연구에 활용 가능함[1][2].
- 본 연구는 CXR 기반 자동 진단 보조 시스템의 임상적 신뢰성과 활용도를 높이는 데 중요한 기여를 함[1][2].

---

**참고:** 데이터 및 코드 공개: [https://github.com/FluteXu/CXR-Rib-Suppression][1][2]

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10177861/
[2]: https://arxiv.org/abs/2302.09696

[1] https://www.mdpi.com/2075-4418/13/9/1652
[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC10177861/
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC8970404/
[4] https://arxiv.org/abs/2302.09696
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC3347229/
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC8151767/
[7] https://arxiv.org/pdf/1810.07500.pdf
[8] http://arxiv.org/pdf/2111.03404.pdf
[9] https://ouci.dntb.gov.ua/en/works/lxmELmP9/
[10] https://www.mdpi.com/2075-4418/13/9/1652/review_report
[11] https://escholarship.org/uc/item/6zq7w1j2
[12] https://ajronline.org/doi/10.2214/AJR.09.2431
[13] https://arxiv.org/abs/2110.09134
[14] https://mayoclinic.elsevierpure.com/en/publications/bone-suppression-technique-for-chest-radiographs
[15] https://pmc.ncbi.nlm.nih.gov/articles/PMC7345724/
[16] https://pmc.ncbi.nlm.nih.gov/articles/PMC11137152/
[17] https://www.semanticscholar.org/paper/An-Efficient-and-Robust-Method-for-Chest-X-ray-Rib-Xu-Xu/080cbe262db769f28eb28de68b56945fb063c4c4
