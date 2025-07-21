# Uncertainty Estimation in Medical Image Denoising with Bayesian Deep Image Prior
   
의료 영상 잡음 제거에서 **Bayesian Deep Image Prior(BDIP)**를 적용해 불확실성을 정량화한 논문 “Uncertainty Estimation in Medical Image Denoising with Bayesian Deep Image Prior”의 핵심 내용과 향후 연구 시사점을 종합적으로 정리한다. 먼저 본 연구의 주장·기여를 압축 요약한 뒤, 해결 과제·수학적 기법·모델 구조·성능·한계를 단계별로 해설하고, 일반화 성능 향상 가능성을 중점적으로 논의한다. 마지막으로 후속 연구에 미칠 영향과 고려사항을 제시한다.  

## 개요  
많은 의료 영상 복원 네트워크는 대규모 학습 데이터에 내재된 통계에 과도하게 의존해 “hallucination” (존재하지 않는 해부학적 구조 생성) 현상을 유발하며, 예측 신뢰도에 대한 실시간 지표가 부재하다[1]. 본 논문은  
- **Deep Image Prior(DIP)** 특유의 데이터 비의존적 구조만으로도 잡음을 억제하면서  
- **Monte Carlo Dropout(MC-Dropout)**를 이용한 베이지안 추론으로 **aleatoric**·**epistemic** 불확실성을 동시 추정해  
- 재구성 오류와 불확실성 간 선형 상관(=잘 보정된 uncertainty)을 달성함을 실험적으로 입증한다[1].  

## 핵심 주장 및 주요 기여 요약  
1. **Hallucination 방지:** 학습 데이터 통계를 사용하지 않는 DIP를 기반으로 해, 감독 학습형 오토인코더 대비 해부학적 왜곡을 크게 줄인다[1].  
2. **이중 불확실성 추정:** 마지막 층을 분기해 픽셀별 평균 $$\hat{x}$$와 분산 $$\hat{\sigma}^2$$를 출력하고, 테스트 단계에서도 MC-Dropout을 반복 적용하여 **aleatoric(데이터 기인)** 과 **epistemic(모델 기인)** 불확실성을 동시에 획득한다[1].  
3. **자동 수렴(early-stopping 불필요):** 베이지안 설계 덕분에 반복 최적화 과정에서 과적합 없이 PSNR·SSIM이 안정 수렴해, 기존 DIP가 요구한 수동 early-stopping 절차를 제거한다[1].  
4. **우수하고 잘 보정된 예측:** 저선량 X-ray, OCT, 초음파 등 다양한 모달리티에서 기존 SGLD-DIP·기본 DIP보다 높은 PSNR(≈29-31 dB)·낮은 Uncertainty Calibration Error(UCE≈0.11)을 기록, 불확실성–오류 선형성이 향상된다[1].  

## 연구가 해결하려는 문제  
### 1. 의료 영상 Denoising 특유의 위험  
- **작은 구조** (맥락적으로 중요한 병변, 층 두께 등) → 잡음·연속적 패턴 사이에 묻히기 쉬움.  
- **Hallucination 위험:** 학습 기반 CNN이 훈련셋 통계로 “가짜 층”을 생성하고도 신뢰 점수를 낮추지 못함[1].  
- **불확실성 부족:** 픽셀 단위로 “이 부분이 불확실하다”는 지표 부재 → 임상의 의사결정 어려움.  

### 2. 딥이미지프라이어(DIP)의 장단점  
- 장점: 랜덤 초기화된 CNN $$f_\theta(\mathbf{z})$$ 자체가 강력한 저주파 이미지 prior → 데이터셋 불필요[2].  
- 단점:  
  - 최적화 $$\hat{\theta}=\arg\min_\theta \| \tilde{x}-f_\theta(\mathbf{z})\|^2 $$ 과정이 계속되면 고주파 잡음까지 복원(오버핏) → **early-stopping** 필요[2].  

  - 불확실성 산출 불가.  

## 제안 방법  
### 1. 수식으로 본 모델링  
1. **입·출력 정의**  
   - 잡음 영상 $$\tilde{x}=c\circ x $$  

   - 네트워크 $$f_\theta: \mathbf{z}\mapsto(\hat{x},\log \hat{\sigma}^2)$$  

2. **Aleatoric 추정용 완전 로그우도**  

$$
   \mathcal{L}(\theta)=\frac1N\sum_{i=1}^{N}\hat{\sigma}_i^{-2}\bigl(\tilde{x}_i-\hat{x}_i\bigr)^2+\log \hat{\sigma}_i^2  
   $$  

3. **Epistemic 추정을 위한 변분 사후**  
   - 파라미터 사전 $$p(\theta)\sim\mathcal{N}(0,\lambda^{-1}I)$$  
   - MC-Dropout으로 $$\tilde{\theta}\sim q(\theta)$$ 샘플 $$T$$ 회($$p=0.3$$)  
   - 예측 평균·분산  
   
  $$
     \hat{x}=\frac1T\sum_{t=1}^{T}\hat{x}^{(t)},\quad  
     \hat{\sigma}^2_{\text{total}}=\underbrace{\frac1T\sum_{t=1}^{T}\bigl(\hat{x}^{(t)}-\hat{x}\bigr)^2}\_{\text{epistemic}}+\underbrace{\frac1T\sum_{t=1}^{T}\hat{\sigma}^{2\,(t)}}_{\text{aleatoric}}
     $$  

4. **Uncertainty Calibration Error(UCE)**  

$$
   \text{UCE}=\sum_{b=1}^{B}\frac{|S_b|}{N}\Bigl|\text{MSE}(S_b)-\text{Unc}(S_b)\Bigr|
   $$  
   
   $$B$$: 구간 수, $$S_b$$: 해당 구간 픽셀 집합.  

### 2. 네트워크 구조  
- **Encoder-Decoder U-Net 변형:** skip connection 포함, 출력 채널을 2배(평균·로그분산)로 분기[1].  
- **입력** $$\mathbf{z}$$: 동일 해상도의 균등분포 $$\mathcal{U}(0,0.1)$$ 노이즈.  
- **Dropout 층**: 인코더·디코더 모든 컨볼루션 블록 후 삽입(keep prob=0.7).  
- **Optimizer**: Adam(learning rate $$1\times10^{-4}$$), 50 k iteration.  

## 실험 설정 및 성능  
### 1. 데이터·잡음  
| 모달리티 | 해상도 | 잡음 유형 | 시뮬레이션 파라미터 |  
|-----------|---------|-------------|---------------------|  
| OCT       |512×512 | Gaussian     |$$\sigma=25$$ gray value[1] |  
| Ultrasound|512×512 | Speckle(Gauss 근사)|$$\sigma=20$$|  
| Chest X-ray|512×512 | Poisson(≈Gaussian)|저선량 λ≈10 photons[1]|  

### 2. 주요 결과(50 k step 기준)  
| 방법 | OCT PSNR (dB) | US PSNR (dB) | X-ray PSNR (dB) | 평균 UCE | Early-Stop 필요 |  
|------|--------------|--------------|----------------|-----------|-----------------|  
| DIP[2]|23.64[1] |23.55 |23.28 |– |필요(엄격) |  
| SGLD-DIP[4]|23.58 |23.81 |23.50 |0.030 |부분 |  
| SGLD+NLL|24.82 |24.55 |24.60 |0.061 |부분 |  
| **MC-Dropout DIP(본 논문)**|**29.88** |**29.67** |**31.19** |**0.109** |불필요 |  

*MC-Dropout DIP가 전 모달리티에서 PSNR 5-8 dB ↑, early-stop 없이도 잡음 재생성 억제[1].*  

### 3. 불확실성–오류 상관  
- **Calibration Diagram**: 예측 불확실성 구간별 실제 MSE가 거의 $$y=x$$ 직선에 근접(UCE≈0.11) → 정량적 오류 예측 가능[1].  
- **Uncertainty Map**: 경계·질감 변화가 큰 위치에서 높은 $$\hat{\sigma}$$, 균일 조직 영역에서 낮은 $$\hat{\sigma}$$ → 임상가가 참고할 수 있는 시각적 가이드 제공.  

## 성능 향상 및 한계  
### 성능 향상 요인  
1. **Dropout-as-Variational Inference:** 반복 학습 단계에서도 파라미터 불확정성을 유지해 고주파 잡음 학습을 억제(regularization 효과)[5].  
2. **공동 학습 분산 Head:** Aleatoric 손실 항(식 1) 덕분에 픽셀별 잡음 분포 자체를 모델이 학습 → 잔여 잡음·구조적 텍스처를 구분.  
3. **학습 데이터 비의존:** 모달리티·도메인이 바뀌어도 네트워크 재사용 가능 → **일반화** 우수.  

### 남은 한계  
| 범주 | 구체적 내용 | 잠재적 해결책 |  
|------|------------|---------------|  
| 계산비용 | 25회 MC 추론 → 실시간 적용엔 부담 | Dropout mask 수 감소·TensorRT 배포 |  
| 공간 해상도 손실 | 과도한 저주파 편향으로 미세 병변이 완전히 회복되지 않는 경우 | Hybrid explicit prior(예: TV) 도입[6] |  
| Dropout 확률 튜닝 | 모달리티별 최적 $$p$$ 상이 — 경험적 선택 | Bayesian Optimization + AutoML |  
| UCE 정의 한계 | 회귀용 UCE 지표가 완전히 표준화되지 않음 | Binning-free ECE 연구 병행[3] |  

## 일반화 성능(Generalization) 논의  
1. **데이터셋 불필요 = 도메인 전이 강인**  
   - 학습-free 특성으로 장비·기관 간 영상 통계 변화에도 성능 유지.  
2. **모델 불확실성 활용**  
   - 높은 $$\hat{\sigma}$$ 영역만 후처리 (예: non-local means) → 성능 보완.  
   - Ensemble 없이도 epistemic 추정 가능 → 경량화.  
3. **Transfer Learning 불필요**  
   - 기존 supervised denoiser처럼 재학습 과정이 없으므로 도메인 shift 시 추가 비용 0.  
4. **우도 기반 손실 = Noise 모델 확장 용이**  
   - Poisson-Gaussian·Rician 등 의료 특유 잡음 분포로 손쉽게 교체 가능 → 범용성.  

## 향후 연구 영향 및 고려 사항  
### 1. 임상 적용 촉진  
- **의사결정 지원:** 불확실성 map을 판독 화면에 오버레이해 진단 신뢰도 지표 제공 → 오진 리스크 완화.  
- **저선량·고속 스캔:** 잡음·artifact가 큰 영상에서도 안전판 역할.  

### 2. 후속 연구 방향  
| 연구축 | 세부 과제 | 기대 효과 |  
|--------|----------|-----------|  
| **Bayesian DIP 고도화** | Mean-Field VI[7], SDE 기반 샘플링 등 | 추론 시간 ↓, 불확실성 품질 ↑ |  
| **다중모달 융합** | CT+MR, OCT+Fundus 동시 DIP | cross-prior로 세밀한 구조 복원 |  
| **Task-aware Loss** | 병변 segmentation joint loss | 임상적 민감도 우선 최적화 |  
| **Adaptive Noise Modeling** | 입력 영상 통계에 따라 $$\sigma$$ 자동 예측 | 매 스캔 별 최적화 필요성 제거 |  

### 3. 적용 시 유의할 점  
1. **MC 샘플 수–지연 트레이드오프:** 실시간 진료용 프로토콜에 맞는 $$T$$ 값 설정 필수.  
2. **불확실성 시각화 표준화:** 색상 범주·범례 통일해 임상의 해석 오류 방지.  
3. **데이터 보호:** 개인 의료 영상은 외부 서버 대신 온프레미스 추론 요구 가능 → 경량 모델 경로 확보.  

## 결론  
본 논문은 기존 DIP에 **베이지안 추론**을 접목해 **잡음 제거 성능**과 **예측 신뢰성**을 동시에 향상시키며, **early-stopping** 의존성을 제거해 실무 친화적 파이프라인을 제시했다[1]. 학습 데이터에 의존하지 않으면서도 두 종류의 불확실성을 정밀 추정하는 이 접근법은, 향후 의료 영상 복원·정합·초해상·분할 전반에서 “모델-내 불확실성”을 표준 기능으로 통합하는 방향을 가속화할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/16064518-e7e0-4878-bdb0-c585ed1e5dba/2008.08837v1.pdf
[2] https://arxiv.org/abs/1711.10925
[3] https://openreview.net/pdf?id=XOuAOv_-5Fx
[4] https://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_A_Bayesian_Perspective_on_the_Deep_Image_Prior_CVPR_2019_paper.pdf
[5] http://proceedings.mlr.press/v48/gal16.pdf
[6] https://ieeexplore.ieee.org/document/10453979/
[7] https://proceedings.mlr.press/v143/tolle21a.html
[8] http://link.springer.com/10.1007/s11263-020-01303-4
[9] https://ieeexplore.ieee.org/document/10629196/
[10] https://ieeexplore.ieee.org/document/10896571/
[11] https://onlinelibrary.wiley.com/doi/10.1002/mrm.30026
[12] https://ieeexplore.ieee.org/document/10138068/
[13] https://arxiv.org/abs/2402.15635
[14] https://ieeexplore.ieee.org/document/10034853/
[15] https://ieeexplore.ieee.org/document/10205276/
[16] https://ieeexplore.ieee.org/document/10845210/
[17] http://arxiv.org/pdf/1711.10925.pdf
[18] https://arxiv.org/html/2404.12142v1
[19] https://en.wikipedia.org/wiki/Deep_image_prior
[20] https://www.geeksforgeeks.org/deep-learning/what-is-monte-carlo-mc-dropout/
[21] https://people.cs.umass.edu/~zezhoucheng/gp-dip/
[22] https://pubmed.ncbi.nlm.nih.gov/35653453/
[23] https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/mc-dropout.html
[24] https://paperswithcode.com/paper/uncertainty-calibration-error-a-new-metric
[25] https://pmc.ncbi.nlm.nih.gov/articles/PMC6584077/
[26] https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf
[27] https://openreview.net/forum?id=XOuAOv_-5Fx
[28] https://arxiv.org/abs/1904.07457
[29] https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2022.995225/full
[30] https://dmitryulyanov.github.io/deep_image_prior
[31] https://tech.onepredict.ai/e0db8c3e-0225-4b2a-8691-1f9e34d01915
[32] https://pypi.org/project/uncertainty-calibration/
[33] https://www.mdpi.com/2076-3417/13/9/5265
[34] https://arxiv.org/pdf/2211.14298.pdf
[35] https://arxiv.org/pdf/2206.02070.pdf
[36] https://arxiv.org/pdf/2203.00479.pdf
[37] http://arxiv.org/pdf/1705.08041.pdf
[38] https://arxiv.org/html/2407.19866v1
[39] http://arxiv.org/pdf/2502.18612.pdf
[40] http://arxiv.org/pdf/2209.08452.pdf
[41] https://arxiv.org/abs/1904.07457v1

# Uncertainty Estimation in Medical Image Denoising with Bayesian Deep Image Prior

# MCDIP
# Bayesian DIP

# Abs

딥 러닝을 사용한 inverse medical imaging의 불확실성 정량화는 거의 주목을 받지 못했습니다.  
그러나 대규모 데이터 세트로 훈련된 심층 모델은 해부학적으로 존재하지 않는 재구성된 이미지에서 환각을 일으키고 결함을 생성하는 경향이 있습니다.  
저희는 무작위로 초기화된 컨볼루션 네트워크를 재구성된 이미지의 매개변수로 사용하고 관찰한 이미지와 일치하도록 경사 하강을 수행하며, 이를 deep image prior라고 합니다.  
이 경우 사전 훈련이 수행되지 않기 때문에 재구성할 때 환각을 겪지 않습니다.  
저희는 이를 Monte Carlo dropout을 사용한 Bayesian approach 방식으로 확장하여 우연에 의한 확실성 및 인식적 불확실성을 모두 정량화합니다.  
제시된 방법은 다양한 의료 영상 양식의 노이즈 제거 작업에 대해 평가됩니다.  
실험 결과는 저희의 접근 방식이 잘 보정된 불확실성을 가져온다는 것을 보여줍니다.  
즉, 예측 불확실성은 예측 오류와 상관관계가 있습니다.  
이를 통해 신뢰할 수 있는 불확실성 추정이 가능하며 inverse medical imaging 작업에서 환각 및 결함 문제를 해결할 수 있습니다. 


# Ref
https://aistudy9314.tistory.com/47
