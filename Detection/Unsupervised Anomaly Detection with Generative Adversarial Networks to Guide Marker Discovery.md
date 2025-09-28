# Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery

# 핵심 요약  
**Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery**(AnoGAN)은 정상 영상의 생성적 분포를 학습하고, 새로운 입력 영상과 학습된 분포 간의 적합도를 정량화하는 **비지도 이상 탐지 프레임워크**를 제안한다. 주요 기여는 다음과 같다:  
- **정상 영상 분포의 대표적 생성 모델**을 GAN으로 학습하여 정상 해부학적 변이(manifold)를 구성.  
- **이미지→잠재공간 매핑 기법**(residual loss + feature matching 기반 discrimination loss)을 도입하여 쿼리 영상의 잠재 벡터를 최적화.  
- **이상 점수(anomaly score)** $$A(x)=\lambda R(x)+D(x)$$로 영상 단위 이상 여부를 판별하고, residual 차이를 통해 픽셀 단위 이상 영역도 검출.  
- 망막 OCT 데이터에서 **망막 삼출(retinal fluid)** 및 **고반사 초점(hyperreflective foci)** 등 병변을 효과적으로 검출.[1]

# 1. 해결 과제  
의료 영상에서 종종 알려진 병변(markers) 이외에도 유의미한 이상 패턴이 존재하지만, 이들의 **주석(labelled data)** 확보는 시간·비용 제약이 크다. 기존 지도 학습 기반 방법은 알려진 마커에 한정되며, 새로운 이상 징후 발굴에 한계가 있다.  

# 2. 제안 방법  
## 2.1 GAN 기반 정상 해부학 변이 학습  
주어진 정상 영상 패치 $$x \in X$$에 대해,  
- Generator $$G$$: 잠재공간 $$z \sim p_z$$ → 영상 $$G(z)$$  
- Discriminator $$D$$: 영상 $$x$$ → 실젯값 $$D(x)$$  
다음 **minimax** 게임으로 학습:  

```math
\min_G \max_D V(D, G) = \mathbb{E}_{x\sim p_\text{data}}\bigl[\log D(x)\bigr]
+ \mathbb{E}_{z\sim p_z}\bigl[\log(1 - D(G(z)))\bigr].
```

(식 1)  

## 2.2 이미지→잠재공간 매핑  
GAN은 $$z\to x$$만 지원하므로, 신규 입력 $$x$$와 유사한 $$z^*$$를 찾기 위해 최적화한다.  
- **Residual Loss**:  

$$
  L_R(z) = \lVert x - G(z)\rVert_1.
  $$  

- **Discrimination Loss (Feature Matching)**:  

$$
  L_D(z) = \lVert f_D(x) - f_D\bigl(G(z)\bigr)\rVert_1,
  $$  
  
  여기서 $$f_D(\cdot)$$는 discriminator의 중간층 출력.[1]

- **총 손실**:  

$$
  L(z) = \lambda L_R(z) + L_D(z),
  $$  
  
$$\lambda=0.1$$로 설정.[1]

500단계 역전파로 $$z$$를 갱신하여 $$z^*$$ 획득.  

## 2.3 이상 탐지  
최적화 후 얻은 손실 값을 기반으로  
- **Residual score** $$R(x)=L_R(z^*)$$  
- **Discrimination score** $$D(x)=L_D(z^*)$$  
- **Anomaly score**  

$$
  A(x)=\lambda R(x)+D(x).
  $$  

$$A(x)$$가 클수록 정상 분포에서 벗어난다고 판정.[1]

# 3. 모델 구조  
- **Generator**: 5×5 필터, 4개 층(strided convolution), 채널 수
- **Discriminator**: 5×5 필터, 4개 층(convolution), 입력 그레이스케일  
- **학습**: Adam, epochs=20, batch normalization, 이미지 크기 64×64  
- **매핑**: 500단계 학습률 0.1로 잠재 벡터 최적화[1]

# 4. 성능 향상 및 한계  
- **성능**: AUC 0.89, 민감도 0.73, 특이도 0.89로 OCT 병변 탐지에 우수.[1]
- **비교 실험**:  
  - 단일 GAN discriminator 출력만 사용 시 AUC 0.72  
  - adversarial CAE 사용 시 AUC 0.73  
  - AnoGAN의 feature matching 추가 시 AUC 0.89로 개선.[1]
- **한계**:  
  - 매핑 단계에서 500회 역전파로 **추론 속도** 느림.  
  - **잠재공간 최적화**가 지역 최적해에 갇힐 수 있음.  
  - **미지 이상**(novel anomalies) 감지시 false positive 평가 어려움.  

# 5. 일반화 성능 향상 가능성  
- **Feature Matching** 기반 loss가 discriminator의 풍부한 표현을 활용하여 다양한 정상 변이를 포착, 일반화 능력 증대.[1]
- **GAN 구조 확장**(e.g., Wasserstein GAN, Spectral Normalization) 적용 시 안정적 학습으로 더 넓은 정상 분포 모델링 가능.  
- **Encoder 네트워크** 추가 학습으로 mapping 단계 생략하고 빠른 추론 가능.  

# 6. 향후 영향 및 고려사항  
이 논문은 GAN을 활용한 **비지도 이상 탐지** 분야를 개척하여, 마커 발굴(marker discovery)을 위한 데이터 마이닝 기반 초석을 마련했다.  
- **임상 적용**: 수백만 장 대규모 의료 데이터에서 잠재적 병변 후보 자동 발굴 가능.  
- **연구 확장**: 다른 영상 모달리티(CT, MRI)와 다양한 이상 유형에도 적용 연구 필요.  
- **고려사항**: 잠재공간 매핑 효율화(encoder 도입), false positive 제어, 이상 점수 해석성 강화 연구 필수.  

---

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/32e7b9ec-f33e-4fb5-805d-97455e2727a0/1703.05921v1.pdf)
