# Adapting Pre-trained Vision Transformers from 2D to 3D through Weight Inflation Improves Medical Image Segmentation

이 논문은 3D 의료 영상 분할을 위해 2D 자연 이미지로 사전 훈련된 Vision Transformer(ViT) 모델을 효과적으로 적용하는 방법을 제안합니다. 기존 접근법의 한계를 해결하고 깊이 정보를 보존하면서도 전이 학습의 이점을 극대화하는 간단하지만 강력한 전략을 소개합니다[1][3].

## 📌 핵심 문제점과 해결 목표
3D 의료 영상(MRI, CT 등) 분석에서 **깊이 정보**는 질병 진단과 치료에 중요합니다. 그러나 대규모 자연 이미지 데이터로 사전 훈련된 2D Vision Transformer 모델을 3D 의료 영상에 직접 적용하기 어려웠습니다[1][4]. 기존 방법은 두 가지 한계가 있었습니다:
- **2D 슬라이스 분할**: 3D 볼륨을 2D 조각으로 나누어 처리해 깊이 정보 손실 발생
- **3D 아키텍처 수정**: 사전 훈련 가중치를 활용하지 못해 학습 효율성 저하[1][5]

## ⚙️ 제안 방법: 가중치 인플레이션(Weight Inflation)
이 논문의 핵심 기여는 **사전 훈련된 2D 가중치를 3D로 확장**하는 간단한 전략입니다:
1. **2D 커널 확장**: 2D 합성곱 필터를 깊이 차원으로 복제하여 3D 필터로 변환
   - 예: `(K, K)` 2D 필터 → `(K, K, 1)` 3D 필터로 확장[1][4]
2. **Transformer 블록 적용**: 인플레이션된 가중치로 ViT의 self-attention 레이어 초기화
3. **미세 조정(Fine-tuning)**: 의료 영상 데이터셋으로 추가 학습 수행[3][2]

이 방식은 **기존 2D 사전 훈련 모델을 최대 98% 재활용**하면서도 3D 공간 정보를 완전히 활용합니다[1].

## 📊 실험 결과 및 성능
다양한 3D 의료 데이터셋(BraTS, LiTS 등)에서 검증된 결과:
- **성능 향상**: 기존 3D 모델 대비 평균 1.95% 정확도 향상[3]
- **효율성**: 계산 비용은 0.75%만 증가시켜 효율적[4][2]
- **일반성**: 11개 의료 데이터셋에서 SOTA(State-of-the-art) 성능 달성[3][5]
- **전이 학습 원천 분석**: 자연 이미지 vs. 의료 이미지 사전 훈련 효과 비교[1][6]

## 💡 의의와 적용 가능성
이 방법은 **의료 영상 분야에 특화된 3가지 장점**을 제공합니다:
1. **깊이 정보 보존**: 3D 구조의 맥락적 정보 유지
2. **전이 학습 최적화**: 대규모 자연 이미지 데이터의 지식 활용
3. **표준화 가능성**: 모든 ViT 기반 3D 의료 모델에 즉시 적용 가능[1][2]

이 연구는 의료 영상 분석 커뮤니티에 **효율적인 3D 모델 설계의 새로운 표준**을 제시하며, 특히 데이터가 제한된 의료 영상 분야에서 전이 학습의 잠재력을 극대화합니다[4][5].

[1] https://arxiv.org/abs/2302.04303
[2] https://marshuang80.github.io/publication/transformers-segmentation/
[3] https://proceedings.mlr.press/v193/zhang22a/zhang22a.pdf
[4] https://github.com/yuhui-zh15/TransSeg
[5] https://openreview.net/forum?id=DGT1MlhfCq
[6] https://paperswithcode.com/paper/adapting-pre-trained-vision-transformers-from
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC9646404/
[8] https://pmc.ncbi.nlm.nih.gov/articles/PMC11381179/
[9] https://arxiv.org/html/2404.10156v1
[10] https://arxiv.org/pdf/2310.19721.pdf
[11] https://pmc.ncbi.nlm.nih.gov/articles/PMC11230947/
[12] https://pmc.ncbi.nlm.nih.gov/articles/PMC10909707/
[13] https://pmc.ncbi.nlm.nih.gov/articles/PMC10376048/
[14] https://arxiv.org/html/2404.10156v2
[15] https://arxiv.org/html/2303.07034v3
[16] https://pmc.ncbi.nlm.nih.gov/articles/PMC9521364/
[17] https://collab.dvb.bayern/spaces/TUMdlma/pages/73379869/Converting+weights+of+2D+Vision+Transformer+for+3D+Image+Classification
[18] https://arxiv.org/pdf/2310.07781.pdf
[19] https://kclpure.kcl.ac.uk/ws/portalfiles/portal/266634273/s11548-024-03140-z.pdf
