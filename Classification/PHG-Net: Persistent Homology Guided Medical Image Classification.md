# PHG-Net: Persistent Homology Guided Medical Image Classification 

## 1. 핵심 주장과 주요 기여

**PHG-Net의 핵심 주장**은 기존 CNN과 Transformer가 픽셀 강도에 최적화된 특징만을 캡처하여 연결된 구성요소와 루프(loop) 같은 핵심적인 해부학적 구조를 무시한다는 문제를 해결하는 것입니다. 논문은 **지속 호몰로지(Persistent Homology)**를 활용하여 의료 영상의 위상학적 특징을 추출하고 이를 딥러닝 모델에 통합하는 새로운 접근법을 제안합니다.[1][2]

**주요 기여는 다음과 같습니다**:[1]

1. **PointNet에서 영감을 받은 새로운 접근법**: 지속성 다이어그램을 벡터화된 특징이 아닌 점 구름으로 처리하는 신경망을 개발
2. **범용적 통합 가능성**: CNN과 Transformer 등 모든 기본 비전 모델에 위상학적 특징을 통합할 수 있는 구조
3. **성능 향상**: 세 개의 공개 데이터셋에서 기존 최첨단 의료 영상 분류 방법 대비 상당한 성능 개선

## 2. 해결하고자 하는 문제와 제안 방법

### 해결하려는 문제
기존 딥러닝 모델들이 의료 영상에서 **전역적이고 견고한 해부학적 구조(연결 구성요소, 루프, 공극 등)**를 간과하는 문제를 해결합니다. 의료 영상은 일반적으로 특정 패턴의 연결 구성요소로 구성된 조직, 기관, 병변을 포함하지만, 기존 DL 모델들은 이러한 구조와 위상을 종종 간과합니다.[1]

### 제안하는 방법
**큐비컬 지속 호몰로지(Cubical Persistent Homology)**를 사용하여 의료 영상의 위상학적 특징을 추출합니다.[1]

**핵심 수식들**:

**1. PD 인코딩 공식**:[2]

$$ f(p_1, p_2, \ldots, p_n) = g(m(p_1), m(p_2), \ldots, m(p_n)) $$

여기서 $$p_i$$는 PD의 데이터 포인트, $$f$$는 출력 특징 벡터, $$m(p_i)$$는 데이터 포인트를 고차원으로 매핑하는 함수, $$g$$는 모든 포인트의 정보를 집계하는 함수입니다.[2]

**2. 위상학적 안내 메커니즘**:[2]

$$ t' = f(t, W) = \sigma(W_1(\text{ReLU}(W_2(t)))) $$

$$ F' = F \otimes t' $$

여기서 $$t$$는 PD 인코더의 출력, $$t'$$는 처리된 위상학적 특징 벡터, $$F$$는 CNN 특징 맵, $$F'$$는 개선된 특징 맵입니다.[2]

**3. 최종 손실 함수**:[2]

$$ \mathcal{L} = \mathcal{L}\_V(\tilde{y}\_v, y) + \alpha \mathcal{L}\_{\text{Topo}}(\tilde{y}_{\text{topo}}, y) $$

여기서 $$\mathcal{L}\_V$$와 $$\mathcal{L}_{\text{Topo}}$$는 각각 비전 모델과 위상학적 브랜치의 교차 엔트로피 손실, $$\alpha$$는 균형 하이퍼파라미터입니다.[2]

## 3. 모델 구조

PHG-Net은 **4단계 파이프라인**으로 구성됩니다:[1]

1. **Stage 1**: 입력 영상에서 큐비컬 지속성 다이어그램 계산
2. **Stage 2**: PD 인코더(PH 모듈)를 통해 위상학적 특징을 벡터로 변환
3. **Stage 3**: FC 블록을 통한 게이트 메커니즘으로 특징 조절
4. **Stage 4**: CNN/Transformer의 각 층에서 특징 맵 개선

**PH 모듈**은 지속성 다이어그램을 점 구름으로 처리하며, PointNet과 유사한 구조를 가집니다. 각 데이터 포인트는 (birth, death, homology group indicator)로 표현되며, 순열 불변성을 보장합니다.[1][2]

## 4. 성능 향상 및 실험 결과

**세 개 공개 데이터셋에서의 성능 향상**:[1][2]

- **ISIC 2018**: SwinV2-B 기준으로 정확도 1.07% 향상 (90.85% → 91.92%)
- **Prostate Cancer**: SwinV2-B 기준으로 정확도 3.43% 향상 (95.21% → 98.64%)  
- **CBIS-DDSM**: SwinV2-B 기준으로 정확도 3.72% 향상 (73.51% → 77.23%)

**통계적 유의성**을 paired t-test로 검증했으며, 대부분의 지표에서 p < 0.01의 유의한 개선을 보였습니다.[2]

**계산 복잡도 분석**에서 추가 파라미터와 계산 비용은 상당히 제한적임을 확인했습니다. PD 인코더를 다중 스케일에서 공유하면 성능 저하 없이 파라미터와 FLOP을 더욱 줄일 수 있습니다.[2]

## 5. 모델의 일반화 성능 향상 가능성

**일반화 성능 향상의 핵심 요인들**:

### 5.1 위상학적 불변성과 견고성
지속 호몰로지는 **연속적 변형에 대한 불변성**을 제공하여, 노이즈나 작은 변형에 대해 견고한 특징을 추출합니다. 이는 의료 영상에서 환자별, 장비별 차이에도 불구하고 일관된 성능을 보장할 수 있습니다.[3][4][5]

### 5.2 다중 스케일 특징 통합
기존 방법들이 마지막 CNN 층에서만 위상학적 특징을 결합하는 반면, PHG-Net은 **다중 스케일에서 특징을 통합**합니다. 이를 통해 저수준부터 고수준까지 다양한 해부학적 구조를 포착할 수 있어 일반화 성능이 향상됩니다.[1][2]

### 5.3 범용적 아키텍처 호환성
PHG-Net의 PH 모듈은 **경량화되어 있으며 임의의 CNN이나 Transformer 아키텍처에 통합 가능**합니다. 이는 다양한 의료 영상 태스크와 아키텍처에 적용할 수 있는 범용성을 제공합니다.[1][2]

### 5.4 데이터 기반 학습 방식
기존의 수학적 인코딩 방식과 달리, **데이터 기반의 학습 가능한 벡터화 과정**을 통해 태스크에 특화된 위상학적 표현을 학습합니다. 이는 다양한 의료 영상 도메인에서 적응적 성능을 보장할 수 있습니다.[2][1]

## 6. 한계점

논문에서 직접적으로 언급된 한계점은 제한적이지만, 다음과 같은 한계점들을 유추할 수 있습니다:

1. **계산 복잡성**: 지속성 다이어그램 계산이 추가적인 계산 비용을 요구함[2]
2. **하이퍼파라미터 민감성**: α 값과 상위 n개 포인트 선택 등의 하이퍼파라미터 조정이 필요[2]
3. **3D 확장성**: 현재는 2D 의료 영상에 초점을 맞춤 (3D 확장에 대한 제한적 논의)[6]
4. **위상학적 노이즈**: 대각선에 가까운 점들을 노이즈로 간주하는 방식의 한계[2]

## 7. 앞으로의 연구에 미치는 영향과 고려사항

### 7.1 연구에 미치는 영향

**위상학적 딥러닝의 발전**: PHG-Net은 위상학적 데이터 분석과 딥러닝의 결합에서 **새로운 패러다임을 제시**합니다. 특히 의료 영상 분야에서 해부학적 구조의 중요성을 재조명하며, 향후 TDA 기반 딥러닝 연구의 기초를 마련했습니다.[5][1]

**범용적 통합 프레임워크**: 임의의 CNN/Transformer 아키텍처에 통합 가능한 구조는 **다양한 컴퓨터 비전 태스크로의 확장 가능성**을 제시합니다. 이는 의료 영상을 넘어 일반적인 영상 분석 태스크에도 적용될 수 있는 잠재력을 보여줍니다.[7][8][1]

**3D 확장 연구 촉진**: 논문의 성공은 3D 의료 영상에서의 위상학적 분석 연구를 촉진할 것으로 예상됩니다. 이미 3D TDA 방법들이 개발되고 있으며, PHG-Net의 접근법이 3D로 확장될 가능성이 높습니다.[6]

### 7.2 앞으로 연구 시 고려사항

**1. 계산 효율성 개선**: 지속성 다이어그램 계산의 **계산 복잡도를 줄이는 연구**가 필요합니다. 특히 대용량 의료 영상이나 실시간 처리가 요구되는 환경에서는 효율성이 중요한 고려사항입니다.[5]

**2. 다양한 위상학적 표현 탐색**: **다양한 TDA 도구들의 통합**을 고려해야 합니다. 지속 호몰로지 외에도 지속 엔트로피, 지속 랜드스케이프 등 다양한 위상학적 서명의 활용 가능성을 탐색할 필요가 있습니다.[4][9][5]

**3. 멀티모달 데이터 통합**: **다양한 모달리티의 의료 영상을 통합**하는 연구가 중요합니다. MRI, CT, X-ray 등 서로 다른 영상 모달리티에서의 위상학적 특징 통합 방법을 개발해야 합니다.

**4. 해석 가능성 향상**: 위상학적 특징이 **임상적으로 어떤 의미를 갖는지에 대한 해석 가능성 연구**가 필요합니다. 의료진이 이해할 수 있는 형태로 위상학적 정보를 제시하는 방법을 개발해야 합니다.[5]

**5. 표준화 및 벤치마킹**: **위상학적 딥러닝 방법들의 표준화된 평가 프레임워크**를 구축하는 것이 중요합니다. 다양한 TDA 기반 방법들을 공정하게 비교할 수 있는 벤치마크 데이터셋과 평가 지표가 필요합니다.[5]

**6. 실제 임상 환경 적용**: **실제 임상 워크플로우에 통합하기 위한 실용적 고려사항**들을 연구해야 합니다. 규제 승인, 임상 검증, 사용자 인터페이스 등이 중요한 연구 과제가 될 것입니다.

PHG-Net은 위상학적 특징과 딥러닝의 효과적인 결합을 통해 의료 영상 분석 분야에 새로운 방향성을 제시했으며, 향후 이 분야의 발전에 중요한 기여를 할 것으로 예상됩니다.

[1] https://ieeexplore.ieee.org/document/10484262/
[2] https://openaccess.thecvf.com/content/WACV2024/papers/Peng_PHG-Net_Persistent_Homology_Guided_Medical_Image_Classification_WACV_2024_paper.pdf
[3] https://ieeexplore.ieee.org/document/11077499/
[4] https://www.semanticscholar.org/paper/cd46d916adf7c07c32895f5d87831df5b5041b66
[5] https://link.springer.com/article/10.1007/s10462-024-10710-9
[6] https://arxiv.org/abs/2408.07905
[7] https://www.semanticscholar.org/paper/02e032378bd5c7d0faf1a8c6799bdee8cfa1d617
[8] https://www.semanticscholar.org/paper/3739770427a1fe75029c2dbf09aeb5778a18910d
[9] https://pmc.ncbi.nlm.nih.gov/articles/PMC10177619/
[10] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/127deb13-a8ff-4950-8ae6-3dabe098c2bd/2311.17243v1.pdf
[11] http://medrxiv.org/lookup/doi/10.1101/2025.02.21.25322669
[12] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10956/2513137/Persistent-homology-for-the-automatic-classification-of-prostate-cancer-aggressiveness/10.1117/12.2513137.full
[13] https://ojs.bonviewpress.com/index.php/JCCE/article/view/3722
[14] https://link.springer.com/10.1007/978-3-030-59728-3_5
[15] http://arxiv.org/pdf/2311.17243.pdf
[16] https://arxiv.org/pdf/2403.11001.pdf
[17] https://pmc.ncbi.nlm.nih.gov/articles/PMC9868648/
[18] http://arxiv.org/pdf/1805.03699.pdf
[19] http://arxiv.org/pdf/2012.12102v2.pdf
[20] https://pmc.ncbi.nlm.nih.gov/articles/PMC11082231/
[21] https://pmc.ncbi.nlm.nih.gov/articles/PMC10067000/
[22] https://www.mdpi.com/2072-6694/15/9/2606
[23] https://arxiv.org/pdf/2110.06295.pdf
[24] https://arxiv.org/abs/2311.17243
[25] https://www.sciencedirect.com/science/article/abs/pii/S0010482523003451
[26] https://www.themoonlight.io/ko/review/position-topological-deep-learning-is-the-new-frontier-for-relational-learning
[27] https://www.sciencedirect.com/science/article/abs/pii/S2214860422006261
[28] https://en.wikipedia.org/wiki/Topological_deep_learning
[29] https://arxiv.org/html/2508.01574v1
[30] https://www.numberanalytics.com/blog/image-analysis-persistent-homology-deep-dive
[31] https://arxiv.org/abs/2302.03836
[32] https://www.phdontrack.net/good-research-practices/research-impact/
[33] https://arxiv.org/abs/2312.05840
[34] https://pmc.ncbi.nlm.nih.gov/articles/PMC8758229/
[35] https://pub.ista.ac.at/~edels/Papers/2016-06-Endoscopy.pdf
[36] https://astron.snu.ac.kr/8053/
[37] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART111002317
[38] https://www.sciencedirect.com/science/article/abs/pii/S0010482525005773
[39] https://academic.oup.com/bib/article/24/5/bbad289/7241306
