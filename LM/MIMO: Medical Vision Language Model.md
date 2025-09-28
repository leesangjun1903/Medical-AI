# MIMO: Medical Vision Language Model

## 배경 및 문제점

기존의 의료 비전-언어 모델(MVLM)은 주로 텍스트 지시만 입력으로 받아 이미지의 시각적 단서를 직접적으로 이해하지 못하고, 출력도 텍스트 답변만 제공하여 이미지의 핵심 영역과의 연결이 부족하다는 한계가 있었습니다[1][2][3]. 즉, 의료 이미지를 해석할 때 "이 부위"와 같이 특정 영역을 언급하거나, 답변에서 해당 부위를 이미지 상에 표시하는 기능이 부족했습니다.

## MIMO의 주요 아이디어

**MIMO**는 이러한 한계를 극복하기 위해 다음 두 가지 핵심 기능을 모두 갖춘 최초의 의료 비전-언어 모델입니다[1][2][3]:

- **Visual Referring Multimodal Input(시각적 지시 멀티모달 입력)**:  
  사용자가 텍스트 질문과 함께 이미지 내의 점, 박스 등 시각적 힌트를 함께 입력할 수 있습니다. 예를 들어, "이 박스 안에 있는 장기는 무엇인가요?"처럼 특정 부위를 정확히 지목하여 질문할 수 있습니다.

- **Pixel Grounding Multimodal Output(픽셀 정렬 멀티모달 출력)**:  
  모델이 답변을 텍스트로만 주는 것이 아니라, 답변 내에서 언급된 의료 개체(예: 종양, 기관 등)를 이미지의 분할(세그멘테이션) 마스크와 연결해줍니다. 즉, "간(liver)"이라는 단어와 실제 이미지 상의 간 위치가 함께 출력됩니다.

## 모델 구조

MIMO의 구조는 다음과 같습니다[1][3]:

- **이미지 인코더**: CLIP ViT-H/14 기반으로 이미지를 고해상도 특성으로 임베딩합니다.
- **시각적 프롬프트 인코더**: 점, 박스 등 시각적 힌트를 임베딩하여 이미지 특성과 같은 공간에 배치합니다.
- **멀티모달 입력 정렬기**: 텍스트, 이미지, 시각적 힌트를 효과적으로 융합해 중요한 정보를 추출합니다.
- **대형 언어 모델(LLM)**: Vicuna 기반 LLM이 텍스트 답변을 생성하며, 특수 토큰( <p>, <SEG> )을 활용해 답변 내 의료 개체와 분할 마스크를 연결합니다.
- **분할 디코더**: SAM(Segment Anything Model) 기반으로, LLM에서 지정한 개체의 픽셀 단위 분할 마스크를 생성합니다.

## 대규모 데이터셋 MIMOSeg

모델 학습을 위해 **MIMOSeg**라는 89.5만 개의 샘플로 구성된 대규모 멀티모달 데이터셋을 자체 구축했습니다.  
이 데이터셋은 CT, X-ray, 안저, 병리 등 8개 의료 이미지 모달리티를 포함하며,  
- 텍스트 지시 기반 분할  
- 시각적 프롬프트 기반 분할  
- 분할 정렬형 질의응답  
- 시각적 프롬프트 보조 질의응답  
등 다양한 복합 시나리오를 포괄합니다[1][3].

## 실험 결과 및 의의

- MIMO는 기존 의료 VLM 및 세그멘테이션 모델(SAM, GLaMM 등)과 비교해 다양한 시나리오에서 더 뛰어난 성능을 보였습니다.
- 특히, 복잡한 질의응답이나 시각적 힌트가 포함된 문제에서 답변의 정확도와 분할의 정밀도가 모두 높았습니다.
- 실제 임상 현장에서 의사들이 이미지의 특정 부위를 지목하고, 해당 부위에 대한 정보를 얻고자 할 때 매우 직관적이고 실용적인 도구가 될 수 있습니다[1][2][3].

## 요약

- **MIMO**는 "시각적 지시가 가능한 입력"과 "텍스트-이미지 연결이 가능한 출력"을 동시에 지원하는 최초의 의료 비전-언어 모델입니다.
- 대규모 멀티모달 데이터셋(MIMOSeg)을 활용해 다양한 의료 이미지 및 복합 질의응답 시나리오에 대응할 수 있습니다.
- 의료 AI의 실용성과 해석 가능성을 크게 높인 혁신적인 연구입니다[1][2][3].

[1] https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_MIMO_A_Medical_Vision_Language_Model_with_Visual_Referring_Multimodal_CVPR_2025_paper.pdf
[2] https://cvpr.thecvf.com/virtual/2025/poster/35156
[3] https://blog.csdn.net/alfred_torres/article/details/148561917
[4] https://arxiv.org/abs/2504.11368
[5] https://arxiv.org/abs/2410.14200
[6] https://ieeexplore.ieee.org/document/10570257/
[7] https://arxiv.org/abs/2410.02615
[8] https://arxiv.org/abs/2504.14692
[9] https://arxiv.org/abs/2407.05131
[10] https://arxiv.org/abs/2406.06007
[11] https://arxiv.org/abs/2410.13085
[12] https://atalupadhyay.wordpress.com/2025/06/15/mimo-vl-7b-the-vision-language-ai-revolution-complete-technical-guide-hands-on-lab/
[13] https://arxiv.org/abs/2403.18996
[14] https://huggingface.co/papers/2311.03356
[15] https://openreview.net/pdf/50e8f8f71a8a58fa6ad0a62f9ff1e412a296ff98.pdf
[16] https://www.youtube.com/watch?v=KmdaVIJi1Nc
[17] https://arxiv.org/abs/2407.15728
[18] https://www.mdpi.com/2306-5354/10/3/380
[19] https://arxiv.org/abs/2504.05575
[20] https://arxiv.org/abs/2407.21788
[21] https://arxiv.org/abs/2504.20343
[22] https://arxiv.org/abs/2303.01615
[23] https://arxiv.org/abs/2502.07409
[24] https://openreview.net/forum?id=lvZrYKLBzH
[25] https://www.winlab.rutgers.edu/~aashok/visualmimo/aashok_ciss11.pdf
[26] https://peterhcharlton.github.io/info/datasets/mimic
[27] https://www.winlab.rutgers.edu/~aashok/visualmimo/Overview.html
[28] https://ieeexplore.ieee.org/document/10816095/
[29] https://arxiv.org/abs/2405.10948
[30] https://proceedings.neurips.cc/paper_files/paper/2024/file/dc6319dde4fb182b22fb902da9418566-Paper-Conference.pdf
[31] https://www.themoonlight.io/en/review/mimo-vl-technical-report
[32] https://link.springer.com/10.1007/s10278-024-01051-8
[33] https://encord.com/blog/vision-language-models-guide/
[34] https://arxiv.org/html/2501.02385v1
[35] https://www.tvba.sk/imageuploads/1207/sprava-kreativny-priemysel-mksr-neulogy.pdf
