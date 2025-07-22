# From Signal to Knowledge: The Diagnostic Value of Raw Data in the Artificial Intelligence Prediction of Human Data for the First Time

# 핵심 요약 및 주요 기여

**주요 주장**  
비파괴적 진단 절차에서 전통적으로는 CT 스캔의 원시 신호(raw data)를 영상으로 재구성한 뒤 그 이미지를 바탕으로 AI가 진단을 수행해 왔으나, 이 과정에서 재구성 오류로 인한 정보 손실이 불가피하다. 본 연구는 원시 신호를 직접 AI로 분석(“signal-to-knowledge”)하여 폐 결절의 양·악성 분류 성능이 기존 이미지 기반 진단과 동등하거나 이를 개선할 수 있음을 세계 최초로 실증하였다.

**주요 기여**  
1. 실제 환자(raw data) 기준 최초의 “signal-to-knowledge” AI 진단 파이프라인 개발  
2. CT 재구성 이미지 단독 모델(CTM) 대비 원시 신호를 결합한 residual fusion 모델(RGM)이 AUC 0.01–0.12 상승  
3. Grad-CAM 및 attention 분석으로 raw data 내 병변 부위 집중 확인  
4. 다양한 CT 기반 모델(CTM1–CTM4) 및 백본(DN, RE, RX)에서 성능·안정성 일관적 개선  

# 문제 정의 및 해결 방법

## 해결하고자 한 문제  
- CT raw data가 영상 재구성 과정에서 폐색전·중첩·필터링으로 인한 진단 정보 손실을 초래함  
- 기존 AI 진단도 영상에 의존하여 동일한 한계를 지님  

## 제안한 방법  
1. **데이터 수집**  
   - 276명 환자의 contrast-enhanced CT 이미지 및 raw data  
   - 레이블: 병리학적 양·악성 확진  
2. **CT 모델(CTM)**  
   - CT 이미지 기반 폐 결절 분류에 사용된 대표 논문 4종(CTM1–CTM4) 재현  
3. **Raw Data Gain 모델(RGM)**  
   - CTM 예측 확률에 raw data 전용 3D CNN(RDM; DenseNet121, ResNet18, ResNeXt18) 출력 확률을 residual 합산  
   -  
$$
       p_{\text{positive}}^{\text{RGM}} = p_{\text{positive}}^{\text{RDM}} + p_{\text{positive}}^{\text{CTM}},
     $$  
     $$
       p_{\text{negative}}^{\text{RGM}} = p_{\text{negative}}^{\text{RDM}} + (1 - p_{\text{positive}}^{\text{CTM}})
     $$  

   - Softmax 후 교차엔트로피로 학습  

4. **Lesion Mapping**  
   - CT–raw data 간 voxel 좌표 변환(공간 보정, scan index 범위 쿼리, detector plane 투영)  
5. **성능 측정**  
   - 훈련·검증·테스트 코호트 AUC 비교, t-SNE 분포, Grad-CAM 집중도, 하위그룹 분석  

# 모델 구조 및 성능 향상

| 모델 종류               | CTM AUC (Test) | 최우수 RGM AUC (Test) | AUC 개선폭 |
|------------------------|----------------|-----------------------|------------|
| CTM1 (multi-scale)     | 0.807          | 0.853 (RGM-RE1)       | +0.046     |
| CTM2 (global/local)    | 0.760          | 0.782 (RGM-DN2)       | +0.022     |
| CTM3 (loss-based)      | 0.773          | 0.800 (RGM-RE3)       | +0.027     |
| CTM4 (multi-view)      | 0.833          | 0.867 (RGM-RE4)       | +0.034     |

- 모든 CTM에서 최소 +0.01, 최대 +0.12(AUC 기준) 성능 향상  
- ResNeXt 백본(RGM-RX)이 평균 최다 개선폭 달성  

**Grad-CAM 집중도**  
병변 영역의 평균 attention score가 비병변 대비 1–2배 높음 → raw data가 실질적 진단 정보를 담고 있음을 시각적·정량적으로 검증.

# 일반화 성능 향상 관련 고찰

- **모델 안정성**: 4개 CTM 모두에서 RGM이 CTM 대비 error rate보다 높은 optimization rate 보이며, 80% 샘플이 최소 2개 RGM에서 개선  
- **하위그룹 분석**:  
  - 고령(>60세), 여성, 소결절(≤23 mm) 등에서 CTM보다 RGM 성능 우위  
  - 전폐엽 위치별(좌·우 상·하엽)에서도 일관된 성능 상승  
→ 다양한 환자군·촬영 조건에 걸쳐 일반화 가능성 시사

# 한계 및 향후 연구 고려사항

1. 대상 환자 수(276명), 단일 센터·단일 기종 CT 한정 → 다기관·대규모 검증 필요  
2. 다결절 환자, 다양한 병변(비폐결절) 확장 연구 필요  
3. 3D CNN 구조 외 raw data 특화 네트워크(Transformer, 스파스 컨볼루션) 탐색  
4. 현실적 컴퓨팅 부담 해소 위한 경량화·프리프로세싱 기법 개발  
5. Raw data 최적 획득을 위한 새로운 스캔 프로토콜 설계 고려

# 향후 영향 및 제언

- **임상 영상진단 패러다임 전환**: 영상 재구성 비의존적 진단 가능성 열어, 스캔·진단 전(前) 자동화  
- **저자원 의료기관 적용**: 인력·전문가 부족 환경에서 CT raw data 직접 분석 솔루션으로 균등 의료 기회 제공  
- **융합 진단 연구**: 원시·영상·임상·유전체 데이터를 통합한 멀티모달 AI 연구 가속화  
- **표준화 요구**: raw data 처리·저장·교환 표준 부재 → 표준 프로토콜 개발 시급

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5bec3486-9c53-49d8-984e-f0f14930043d/1-s2.0-S209580992300156X-main.pdf
