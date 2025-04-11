# Bone Suppression on Chest Radiographs for Pulmonary Nodule Detection: Comparison between a Generative Adversarial Network and Dual-Energy Subtraction, 인용수 : 29
To compare the effects of bone suppression imaging using deep learning (BSp-DL) based on a generative adversarial network (GAN) and bone subtraction imaging using a dual energy technique (BSt-DE) on radiologists’ performance for pulmonary nodule detection on chest radiographs (CXRs).  

생성적 적대 신경망(GAN)을 기반으로 한 딥러닝(BSP-DL)과 이중 에너지 기법(BST-DE)을 사용한 뼈 제거 영상이 흉부 방사선 사진(CXR)에서 폐 결절 검출에 대한 방사선 전문의의 성과에 미치는 영향을 비교하기 위해 이중 에너지 기법(bone subtraction imaging using a dual energy technique)과 흉부 CT를 사용하여 CXR을 모두 받은 83개의 폐 결절 환자 49명을 포함하여 총 111명의 성인이 등록했습니다.  
CT를 참고하여 두 명의 독립적인 방사선 전문의가 세 번의 판독 세션(표준 CXR, BSt-DE CXR, BSp-DL CXR)에서 폐 결절 유무에 대해 CXR 영상을 평가했습니다.

사람별 및 결절별 성능은 각각 eceiver-operating characteristic (ROC) and alternative free-response ROC (AFROC) curve 분석을 사용하여 평가했습니다.  

## ROC Curve, AFROC
```
ROC Curve
x축(1 - specificity = FPR(False Positive Rate))은 가짜 중에 진짜를 찾은 비율(가짜 중에 잘못 예측한 비율)이고, y축(Sensitivity)이 의미하는 바는 진짜 중에 진짜를 찾은 비율(진짜 중에 진짜를 잘 찾은 비율)이 된다.
진짜로 예측한 값들 중에서 실제로도 진짜일 경우가 실제로 가짜일 경우보다 높아야 AUC가 높게 나온다.  
곡선이 굽어지면 굽어질수록 AUC가 넓어지므로, 더욱 정확한 모델임을 알 수 있다. 
[[출처] ROC (Receiver Operating Characteristic) Curve  완벽 정리!|작성자 PN](https://blog.naver.com/sw4r/221015817276)

AFROC Curve
X축을 위양성이 포함된 영상의 비율로, Y축을 병변의 유무와 위치를 정확히 평가한 비율로 정의하며, 두 축 모두 0에서 1 사이의 값을 가집니다.
https://www.jnpmedi.com/media/?bmode=view&idx=140072145
```
BSt-DE with an area under the AFROC curve (AUAFROC) of 0.996 and 0.976 for readers 1 and 2, respectively, and BSp-DL with AUAFROC of 0.981 and 0.958, respectively, showed better nodule-wise performance than standard CXR (AUAFROC of 0.907 and 0.808, respectively; p ≤ 0.005).

AFROC 곡선 아래 면적(AUAFROC)이 1명과 2명일 때 각각 0.996과 0.976인 BSt-DE와 AUAFROC가 각각 0.981과 0.958인 BSp-DL은 표준 CXR(AUAFROC 각각 0.907과 0.808, p≤ 0.005)보다 결절별 성능이 우수했습니다.

We developed a new bone suppression model based on the wavelet transform and generative adversarial networks (GANs) [20].  
Although the performance of this model was evaluated by comparing quality metrics with those of other convolutional neural network-based models, its potential for clinical application has not yet been assessed.  
In addition, the performance of software-based bone suppression images compared to that of bone subtraction images using the dual energy technique remains unclear.  
Therefore, the purpose of the present study was to evaluate the effect of bone suppression imaging using deep learning (BSp-DL) based on GAN compared to bone subtraction imaging using a dual energy technique (BSt-DE) on radiologists’ performances for pulmonary nodule detection on CXR.

우리는 웨이블릿 변환과 생성적 적대 신경망(GAN)을 기반으로 한 새로운 뼈 억제 모델을 개발했습니다 [20].  
이 모델의 성능은 다른 합성곱 신경망 기반 모델과 품질 지표를 비교하여 평가되었지만, 임상 적용 가능성은 아직 평가되지 않았습니다.  
또한 소프트웨어 기반 골 억제 이미지의 성능은 이중 에너지 기법을 사용한 골 제거 이미지와 비교했을 때 아직 명확하지 않습니다.  
따라서 본 연구의 목적은 이중 에너지 기법(BST-DE)을 사용한 골 제거 영상과 비교하여 GAN 기반의 딥러닝(BSP-DL)을 사용한 골 억제 영상이 방사선 전문의의 폐 결절 검출 성능에 미치는 영향을 평가하는 것이었습니다.

Our bone suppression model used adversarial training in the GAN framework to learn the conditional probability distribution of the output images according to the input images.  
The Haar wavelet decomposition was adopted as an input system.  
It pre-defined features that the network should learn via wavelet transformation of an image into four directional feature images (Fig. 2).  
This allowed the model to exploit high-frequency details of CXR and to converge more quickly and efficiently [21].  
In the GAN framework, the generator attempted to deceive the discriminator by creating an image similar to the training set.  
This bone suppression model was trained and validated using a total of 348 pairs of composite and soft tissue selective images obtained by dual-energy radiography that were available in public domain [20].  
Technical details of the model, including network architecture, quantitative performance metrics, and training process, have been described previously [20].  
No additional training was imparted for this study. A total of 111 CXRs obtained using the dual energy technique were postprocessed.  
The generated bone-suppressed images (BSpDL) were anonymized and stored for evaluation.

우리의 뼈 억제 모델은 입력 이미지에 따른 출력 이미지의 조건부 확률 분포를 학습하기 위해 GAN 프레임워크에서 적대적 훈련을 사용했습니다. Haar 웨이블릿 분해는 입력 시스템으로 채택되었습니다. 이는 네트워크가 이미지를 네 방향 특징 이미지로 웨이블릿 변환하여 학습해야 할 특징을 미리 정의한 것입니다(그림 2).  
이를 통해 모델은 CXR의 고주파 세부 사항을 활용하고 더 빠르고 효율적으로 수렴할 수 있었습니다 [21].  
GAN 프레임워크에서 생성기는 훈련 세트와 유사한 이미지를 생성하여 판별기를 속이려고 시도했습니다.  
이 뼈 억제 모델은 공개 도메인에서 사용할 수 있는 이중 에너지 방사선 촬영을 통해 얻은 총 348쌍의 복합 및 연조직 선택 이미지를 사용하여 훈련 및 검증되었습니다[20].  

![image](https://github.com/user-attachments/assets/b16b1699-16e8-4674-a428-f43e669630d1)

A. 원본 이미지를 수신하고 뼈가 억제된 이미지를 생성하는 생성기의 아키텍처.  
이 시스템은 네트워크가 학습해야 할 특징을 사전에 정의한 Haar 웨이블릿 변환에서 얻은 주파수 정보를 사용하여 네트워크가 더 빠르고 효율적으로 수렴할 수 있도록 합니다.  
생성기는 Haar 2D 웨이블릿 분해를 통해 얻은 512 x 512 소스(원본) 이미지의 4채널을 가져와 이미지 블러링을 방지하여 판별기를 속일 수 있는 출력(뼈가 억제된) 이미지를 생성하려고 합니다.  
출력 이미지는 Haar 2D 웨이블릿 재구성을 통해 최종적으로 1024 x 1024로 재구성됩니다.  
각 컨볼루션 블록 아래의 값은 채널 수에 따른 이미지 압축 비율입니다.  

이미지 블러링을 방지하기 위해, 입력 데이터의 Harr 2D 웨이블릿 분해를 통해 원본 이미지의 주파수 세부 사항을 보다 효과적으로 학습하도록 설계되었습니다.

B. 판별기의 아키텍처. 판별기는 입력이 생성기에서 나온 가짜 이미지인지, 아니면 훈련 세트에서 나온 실제 이미지인지 구별합니다.  
훈련 과정에서 두 네트워크가 경쟁합니다. 판별기는 생성기가 흐릿한 이미지를 생성하지 못하도록 하는 데 중요한 역할을 하며, 생성기와 마찬가지로 이미지 배치 s의 분포도 고려합니다.  

## Haar Wavelet Transform
```
This wavelet transform finds its most appropriate use in non-stationary signals.
This transformation achieves good frequency resolution for low-frequency components and high temporal resolution for high-frequency components.
비정상 신호에서 가장 적합한 용도를 찾습니다. 이 변환은 저주파 성분에 대한 좋은 주파수 분해능과 고주파 성분에 대한 높은 시간 분해능을 달성합니다.
Dimensionality reduction: The Haar wavelet transform can reduce the dimensionality of a dataset while preserving important features. This can lead to faster computation and improved performance.
차원 감소 : Haar 웨이블릿 변환은 중요한 특징을 보존하면서 데이터 세트의 차원을 줄일 수 있습니다. 이를 통해 더 빠른 계산과 향상된 성능을 얻을 수 있습니다.

웨이블릿 분석은 이미지에 존재하는 정보(신호)를 근사치와 세부 정보(하위 신호)라는 두 가지 개별 구성 요소로 나누는 데 사용됩니다.
신호는 고역 통과 필터와 저역 통과 필터, 두 개의 필터를 통과합니다.
그런 다음 이미지는 고주파(디테일)와 저주파(근사) 성분으로 분해됩니다. 각 레벨에서 4개의 하위 신호를 얻습니다.
근사값은 픽셀 값의 전반적인 추세와 수평, 수직, 대각선 성분으로 표현된 디테일을 보여줍니다.

https://medium.com/@koushikc2000/2d-discrete-wavelet-transformation-and-its-applications-in-digital-image-processing-using-matlab-1f5c68672de3
https://en.wikipedia.org/wiki/Haar_wavelet
https://rla020.tistory.com/16
https://blog.naver.com/PostView.nhn?blogId=skkong89&logNo=222093854309
```

두 명의 보드 인증 방사선 전문의(각각 5년 및 7년 경력)는 2주 간격으로 세 번의 판독 세션에서 폐 결절을 감지하기 위해 333개의 이미지 세트를 독립적으로 평가

![image](https://github.com/user-attachments/assets/77546761-3516-4969-bbdd-41d406db28d7)


![image](https://github.com/user-attachments/assets/7848d592-3624-4744-93c2-fcb9acb1cf39)


While there was no significant difference in the detection of nodules in the central lung zone or nodules > 10 mm, between the two techniques, BSp-DL was inferior to BSt-DE in detecting the sub-centimeter and peripheral nodules.  

중앙 폐 영역이나 결절 > 10mm에서 결절 검출에는 큰 차이가 없었지만, 두 기술 간에는 BSp-DL이 센티미터 이하와 주변 결절 검출에서 BSt-DE보다 열등했습니다.

Although our program outperformed existing state-of-the-art methods in preserving the frequency details of original images, small peripheral nodules could fade out while de-noising the overlying bones.

우리 프로그램은 원본 영상의 주파수 세부 사항을 보존하는 데 있어 기존의 최첨단 방법보다 우수했지만, 작은 주변 결절은 위에 있는 뼈를 노이즈 제거하면서 희미해질 수 있습니다.

이 연구의 한계는 각 하위 그룹의 사례 수가 적다는 점이며, 이는 결과에 영향을 미쳤을 수 있습니다.  

In conclusion, BSp-DL (a bone suppression imaging based on the GAN framework) can improve radiologists’ detection of nodules on CXRs.  
It showed comparable performance to that of the DES technique.  
Nevertheless, further technical improvements are needed for better delineation of small and peripherally located nodules on CXRs.

결론적으로, BSp-DL(간 프레임워크를 기반으로 한 골 억제 영상)은 방사선 전문의의 CXR 결절 검출을 개선할 수 있습니다.  
이는 DES 기법과 비슷한 성능을 보였습니다. 그럼에도 불구하고, CXR에서 작고 말초에 위치한 결절을 더 잘 묘사하기 위해서는 추가적인 기술적 개선이 필요합니다.

