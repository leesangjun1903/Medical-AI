# Main Project : Bone suppression, X-ray 이미지에서 갈비뼈 제거
## Survey
https://github.com/diaoquesang/A-detailed-summarization-about-bone-suppression-in-Chest-X-rays?tab=readme-ov-file 

## DATASETS
### Stanford AIMI Shared Datasets
https://stanfordaimi.azurewebsites.net/datasets?domain=CHEST&page=2

### X-ray Bone Shadow Supression (Useful kaggle challenge )
https://www.kaggle.com/datasets/hmchuong/xray-bone-shadow-supression/data

# Papers and Models
- Dual energy subtraction: Principles and clinical applications
- Deep learning models for bone suppression in chest radiographs, Maxim Gusarev
- Bone Suppression on Chest Radiographs for Pulmonary Nodule Detection: Comparison between a Generative Adversarial Network and Dual-Energy Subtraction, Kyungsoo Bae
- Chest X‐Ray Bone Suppression for Improving Classification of Tuberculosis‐Consistent Findings, Sivaramakrishnan Rajaraman(2021) : https://github.com/sivaramakrishnan-rajaraman/CXR-bone-suppression
- DeBoNet: A deep bone suppression model ensemble to improve disease detection in chest radiographs (2022) : https://github.com/sivaramakrishnan-rajaraman/Bone-Suppresion-Ensemble
- Development and Validation of a Deep Learning–Based Synthetic Bone-Suppressed Model for Pulmonary Nodule Detection in Chest Radiographs, Hwiyoung Kim
- An Efficient and Robust Method for Chest X-ray Rib Suppression That Improves Pulmonary Abnormality Diagnosis : https://github.com/FluteXu/CXR-Rib-Suppression/tree/master
- Diffusion 기반 모델 : BS-LDM: Effective Bone Suppression in High-Resolution Chest X-Ray Images with Conditional Latent Diffusion Models : https://github.com/diaoquesang/BS-LDM
- Remove Bones from X-Ray images with fastai : https://github.com/mmiv-center/deboning/tree/master
- Suzuki 저자의 논문 : 읽어볼 여지가 있으나 2020년대 이후 BS 작업은 확인되지 않음. : https://suzukilab.first.iir.titech.ac.jp/publications/journal-papers/ : Enhancement of chest radiographs obtained in the intensive care unit through bone suppression and consistent processing(2016)
- U-Net 기반 모델 : xU-NetFullSharp: ANN for Chest X-Ray Bone Shadow Suppression : https://github.com/xKev1n/xU-NetFullSharp/tree/main
- xU-NetFullSharp (Paper Implementation) : https://www.kaggle.com/code/arjunbasandrai/xu-netfullsharp-paper-implementation
- Deep Learning Models for bone suppression in chest radiographs (Gusarev's bone suppression method 사용. ConV , AE 기반 모델.) : https://github.com/danielnflam/Deep-Learning-Models-for-bone-suppression-in-chest-radiographs?tab=readme-ov-file
- High-Resolution Chest X-ray Bone Suppression Using Unpaired CT Structural Priors : https://github.com/MIRACLE-Center/High-Resolution-Chest-X-ray-Bone-Suppression?tab=readme-ov-file
- Bone Suppression on Chest Radiographs With Adversarial Learning | Pix2Pix, CycleGAN
- Generation of Virtual Dual Energy Images from Standard Single-Shot Radiographs using Multi-scale and Conditional Adversarial Network
- Feasibility Study of Deep-Learning-Based Bone Suppression with Single-Energy Material Decomposition in Chest X-Rays

## Object Removal or Image Inpainting : 특정 물체 제거 작업 -> Suppression 작업에 영향을 줄 수 있을 것.
- An object removal from image system using deep learning image segmentation and inpainting techniques(Free-Form Image Inpainting with Gated Convolution) : https://github.com/treeebooor/object-remove
- Sementic segmentation : (Mask-R CNN : https://herbwood.tistory.com/20)  
- Semantic segmenator model of deeplabv3/fcn resnet 101 has been combined with EdgeConnect(EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning) : https://medium.com/analytics-vidhya/removing-objects-from-pictures-with-deep-learning-5e7c35f3f0dd , https://github.com/sujaykhandekar/Automated-objects-removal-inpainter/tree/master
- Object Removal using Image Processing : https://github.com/KnowledgePending/Object-Removal-using-Image-Processing/tree/master  

## Bone fracture detection : 뼈 탐지 알고리즘

