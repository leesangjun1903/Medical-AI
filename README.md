# Computer-Tomograpy reconstruction

## Computer Tomography(Theory)
- Radon Transform  
- Filtered Back Projection(FBP)  
- Deep Filtered Back Projection(DFBP)
- Low pass filters
- Sampling Theory
- Parallel beam geometry
- Fan beam geometry
- Multivariable Radon transform, X-ray transform
- Orlov's fomula
- Grageat's method
- FDK algorithm

## CT reconstruction(Super-resolution, Image Denoising), 연구해볼 모델
### Useful paper(survey) :
- Deep Learning for Single Image Super-Resolution: A Brief Review
- A Deep Journey into Super-resolution: A survey
- Deep learning for image super-resolution: A survey
- Real-World Single Image Super-Resolution:A Brief Review
- Blind Image Super-Resolution: A Survey and Beyond
- Generative Adversarial Networks for Image Super-Resolution: A Survey
- Hitchhiker's Guide to Super-Resolution: Introduction and Recent Advances
- Diffusion Models, Image Super-Resolution And Everything: A Survey
- (Video Super Resolution Based on Deep Learning: A comprehensive survey)
### GAN models
- SRGAN
- ESRGAN
- Real-ESRGAN
- BSRGAN
- DPSRGAN
- EdgeSRGAN
- LSMGAN : https://github.com/BehzadBozorgtabar/Retinal_Super_Resolution/tree/master
- PGGAN -> StyleGAN
- ESRGAN+ : https://github.com/ncarraz/ESRGANplus
- Enlighten-GAN : https://github.com/VITA-Group/EnlightenGAN
- TWIST-GAN
- SCSE-GAN
- MFAGAN
- TGAN
- DGAN
- DMGAN
- G-GANISR
- RaGAN
- LE-GAN
- NCSR
- Beby-GAN
- CMRI-CGAN
- D-SRGAN
- LMISR-GAN
- SRDGAN
- GLEAN
- I-SRGAN
- RankSRGAN
- GMGAN
- FSLSR
- I-WAGAN
- CESR-GAN
- RTSRGAN
- MSSRGAN
- RSISRGAN
- JPLSRGAN
- SRR-GAN
- MRD-GAN
- MESRGAN+
- SNPE-SRGAN
- SOUP-GAN

- SWAGAN(super-resolution용은 아님)
- Lightweight GAN https://github.com/lucidrains/lightweight-gan/tree/main?tab=readme-ov-file
### 기타 models
- NAFNet
- Wavelet Transform
- DIP, Deep Image Prior
- Autoencoder
- Wavelet Transform
- DBPN
- MIRnet
- MPRNet
- FSRCNN
- SRDenseNet
- EDSR

#### I want to find Lightweight models
Useful paper : Lightweight image super-resolution based on deep learning: State-of-the-art and future directions
- Light ESPCN
- DPSRGAN
- EdgeSRGAN
- FSRCNN
- VDSR
- DRCN
- GhostSR
- CARN
- CBPN
- SRRFN
- MFIN
- WMRN
- ELCRN

### Diffusion models
- DPM, DDPM
- DDIM
- SR3
- Latent Diffusion
- Stable Diffusion

# Papers
- Awesome-CT-Reconstruction https://github.com/LoraLinH/Awesome-CT-Reconstruction?tab=readme-ov-file
- Awesome Diffusion Models in Medical Imaging https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging?tab=readme-ov-file#reconstruction
- Awesome Diffusion models for image processing https://github.com/lixinustc/awesome-diffusion-model-for-image-processing
- Awesome articles about Implicit Neural Representation networks in medical imaging https://github.com/xmindflow/Awesome-Implicit-Neural-Representations-in-Medical-imaging?tab=readme-ov-file#tomography-and-ct
- Awesome-Diffusion-Models : https://github.com/diff-usion/Awesome-Diffusion-Models
- Awesome-Super-Resolution https://github.com/ChaofWang/Awesome-Super-Resolution?tab=readme-ov-file
- Single-Image-Super-Resolution(7 years ago) : https://github.com/YapengTian/Single-Image-Super-Resolution?tab=readme-ov-file

## Paper with name ( * ~ ***** : 성능)
- Medical image denoising using convolutional denoising autoencoders(Autoencoder) https://paperswithcode.com/paper/medical-image-denoising-using-convolutional
- Content-Noise Complementary Learning for Medical Image Denoising(CNCL) https://github.com/gengmufeng/CNCL-denoising/tree/main, https://github.com/kiananvari/Content-Noise-Complementary-Learning-for-Medical-Image-Denoising?tab=readme-ov-file
- Uncertainty Estimation in Medical Image Denoising with Bayesian Deep Image Prior(DIP) https://github.com/mlaves/uncertainty-deep-image-prior?tab=readme-ov-file
- Learning Medical Image Denoising with Deep Dynamic Residual Attention Network https://github.com/sharif-apu/MID-DRAN
- Transformers in Medical Imaging: A Survey(Transformers) https://github.com/fahadshamshad/awesome-transformers-in-medical-imaging?tab=readme-ov-file#reconstruction  
- Adversarial Distortion Learning for Medical Image Denoising(수정필) https://github.com/mogvision/ADL/tree/main
- Physics-informed deep neural network for image denoising(PINN) https://codeocean.com/capsule/9043085/tree/v1
- Simple Baselines for Image Restoration(NAFNet) https://github.com/megvii-research/NAFNet
- Multi-stage image denoising with the wavelet transform(Wavelet Transform)  https://github.com/hellloxiaotian/MWDCNN
- CT Image Denoising with Perceptive Deep Neural Networks
- Dynamic Convolution: Attention over Convolution Kernels
- Recent Advances in CT Image Reconstruction
- The evolution of image reconstruction for CT—from filtered back projection to artificial intelligence
- EDCNN: Edge enhancement-based Densely Connected Network with Compound Loss for Low-Dose CT Denoising(EDCNN) https://github.com/workingcoder/EDCNN
- X2CT-GAN: Reconstructing CT from Biplanar X-Rays with Generative Adversarial Networks(X2CT-GAN) https://github.com/kylekma/X2CT  
- Efficient Face Super-Resolution via Wavelet-based Feature Enhancement Network(Wavelet) : https://github.com/pris-cv/wfen?tab=readme-ov-file
- CAMixerSR: Only Details Need More “Attention”(CAMixerSR) : https://github.com/icandle/CAMixerSR/tree/main?tab=readme-ov-file
- DBPN : Deep Back-Projection Networks for Single Image Super-resolution(DBPN) : https://github.com/Lornatang/DBPN-PyTorch?tab=readme-ov-file
- Image Denoising Using a Generative Adversarial Network(SRGAN) : https://github.com/manumathewthomas/ImageDenoisingGAN/tree/master?tab=readme-ov-file
- Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data(Real-ESRGAN) : https://github.com/xinntao/Real-ESRGAN
- Learning Enriched Features for Fast Image Restoration and Enhancement(MIRnet) : https://github.com/swz30/MIRNetv2/tree/main
- Efficient Face Super-Resolution via Wavelet-based Feature Enhancement Network : https://github.com/PRIS-CV/WFEN/tree/main?tab=readme-ov-file
- FreeSeed: Frequency-band-aware and Self-guided Network for Sparse-view CT Reconstruction(FreeSeed) : https://github.com/masaaki-75/freeseed
- Image Denoising Using a Generative Adversarial Network(ImageDenoisingGAN) : https://github.com/manumathewthomas/ImageDenoisingGAN/tree/master?tab=readme-ov-file
- SwinIR: Image Restoration Using Swin Transformer(SwinIR) : https://github.com/JingyunLiang/SwinIR/tree/main
- Lightweight Image Super-Resolution with Enhanced CNN(LESRCNN) : https://github.com/hellloxiaotian/LESRCNN/tree/master?tab=readme-ov-file
- Denoising Diffusion Probabilistic Models(DDPM) : https://github.com/lucidrains/denoising-diffusion-pytorch?tab=readme-ov-file
- Deep Unsupervised Learning using Nonequilibrium Thermodynamics(DPM) https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models
- Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising(DnCNN) https://github.com/anushkayadav/Denoising_cifar10
- Deep Image Prior https://github.com/DmitryUlyanov/deep-image-prior/tree/master
- DPSRGAN: Dilation Patch Super-Resolution Generative Adversarial Networks https://github.com/kushalchordiya216/DPSRGAN/tree/master?tab=readme-ov-file
- Generative Adversarial Super-Resolution at the Edge with Knowledge Distillation(EdgeSRGAN) : https://github.com/PIC4SeR/EdgeSRGAN?tab=readme-ov-file

## Image reconstruction with Super-resolution
- X-ray and CT image processing using machine learning and deep learning https://github.com/YIZHE12/ML_DeepCT?tab=readme-ov-file
- pix2pix_super_resolution(pix2pix) https://www.kaggle.com/code/chaimaemoumou/pix2pix-super-resolution/notebook
- Computed-tomography Fan-beam FBP reconstruction https://github.com/kk17m/CT-Fan-beam-FBP-reconstruction/tree/master
- Single Image Super-Resolution(EDSR, WDSR, SRGAN) : https://github.com/krasserm/super-resolution/tree/master?tab=readme-ov-file
- Image-Super-Resolution(SRCNN) : https://github.com/sharmaroshan/Image-Super-Resolution/tree/master
- Image Super Resolution Using Deep Convolution Neural Network Computer Science Graduation Project : https://github.com/AmrShaaban99/super-resolution/tree/main?tab=readme-ov-file#abstract
- BCRN：A Very Lightweight Network for single Image Super-Resolution(BCRN) https://github.com/kptx666/BCRN/tree/main
- Image Reconstructor that applies fast proximal gradient method (FISTA) to the wavelet transform of an image using L1 and Total Variation (TV) regularizations : https://github.com/EliaFantini/Image-Reconstructor-FISTA-proximal-method-on-wavelets-transform/tree/main
- 'Lightweight' GAN : https://github.com/lucidrains/lightweight-gan
- BEDMAP2 using a super resolution deep neural network : https://github.com/weiji14/deepbedmap/tree/master?tab=readme-ov-file
- Deep Learning Image Generation with GANs and Diffusion Model : https://github.com/kanru-wang/Udemy_GAN_and_Diffusion_models/tree/main
- Training and testing codes for USRNet, DnCNN, FFDNet, SRMD, DPSR, MSRResNet, ESRGAN, BSRGAN, SwinIR, VRT, RVRT : https://github.com/cszn/KAIR/tree/master
- https://github.com/hellloxiaotian/Deep-Learning-on-Image-Denoising-An-overview
- Annotated Research Paper Implementations: Transformers, StyleGAN, Stable Diffusion, DDPM/DDIM, LayerNorm, Nucleus Sampling and more https://nn.labml.ai/index.html

- AI 양재 허브 인공지능 오픈소스 경진대회 : https://dacon.io/en/competitions/official/235977/codeshare?page=1&dtype=view&ptype=pub&keyword
- 카메라 이미지 품질 향상 AI 경진대회 : https://dacon.io/en/competitions/official/235746/codeshare?page=1&dtype=view&ptype=pub&keyword

## Image Denoising
- Awesome Image or Video Denoising Algorithms https://github.com/z-bingo/awesome-image-denoising-state-of-the-art
- Image Denoising State-of-the-art https://github.com/flyywh/Image-Denoising-State-of-the-art
- reproducible-image-denoising-state-of-the-art https://github.com/wenbihan/reproducible-image-denoising-state-of-the-art?tab=readme-ov-file
- MR-self Noise2Noise: self-supervised deep learning–based image quality improvement of submillimeter resolution 3D MR images https://link.springer.com/article/10.1007/s00330-022-09243-y
- Multi-Stage Progressive Image Restoration https://arxiv.org/abs/2102.02808 https://github.com/swz30/MPRNet
- CycleISP: Real Image Restoration via Improved Data Synthesis https://arxiv.org/abs/2003.07761 https://github.com/swz30/CycleISP
- Image Denoising https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html
- Learning Medical Image Denoising with Deep Dynamic Residual Attention Network https://github.com/sharif-apu/MID-DRAN
- SwinIA: Self-Supervised Blind-Spot Image Denoising with Zero Convolutions https://arxiv.org/abs/2305.05651
- MR Image Denoising and Super-Resolution Using Regularized Reverse Diffusion https://arxiv.org/abs/2203.12621
- Image Denoising with Autoencoder (as a baseline model), CBDNet, PRIDNet, RIDNet https://github.com/sharathsolomon/ImageDenoising/tree/main?tab=readme-ov-file

## Sparse View CT
- Scientific Computational Imaging Code (SCICO) https://github.com/lanl/scico
- 2-Step Sparse-View CT Reconstruction with a Domain-Specific Perceptual Network https://github.com/anonyr7/Sinogram-Inpainting/tree/master?tab=readme-ov-file
- Improving Hemorrhage Detection in Sparse-view CTs via Deep Learning https://github.com/J-3TO/Sparse-View-Cranial-CT-Reconstruction
- Low-Dose X-Ray Ct Reconstruction on X3D https://paperswithcode.com/sota/low-dose-x-ray-ct-reconstruction-on-x3d
- WalnutReconstructionCodes (FDK) : https://github.com/cicwi/WalnutReconstructionCodes?tab=readme-ov-file

# Main Project(연구과제)
  1.  Lower-Dose reduction Pediatric CT reconstruction : 적은 방사선량으로 데이터 추출하여 최적화된 CT 이미지 재건축
  2.  Metal artifact reduction techniques : CT 이미지의 결함 문제 : 배터리 노화, 인체 속 다른 물질 출력으로 인한 이미지 결함 해결
  3.  motion-induced blur in cone-beam computed tomography(CB-CT) : CB-CT 를 사용했을 때 먼 거리에서 뿌연 현상(blur) 발생 문제
  4.  segmentation task of oral CBCT image with Deep Learning : 딥러닝을 이용하여 이미지의 인체기관 위치 식별
  5.  3D image stitching with CB-CT : CB-CT 를 이용하여 추출한 3D 이미지 연결(스티칭)

# Lecture   
- Computer Tomography, Matthias Beckmann  
- Timothy G. Feeman, The Mathematics of Medical Imaging, Second Edition
- 의학영상기기 https://www.youtube.com/playlist?list=PLSN_PltQeOyj-XhgiWmtqjGo08isXUlSy


# 3D reconstruction

## 3D image stitching
- 🧩Image Stitching 프로젝트 https://velog.io/@davkim1030/Image-Stitching 

## 3D Reconstruction
- Awesome 3D Reconstruction Papers https://github.com/bluestyle97/awesome-3d-reconstruction-papers
- 3D-Reconstruction-with-Deep-Learning-Methods https://github.com/natowi/3D-Reconstruction-with-Deep-Learning-Methods
- 3D Machine Learning Study notes https://github.com/timzhang642/3D-Machine-Learning
- 3D Reconstruction & visualistaion https://www.kaggle.com/code/aatamikorpi/3d-reconstruction-visualistaion/notebook  
- Pulmonary Dicom Preprocessing https://www.kaggle.com/code/allunia/pulmonary-dicom-preprocessing  
- Advanced DICOM-CT 3D visualizations with VTK https://www.kaggle.com/code/wrrosa/advanced-dicom-ct-3d-visualizations-with-vtk  
- Covid19 segmentation and 3D reconstruction https://www.kaggle.com/code/qiyuange/covid19-segmentation-and-3d-reconstruction  
- OpenSfM https://github.com/Unity-Technologies/ind-bermuda-opensfm  
- NeRF모델을 이용해 뽑은 3D Model https://github.com/ProtossDragoon/PlankHyundong  
- Benchmark for visual localization on 3D mesh models https://github.com/v-pnk/cadloc?tab=readme-ov-file
- Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network https://github.com/yfeng95/PRNet
- 3D reconstruction of a mouse CT https://github.com/jvirico/mouse_CT_3D_reconstruction/tree/master   
- TripoSR : a state-of-the-art open-source model for fast feedforward 3D reconstruction from a single image https://github.com/vast-ai-research/triposr?tab=readme-ov-file
- 2차원 이미지에서 3차원 모델 자동 생성하는 SFM기반 OpenMVG https://daddynkidsmakers.blogspot.com/2019/11/2-3-sfm-openmvg.html https://github.com/openMVG/openMVG/tree/develop
- RSNA AI Deep Learning Lab 2019 https://github.com/mohannadhussain/AI-Deep-Learning-Lab/tree/master

## Physically Based Rendering method
- Deep Learning with Cinematic Rendering: Fine-Tuning Deep Neural Networks Using Photorealistic Medical Images https://durr.jhu.edu/pubs/dl-cin-ren/

# Studies 
- Notebooks, datasets, other content for the Radiology:AI series known as Magicians Corner by Brad Erickson https://github.com/RSNA/MagiciansCorner/tree/master?tab=readme-ov-file
- 
- Lower-Dise
- 
- CB-CT
- 
- Computer Tomography  
https://tristanvanleeuwen.github.io/IP_and_Im_Lectures/tomography.html  
https://ok97465.github.io/2019/10/191019_PrincipleOfCT
https://circlezoo.tistory.com/83  
https://slideplayer.com/slide/4044214/  
https://slideplayer.com/slide/4881743/  
https://lme.tf.fau.de/category/lecture-notes/lecture-notes-me/
https://unist.tistory.com/5

- Radon Transform  
https://ghebook.blogspot.com/2020/11/radon-transform.html  
https://ebrary.net/207759/engineering/radon_transform_approach_solution_elastodynamic_greens_function  
Object Recognition Using Radon Transform-Based RST Parameter Estimation https://www.researchgate.net/publication/279770487_Object_Recognition_Using_Radon_Transform-Based_RST_Parameter_Estimation  
Translation Invariant Global Estimation of Heading Angle Using Sinogram of LiDAR Point Cloud    https://www.researchgate.net/publication/358975239_Translation_Invariant_Global_Estimation_of_Heading_Angle_Using_Sinogram_of_LiDAR_Point_Cloud  
https://blog.naver.com/yunuri1012/80094694153  
https://blog.naver.com/gwl0711/222187454928

- Sinogram
https://blog.naver.com/seungwan777/100210900086

- Hough transform  
https://en.wikipedia.org/wiki/Hough_transform

- Fourier transform  
https://www.theoretical-physics.com/dev/math/transforms.html  
https://en.wikipedia.org/wiki/Fourier_transform

- Dirac-Delta Function
https://en.wikipedia.org/wiki/Dirac_delta_function

- Interpolation
https://blog.naver.com/PostView.nhn?blogId=libeor06&logNo=221966447062

- Machine learning
https://kimbg.tistory.com/category/machine%20learning

- Artifact
https://m.blog.naver.com/na1se/80176910730
