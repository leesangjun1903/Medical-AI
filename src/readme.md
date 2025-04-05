## LDM VQ-GAN 과 유사한 구조의 Quantilization layer 의미

VQ-reg: decoder에 vector quantization layer를 사용하여 vector quantization layer가 decoder 내부에 위치하는 VQGAN의 형태.

VQ-VAE 처럼 Encoder에서 나온 vector 값과 codebook 간의 유클리디안 distance를 비교한 후 distance가 가장 작은 vector 들의 값으로 quantized vector $\(z_q\)$를 구성한다.

이렇게 구성한 $\( z_q \)$를 decoder에 넣어 reconstruction image를 생성합니다. 그리고 이를 discriminator에 넣어 patch 단위로 real 인지 fake 인지 판단한다.

다음으로 VQ loss는 codebook만 update 하는 loss로 codebook vector가 encoder의 출력과 비슷하게 만들도록 하는 목적을 가진다. 여기서 \( sg \)는 stop gradient 라는 표기로 encoder \( z_{e}(x) \)를 update하지 않는다. 

마지막으로 commitment loss이다. commitment loss 는 Encoder만 update 하는 loss로 Encoder의 출력이 codebook vector와 가까운 값을 출력하는 것이 목적인 loss이다.

https://dlaiml.tistory.com/entry/LDM-High-Resolution-Image-Synthesis-with-Latent-Diffusion-Models
https://bigdata-analyst.tistory.com/m/349  
https://github.com/CompVis/latent-diffusion/issues/218
https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf  
https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py  
https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/models/autoencoder.py#L8
