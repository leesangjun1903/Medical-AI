## LDM VQ-GAN 과 유사한 구조의 Quantilization layer 의미

VQ-VAE 처럼 Encoder에서 나온 vector 값과 codebook 간의 유클리디안 distance를 비교한 후 distance가 가장 작은 vector 들의 값으로 quantized vector $\(z_q\)$를 구성한다.

이렇게 구성한 $\( z_q \)$를 decoder에 넣어 reconstruction image를 생성합니다. 그리고 이를 discriminator에 넣어 patch 단위로 real 인지 fake 인지 판단한다.



https://bigdata-analyst.tistory.com/m/349  
https://github.com/CompVis/latent-diffusion/issues/218
