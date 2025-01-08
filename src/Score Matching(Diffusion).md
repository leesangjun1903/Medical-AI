SM 기법은 기계학습 모델이 데이터 분포의 Score 함수랑 매칭하도록 학습하는 목적식을 제안한다[14].  
이 기법은 본래 정규화(normalization)가 어려운 통계 모형의 패러미터를 추정할 때 사용하는 기법으로, MCMC(Markov Chain Monte Carlo) 샘플링 기법이 가진 단점을 보완하기 위해 고안되었다.

Score function은 log-likelihood의 gradient를 의미하고, score matching은 실제 probability density function을 구하는 것이 아닌, 이러한 score 값을 활용하여 probability density function을 추정하는 것을 의미한다.  
이러한 score matching 기법은 최근 SDE를 활용하여 diffusion model을 모델링할 때에 널리 활용된다. 

# Reference
- https://horizon.kias.re.kr/25133/
- https://process-mining.tistory.com/211
