---
layout: post
title: MCL_Day 2,3
subtitle: Backpropagation, Optimizer, Regression, DataLoader
gh-repo: daattali/beautiful-jekyll
gh-badge: 
tags: [MCL_Internship]
comments: true
---

 MCL_Internship 2일차와 3일차 내용들을 간략하게 정리한 글들이다. (사진의 출처들은 모두 [MCL](https://mcl.korea.ac.kr/) 의 자료들이다)

## 역전파(Backpropagation)

   역전파를 설명하기 이전에 **Machine Learning**을 조금 더 살펴보자.
   **Machine Learning**은 인체의 뉴런 구조를 모방했다고 많이들 말한다. 우선 입력을 _x1_, _x2_, _x3_, ... ,_xn_ 총 n개를 생각해보자. 해당 입력들이 어떤 함수를 거쳐 하나의 출력을 만들어내는 구조인데 각 입력들에게 가중치(weight)가 붙게 된다. 즉, _x1_ 에 w1 이라는 가중치가 붙게 되면 함수 내에서는 _x1_ X w1 = _X1_ 가 된다. n개의 입력에 대해 모두 가중치가 각각 다르게 붙고(동일하게 붙을 수도 있다!) 이를 모두 더한 값이 출력으로 향하게 된다. 하지만 여기서 편향치(bias) _b_ 라는 상수를 마지막에 더해줘야 원하는 출력이 되도록 만들 수 있다.  
 
 <img src="/assets/img/MCL/MCL_day2_ANN.png" width="50%" height="50%"> 
 
   사진과 같이 n 개의 입력 layer가 하나의 출력 layer로 향하는 구조이지만 사실 2 개의 layer 사이에 hidden layer들이 존재한다. 이 hidden layer 들이 가중치 값들이라고 생각하면 된다. 이런 결과를 내는 함수를 활성화 함수(Activation Function)이라고 하는데 DL에서는 데이터를 **비선형** 으로 바꾸기 위해 비선형 함수를 사용한다고 한다. 
   
   예를 들어 선형 활성화 함수 h(x) = ax (a는 상수)를 가정해보자. 정말 간단하게 한 개의 hidden layer가 있다고 생각하면 출력 y(x) = h(h(x)) 가 되는데 이는 y(x) = bx (b도 상수) 형태로 선형 결과를 내게 된다. 이런 경우 층을 쌓는 혜택이 없어지게 되므로 비선형 함수를 사용한다. 대표적으로 Sigmoid 함수 (결과값: 0 ~ 1 사이), Tanh 함수 (결과값: -1 ~ 1 사이) 두 개는 출력 범위 제한용으로 마지막 layer 쯤에 사용되고 hidden layer들에는 ReLU(Rectified Linear Unit) 함수를 사용하게 된다.
 > 보통 network 층의 개수는 hidden + output layer 의 개수
 
 **역전파(Backpropagation)** 은 예측 값과 정답과의 차이를 역방향으로 전파시키면서 최적의 가중치를 찾아가는 방법이다. 즉, 가중치와 bias 값을 조정하는 것이 학습 과정이 되는 방식이다. 공부도 그렇듯이 목표를 명확히 한 후에 그 목표를 향해 공부 방법을 찾아야 한다. 역전파 역시 목표(label)을 향해 학습해야 하는데 이것이 잘 되고 있는지 확인하는 방법은 Loss function 이다.
 
 **Loss function**는 정답(label)과 예측 값의 차에 대한 수치를 구하는 함수이다. 가장 대표적으로 MSE(Mean Square Error)가 있으며 이 외에도 MAE(Mean Absolute Error), CE(Cross Entropy) 등이 있다. 해당 함수를 최소값이 되도록 하는게 기계학습의 목표가 된다! 그렇다면 잘 되고 있는지 확인하는 지표가 Loss function이라면 실제로 학습을 조정하기 위해 가중치 값들을 조절하는 것을 Optimization이라고 한다. 그리고 그 방법들 중 Gradient-descent를 간단하게 알아보자.
 
 **Gradient-descent**는 미분하였을 때 기울기가 가장 큰 방향으로 움직이도록 가중치를 조절하는 방법이다.
 
  <img src="/assets/img/MCL/MCL_day2_GD.png" width="50%" height="50%"> 
  
  <img src="/assets/img/MCL/MCL_day2_GD_calc.png" > 

 수식은 위와 같은데 이 때 alpha 값이 Learning rate가 된다. 해당 값이 클수록 가중치가 크게 바뀌지만 반대로 너무 크게 바뀌어서 최소값을 찾는데 더 어려움을 겪을 수도 있다(Overfitting). 반대로 너무 작으면 학습이 너무 천천히 되므로(Underfitting) 적당한 값을 찾는 것이 중요하다고 한다. (_그래도 0.01 ~ 0.001 정도 사용하신다고 한다_) 
 
 <img src="/assets/img/MCL/MCL_day2_Overfit.png" width="50%" height="50%"> 
 
 Pytorch 에서는 이미 구현되어 있으므로 다음 코드들을 응용하면 된다.

 ~~~
torch.optim.SGD(params, learning rate, ...)
~~~

 그리고 이는 또 다른 Optimization 함수이다.
 ~~~
torch.optim.Adam(params, learning rate, ...)
~~~

## 프로젝트 진행에 있어...

<img src="/assets/img/MCL/MCL_day2_project.png" width="50%" height="50%"> 

 보통 프로젝트는 위와 같은 순서로 진행된다고 한다. ML Model을 만들면 Train과 Test를 모두 거쳐야 한다. 이를 위해 예시 Dataset을 준비하고 80%는 Training에, 20%를 Test용으로 사용한다고 한다. 이를 분리하지 않으면 일반적인 상황에 대한 대처를 확인하기 어렵다. 마치 교과서 위주로 공부하였는데 교과서 예시들로 시험을 내면 크게 분별력이 없는 것처럼 말이다.
 
 대부분의 영상들은 FHD인데 보통 1920(가로) X 1080(세로) pixels로 이루어져 있다. 하지만 이 때 한 픽셀을 구현할 때 3가지 색상 RGB들을 조합해야 하는데 이로써 실제 영상 1080p 는 1920(가로) X 1080(세로) X 3 (channels) 로 구성된다. 이를 계산하면 약 622만으로 1 프레임이 이정도니까 초당 30프레임 10초 짜리 영상은 18.6억 개의 값들을 처리해야 한다. 이를 한 번에 처리하는 것은 어려우므로 앞서 언급한 batch 단위로 나누어서 학습을 진행하게 된다. 데이터 형태는 (batch_size)x(channel)x(height)x(width) 가 되고 Pytorch에서는 다음과 같은 함수를 제공하고 있다.

~~~
torch.utils.data.DataLoader(dataset, batch_size, shuffle, ...)
~~~

_(shuffle = True 면 랜덤으로 섞어서 들어오므로 Training 할 때 좋으나 Test 할 때는 필요없으므로 False로 둔다)_
