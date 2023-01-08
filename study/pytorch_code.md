## Table of Contents

- [view 와 reshape의 차이 (cf. flatten,contiguous,clone)](#1)
- []()

---

## #1

### view 와 reshape의 차이 (cf. flatten,contiguous,clone)
- 우선적으로 contiguous 개념을 알아야함
    - contiguous : contiguous란 matrix의 순차적인 shape information과 실제 matrix의 각 데이터가 저장된 위치가 같은지 여부를 말함
    - contigous가 False가 되는 가장 대표적인 예시로 transpose 시킬 경우 contiguous가 False가 됨
    - 만약 RuntimeError : input is not contiguous 하다고 나올경우 input = input.contiguous()를 통해 변경후 사용가능
    - 코드
        ```python
        a = torch.randn(4,3,2)
        print(a.is_contiguous())
        print(a.size())
        '''
        True
        torch.Size([4, 3, 2])
        '''
        ```
        ```python
        b = a.transpose(1,2)
        print(b.is_contiguous())
        print(b.size())
        '''
        False
        torch.Size([4, 2, 3])
        '''
        ```
- view vs reshape(flatten)
    - flatten은 reshape 기반으로 작성된것이므로 reshape으로 코드를 사용해도 됨
    - view는 contiguous하지 못한 데이터에는 적용 불가(.contiguous().view() 를 통해 강제적으로 적용가능)
    - reshape은 contiguous 한 데이터는 view와 똑같은 기능을 수행하지만 view와는 다르게 contiguous하지 못한 데이터가 들어오더라도 그 데이터를 알아서 contiguous하게 변경후 view 시켜줌
    - 여기서 주의할 점이 view나 contiguous한 데이터에 reshape 적용하는경우 데이터를 공유하게 됨 ( 하지만 view에 contiguous를 적용해주거나, contiguous하지 않은 데이터에 reshape을 적용한 경우는 데이터를 공유하지 않게 됨 )
        - reshape과 view이 데이터를 공유하는 경우(contigous()가 적용되지 않은 경우)
            ```python
            z = torch.zeros(3,2)
            y = z.reshape(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([1., 1., 1., 1., 1., 1.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```
            ```python
            z = torch.zeros(3,2)
            y = z.reshape(6)
            y.fill_(1)
            print(y)
            print(z)
            '''
            tensor([1., 1., 1., 1., 1., 1.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```
            ```python
            z = torch.zeros(3,2)
            y = z.view(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([1., 1., 1., 1., 1., 1.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```
            ```python
            z = torch.zeros(3,2)
            y = z.view(6)
            y.fill_(1)
            print(y)
            print(z)
            '''
            tensor([1., 1., 1., 1., 1., 1.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```
        - 데이터를 공유하지 않는 경우 (contiguous()가 적용된 경우, 단, 그전 데이터가 contiguous하지 않아야지만 contiguous가 적용이 됨)
            ```python
            z = torch.zeros(3,2)
            y = z.contiguous().view(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([1., 1., 1., 1., 1., 1.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```
            (contiguous한 데이터에 contiguous를 적용하면 아무런 변화가 없음(즉 데이터를 공유함))
            ```python
            z = torch.zeros(3,2).transpose(0,1)
            y = z.contiguous().view(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([0., 0., 0., 0., 0., 0.])
            tensor([[1., 1., 1.],
                    [1., 1., 1.]])
            '''
            ```
            (contiguous 하지 않은 데이터에 contiguous를 적용하니까 데이터를 공유하지 않는것을 확인할 수 있음)
            ```python
            z = torch.zeros(3,2)
            y = z.reshape(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([1., 1., 1., 1., 1., 1.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```
            (contiguous한 데이터에 reshape이 적용되었으므로 contiguous()가 적용되지 않음 따라서 데이터를 공유하게 됨)
            ```python
            z = torch.zeros(3,2).transpose(0,1)
            y = z.reshape(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([0., 0., 0., 0., 0., 0.])
            tensor([[1., 1., 1.],
                    [1., 1., 1.]])
            '''
            ```
            (contiguous하지 않은 데이터에 reshape이 적용되면 contiguous()가 자동으로 적용되어서 데이터를 공유하지 않게 됨)
        - 참고 : contiguous 한지 고려하지 않고 무조건으로 데이터를 공유하지 않고 싶다면 clone을 통해 복사하여 사용하면됨
            ```python
            z = torch.zeros(3,2)
            y = z.clone().reshape(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([0., 0., 0., 0., 0., 0.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```
            ```python
            z = torch.zeros(3,2)
            y = z.clone().view(6)
            z.fill_(1)
            print(y)
            print(z)
            '''
            tensor([0., 0., 0., 0., 0., 0.])
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]])
            '''
            ```

    - 결론 : .contiguous().view() 와 .reshape() 완전히 동일한 기능을 수행하게 됨. reshape()이 더 강력한 기능을 하기 때문에 reshape을 사용하는 것을 추천
    
#### References
- https://jimmy-ai.tistory.com/122

---

## #2

### weight 초기화
- 가중치 초기화 이론
    - 가중치 초기값이 0이거나 동일한 경우
        - 가중치의 초기값을 모두 0으로 초기화하거나 동일한 값으로 초기화할 경우 모든 뉴런의 동일한 출력값을 내보낼것임. 이럴 경우 역전파에서 각 뉴런이 모두 동일한 그래디언트 값을 가지게 됨(이럴 경우 뉴런의 개수가 아무리 많아도 뉴런이 하나뿐인 것처럼 작동하기 때문에 학습이 제대로 이루어지지 않음)
        - 따라서 가중치 초기값을 동일한 값으로 초기화 해서는 안됨.
    - 작은 난수
        - 가중치 초기값은 작은 값으로 초기화 해야하는데, 그 이유는 활성화 함수가 sigmoid일 경우 가중치 초기값을 절대값이 큰 값으로 한다면 0과 1로 수렴하기 때문에 그래디언트 소실이 발생함. 또한 ReLu일 경우도 절대값이 클 경우 음수일 때는 dead ReLU 문제가 발생하고, 양수일 때는 그래디언트가 폭주하게 됨.
        - 따라서 가중치 초기값을 작게 초기화 해야하며 동일한 초기값을 가지지 않도록 랜덤하게 초기화 해야함
        - 일반적으로 가중치 초기값은 평균이 0이고 표준편차가 0.01인 정규 분포를 따르는 값으로 랜덤하게 초기화함
        - 하지만 이러한 가중치 초기화 방법은 얕은 신경망에서만 괜찮게 작동할지 모르지만, 신경망의 깊이가 깊어질수록 문제가 발생하게 됨. 
            - 예를 들어 평균이 0 이고 표준편차가 0.01 정규분포를 따르는 값으로 랜덤하게 초기화하고 tanh를 활성화 함수로 사용하게 되면 첫번째 hidden layer를 제외한 나머지 레이어들이 모두 0을 출력하게 됨(학습이 이루어지지 않게 됨)
                ![](./img/weight_init.jpg)
            - 또한 평균이 0 이고 표준편차가 1인 정규 분포를 따르는 값으로 랜덤하게 초기화하고 tanh를 활성화 함수로 사용하였을 경우 아래의 그림처럼 tanh의 출력이 -1과 1로 집중되면서 그래디언트 소실 문제가 발생함.
                ![](./img/weight_init1.jpg)
    - Xavier 초기화
        - 입출력 레이어의 개수 고려
            ![](./img/weight_init2.jpg)
        - tanh 활성화 함수에 xavier 초기값 설정
            ![](./img/weight_init3.jpg)
        - ReLU 활성화 함수에 xavier 초기값 설정 -> 레이어가 깊어질수록 출력값이 0으로 치우치는 문제 발생
            ![](./img/weight_init4.jpg)
    - He 초기화
        - Xavier의 ReLU 활성화 함수에서의 문제를 해결한것
            ![](./img/weight_init5.jpg)
            ![](./img/weight_init6.jpg)
    - 결론
        - 활성화 함수로는 ReLU를 먼저 사용하는 것이 좋음
        - 가중치 초기화는 Sigmoid일 경우 Xavier, ReLU일 경우 He 초기값을 사용하는 것이 좋음
- 다양한 초기화 방법
    - `torch.nn.init.constant_(tensor, val)` : 상수로 설정
    - `torch.nn.init.unifiom_(tensor, a=0.0, b=1.0)` : a부터 b사이의 값을 균일한 분포로 설정. 디폴트 설정은 a=0.0, b=1.0
    - `torch.nn.init.normal_(tensor, mean=0.0, std=1.0)` : 평균이 0이고 표준편차가 1인 분포로 설정
    - Xavier(= Glorot initialization)
        - 기존의 무작위 수로 초기화와 다르게 layer의 특성에 맞춰서 초기화 하는 방법
        - `torch.nn.init.xavier_uniform_(tensor, gain=1.0)`
        - `torch.nn.init.xavier_normal_(tensor, gain=1.0)`
    - Kaiming(= He-initialization)
        - relu나 leaky_relu를 activation function으로 사용하는 경우 많이 사용함
        - `torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`
        - `torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`
- 일반적으로 아래 방식 사용
    ```python
    # 모든 neural network module, nn.Linear, nn.Conv2d, BatchNorm, Loss function 등.
    import torch.nn as nn 
    # 파라미터가 없는 함수들 모음
    import torch.nn.functional as F 

    class CNN(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=6,
                    kernel_size=(3,3),
                    stride=(1,1),
                    padding=(1,1)
            )
            self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
            self.conv2 = nn.Conv2d(
                    in_channels=6,
                    out_channels=16,
                    kernel_size=(3,3),
                    stride=(1,1),
                    padding=(1,1)
            )
            self.fc1 = nn.Linear(16*7*7, num_classes)
            # 예제의 핵심인 initialize_weights()로 __init__()이 호출될 때 실행됩니다.
            self.initialize_weights()

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc1(x)
            
            return x
        
        # 각 지정된 연산(Conv2d, BatchNorm2d, Linear)에 따라 각기 다른 초기화를 줄 수 있습니다.
        def initialize_weights(self):
            for m in self.modules():
                # convolution kernel의 weight를 He initialization을 적용한다.
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight)
                    
                    # bias는 상수 0으로 초기화 한다.
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

            
    if __name__ == '__main__':
        model = CNN(in_channels=3,num_classes=10)
        
        # He initialization과 Constant로 초기화 한것을 확인할 수 있습니다.
        for param in model.parameters():
            print(param)
    ```
- 간단한 예시 코드
    ```python
    class Custom_Net(nn.Module):
        def __init__(self):
            super(Custom_Net, self).__init__()

            self.linear_1 = nn.Linear(1024, 1024, bias=False)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight,mean=0,std=1)

    model = Custom_Net()

    for param in model.parameters():
        print(param)
        print(param.mean())
        print(param.std())
    '''
    Parameter containing:
    tensor([[ 0.0336, -0.3054, -0.5625,  ..., -0.8603,  0.3355, -0.1826],
            [ 0.3902, -0.1708, -0.2713,  ..., -0.5215,  0.7843, -1.2075],
            [-0.5171,  0.6536,  0.7110,  ...,  0.1070,  0.3790,  0.3462],
            ...,
            [ 0.2072, -0.7820, -0.5950,  ...,  1.6191,  0.2633,  0.2891],
            [-0.2336,  0.9174, -0.7213,  ...,  1.3410,  0.2403,  0.4809],
            [ 0.0212, -0.5089,  0.2311,  ..., -0.1621,  0.3824, -0.5731]],
        requires_grad=True)
    tensor(0.0010, grad_fn=<MeanBackward0>)
    tensor(0.9997, grad_fn=<StdBackward0>)
    '''
    ```
#### References
- https://freshrimpsushi.github.io/posts/weights-initialization-in-pytorch/
- https://gaussian37.github.io/dl-pytorch-snippets/#weight-%EC%B4%88%EA%B8%B0%ED%99%94-%EB%B0%A9%EB%B2%95-1
- https://supermemi.tistory.com/121
- https://excelsior-cjh.tistory.com/177
- https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_