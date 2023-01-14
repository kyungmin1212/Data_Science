## Table of Contents

- [view 와 reshape의 차이 (cf. flatten,contiguous,clone)](#1)
- [가중치 초기화(weight init)](#2)
- [Batch Norm과 Layer Norm](#3)
- [pad_sequence(길이가 다른 데이터를 하나의 텐서로 묶어주기(+ collate_fn))](#4)
- [ne,repeat](#5)
- [이미지에 패딩처리해주기 (torch.nn.functional.pad) (+collate_fn)](#6)
- [argmax와 topk](#7)

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

### 가중치 초기화(weight init)
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
            
            # model.modules()-> generator / for 문 사용하면 
            # Custom_Net((linear_1): Linear(in_features=1024, out_features=1024, bias=False))
            # Linear(in_features=1024, out_features=1024, bias=False)
            for m in self.modules(): 
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight,mean=0,std=1) # m.weight.data = m.weight

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

- 다른 대표적으로 사용하는 초기화 방법(이것만 적용해도 어느정도 커버 가능)
    ```python
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1: # m.weight.dim()는 shape의 차원수 즉 torch.size([1024,1024])-> 2,torch.Size([20,100,20,20])-> 4
            nn.init.kaiming_uniform_(m.weight.data)
        if hasattr(m,'bias') and m.bias is not None: # bias가 False일 경우는 None으로 나옴
            nn.init.constant_(m.bias, 0)

    model = Transformer().to(device)

    # model.apply를 하게되면 함수의 인자로 module이 들어가게됨
    model.apply(initialize_weights)
    ```
    ```python
    class Custom_Net(nn.Module):
        def __init__(self):
            super(Custom_Net, self).__init__()

            self.linear_1 = nn.Linear(1024, 1024, bias=False)
            self.cnn = nn.Conv2d(100,20,20)
            
            self.embedding = nn.Embedding(100,20)
    model = Custom_Net().to(device)

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1: # m.weight.dim()는 shape의 차원수 즉 torch.size([1024,1024])-> 2,torch.Size([20,100,20,20])-> 4
            nn.init.kaiming_uniform_(m.weight.data,nonlinearity='leaky_relu')
        if hasattr(m,'bias') and m.bias is not None: # bias가 False일 경우는 None으로 나옴
            nn.init.constant_(m.bias, 0)

    model.apply(initialize_weights)
    ```

- 정리본
    ```python
    def initialize_weights(m):
        # convolution kernel의 weight를 He initialization을 적용한다.
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data,nonlinearity='leaky_relu')

            # bias는 상수 0으로 초기화 한다.
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data,nonlinearity='leaky_relu')
            
            # bias는 상수 0으로 초기화 한다.
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

            
    model = Custom_Net().to(device)
    model.apply(initialize_weights)
    ```
    - layernorm : weight = 1 bias = 0으로 초기화 되어있음 (https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
    - Embedding : 평균0, 표준편차 1로 초기화 되어져있음 (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
    - BatchNorm2d는 초기값이 torch에 정의되어져 있지 않음 (https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
    - Conv2d : He 초기화 되어져 있음
    - Linear : He 초기화 되어져 있음

- transformer에서 가중치 초기화 실험결과
    - 1번
        ```python
        def initialize_weights_base(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
        ```
    - 2번
        ```python
        def initialize_weights(m):
            # convolution kernel의 weight를 He initialization을 적용한다.
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

                # bias는 상수 0으로 초기화 한다.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

                # bias는 상수 0으로 초기화 한다.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        ```
    - 단순히 1번처럼 사용하는것 보다 2번처럼 직접 작성해주는것이 보다 효율적이였음(하지만 정답은 없음 -> 선택적으로 사용)
    - 다른 후보
        ```python
        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        ```
        ```python
        def initialize_weights(m):
            # convolution kernel의 weight를 He initialization을 적용한다.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')

                # bias는 상수 0으로 초기화 한다.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')

                # bias는 상수 0으로 초기화 한다.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        ```
#### References
- https://freshrimpsushi.github.io/posts/weights-initialization-in-pytorch/
- https://gaussian37.github.io/dl-pytorch-snippets/#weight-%EC%B4%88%EA%B8%B0%ED%99%94-%EB%B0%A9%EB%B2%95-1
- https://supermemi.tistory.com/121
- https://excelsior-cjh.tistory.com/177
- https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_

---

## #3

### Batch Norm과 Layer Norm
- 그림을 통한 이해    
    ![](./img/batchnorm_layernorm.jpg)    
- Batch Norm과 Layer Norm의 간단한 설명
    - Batch Norm 
        - 채널별로 배치를 묶어 평균이 0 표준편차를 1로 정규화. 
        - (Batch,Channel,H,W)일경우 Batch,H,W의 평균이 0 표준편차를 1로 만드는 것. 
        - 감마와 베타의 shape은 (Channel)와 같음 (nn.BatchNorm1d 나 2d의 코드를 보게되면 입력인자가 channel만 들어감 즉 감마와 베타를 channel shape을 가지는 벡터만 만드는것).
        - 즉 채널별로 (batch,h,w)의 평균이 0 분산이 1로 만들었는데 이것을 얼마만큼씩 scale,bias를 적용해줄지를 감마와 베타가 정해주는것 (이미지의 경우 채널별로 이미지에 대해 배치사이즈로 묶어준 것들의 평균과 분산을 구해준후 적절하게 채널별로 감마와 베타로 scale, bias 해주는 것)    
        ![](./img/batchnorm_layernorm2.jpg)    
    - Layer Norm 
        - 배치별로 layer를 묶어 평균이 0 표준편차를 1로 정규화.
        - (Batch,Channel,H,W)일경우 Channle,H,W의 평균이 0 표준편차를 1로 만드는 것. 
        - 감마와 베타의 shape은 (Channle,H,W)와 같음 (nn.LayerNorm코드를 보면 입력인자가 여러개의 차원이 들어감. 즉 감마와 벡타도 그 shape에 맞게 생성됨)
        - 즉, 이미지에 대한 경우는 모든 이미지에 대해 각각 이미지를 평균0 표준편차1로 어느정도 안정화된 값으로 만든다음에 모든 이미지의 같은 위치의 모든채널안에서의 픽셀마다(즉, (0,0,0) 이라면 0번째 채널의 0,0의 픽셀 끼리의 이미지의 관계,(1,10,30) 이라면 1번재 채널의 10,30 픽셀에서 모든 이미지의 관계) 감마와 베타를 학습해 그 위치에서는 어떤 scale, bias가 좋은지를 적용시켜줌(이럴 경우 각각의 픽셀위치가 어떤 역할을 하는지 어느정도 의미를 파악가능. 사실 이미지에서는 픽셀위치가 그렇게 유의미한 경우가 없기 때문에 이미지에서는 Batch Norm을 많이 사용)    
            ![](./img/batchnorm_layernorm3.jpg)    
        - NLP에 대한 경우도 마찬가지임 NLP는 보통 (Batch,Length,embedding)인 경우가 많은데 nn.LayerNorm에 embedding 1차원만 들어가는 경우가 대부분임. 이럴 경우는 전체 Length에 대해서 각각의 단어 임베딩차원 값들을 평균 0 표준편차 1로 만들어준다음에 각각의 임베딩 차원에서(즉, embedding이 5차원일 경우 전체 length 문장에서 각 단어마다 1번째 차원끼리의 상관관계, 2번재 차원끼리의 상관관계 , ... 5번재 차원끼리의 상관관계를 감마 베타가 학습하는것) 전체 단어에 대해 각각의 차원의 상관관계를 감마 베타가 구해줌(이럴 경우 각각의 임베딩차원이 어떤 역할을 하는지 어느정도 더 정확하게 설정가능해짐)    
            ![](./img/batchnorm_layernorm4.jpg)     
    - Batch Norm 은 채널에 대한 감마와 베타의 벡터에 대해 파라미터를 학습하지만 , Layer Norm은 감마와 베타의 affine transform를 학습하는 것

- Batch Norm
    - nn.BatchNorm1d 와 nn.BatchNorm2d
        - nn.BatchNorm1d
            - input과 output
                - input : (N,C) or (N,C,L) (N : batch size, C : channels, L : sequence length)
                - output : (N,C) or (N,C,L)
            - 코드
                ```python
                # With Learnable Parameters
                m = nn.BatchNorm1d(100)

                # Without Learnable Parameters
                m = nn.BatchNorm1d(100, affine=False)
                input = torch.randn(20, 100)
                output = m(input)
                print(output.shape)
                '''
                torch.Size([20, 100])
                '''
                ```
        - nn.BatchNorm2d
            - input과 output
                - input : (N,C,H,W)
                - output : (N,C,H,W)
            - 코드
                ```python
                # With Learnable Parameters
                m = nn.BatchNorm2d(100)

                # Without Learnable Parameters
                m = nn.BatchNorm2d(100, affine=False)
                input = torch.randn(20, 100, 35, 45)
                output = m(input)
                print(output.shape)
                '''
                torch.Size([20, 100, 35, 45])
                '''
                ```
- Layer Norm
    - 대표적으로 NLP transformer에 존재
    - 코드를 통한 설명(NLP)
        - nn.LayerNorm 사용
            ```python
            # NLP Example
            batch, sentence_length, embedding_dim = 20, 50, 100
            embedding = torch.randn(batch, sentence_length, embedding_dim)
            layer_norm = nn.LayerNorm(embedding_dim)

            # Activate module
            output_1 = layer_norm(embedding)
            print(output_1.shape)
            '''
            torch.Size([20, 50, 100])
            '''
            ```
        - 직접 구현
            ```python
            class LayerNorm(nn.Module):
                def __init__(self, d_model, eps=1e-12):
                    super(LayerNorm, self).__init__()
                    self.gamma = nn.Parameter(torch.ones(d_model))
                    self.beta = nn.Parameter(torch.zeros(d_model))
                    self.eps = eps

                def forward(self, x):
                    mean = x.mean(-1, keepdim=True)
                    var = x.var(-1, unbiased=False, keepdim=True)
                    # '-1' means last dimension. 

                    out = (x - mean) / torch.sqrt(var + self.eps)
                    out = self.gamma * out + self.beta
                    return out

            batch, sentence_length, embedding_dim = 20, 50, 100
            embedding = torch.randn(batch, sentence_length, embedding_dim)
            layer_norm = LayerNorm(embedding_dim)
            # Activate module
            output_2 = layer_norm(embedding)
            print(output_2.shape)
            '''
            torch.Size([20, 50, 100])
            '''
            ```
        - 감마와 베타는 nn.LayerNorm에서 elementwise_affine이 True일 경우 학습되는 요소임(elementwise_affine의 default는 True)
        - 직접 구현한 코드를 보게 되면 nn.Parameter를 통해 선언해서 감마와 베타를 학습가능한 상수를 선언해줌
    - 코드를 통한 설명(Image)
        - nn.LayerNorm
            ```python
            # Image Example
            N, C, H, W = 20, 5, 10, 10
            input = torch.randn(N, C, H, W)
            # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
            # as shown in the image below
            layer_norm = nn.LayerNorm([C, H, W])
            output = layer_norm(input)
            print(output.shape)
            '''
            torch.Size([20, 5, 10, 10])
            '''
            ```
    - nn.LayerNorm 설명
        - 예를 들어, nn.LayerNorm에 들어가는 shape이 (3,5)(2차원 shape)일 경우, 평균과 표준편차는 input의 마지막 2차원으로 계산이 됨(예를 들면 input.mean((-2,-1))). 여기서 감마와 베타는 nn.LayerNorm에 들어가는 shape과 동일한 shape을 가짐(elementwise_affine이 True일 경우 학습되는 요소임(elementwise_affine의 default는 True))
        - 그림을 통한 이해    
            ![](./img/batchnorm_layernorm1.jpg)    


#### References
- https://gaussian37.github.io/dl-concept-batchnorm/
- https://yonghyuc.wordpress.com/2020/03/04/batch-norm-vs-layer-norm/
- https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
- https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
- https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- https://github.com/hyunwoongko/transformer

---

## #4

### pad_sequence(길이가 다른 데이터를 하나의 텐서로 묶어주기(+ collate_fn))
- 예시 코드
    ```python
    from torch.nn.utils.rnn import pad_sequence

    a = torch.ones(25, 300)
    b = torch.ones(22, 300)
    c = torch.ones(15, 300)
    print(pad_sequence([a, b, c]).size())

    '''
    torch.Size([25, 3, 300])
    '''
    ```
    ```python
    from torch.nn.utils.rnn import pad_sequence

    a = torch.ones(4, 6) # a문장의 길이는 4 , embedding은 6
    b = torch.ones(5, 6) # b문장의 길이는 5 , embedding은 6
    c = torch.ones(1, 6) # c문장의 길이는 1 , embedding은 6
    print(pad_sequence([a, b, c]).size()) # 전체 문장길이는 가장 긴 문장인 b에 맞춰 5가 됨
    print(pad_sequence([a, b, c]))

    '''
    torch.Size([5, 3, 6])
    tensor([[[1., 1., 1., 1., 1., 1.],  -> a문장 1번째 단어 임베딩
            [1., 1., 1., 1., 1., 1.],  -> b문장 1번째 단어 임베딩
            [1., 1., 1., 1., 1., 1.]], -> c문장 1번째 단어 임베딩
    
            [[1., 1., 1., 1., 1., 1.],  -> a문장 2번째 단어 임베딩
            [1., 1., 1., 1., 1., 1.],  -> b문장 2번째 단어 임베딩
            [0., 0., 0., 0., 0., 0.]], -> c문장 2번째 단어 임베딩 # c문장은 1단어로 이루어진 문장이므로 값이 없어서 0을 넣어줌
    
            [[1., 1., 1., 1., 1., 1.],  -> a문장 3번째 단어 임베딩
            [1., 1., 1., 1., 1., 1.],  -> b문장 3번째 단어 임베딩
            [0., 0., 0., 0., 0., 0.]], -> c문장 3번째 단어 임베딩
    
            [[1., 1., 1., 1., 1., 1.],  -> a문장 4번째 단어 임베딩
            [1., 1., 1., 1., 1., 1.],  -> b문장 4번째 단어 임베딩
            [0., 0., 0., 0., 0., 0.]], -> c문장 4번째 단어 임베딩
    
            [[0., 0., 0., 0., 0., 0.],  -> a문장 5번째 단어 임베딩 # a문장은 4단어로 이루어진 문장이므로 값이 없어서 0을 넣어줌
            [1., 1., 1., 1., 1., 1.],  -> b문장 5번째 단어 임베딩
            [0., 0., 0., 0., 0., 0.]]])-> c문장 5번째 단어 임베딩
    '''
    ```
- Data Loader를 할경우 길이가 다르게 나오는 경우 collate_fn을 통해 배치내 문장 길이 맞춰주기
    ```python
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)

        src_batch = pad_sequence(src_batch, padding_value=0)
        tgt_batch = pad_sequence(tgt_batch, padding_value=0)
        return src_batch, tgt_batch
    ```
- 실전 예시
    ```python
    pad_id = 0
    vocab_size = 100

    src_data = [
    [62, 13, 47, 39, 78, 33, 56, 13],
    [60, 96, 51, 32, 90],
    [35, 45, 48, 65, 91, 99, 92, 10, 3, 21],
    [66, 88, 98, 47],
    [77, 65, 51, 77, 19, 15, 35, 19, 23]
    ]

    trg_data = [
    [33, 11, 49, 10],
    [88, 34, 5, 29, 99, 45, 11, 25],
    [67, 25, 15, 90, 54, 4, 92, 10, 46, 20, 88 ,19],
    [16, 58, 91, 47, 12, 5, 8],
    [71, 63, 62, 7, 9, 11, 55, 91, 32, 48]
    ]

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self,src_data,trg_data):
            super().__init__()
            self.src_data = src_data
            self.trg_data = trg_data

        def __getitem__(self,index):
            return torch.LongTensor(self.src_data[index]),torch.LongTensor(self.trg_data[index])

        def __len__(self):
            return len(self.src_data)
    
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)

        src_batch = pad_sequence(src_batch, padding_value=0)
        tgt_batch = pad_sequence(tgt_batch, padding_value=0)
        return src_batch, tgt_batch

    dataset = CustomDataset(src_data,trg_data)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
    src,trg = next(iter(train_loader))
    print(src.shape) # L,B
    print(trg.shape) # L,B
    print(src)
    print(trg)
    '''
    torch.Size([9, 2])
    torch.Size([10, 2])
    tensor([[66, 77],
            [88, 65],
            [98, 51],
            [47, 77],
            [ 0, 19],
            [ 0, 15],
            [ 0, 35],
            [ 0, 19],
            [ 0, 23]])
    tensor([[16, 71],
            [58, 63],
            [91, 62],
            [47,  7],
            [12,  9],
            [ 5, 11],
            [ 8, 55],
            [ 0, 91],
            [ 0, 32],
            [ 0, 48]])
    '''
    ```
#### References
- https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

---

## #5

### ne,repeat
- ne
    - not equal (같으면 False, 다르면 True)
    - 코드
        ```python
        torch.ne(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
        '''
        tensor([[False,  True],
                [ True, False]])
        '''
        ```
        ```python
        a = torch.tensor([[1, 2], [3, 4]])
        print(a.ne(torch.tensor([[1, 1], [4, 4]])))
        '''
        tensor([[False,  True],
                [ True, False]])
        '''
        ```

- repeat
    - torch.repeat(*sizes)
    - 특정 텐서의 sizes 차원으로 데이터를 반복함
        - x의 차원이 repeat하는 차원에 맞지 않을 경우 앞에 1차원을 추가시켜줌
    - 코드
        ```python
        x = torch.tensor([1, 2, 3]) # torch.Size([3]) -> repeat안에 차원에 맞춰 [1,3]이 됨
        print(x.repeat(4, 2))
        print(x.repeat(4, 2).size())
        '''
        tensor([[ 1,  2,  3,  1,  2,  3],
               [ 1,  2,  3,  1,  2,  3],
               [ 1,  2,  3,  1,  2,  3],
               [ 1,  2,  3,  1,  2,  3]])
        torch.Size([4, 6])
        '''

        print(x.repeat(4, 2, 1).size())
        '''
        torch.Size([4, 2, 3])
        '''
        ```
        ```python
        x = torch.tensor([[1, 2, 3],[4,5,6]]) # torch.Size([2, 3])
        print(x.repeat(4, 2))
        print(x.repeat(4, 2).size())
        '''
        tensor([[1, 2, 3, 1, 2, 3],
                [4, 5, 6, 4, 5, 6],
                [1, 2, 3, 1, 2, 3],
                [4, 5, 6, 4, 5, 6],
                [1, 2, 3, 1, 2, 3],
                [4, 5, 6, 4, 5, 6],
                [1, 2, 3, 1, 2, 3],
                [4, 5, 6, 4, 5, 6]])
        torch.Size([8, 6])
        '''
        ```
        ```python
        x = torch.tensor([[1, 2, 3],[4,5,6]]) # torch.Size([2, 3]) -> torch.Size([1,1,2,3]) repeat 차원에 맞춰 앞에 1차원을 추가 시켜줌
        print(x.repeat(4, 2, 5,10).size())
        '''
        torch.Size([4, 2, 10, 30])
        '''
        ```
        ```python
        x = torch.tensor([[1, 2, 3],[4,5,6]]) # torch.Size([2, 3])
        print(x.repeat(1,2).size())
        '''
        torch.Size([2, 6])
        '''
        ```
        ```python
        x = torch.tensor([[[1, 2, 3],[4,5,6]]]) # torch.Size([1, 2, 3])
        print(x.repeat(2,3).size()) # 차원수가 더 작으면 안됨.
        '''
        RuntimeError: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor
        '''
        ```

#### References
- https://pytorch.org/docs/stable/generated/torch.ne.html
- https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html

---

## #6

### 이미지에 패딩처리해주기 (torch.nn.functional.pad) (+collate_fn)
- 코드
    - 대표 코드 
        ```python
        torch.nn.functional.pad(input, pad, mode='constant', value=0)
        ```
    - 예시 코드
        - pad 를 마지막 dim에만 줄 경우 (pad_left, pad_right) 모양으로 주기
        - pad 를 마지막 2개의 dim에 줄 경우 (pad_left, pad_right, pad_top, pad_bottom) 모양으로 주기
        - pad 를 마지막 3개의 dim에 줄 경우 (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back) 모양으로 주기
        - constant : 내가 설정한 값으로 패딩이 들어감
            ```python
            import torch.nn.functional as F
            import torch

            t4d = torch.ones((2, 3, 3, 2)) # (batch, channel, height, width)
            print(t4d)

            p1d = (1, 1) # 왼쪽에 1만큼 오른쪽에 1만큼 패딩추가
            out = F.pad(t4d, p1d, "constant", 0) # effectively zero padding, 
            print(out)
            print(out.size()) 
            '''
            tensor([[[[1., 1.],
                      [1., 1.],
                      [1., 1.]],

                     [[1., 1.],
                      [1., 1.],
                      [1., 1.]],

                     [[1., 1.],
                      [1., 1.],
                      [1., 1.]]],


                    [[[1., 1.],
                      [1., 1.],
                      [1., 1.]],

                     [[1., 1.],
                      [1., 1.],
                      [1., 1.]],

                     [[1., 1.],
                      [1., 1.],
                      [1., 1.]]]])
            tensor([[[[0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.]],

                     [[0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.]],

                     [[0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.]]],


                    [[[0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.]],

                     [[0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.]],

                     [[0., 1., 1., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 1., 0.]]]])
            torch.Size([2, 3, 3, 4])
            '''
            ```
        - replicate : pad값에 상관없이 제일 가장자리에있는 값으로 패딩해서 들어감 (단 replicate는 2d부터 지원,1d만 늘리고싶으면 넣지 않을곳은 0으로 넣어주면 됨)
            ```python
            import torch.nn.functional as F
            import torch

            t4d = torch.Tensor(2, 3, 3, 2) # (batch, channel, height, width)
            print(t4d)

            # replicate
            p1d = (2, 2,0,0) 
            out = F.pad(t4d, p1d, "replicate",0) # replicate는 value에 상관없이 제일 가장자리값으로 채워짐
            print(out)

            '''
            tensor([[[[1.9006e-35, 0.0000e+00],
                      [3.1529e-43, 0.0000e+00],
                      [1.9234e-35, 0.0000e+00]],

                     [[1.2057e+17, 4.5698e-41],
                      [       nan, 0.0000e+00],
                      [4.4721e+21, 3.9891e+24]],

                     [[4.1996e+12, 7.5338e+28],
                      [1.5975e-43, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00]]],


                    [[[2.2561e-43, 0.0000e+00],
                      [1.9234e-35, 0.0000e+00],
                      [1.2057e+17, 4.5698e-41]],

                     [[       nan, 0.0000e+00],
                      [4.4721e+21, 2.3079e+20],
                      [6.2689e+22, 4.7428e+30]],

                     [[0.0000e+00, 0.0000e+00],
                      [9.1845e-41, 1.3079e+22],
                      [1.3593e-43, 0.0000e+00]]]])

            tensor([[[[1.9006e-35, 1.9006e-35, 1.9006e-35, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [3.1529e-43, 3.1529e-43, 3.1529e-43, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [1.9234e-35, 1.9234e-35, 1.9234e-35, 0.0000e+00, 0.0000e+00, 0.0000e+00]],

                     [[1.2057e+17, 1.2057e+17, 1.2057e+17, 4.5698e-41, 4.5698e-41, 4.5698e-41],
                      [       nan,        nan,        nan, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [4.4721e+21, 4.4721e+21, 4.4721e+21, 3.9891e+24, 3.9891e+24, 3.9891e+24]],

                     [[4.1996e+12, 4.1996e+12, 4.1996e+12, 7.5338e+28, 7.5338e+28, 7.5338e+28],
                      [1.5975e-43, 1.5975e-43, 1.5975e-43, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]]],


                    [[[2.2561e-43, 2.2561e-43, 2.2561e-43, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [1.9234e-35, 1.9234e-35, 1.9234e-35, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [1.2057e+17, 1.2057e+17, 1.2057e+17, 4.5698e-41, 4.5698e-41, 4.5698e-41]],

                     [[       nan,        nan,        nan, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [4.4721e+21, 4.4721e+21, 4.4721e+21, 2.3079e+20, 2.3079e+20, 2.3079e+20],
                      [6.2689e+22, 6.2689e+22, 6.2689e+22, 4.7428e+30, 4.7428e+30, 4.7428e+30]],

                     [[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                      [9.1845e-41, 9.1845e-41, 9.1845e-41, 1.3079e+22, 1.3079e+22, 1.3079e+22],
                      [1.3593e-43, 1.3593e-43, 1.3593e-43, 0.0000e+00, 0.0000e+00, 0.0000e+00]]]])
            '''
            ```
- 실전 코드 
    - 데이터의 이미지 크기가 다른경우가 존재하는데 이 데이터를 무조건 resize를 하게 된다면 이미지의 비율이 깨져 문제가 발생할수 있음
    - 예를 들어 종이에 써진 글씨를 체크하는 ocr task를 할때 무작정으로 resize를 하게되면 원본 이미지가 비율이 이상해져서 학습이 잘못될수도 있음
    - collate_fn에서 해결가능
        - 배치별로 이미지 크기를 다르게 하고 싶은 경우
            ```python
            def collate_fn(batch):
                img_batch, tgt_batch = [], []
                
                max_w = 0
                max_h = 0
                for img, tgt_sample in batch:
                    w = img.size(2)
                    h = img.size(1)
                    if w>max_w:
                        max_w = w
                    if h>max_h:
                        max_h = h
                    tgt_batch.append(tgt_sample)

                for img, _ in batch:
                    w= img.size(2)
                    h= img.size(1)
                    new_img = torch.nn.functional.pad(img, (0,max_w-w,0,max_h-h), mode='replicate')
                    img_batch.append(new_img)

                tgt_batch = pad_sequence(tgt_batch,batch_first=True, padding_value=0)
                return torch.stack(img_batch), tgt_batch
            ```
        - 모든 이미지 크기를 동일하게 하고 싶은 경우(미리 max_w,max_h 지정해놔야함)
            ```python
            def collate_fn(batch):
                img_batch, tgt_batch = [], []
                
                for img, tgt_sample in batch:
                    w = img.size(2)
                    h = img.size(1)
                    new_img = torch.nn.functional.pad(img, (0,max_w-w,0,max_h-h), mode='replicate')
                    img_batch.append(new_img)
                    tgt_batch.append(tgt_sample)
                    

                tgt_batch = pad_sequence(tgt_batch,batch_first=True, padding_value=0)
                return torch.stack(img_batch), tgt_batch
            ```
    - 이미지를 transformer에 적용할경우 mask까지 생각가능함(이미지에서 패딩처리된부분을 체크해주기)
        ```python
        def collate_fn(batch):
            img_batch, tgt_batch = [], []
            pad_mask_batch =[]
            max_w = 0
            for img, tgt_sample in batch:
                w = img.size(2)
                if w>max_w:
                    max_w=w        
                tgt_batch.append(tgt_sample)
            if max_w%4!=0:
                max_w+=(4-max_w%4)
            for img, _ in batch:
                w= img.size(2)
                new_img = torch.nn.functional.pad(img, (0,max_w-w,0,0), mode='replicate', value=0)
                pad_mask_batch.append(max_w-w)
                img_batch.append(new_img)

            tgt_batch = pad_sequence(tgt_batch,batch_first=True, padding_value=0)
            return torch.stack(img_batch), tgt_batch , pad_mask_batch
        ```
        - 트레인 부분에서 아래와 같이 처리
        ```python
        for i, batch in enumerate(tqdm(iterator)):
            image = batch[0].to(device) # Batch,channel,height,width
            text = batch[1].to(device) # Batch,Lenght
            pad_mask = batch[2]
            batch_w = image.size(-1)
            batch_img_mask=[]
            for pad_len in pad_mask:
                img_mask = [1 for _ in range(batch_w//4-pad_len//4)]+[0 for _ in range(pad_len//4)]
                batch_img_mask.append(img_mask)
            batch_img_mask=torch.LongTensor(batch_img_mask).to(device)

        ```
#### References
- https://hichoe95.tistory.com/116
- https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad

---

## #7

### argmax와 topk
- argmax는 인덱스 값중에서 제일 큰 값만 남겨줌
    - 코드
        ```python
        a = torch.randn(4, 4)
        print(a)
        print(torch.argmax(a,dim=-1))
        print(a.argmax(-1))
        '''
        tensor([[ 0.5596, -0.1956, -1.8283,  0.9479],
                [ 1.3220, -0.9693,  0.0772,  0.3041],
                [ 1.9671,  0.2095,  1.2014, -0.8649],
                [-0.7770,  0.5726,  1.1960, -0.7411]])
        tensor([3, 0, 0, 2])
        tensor([3, 0, 0, 2])
        '''
        ```
- topk : argmax는 한차원이 감소하는 반면 topk는 입력텐서와 동일한 차원으로 결과가 나옴(value와 index 모두 반환)
    - 코드
        ```python
        x = torch.randn(4, 4)
        print(x)
        print(torch.topk(x, 3))
        print(torch.topk(x,3)[0],torch.topk(x,3)[1])
        '''
        tensor([[-0.7079,  1.4738, -0.8523, -0.6596],
                [ 0.2957, -0.2204,  0.3734,  0.0425],
                [ 1.2357, -0.4194, -0.8848,  0.1949],
                [ 0.2156, -1.0695,  0.7949, -0.0830]])
        torch.return_types.topk(
                values=tensor([[ 1.4738, -0.6596, -0.7079],
                                [ 0.3734,  0.2957,  0.0425],
                                [ 1.2357,  0.1949, -0.4194],
                                [ 0.7949,  0.2156, -0.0830]]),
                indices=tensor([[1, 3, 0],
                                [2, 0, 3],
                                [0, 3, 1],
                                [2, 0, 3]]))
        tensor([[ 1.4738, -0.6596, -0.7079],
                [ 0.3734,  0.2957,  0.0425],
                [ 1.2357,  0.1949, -0.4194],
                [ 0.7949,  0.2156, -0.0830]]) 
        tensor([[1, 3, 0],
                [2, 0, 3],
                [0, 3, 1],
                [2, 0, 3]])
        '''
        ```
#### References
- https://pytorch.org/docs/stable/generated/torch.argmax.html
- https://pytorch.org/docs/stable/generated/torch.topk.html
