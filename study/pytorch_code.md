## Table of Contents

- [view 와 reshape의 차이 (cf. flatten,contiguous,clone)](#1)

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