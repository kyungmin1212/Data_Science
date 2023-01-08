## Table of Contents

- [Bag-of-words 와 NaiveBayes Classifier](#1)
- [Word Embedding(Word2Vec, Glove)(cf. CBOW, Skip-gram)](#2)
- [RNN(Recurrent Neural Network)](#3)
- [LSTM(Long Short-Term Memory), GRU(Gated Recurrent Unit)](#4)
- [Sequence to Sequence with Attention](#5)
- [Beam Search](#6)
---

## #1

### Bag-of-words 와 NaiveBayes Classifier
- Bag-of-words : 딥러닝 기술이 적용되기 이전에 많이 활용되던 단어 및 문서를 숫자 형태로 나타내는 가장 간단한 기법
    - Bag-of-words 과정
        1. 텍스트 데이터셋에서 unique한 word들을 모아서 vocabulary(사전)을 구축
        2. 사전에 단어 개수가 n개라면 n dimension의 벡터를 만들고, 각 단어에 해당하는 값은 1, 나머지는 0으로 표현(원핫벡터)
        (원핫벡터는 워드 임베딩 기법과 대비되는 특성으로 어떤 단어쌍이든 모두 유클리드 거리가 루트2로 표현되고 내적값 혹은 내적 코사인유사도는 0으로 모두 동일하게 계산됨 -> 즉 단어의 의미와 상관없이 모두가 동일한 관계를 가지는 형태로 단어 표현)
        3. 문장에 있는 단어들의 원핫벡터를 모두 더하여 문장 표현
        (즉, 단어들을 Bag에 넣는다고 생각)
    - Bag-of-words 예시    
        ![](./img/Bag-of-words.jpg)
- NaiveBayes Classifier : Bag-of-words를 활용한 대표적인 문서 분류 기법
    - 이론    
        ![](./img/NaiveBayes_Classifier.jpg)
    - 예시    
        ![](./img/NaiveBayes_Classifier1.jpg)
        ![](./img/NaiveBayes_Classifier2.jpg)
  

#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)

---

## #2

### Word Embedding(Word2Vec, Glove)(cf. CBOW, Skip-gram)
- Word Embedding의 기본 아이디어는 비슷한 의미를 가지는 단어가 좌표공간상에 비슷한 위치의 점으로 매핑 되도록 함으로써 의미상의 유사도를 잘 반영한 벡터 표현을 만들도록 해줌
- CBOW와 Skip-gram
    - CBOW(Continuous Bag-of-Words)
        - 주변 단어들을 가지고 중심 단어를 예측하는 방식
        - 예시
            - **A cute puppy is walking** in the park. & window size: 2
                - Input(주변 단어): "A", "cute", "is", "walking"
                - Output(중심 단어): "puppy"
    - Skip-gram
        - 중심 단어를 가지고 주변 단어들을 예측하는 방식
        - 예시
            - **A cute puppy is walking** in the park. & window size: 2
                - Input(중심 단어): "puppy"
                - Output(주변 단어): "A", "cute", "is", "walking"
    - CBOW는 입출력 쌍이 (윈도우내 주변단어 전체,중심단어)(즉, (2*windowsize,1) 크기)이고, Skip-gram은 입출력 쌍이 (주변단어1개,중심단어)(즉, (1,1) 크기)로 이루어져있음(CBOW는 모든 입력쌍이 임베딩 된 후에 sum을 통해 합쳐줌)
- Word2Vec
    - 한 단어가 주변에 등장하는 단어를 통해 그 의미를 알 수 있다는 점에서 착안
    - Word2Vec 과정 (예시 문장 : `I study math`)
        1. 문장에서 unique한 단어를 추출하여 사전 구축(one-hot vector)
        2. 문장에서 입출력 순서쌍 구하기(Skip-gram적용, (여기선 window size = 3 으로 설정(앞,자기자신,뒤)))
            - 입출력 쌍 : (I,study) // (study,I), (study,math) // (math,study)
        3. 입출력 레이어의 노드 수는 사전의 사이즈와 같음 (여기선 문장의 단어수가 3개이므로 vocabulary size = 3)
        4. 입출력 순서쌍 사이에 두개의 입출력 레이어와 소프트맥스를 넣고 학습 진행(vocabulary size 차원 -> vocabulary size 보다 작은 차원 -> vocabulary size 차원)    
        ![](./img/Word2Vec.jpg)

        (우리는 이 중에서 input 단어가 들어갔을때 W1을 거쳐 hidden layer에 존재하는 벡터를 임베딩 벡터로 사용)
    - Property of Word2Vec
        - 단어들 사이에 관계를 표현해보면 비슷한 관계는 같은 벡터를 가짐    
            ![](./img/Word2Vec1.jpg)

- GloVe
    - Word2Vec과 더불어 많이 쓰이는 워드 임베딩 방법
    - GloVe는 어떠한 단어쌍이 동시에 등장하는 횟수를 미리 계산해서 중복되는 계산을 줄여줄수 있다는 장점이 존재하여 상대적으로 더 빠르고 더 적은 데이터에 대해서도 더 잘 동작하는 특성을 보임
    - Glove 과정
        - 각 입출력 단어 쌍들에 대해 그 학습 데이터에서 그 두 단어가 한 윈도우 내에서 총 몇 번 동시에 등장했는지를 사전에 미리 계산 $P_{ij}$
        - 입력워드의 임베딩 벡터 $u_i$ 와 출력워드의 임베딩 벡터 $v_j$의 내적값이 한 윈도우 안에서 두 단어가 동시에 나타난 횟수인 $P_{ij}$에 가까워질 수 있도록 학습 진행함
    - Glove 수식    
        ![](./img/glove.jpg)

- Word2Vec 실습 (+ CBOW, SkipGram)
    - 실습 데이터
        ```python
        train_data = [
        "정말 맛있습니다. 추천합니다.",
        "기대했던 것보단 별로였네요.",
        "다 좋은데 가격이 너무 비싸서 다시 가고 싶다는 생각이 안 드네요.",
        "완전 최고입니다! 재방문 의사 있습니다.",
        "음식도 서비스도 다 만족스러웠습니다.",
        "위생 상태가 좀 별로였습니다. 좀 더 개선되기를 바랍니다.",
        "맛도 좋았고 직원분들 서비스도 너무 친절했습니다.",
        "기념일에 방문했는데 음식도 분위기도 서비스도 다 좋았습니다.",
        "전반적으로 음식이 너무 짰습니다. 저는 별로였네요.",
        "위생에 조금 더 신경 썼으면 좋겠습니다. 조금 불쾌했습니다."       
        ]

        test_words = ["음식", "맛", "서비스", "위생", "가격"]
        ```
    - 데이터를 Tokenization을 진행한 후에 token 별로 embedding에 집어넣기 위한 숫자로 변경시켜주기(token id화 시켜주기)
        ```python
        from konlpy.tag import Okt
        from collections import defaultdict

        tokenizer = Okt()

        def make_tokenized(data):
            tokenized = []
            for sent in data:
                tokens = tokenizer.morphs(sent, stem=True)
                tokenized.append(tokens)

            return tokenized

        train_tokenized = make_tokenized(train_data) # [['정말', '맛있다', '.', '추천', '하다', '.'], ['기대하다', '것', '보단', '별로', '이다', '.'], ...]

        word_count = defaultdict(int)

        for tokens in train_tokenized:
            for token in tokens:
                word_count[token] += 1
        
        w2i = {}
        for pair in word_count:
            if pair[0] not in w2i:
                w2i[pair[0]] = len(w2i)

        print(w2i)
        '''
        {'.': 0, '도': 1, '이다': 2, '좋다': 3, '별로': 4, '다': 5, '이': 6, '너무': 7, '음식': 8, '서비스': 9, '하다': 10, '방문': 11, '위생': 12, '좀': 13, '더': 14, '에': 15, '조금': 16, '정말': 17, '맛있다': 18, '추천': 19, '기대하다': 20, '것': 21, '보단': 22, '가격': 23, '비싸다': 24, '다시': 25, '가다': 26, '싶다': 27, '생각': 28, '안': 29, '드네': 30, '요': 31, '완전': 32, '최고': 33, '!': 34, '재': 35, '의사': 36, '있다': 37, '만족스럽다': 38, '상태': 39, '가': 40, '개선': 41, '되다': 42, '기르다': 43, '바라다': 44, '맛': 45, '직원': 46, '분들': 47, '친절하다': 48, '기념일': 49, '분위기': 50, '전반': 51, '적': 52, '으로': 53, '짜다': 54, '저': 55, '는': 56, '신경': 57, '써다': 58, '불쾌하다': 59}
        '''
    - CBOW Dataset 과 SkipGram Dataset 생성 (실제 embedding 모델에 들어가기 위한 input과 output 쌍 생성)
        - CBOW Dataset
            ```python
            class CBOWDataset(Dataset):
                def __init__(self, train_tokenized, window_size=2):
                    self.x = []
                    self.y = []

                    for tokens in tqdm(train_tokenized):
                        # token_ids 는 한 문장 안에 있는 token들의 id 리스트
                        token_ids = [w2i[token] for token in tokens]
                        for i, id in enumerate(token_ids):
                            if i-window_size >= 0 and i+window_size < len(token_ids): # 범위 안에 있는것들만 가져온다.
                                self.x.append(token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1]) # 주변단어 4개
                                self.y.append(id) # 중심단어 1개

                    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수 * 2 * window_size)
                    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

                def __len__(self):
                    return self.x.shape[0]

                def __getitem__(self, idx):
                    return self.x[idx], self.y[idx]
            ```
        - SkipGram Dataset
            ```python
            class SkipGramDataset(Dataset):
                def __init__(self, train_tokenized, window_size=2):
                    self.x = []
                    self.y = []

                    for tokens in tqdm(train_tokenized):
                        token_ids = [w2i[token] for token in tokens]
                        for i, id in enumerate(token_ids):
                            if i-window_size >= 0 and i+window_size < len(token_ids):
                                self.y += (token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
                                self.x += [id] * 2 * window_size

                    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수 * 2 * window_size)
                    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수 * 2 * window_size)

                def __len__(self):
                    return self.x.shape[0]

                def __getitem__(self, idx):
                    return self.x[idx], self.y[idx]
            ```
        - Dataset 생성
            ```python
            cbow_set = CBOWDataset(train_tokenized)
            skipgram_set = SkipGramDataset(train_tokenized)
            print(list(cbow_set)[:5])
            print()
            print(list(skipgram_set)[:5])
            print()
            print(len(cbow_set))
            print(len(skipgram_set))
            '''
            [(tensor([17, 18, 19, 10]), tensor(0)), (tensor([18,  0, 10,  0]), tensor(19)), (tensor([20, 21,  4,  2]), tensor(22)), (tensor([21, 22,  2,  0]), tensor(4)), (tensor([5, 3, 6, 7]), tensor(23))]

            [(tensor(0), tensor(17)), (tensor(0), tensor(18)), (tensor(0), tensor(19)), (tensor(0), tensor(10)), (tensor(19), tensor(18))]

            64
            256
            '''
            ```
    - Word2Vec 모델
        - CBOW Word2Vec 모델
            ```python
            class CBOW(nn.Module):
                def __init__(self, vocab_size, dim):
                    super(CBOW, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
                    self.linear = nn.Linear(dim, vocab_size)

                # B: batch size, W: window size, d_w: word embedding size, V: vocab size
                def forward(self, x):  # x: (B, 2W)
                    embeddings = self.embedding(x)  # (B, 2W, d_w)
                    embeddings = torch.sum(embeddings, dim=1)  # (B, d_w)
                    output = self.linear(embeddings)  # (B, V)
                    return output
                
            cbow = CBOW(vocab_size=len(w2i), dim=256)
            ```
        - SkipGram Word2Vec 모델
            ```python
            class SkipGram(nn.Module):
                def __init__(self, vocab_size, dim):
                    super(SkipGram, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
                    self.linear = nn.Linear(dim, vocab_size)

                # B: batch size, W: window size, d_w: word embedding size, V: vocab size
                def forward(self, x): # x: (B)
                    embeddings = self.embedding(x)  # (B, d_w)
                    output = self.linear(embeddings)  # (B, V)
                    return output
                
            skipgram = SkipGram(vocab_size=len(w2i), dim=256)
            ```
    - 학습 진행
        - CBOW Word2Vec 모델 학습
            ```python
            batch_size=4
            learning_rate = 5e-4
            num_epochs = 5
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            cbow_loader = DataLoader(cbow_set, batch_size=batch_size)

            cbow.train()
            cbow = cbow.to(device)
            optim = torch.optim.SGD(cbow.parameters(), lr=learning_rate)
            loss_function = nn.CrossEntropyLoss()

            for e in range(1, num_epochs+1):
                print("#" * 50)
                print(f"Epoch: {e}")
                for batch in tqdm(cbow_loader):
                    x, y = batch
                    x, y = x.to(device), y.to(device) # (B, 2*W), (B)
                    output = cbow(x)  # (B, V)

                    optim.zero_grad()
                    loss = loss_function(output, y)
                    loss.backward()
                    optim.step()

                    print(f"Train loss: {loss.item()}")

            print("Finished.")
            ```
        - SkipGram Word2Vec 모델 학습
            ```python
            batch_size=4
            learning_rate = 5e-4
            num_epochs = 5
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            skipgram_loader = DataLoader(skipgram_set, batch_size=batch_size)

            skipgram.train()
            skipgram = skipgram.to(device)
            optim = torch.optim.SGD(skipgram.parameters(), lr=learning_rate)
            loss_function = nn.CrossEntropyLoss()

            for e in range(1, num_epochs+1):
                print("#" * 50)
                print(f"Epoch: {e}")
                for batch in tqdm(skipgram_loader):
                    x, y = batch
                    x, y = x.to(device), y.to(device) # (B), (B)
                    output = skipgram(x)  # (B, V)

                    optim.zero_grad()
                    loss = loss_function(output, y)
                    loss.backward()
                    optim.step()

                    print(f"Train loss: {loss.item()}")

            print("Finished.")
            ```
    - test 단어들의 word embedding 확인
        - 모델.embedding을 이용하여 단어의 embedding을 확인
        ```python
        for word in test_words:
            input_id = torch.LongTensor([w2i[word]]).to(device)
            emb = cbow.embedding(input_id)

            print(f"Word: {word}")
            print(emb.squeeze(0))
        ```
        ```python
        for word in test_words:
            input_id = torch.LongTensor([w2i[word]]).to(device)
            emb = skipgram.embedding(input_id)

            print(f"Word: {word}")
            print(emb.squeeze(0))
        ```

#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)

---

## #3

### RNN(Recurrent Neural Network)
- RNN 종류    
    ![](./img/rnn.jpg)
- RNN 기본 구조    
    ![](./img/RNN1.jpg)
- RNN 연산 과정    
    ![](./img/RNN2.jpg)
- RNN many-to-many 학습 추론 과정 예시
    - hello 단어를 통한 설명
    - 학습과정    
        ![](./img/RNN3.jpg)
    - 추론과정
        - h라는 문자열 하나가 들어가게 되면 h의 결과값 y가 다음 input으로 들어가게 되고 또 다시 그 input의 결과값이 다시 다음 input으로 들어가게 됨    
        ![](./img/RNN4.jpg)
- Backpropagation through time (BPTT)
    - 각 타임 스텝마다 예측값과 실제값의 비교를 통한 loss function을 통해서 전체 네트워크가 학습을 진행됨 -> 전체 시퀀스의 길이가 길어지게 되면 메모리 문제등으로 인하여 학습이 어려워짐
    - 실제 한번 학습을 진행하기 위해서는 하나의 입력의 output을 구하고 그 output과 입력을 통해서 다시 output을 구하고 이런식으로 모든 output을 구하게 되면 마지막 타임스텝쯤에는 제일 처음의 타임 스텝부터 동일한 matrix가 매 타임 스텝마다 곱해지게 되면서 메모리 문제가 발생할수 있음 또는 Vanishing/Exploding Gradient Problem 발생 가능 
    - truncation을 이용하여 제한된 길이의 시퀀스 만으로 학습을 진행하는 방법을 사용(Truncated-BPTT)    
        ![](./img/RNN5.jpg)

- RNN 실습
    - 샘플 데이터 (전체 vocab_size = 100, pad_id = 0)
        ```python
        vocab_size = 100
        pad_id = 0

        data = [
        [85,14,80,34,99,20,31,65,53,86,3,58,30,4,11,6,50,71,74,13],
        [62,76,79,66,32],
        [93,77,16,67,46,74,24,70],
        [19,83,88,22,57,40,75,82,4,46],
        [70,28,30,24,76,84,92,76,77,51,7,20,82,94,57],
        [58,13,40,61,88,18,92,89,8,14,61,67,49,59,45,12,47,5],
        [22,5,21,84,39,6,9,84,36,59,32,30,69,70,82,56,1],
        [94,21,79,24,3,86],
        [80,80,33,63,34,63],
        [87,32,79,65,2,96,43,80,85,20,41,52,95,50,35,96,24,80]
        ]
        ```
    - padding 처리
        ```python
        max_len = len(max(data, key=len))
        print(f"Maximum sequence length: {max_len}")

        valid_lens = []
        for i, seq in enumerate(tqdm(data)):
            valid_lens.append(len(seq)) # padding 전 길이 저장
            if len(seq) < max_len:
                data[i] = seq + [pad_id] * (max_len - len(seq)) # pad_id로 data max 길이와 동일하게 padding 처리
        ```
        ```python
        # B: batch size, L: max_len
        batch = torch.LongTensor(data)  # (B, L)
        batch_lens = torch.LongTensor(valid_lens)  # (B)
        ```
        (주어진 데이터 전체를 batch 1개로 보기(batch size = len(data)))
    - RNN에 넣기 위한 word embedding 처리(이미 학습된 word2vec이나 glove같은 embedding 모델을 불러와서 사용해도 됨)
        ```python
        embedding_size = 256
        embedding = nn.Embedding(vocab_size, embedding_size)

        # d_w: embedding size
        batch_emb = embedding(batch)  # (B, L, d_w)
        ```
    - RNN 모델 생성 (배치 1개 넣어주기)
        ```python
        hidden_size = 512  # RNN의 hidden size
        num_layers = 1  # 쌓을 RNN layer의 개수
        num_dirs = 1  # 1: 단방향 RNN, 2: 양방향 RNN

        rnn = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # batch_first = True #(False 가 default) -> False면 rnn의 입력순서가 (seq,batch,feature) 이어야함. feature은 embedding한 데이터
            bidirectional=True if num_dirs > 1 else False
        )

        h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)

        # hidden_states: 각 time step에 해당하는 hidden state들의 묶음.
        # h_n: 모든 sequence를 거치고 나온 마지막 hidden state.
        hidden_states, h_n = rnn(batch_emb.transpose(0, 1), h_0) # batch_first 가 False 이므로 transpose시켜줌

        # d_h: hidden size, num_layers: layer 개수, num_dirs: 방향의 개수
        print(hidden_states.shape)  # (L, B, d_h)
        print(h_n.shape)  # (num_layers*num_dirs, B, d_h) = (1, B, d_h)
        '''
        torch.Size([20, 10, 512])
        torch.Size([1, 10, 512])
        '''
        ```    
        ![](./img/RNN6.jpg)     
    - RNN 활용법
        - 마지막 hidden state 만을 이용하여 text classification task 적용 가능
            ```python
            num_classes = 2
            classification_layer = nn.Linear(hidden_size, num_classes)

            # C: number of classes
            output = classification_layer(h_n.squeeze(0))  # (1, B, d_h) => (B, C)
            print(output.shape)
            '''
            torch.Size([10, 2])
            '''
            ```
        - 각 time step에 대한 hidden state를 이용하여 token-level의 task 수행 가능(각 품사를 태깅하는 POS 태깅이나 named_entity_recognition(각 토큰이 어떤 의미를 가지는지를 분류)등에 사용 가능)
            ```python
            num_classes = 5
            entity_layer = nn.Linear(hidden_size, num_classes)

            # C: number of classes
            output = entity_layer(hidden_states)  # (L, B, d_h) => (L, B, C)
            print(output.shape)
            '''
            torch.Size([10, 20, 5])
            '''
            ```
            (단, Language modeling(특정 토큰 뒤에 다음 토큰이 무엇이 올지를 순차적으로 예측하는 task)을 한다고 가정하면 위와 같이 토큰 레벨로 classification 해주면 안됨 -> 직접 for loop 를 호출해서 맨 처음 input의 결과를 다음 input으로 넣어주고 그것의 결과를 다시 input으로 넣어주는 반복작업이 필요)    
            (예를 들면 I want to go home 에서 위의 예시에서는 그냥 I want to go home. 문장 전체를 넣어주면 전체 각 token에 대한 품사 결과가 나올 수 있지만, Language modeling은 I를 넣고 그것으로 인한 결과인 want를 모르면 다음 input을 넣을수가 없음.(Language modeling은 처음 단어를 넣으면 그에 맞는 다음 단어를 생성하고 또 그 다음 단어를 생성하는 것인데 I want to go home.을 바로 input에 넣는다는 것은 정답을 알고 있으면서 문제를 풀어달라고 하는 것.))
    - PackedSequence 사용법
        - 앞서 pad_id 0을 길이를 맞춰주기 위해 넣어주었는데 이 0은 아무런 의미가 없는 dummy 데이터임, 의미적으로 아무런 중요도 없고 굳이 계산을 하지 않아도 되는 부분 (메모리와 연산량 낭비) -> PackedSequence를 통해 해결 가능
        - 정렬을 하지않고 PackedSequence를 사용하는 경우
            - T=2,3인 부분은 중간에 pad가 끼어 있어 어쩔수 없이 0을 넣어서 연산을 해주어야함(정렬을 하지 않으면 PackedSequence의 장점을 살리지 못할수도 있음)    
            ![](./img/packedsequence.gif) 
        - 정렬 후에 PackedSequence 적용
            - 배치내의 문장의 길이를 기준으로 정렬해주고 넣어주게 된다면 RNN에서 다음 타임 스텝으로 넘어갈때마다 배치사이즈를 조절하게 된다면 메모리와 연산량 낭비 문제를 최대한 해결가능(더 빠른 연산 가능)(정렬을 하게 된다면 pad를 넣은 부분을 하나도 사용하지 않게 됨)
            - 연산량이 (5 x 6 x 1) = 30 에서 (5+4+3+3+2+1) = 18로 크게 줄어들게 됨    
            ![](./img/packedsequence1.gif)
        - PackedSequence의 마지막 hidden 부분은 아래와 같이 알아서 마지막 부분을 선택해서 출력하게됨
            ![](./img/packedsequence2.jpg)
        - 코드
            ```python
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

            # batch_lens는 배치내의 데이터 길이가 담긴 torch.LongTensor

            sorted_lens, sorted_idx = batch_lens.sort(descending=True)

            # torch의 tensor는 정렬하게 되면 value와 indices가 같이 나옴
            # import torch

            # a = torch.LongTensor([3,4,2,1])
            # a.sort(descending=True)
            ### torch.return_types.sort(values=tensor([4, 3, 2, 1]),indices=tensor([1, 0, 2, 3]))

            sorted_batch = batch[sorted_idx]

            print(sorted_batch)
            print(sorted_lens)
            '''
            tensor([[85, 14, 80, 34, 99, 20, 31, 65, 53, 86,  3, 58, 30,  4, 11,  6, 50, 71,  74, 13],
                    [58, 13, 40, 61, 88, 18, 92, 89,  8, 14, 61, 67, 49, 59, 45, 12, 47,  5,  0,  0],
                    [87, 32, 79, 65,  2, 96, 43, 80, 85, 20, 41, 52, 95, 50, 35, 96, 24, 80,  0,  0],
                    [22,  5, 21, 84, 39,  6,  9, 84, 36, 59, 32, 30, 69, 70, 82, 56,  1,  0,  0,  0],
                    [70, 28, 30, 24, 76, 84, 92, 76, 77, 51,  7, 20, 82, 94, 57,  0,  0,  0,  0,  0],
                    [19, 83, 88, 22, 57, 40, 75, 82,  4, 46,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [93, 77, 16, 67, 46, 74, 24, 70,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [94, 21, 79, 24,  3, 86,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [80, 80, 33, 63, 34, 63,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                    [62, 76, 79, 66, 32,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
            tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])
            '''
            ```
            ```python
            # 정렬된 batch를 embedding후에 전체 문장별 길이데이터와 함께 pack_padded_sequence에 넣어주기 
            sorted_batch_emb = embedding(sorted_batch)
            packed_batch = pack_padded_sequence(sorted_batch_emb.transpose(0, 1), sorted_lens)

            print(packed_batch)
            print(packed_batch[0].shape)
            '''
            PackedSequence(data=tensor([[-0.9775, -0.0275,  0.2590,  ...,  0.5289,  0.5520, -0.0085],
                    [ 0.9082, -1.4621, -0.7293,  ...,  0.9566,  1.0870,  1.0706],
                    [ 0.3689,  0.2891,  1.2888,  ..., -1.0620,  0.1863,  1.2684],
                    ...,
                    [ 1.3904,  1.3372, -1.8687,  ..., -1.9423, -0.7393, -0.2358],
                    [-0.5254,  1.5967,  1.3016,  ...,  1.1077, -0.9597,  1.4632],
                    [ 0.8475,  0.3587, -0.9515,  ..., -0.3165, -0.6473, -0.4341]],
                grad_fn=<PackPaddedSequenceBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
                    1,  1]), sorted_indices=None, unsorted_indices=None)
            torch.Size([123, 256])
            '''
            ```
            ```python
            packed_outputs, h_n = rnn(packed_batch, h_0)

            print(packed_outputs)
            print(packed_outputs[0].shape)
            print(h_n.shape)
            '''
            PackedSequence(data=tensor([[-0.1106, -0.1299,  0.5338,  ..., -0.5201, -0.7577,  0.5029],
                    [-0.3075, -0.0623, -0.6321,  ..., -0.0955,  0.3264, -0.4315],
                    [-0.3981,  0.0435,  0.1340,  ..., -0.1250,  0.1168,  0.3860],
                    ...,
                    [-0.1111,  0.5823, -0.4191,  ..., -0.0901, -0.1965, -0.6291],
                    [-0.0949,  0.2718,  0.1046,  ..., -0.0153,  0.4186,  0.3825],
                    [ 0.1410, -0.1145,  0.0253,  ..., -0.3859,  0.3022,  0.6851]],
                grad_fn=<CatBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
                    1,  1]), sorted_indices=None, unsorted_indices=None)
            torch.Size([123, 512])
            torch.Size([1, 10, 512])
            '''
            ```
            ```python
            # packed_output은 PackedSequence 이므로 원래 output 형태와 다름 이를 다르 원래 형태로 바꿔주기 위해 pad_packed_sequence를 이용
            outputs, outputs_lens = pad_packed_sequence(packed_outputs)

            print(outputs.shape)  # (L, B, d_h)
            print(outputs_lens)
            '''
            torch.Size([20, 10, 512])
            tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])
            '''
            ```
        - 코드부분을 그림을 통한 이해     
            ![](./img/packedsequence3.jpg)
            
#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)
- https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
---

## #4

### LSTM(Long Short-Term Memory), GRU(Gated Recurrent Unit)
- LSTM
    - LSTM은 original RNN이 가지는 문제인 Gradient Vanishing/Explosion 를 해결하고 타임 스텝이 먼 경우에도 필요로 하는 정보를 보다 효과적으로 처리하고 학습할 수 있도록 하는 모델
    - 단기기억(Short-Term Memory)을 보다 오래(Long) 기억할 수 있도록 한다는 뜻으로 Long Short-Term Memory 라고 이름을 지음
    - 기본 구조    
        ![](./img/LSTM.jpg)
    - 전체 연산 과정
        - c는 장기 기억 상태, h는 단기 기억 상태라고 볼 수 있음.
        - forget gate를 통해서 0과 1 사이값을 얻어내는데 이전 기억 $c_{t-1}$ 을 얼마나 기억하고 가져갈지를 정해줌(0이면 완전히 이전 기억 삭제, 1이면 기존의 기억을 유지한채로 지나감)
        - input gate를 통해서 현재 입력과 이전 출력의 결과를 시그모이드와 tanh를 각각 통과시켜 0과 1사이 값과 -1과 1 사이값을 얻어내는데 현재 입력과 이전 출력의 결과를 얼마나 c에 저장할지(0과 1 사이값)와 어떤 정보를 c에 저장할지 후보(-1과 1 사이값)를 정함
        - output gate를 통해서 현재 입력과 이전 출력의 결과를 0과 1 사이값을 얻어내는데 장기기억부분의 정보인 $c_t$ 를 얼마나 꺼내쓸지를 정해줌    
        ( $c_t$ 에 tanh를 적용시켜 -1과 1 사이값으로 만들어주고 $c_t$에는 이미 현재 셀의 정보가 들어가 있기 때문에(input gate를 통해) 현재 입력과 이전 출력의 결과를 통해 얼마만큼 $c_t$에서 정보를 꺼내써야 좋을지를 output gate가 정함)    
        ![](./img/LSTM1.jpg)
        ![](./img/LSTM2.jpg)
        ![](./img/LSTM3.jpg)
        ![](./img/LSTM4.jpg)
        ![](./img/LSTM5.jpg)
    
- GRU
    - GRU는 LSTM의 모델 구조를 조금 더 경량화 해서 적은 메모리 요구량과 빠른 계산시간이 가능하도록 만든 모델
    - LSTM에서 두가지 종류의 벡터로 존재하던 cell state vector(장기기억)와 hidden state vector(단기기억)를 하나로 합쳐서 hidden state vector(cell state vector와 비슷한 역할을 함)만이 존재한다는것이 가장 큰 특징
    - 경량화를 진행했음에도 LSTM에 뒤지지 않고 비슷한 성능을 보여줌
    - 구조    
        ![](./img/GRU.jpg)

- LSTM, GRU 코드 실습
    - 데이터 전처리 (0 pad + 문장길이 정렬)
        ```python
        vocab_size = 100
        pad_id = 0

        data = [
        [85,14,80,34,99,20,31,65,53,86,3,58,30,4,11,6,50,71,74,13],
        [62,76,79,66,32],
        [93,77,16,67,46,74,24,70],
        [19,83,88,22,57,40,75,82,4,46],
        [70,28,30,24,76,84,92,76,77,51,7,20,82,94,57],
        [58,13,40,61,88,18,92,89,8,14,61,67,49,59,45,12,47,5],
        [22,5,21,84,39,6,9,84,36,59,32,30,69,70,82,56,1],
        [94,21,79,24,3,86],
        [80,80,33,63,34,63],
        [87,32,79,65,2,96,43,80,85,20,41,52,95,50,35,96,24,80]
        ]
        ```
        ```python
        max_len = len(max(data, key=len))
        print(f"Maximum sequence length: {max_len}")

        valid_lens = []
        for i, seq in enumerate(tqdm(data)):
            valid_lens.append(len(seq))
            if len(seq) < max_len:
                data[i] = seq + [pad_id] * (max_len - len(seq))
        ```
        ```python
        # B: batch size, L: maximum sequence length
        batch = torch.LongTensor(data)  # (B, L)
        batch_lens = torch.LongTensor(valid_lens)  # (B)

        batch_lens, sorted_idx = batch_lens.sort(descending=True)
        batch = batch[sorted_idx]

        print(batch)
        print(batch_lens)
        '''
        tensor([[85, 14, 80, 34, 99, 20, 31, 65, 53, 86,  3, 58, 30,  4, 11,  6, 50, 71,  74, 13],
                [58, 13, 40, 61, 88, 18, 92, 89,  8, 14, 61, 67, 49, 59, 45, 12, 47,  5,  0,  0],
                [87, 32, 79, 65,  2, 96, 43, 80, 85, 20, 41, 52, 95, 50, 35, 96, 24, 80,  0,  0],
                [22,  5, 21, 84, 39,  6,  9, 84, 36, 59, 32, 30, 69, 70, 82, 56,  1,  0,  0,  0],
                [70, 28, 30, 24, 76, 84, 92, 76, 77, 51,  7, 20, 82, 94, 57,  0,  0,  0,  0,  0],
                [19, 83, 88, 22, 57, 40, 75, 82,  4, 46,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [93, 77, 16, 67, 46, 74, 24, 70,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [94, 21, 79, 24,  3, 86,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [80, 80, 33, 63, 34, 63,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [62, 76, 79, 66, 32,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])        
        '''
        ```
    - LSTM 실습
        ```python
        embedding_size = 256
        hidden_size = 512
        num_layers = 1
        num_dirs = 1

        embedding = nn.Embedding(vocab_size, embedding_size)
        lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True if num_dirs > 1 else False
        )

        h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)
        c_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h)
        ```
        ```python
        # d_w: word embedding size
        batch_emb = embedding(batch)  # (B, L, d_w)

        packed_batch = pack_padded_sequence(batch_emb.transpose(0, 1), batch_lens)

        packed_outputs, (h_n, c_n) = lstm(packed_batch, (h_0, c_0))
        print(packed_outputs)
        print(packed_outputs[0].shape)
        print(h_n.shape)
        print(c_n.shape)
        '''
        PackedSequence(data=tensor([[ 0.0352,  0.0205,  0.0447,  ..., -0.2041,  0.0185,  0.1069],
                [-0.1200,  0.0170, -0.0072,  ...,  0.0777,  0.1415,  0.1520],
                [-0.0085,  0.0413, -0.0811,  ..., -0.0872, -0.0502, -0.0352],
                ...,
                [-0.0789,  0.0547, -0.0574,  ...,  0.1634, -0.0458,  0.2277],
                [-0.0331, -0.0151, -0.0688,  ...,  0.1258, -0.0205,  0.1043],
                [-0.1461, -0.0346, -0.0994,  ...,  0.0647, -0.0718,  0.0225]],
            grad_fn=<CatBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
                1,  1]), sorted_indices=None, unsorted_indices=None)
        torch.Size([123, 512])
        torch.Size([1, 10, 512])
        torch.Size([1, 10, 512])
        '''
        ```
        ```python
        outputs, output_lens = pad_packed_sequence(packed_outputs)
        print(outputs.shape)
        print(output_lens)
        '''
        torch.Size([20, 10, 512])
        tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])
        '''
        ```
    - GRU 실습
        - cell state가 없어 RNN과 동일하게 사용 가능
        - RNN 실습과 동일하므로 RNN에서는 하지 못했던 LM task 수행해보기(for loop 를 통해 하나의 입력을 계속 넣어주기)
            ```python
            gru = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True if num_dirs > 1 else False
            )
            output_layer = nn.Linear(hidden_size, vocab_size)
            ```
            ```python
            input_id = batch.transpose(0, 1)[0, :]  # (B) (전체 타임스텝중 제일 앞 타임스텝만 가져오기)
            hidden = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (1, B, d_h)
            ```
            ```python
            # teacher forcing 없이 이전에 얻은 output을 다음 input으로 이용하기
            for t in range(max_len):
                input_emb = embedding(input_id).unsqueeze(0)  # (1, B, d_w)
                output, hidden = gru(input_emb, hidden)  # output: (1, B, d_h), hidden: (1, B, d_h)

                # V: vocab size
                output = output_layer(output)  # (1, B, V)
                probs, top_id = torch.max(output, dim=-1)  # probs: (1, B), top_id: (1, B)
                
                # a = torch.LongTensor([[1,3,4],[5,1,0]])
                # print(torch.max(a,dim=-1))
                ## torch.return_types.max(values=tensor([4, 5]),indices=tensor([2, 0]))
                
                print("*" * 50)
                print(f"Time step: {t}")
                print(output.shape)
                print(probs.shape)
                print(top_id.shape)

                input_id = top_id.squeeze(0)  # (B)
            '''
            **************************************************
            Time step: 0
            torch.Size([1, 10, 100])
            torch.Size([1, 10])
            torch.Size([1, 10])
            **************************************************
            Time step: 1
            torch.Size([1, 10, 100])
            torch.Size([1, 10])
            torch.Size([1, 10])
            .
            .
            .
            **************************************************
            Time step: 19
            torch.Size([1, 10, 100])
            torch.Size([1, 10])
            torch.Size([1, 10])
            '''
            ```
            - 참고 (teacher forcing)
                - I want to go home 이러한 단어들을 순차적으로 생성하는 Language modeling을 할때 I를 넣어주면 want가 나오고 그 want를 다시 input에 넣어주고 이런식으로 진행을 함. 그런데 완전히 초기화 상태에서 학습을 시작하게 되면 I를 넣었을때 완전 이상한 값이 나올것이고 그 이상한 값을 다음 input으로 넣어 결국 계속 오류가 누적되게 됨(학습이 더디고 전체 결과가 안좋게 나옴)
                - 따라서 처음에는 이전에 생성된 단어가 아니라 ground truth 단어들을 기존학습 방식인 output을 input에 넣어주는 연결부분을 끊고 그냥 입력으로 넣어주게 됨(처음 학습할때만 이런식으로 output을 input으로 넣지 않고 정답을 input으로 넣어주기(cheating))
    - 양방향 및 여러 layer 사용 실습
        ```python
        num_layers = 2
        num_dirs = 2
        dropout=0.1

        gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True if num_dirs > 1 else False
        )
        ```
        ```python
        # d_w: word embedding size, num_layers: layer의 개수, num_dirs: 방향의 개수
        batch_emb = embedding(batch)  # (B, L, d_w)
        h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers * num_dirs, B, d_h) = (4, B, d_h)

        packed_batch = pack_padded_sequence(batch_emb.transpose(0, 1), batch_lens)

        packed_outputs, h_n = gru(packed_batch, h_0)
        print(packed_outputs)
        print(packed_outputs[0].shape)
        print(h_n.shape)
        '''
        PackedSequence(data=tensor([[-0.0102, -0.1831, -0.0607,  ...,  0.2596, -0.1109,  0.0306],
                [ 0.0233, -0.0525,  0.0753,  ..., -0.0415, -0.1773,  0.1906],
                [-0.0531,  0.0034, -0.0487,  ...,  0.0955,  0.0615,  0.2063],
                ...,
                [ 0.0157, -0.0067,  0.0083,  ...,  0.1532, -0.1595, -0.0090],
                [-0.0587, -0.0798, -0.0590,  ..., -0.0762,  0.0124, -0.1363],
                [-0.0633, -0.2451,  0.0214,  ...,  0.1049, -0.0192, -0.0980]],
            grad_fn=<CatBackward>), batch_sizes=tensor([10, 10, 10, 10, 10,  9,  7,  7,  6,  6,  5,  5,  5,  5,  5,  4,  4,  3,
                1,  1]), sorted_indices=None, unsorted_indices=None)
        torch.Size([123, 1024])
        torch.Size([4, 10, 512])
        '''
        ```
        ```python
        outputs, output_lens = pad_packed_sequence(packed_outputs)

        print(outputs.shape)  # (L, B, num_dirs*d_h)
        print(output_lens)
        '''
        torch.Size([20, 10, 1024])
        tensor([20, 18, 18, 17, 15, 10,  8,  6,  6,  5])
        '''
        ```
        ```python
        # h_n을 순방향 역방향 ,layer 별로 분리해주고 싶을 경우
        batch_size = h_n.shape[1]
        print(h_n.view(num_layers, num_dirs, batch_size, hidden_size))
        print(h_n.view(num_layers, num_dirs, batch_size, hidden_size).shape)
        '''
        torch.Size([2, 2, 10, 512])
        '''
        ```
        - output shape을 이해하기 위한 그림
            - bidirectional = True : 양방향을 concat 하기때문에 output의 shape이 2배가 됨    
                ![](./img/bidirectional.jpg)
            - multilayer : layer가 쌓이지만 output의 shape은 그대로 나오기 때문에 변함없고 마지막 단에 나오는 h_n 만 쌓인 layer만큼 증가하여 나오게 됨    
                ![](./img/multilayer.jpg)

        - outputs[-1]과 h_n은 전혀 다르다는 것을 명심해야함
            - outputs는 차원이 1024로 multilayer의 마지막 layer에서의 순방향과 역방향의 값이 concat되는것을 알수 있음.
            - 여기서 주의할점이 h_n[-1]이 multilayer마지막 단의 역방향, h_n[-2]가 순방향인데 이 둘이 concat되어서 outputs[-1]이 만들어진게 아닌가 하는 생각이 들수 있음. -> 하지만 여기서 h_n[-1]은 역방향이므로 outputs[0]에서 [512:] 값이랑 동일함. (역방향은 순방향 뒷부분에 concat 됨. 즉 [:512] 는 순방향 , [512:]는 역방향을 의미) h_n[-2]는 순방향이므로 outputs[-1][:512]랑 동일
            ![](./img/bidirectional_multilayer2.jpg)
            - 코드
                ```python
                # 1배치내의 첫번째 문장으로 테스트

                print(outputs[-1][0][:512][:5]) # 마지막 타임스텝
                print(h_n[-2][0][:512][:5]) # 순방향

                print(outputs[0][0][512:][:5]) # 첫번째 타임스텝
                print(h_n[-1][0][:512][:5]) # 역방향
                '''
                tensor([ 0.3047, -0.2413,  0.3590, -0.2214,  0.0414], grad_fn=<SliceBackward>)
                tensor([ 0.3047, -0.2413,  0.3590, -0.2214,  0.0414], grad_fn=<SliceBackward>)
                tensor([-0.2429, -0.2235, -0.0296, -0.1823, -0.0725], grad_fn=<SliceBackward>)
                tensor([-0.2429, -0.2235, -0.0296, -0.1823, -0.0725], grad_fn=<SliceBackward>)
                '''
                ```
            

- Backpropagation in LSTM/GRU
    - 전 타임스텝의 cell state vector에 현재 입력값에 따른 매번 다르게 나오는 forget gate 결과값을 곱해주게 되면 반복적인 연산이 아니게 되고, 현재 타임 스텝에서 필요로 하는 정보를 곱셈이 아닌 덧셈을 통해서 만들어주기 때문에 gradient vanishing/explosion 문제가 사라지게 됨 (original RNN 처럼 단순히 똑같은 행렬을 계속해서 곱해주는 연산이 아님)

#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)
- https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677
- https://github.com/pytorch/pytorch/issues/4930

---

## #5

### Sequence to Sequence with Attention
- Seq2Seq Model은 RNN 구조 중 many to many 형태에 해당함 (many to many 중에서도 입력 Sequence를 모두 다 읽은 후(Encoder) 그 다음에 출력 Sequence를 생성(Decoder하는 모델)    
    ![](./img/seq2seq.jpg)
- Are you free tomorrow? 라고 질문했을때 그에 대응하는 문장으로써 Yes, what`s up? 이라는 문장을 예측하여 생성하는 예시    
    ![](./img/seq2seq1.jpg)
    - start 토큰를 vocabulary 상에 정의해두고 디코더 타임스텝의 제일 처음에 넣어줌으로써 실직적인 단어 예측이 수행됨. 
    - end 토큰이 나올때까지 디코더 RNN을 구동하고 이 토큰이 실제로 생성되면 여기까지 최종적인 출력을 해서 더 이상 단어를 생성하지 않고 종료하도록 함.
- Attention
    - 인코더의 마지막 hidden state vector에 앞서 나온 모든 많은 정보들을 우겨넣게 되면 아무리 LSTM 등으로 의존도를 해결했다고 하더라도 마지막 타임스텝으로 갈수록 앞쪽 정보는 변질되거나 소실될 수 있음 -> 따라서 Attention 모듈을 추가로 사용하여 디코더에서 인코더의 마지막 타임스텝에서 나온 hidden state vector에만 의존하는것이 아니라 입력문장에서 주어졌던 각각의 단어들을 인코딩한 각각의 encoding hidden state vector를 선별적으로 가져가서 사용할수 있도록 만듦
        ![](./img/seq2seq2.gif)
        ![](./img/seq2seq3.gif)
    - 디코더에서 생성된 hidden state vector와 인코더 단의 각 워드별로의 hidden state vector 둘간의 score를 구할때는 여러가지 방법이 존재함
        - 일반적인 내적(dot), 좀더 확장된 generalized dot product, concat을 통한 연산 ( $h_t$ 는 디코더에서 주어지는 hidden state vector , $h_s$ 는 인코더 단에서 각 워드별로의 hidden state vector )
        - 일반적인 내적은 같은 차원에 있는 값들끼리만 곱해짐. -> 가운데에 행렬을 추가하여 같은 차원이 아니더라도 값에 가중치를 부여할수 있도록 해줌    
            ((1,3)과 (2,-5)롤 단순 내적하면 1과 2 끼리 연산, 3과 -5 끼리 연산, 즉 같은 차원끼리만 연산이 됨. 하지만 그 사이에 (1,3) ((a,b),(c,d)) (2,-5) 를 추가해주게 된다면 (1a+3c,1b+3d) (2,-5) -> (2(1a+3c),-5(1b+3d)) 로 다른 차원끼리도 가중치를 부여해줄수 있음
        - concat 방식은 두개의 hidden state vector를 concat하고 새로운 neural net을 만드는 방식임. $W_1$ 는 concat된 벡터를 특정 차원으로 만들어주고 $W_2$ 는 그 특정 차원을 하나의 score 값, 즉 scalar 값으로 만들어줌(그러므로 $W_1$ 은 행렬이지만 , $W_2$ 는 벡터가 됨)    
            ![](./img/seq2seq4.jpg)

- Seq2Seq 실습
    - src_data를 trg_data로 변환 train 후 src_sent 에 대해 test 결과 생성
    - import & data
        ```python
        from tqdm import tqdm
        from torch import nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        import torch
        import random
        ```
        ```python
        vocab_size = 100
        pad_id = 0 
        sos_id = 1 # start token
        eos_id = 2 # end token

        src_data = [
        [3, 77, 56, 26, 3, 55, 12, 36, 31],
        [58, 20, 65, 46, 26, 10, 76, 44],
        [58, 17, 8],
        [59],
        [29, 3, 52, 74, 73, 51, 39, 75, 19],
        [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93],
        [39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99, 5],
        [75, 34, 17, 3, 86, 88],
        [63, 39, 5, 35, 67, 56, 68, 89, 55, 66],
        [12, 40, 69, 39, 49]
        ]

        trg_data = [
        [75, 13, 22, 77, 89, 21, 13, 86, 95],
        [79, 14, 91, 41, 32, 79, 88, 34, 8, 68, 32, 77, 58, 7, 9, 87],
        [85, 8, 50, 30],
        [47, 30],
        [8, 85, 87, 77, 47, 21, 23, 98, 83, 4, 47, 97, 40, 43, 70, 8, 65, 71, 69, 88],
        [32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18],
        [37, 14, 49, 24, 93, 37, 54, 51, 39, 84],
        [16, 98, 68, 57, 55, 46, 66, 85, 18],
        [20, 70, 14, 6, 58, 90, 30, 17, 91, 18, 90],
        [37, 93, 98, 13, 45, 28, 89, 72, 70]
        ]
        ```
    - 데이터 전처리
        ```python
        trg_data = [[sos_id]+seq+[eos_id] for seq in tqdm(trg_data)]

        def padding(data, is_src=True):
            max_len = len(max(data, key=len))
            print(f"Maximum sequence length: {max_len}")

            valid_lens = []
            for i, seq in enumerate(tqdm(data)):
                valid_lens.append(len(seq))
                if len(seq) < max_len:
                    data[i] = seq + [pad_id] * (max_len - len(seq))

            return data, valid_lens, max_len

        src_data, src_lens, src_max_len = padding(src_data)
        trg_data, trg_lens, trg_max_len = padding(trg_data)
        '''
        Maximum sequence length: 15
        Maximum sequence length: 22
        '''
        ```
        ```python
        # B: batch size, S_L: source maximum sequence length, T_L: target maximum sequence length
        src_batch = torch.LongTensor(src_data)  # (B, S_L)
        src_batch_lens = torch.LongTensor(src_lens)  # (B)
        trg_batch = torch.LongTensor(trg_data)  # (B, T_L)
        trg_batch_lens = torch.LongTensor(trg_lens)  # (B)

        print(src_batch.shape)
        print(src_batch_lens.shape)
        print(trg_batch.shape)
        print(trg_batch_lens.shape)
        '''
        torch.Size([10, 15])
        torch.Size([10])
        torch.Size([10, 22])
        torch.Size([10])
        '''
        ```
        ```python
        # PackedSquence를 사용을 위해 source data를 기준으로 정렬.

        src_batch_lens, sorted_idx = src_batch_lens.sort(descending=True)
        src_batch = src_batch[sorted_idx]
        trg_batch = trg_batch[sorted_idx]
        trg_batch_lens = trg_batch_lens[sorted_idx]

        print(src_batch)
        print(src_batch_lens)
        print(trg_batch)
        print(trg_batch_lens)
        '''
        tensor([[39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99,  5],
                [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93,  0,  0,  0,  0],
                [63, 39,  5, 35, 67, 56, 68, 89, 55, 66,  0,  0,  0,  0,  0],
                [ 3, 77, 56, 26,  3, 55, 12, 36, 31,  0,  0,  0,  0,  0,  0],
                [29,  3, 52, 74, 73, 51, 39, 75, 19,  0,  0,  0,  0,  0,  0],
                [58, 20, 65, 46, 26, 10, 76, 44,  0,  0,  0,  0,  0,  0,  0],
                [75, 34, 17,  3, 86, 88,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [12, 40, 69, 39, 49,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [58, 17,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [59,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        tensor([15, 11, 10,  9,  9,  8,  6,  5,  3,  1])
        tensor([[ 1, 37, 14, 49, 24, 93, 37, 54, 51, 39, 84,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1, 32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18,  2,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1, 20, 70, 14,  6, 58, 90, 30, 17, 91, 18, 90,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1, 75, 13, 22, 77, 89, 21, 13, 86, 95,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  8, 85, 87, 77, 47, 21, 23, 98, 83,  4, 47, 97, 40, 43, 70,  8, 65,  71, 69, 88,  2],
                [ 1, 79, 14, 91, 41, 32, 79, 88, 34,  8, 68, 32, 77, 58,  7,  9, 87,  2,  0,  0,  0,  0],
                [ 1, 16, 98, 68, 57, 55, 46, 66, 85, 18,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1, 37, 93, 98, 13, 45, 28, 89, 72, 70,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1, 85,  8, 50, 30,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                [ 1, 47, 30,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        tensor([12, 14, 13, 11, 22, 18, 11, 11,  6,  4])
        '''
        ```
    - Encoder 구현
        ```python
        embedding_size = 256
        hidden_size = 512
        num_layers = 2
        num_dirs = 2
        dropout = 0.1

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()

                self.embedding = nn.Embedding(vocab_size, embedding_size)
                self.gru = nn.GRU(
                    input_size=embedding_size, 
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=True if num_dirs > 1 else False,
                    dropout=dropout
                )
                self.linear = nn.Linear(num_dirs * hidden_size, hidden_size)

            def forward(self, batch, batch_lens):  # batch: (B, S_L), batch_lens: (B)
                # d_w: word embedding size
                batch_emb = self.embedding(batch)  # (B, S_L, d_w)
                batch_emb = batch_emb.transpose(0, 1)  # (S_L, B, d_w)

                packed_input = pack_padded_sequence(batch_emb, batch_lens)

                h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers*num_dirs, B, d_h) = (4, B, d_h)
                packed_outputs, h_n = self.gru(packed_input, h_0)  # h_n: (4, B, d_h)
                outputs = pad_packed_sequence(packed_outputs)[0]  # outputs: (S_L, B, 2d_h)

                forward_hidden = h_n[-2, :, :]
                backward_hidden = h_n[-1, :, :]
                hidden = self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1)).unsqueeze(0)  # (1, B, d_h) # sequence,batch,feature 순이기 때문에 unsqueeze 필요

                return outputs, hidden

        encoder = Encoder()
        ```
    - Decoder 구현
        ```python
        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()

                self.embedding = nn.Embedding(vocab_size, embedding_size)
                self.gru = nn.GRU(
                    input_size=embedding_size, 
                    hidden_size=hidden_size,
                )
                self.output_layer = nn.Linear(hidden_size, vocab_size)

            def forward(self, batch, hidden):  # batch: (B), hidden: (1, B, d_h)
                batch_emb = self.embedding(batch)  # (B, d_w)
                batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w) # sequence,batch,feature 순이기 때문에 unsqueeze 필요

                outputs, hidden = self.gru(batch_emb, hidden)  # outputs: (1, B, d_h), hidden: (1, B, d_h)

                # V: vocab size
                outputs = self.output_layer(outputs)  # (1, B, V)

                return outputs.squeeze(0), hidden
        
        decoder = Decoder()
        ```

    - Seq2Seq 모델 구축
        ```python
        class Seq2seq(nn.Module):
            def __init__(self, encoder, decoder):
                super(Seq2seq, self).__init__()

                self.encoder = encoder
                self.decoder = decoder

            def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
                # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L)

                _, hidden = self.encoder(src_batch, src_batch_lens)  # hidden: (1, B, d_h)

                input_ids = trg_batch[:, 0]  # (B) # 첫번째 타임스텝 (원래는 start token 이지만 여기서는 매번 다른값이 들어감)
                batch_size = src_batch.shape[0]
                outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

                for t in range(1, trg_max_len): # 결과는 trg_max_len - 1 짜리 길이로 나오게됨(첫번째는 입력으로만들어가기 때문), 여기서는 outputs[1:] 가 진짜 결과임
                    decoder_outputs, hidden = self.decoder(input_ids, hidden)  # decoder_outputs: (B, V), hidden: (1, B, d_h)

                    outputs[t] = decoder_outputs
                    _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)
                    # index값이 즉 id값임 , decoder 의 output의 dimension이 vocab size로 되어있기때문 
                    
                    input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

                return outputs
        
        seq2seq = Seq2seq(encoder, decoder)
        ```

    - 모델 학습과정 살펴보기
        ```python
        # 학습 과정이라고 생각하고 모델에 input 넣기
        outputs = seq2seq(src_batch, src_batch_lens, trg_batch)

        print(outputs.shape)
        '''
        torch.Size([22, 10, 100])
        '''
        ```
        ![](./img/seq2seq5.jpg)
        ```python
        loss_function = nn.CrossEntropyLoss()

        preds = outputs[1:, :, :].transpose(0, 1)  # (B, T_L-1, V)
        loss = loss_function(preds.reshape(-1, vocab_size), trg_batch[:,1:].reshape(-1,1).squeeze(1)) # trg_batch -> (10,20)
        # cross entropy 는 target이 one-hot vector로 표현되어져 있지 않더라도 알아서 one-hot vector로 변경시켜서 pred과 비교해줌
        # cross entropy 는 분류 문제에서 사용. 숫자간의 연관성이 없음 (class 10과 class 40은 숫자가 크다고 더 좋고 그런것이 없음)
        # 회귀문제(회귀문제는 숫자가 의미가 있을경우 사용함. 숫자 10보다 40이 큰것이 더 좋을경우 회귀로 생각)와 달리 one-hot encoding이나 embedding 된 경우가 많음
        # trg_batch[:,1:].reshape(-1,1).squeeze(1) 를 trg_batch[:,1:].reshape(-1)로 작성해도 됨(단지 preds.reshape(-1,vocab_size) 와 통일시켜주기 위함)
        print(loss)
        '''
        tensor(4.5950, grad_fn=<NllLossBackward>)
        '''
        ```
    - 모델 추론과정 살펴보기
        - 실제 inference 에선 teacher forcing 없이 이전 결과만을 가지고 생성함
        ```python
        src_sent = [4, 10, 88, 46, 72, 34, 14, 51]
        src_len = len(src_sent)

        src_batch = torch.LongTensor(src_sent).unsqueeze(0)  # (1, L)
        src_batch_lens = torch.LongTensor([src_len])  # (1)

        _, hidden = seq2seq.encoder(src_batch, src_batch_lens)  # hidden: (1, 1, d_h)

        input_id = torch.LongTensor([sos_id]) # (1)
        output = []

        for t in range(1, trg_max_len):
            decoder_output, hidden = seq2seq.decoder(input_id, hidden)  # decoder_output: (1, V), hidden: (1, 1, d_h)

            _, top_id = torch.max(decoder_output, dim=-1)  # top_ids: (1)

            if top_id == eos_id:
                break
            else:
                output += top_id.tolist()
                input_id = top_id
        
        print(output)
        '''
        [21, 77, 84, 89, 1, 21, 77, 84, 89, 1, 21, 77, 84, 89, 1, 21, 77, 84, 89, 1, 21]
        '''
        ```
- Seq2Seq with Attention 실습
    - import & data
        ```python
        from tqdm import tqdm
        from torch import nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from torch.nn import functional as F

        import torch
        import random
        ```
        ```python
        vocab_size = 100
        pad_id = 0
        sos_id = 1
        eos_id = 2

        src_data = [
        [3, 77, 56, 26, 3, 55, 12, 36, 31],
        [58, 20, 65, 46, 26, 10, 76, 44],
        [58, 17, 8],
        [59],
        [29, 3, 52, 74, 73, 51, 39, 75, 19],
        [41, 55, 77, 21, 52, 92, 97, 69, 54, 14, 93],
        [39, 47, 96, 68, 55, 16, 90, 45, 89, 84, 19, 22, 32, 99, 5],
        [75, 34, 17, 3, 86, 88],
        [63, 39, 5, 35, 67, 56, 68, 89, 55, 66],
        [12, 40, 69, 39, 49]
        ]

        trg_data = [
        [75, 13, 22, 77, 89, 21, 13, 86, 95],
        [79, 14, 91, 41, 32, 79, 88, 34, 8, 68, 32, 77, 58, 7, 9, 87],
        [85, 8, 50, 30],
        [47, 30],
        [8, 85, 87, 77, 47, 21, 23, 98, 83, 4, 47, 97, 40, 43, 70, 8, 65, 71, 69, 88],
        [32, 37, 31, 77, 38, 93, 45, 74, 47, 54, 31, 18],
        [37, 14, 49, 24, 93, 37, 54, 51, 39, 84],
        [16, 98, 68, 57, 55, 46, 66, 85, 18],
        [20, 70, 14, 6, 58, 90, 30, 17, 91, 18, 90],
        [37, 93, 98, 13, 45, 28, 89, 72, 70]
        ]
        ```
    - 데이터 전처리
        - seq2seq 실습과 동일
        ```python
        trg_data = [[sos_id]+seq+[eos_id] for seq in tqdm(trg_data)]

        def padding(data, is_src=True):
            max_len = len(max(data, key=len))
            print(f"Maximum sequence length: {max_len}")

            valid_lens = []
            for i, seq in enumerate(tqdm(data)):
                valid_lens.append(len(seq))
                if len(seq) < max_len:
                    data[i] = seq + [pad_id] * (max_len - len(seq))

            return data, valid_lens, max_len

        src_data, src_lens, src_max_len = padding(src_data)
        trg_data, trg_lens, trg_max_len = padding(trg_data)
        '''
        Maximum sequence length: 15
        Maximum sequence length: 22
        '''
        ```
        ```python
        # B: batch size, S_L: source maximum sequence length, T_L: target maximum sequence length
        src_batch = torch.LongTensor(src_data)  # (B, S_L)
        src_batch_lens = torch.LongTensor(src_lens)  # (B)
        trg_batch = torch.LongTensor(trg_data)  # (B, T_L)
        trg_batch_lens = torch.LongTensor(trg_lens)  # (B)

        print(src_batch.shape)
        print(src_batch_lens.shape)
        print(trg_batch.shape)
        print(trg_batch_lens.shape)
        '''
        torch.Size([10, 15])
        torch.Size([10])
        torch.Size([10, 22])
        torch.Size([10])
        '''
        ```
        ```python
        src_batch_lens, sorted_idx = src_batch_lens.sort(descending=True)
        src_batch = src_batch[sorted_idx]
        trg_batch = trg_batch[sorted_idx]
        trg_batch_lens = trg_batch_lens[sorted_idx]

        print(src_batch)
        print(src_batch_lens)
        print(trg_batch)
        print(trg_batch_lens)
        ```
    - Encoder
        - Encoder 역시 기존 Seq2Seq 모델과 동일
        ```python
        embedding_size = 256
        hidden_size = 512
        num_layers = 2
        num_dirs = 2
        dropout = 0.1

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()

                self.embedding = nn.Embedding(vocab_size, embedding_size)
                self.gru = nn.GRU(
                    input_size=embedding_size, 
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=True if num_dirs > 1 else False,
                    dropout=dropout
                )
                self.linear = nn.Linear(num_dirs * hidden_size, hidden_size)

            def forward(self, batch, batch_lens):  # batch: (B, S_L), batch_lens: (B)
                # d_w: word embedding size
                batch_emb = self.embedding(batch)  # (B, S_L, d_w)
                batch_emb = batch_emb.transpose(0, 1)  # (S_L, B, d_w)

                packed_input = pack_padded_sequence(batch_emb, batch_lens)

                h_0 = torch.zeros((num_layers * num_dirs, batch.shape[0], hidden_size))  # (num_layers*num_dirs, B, d_h) = (4, B, d_h)
                packed_outputs, h_n = self.gru(packed_input, h_0)  # h_n: (4, B, d_h)
                outputs = pad_packed_sequence(packed_outputs)[0]  # outputs: (S_L, B, 2d_h)
                outputs = torch.tanh(self.linear(outputs))  # (S_L, B, d_h)

                forward_hidden = h_n[-2, :, :]
                backward_hidden = h_n[-1, :, :]
                hidden = torch.tanh(self.linear(torch.cat((forward_hidden, backward_hidden), dim=-1))).unsqueeze(0)  # (1, B, d_h)

                return outputs, hidden
        
        encoder = Encoder()
        ```
    - Dot-product Attention 구현
        ```python
        class DotAttention(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h) ,decoder_hidden 은 1개의 time step 이므로 (1,B,d_h) 임
                query = decoder_hidden.transpose(0,1)  # (B, 1, d_h) 
                key = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

                energy = torch.sum(torch.mul(key, query), dim=-1)  # (B, S_L) , mul 은 element-wise 곱셈(차원이 다를 경우 브로드캐스팅된 후에 곱셈이 수행됨)
                
                
                attn_scores = F.softmax(energy, dim=-1)  # (B, S_L)
                attn_values = torch.sum(torch.mul(encoder_outputs.transpose(0, 1), attn_scores.unsqueeze(2)), dim=1)  # (B, d_h)

                return attn_values, attn_scores

        dot_attn = DotAttention()
        ```
    - Decoder 구현
        ```python
        class Decoder(nn.Module):
            def __init__(self, attention):
                super().__init__()

                self.embedding = nn.Embedding(vocab_size, embedding_size)
                self.attention = attention
                self.rnn = nn.GRU(
                    embedding_size,
                    hidden_size
                )
                self.output_linear = nn.Linear(2*hidden_size, vocab_size)

            def forward(self, batch, encoder_outputs, hidden):  # batch: (B), encoder_outputs: (L, B, d_h), hidden: (1, B, d_h)  
                # batch는 하나의 타임스텝에 대한것이기때문에 (1,B)에서 (B)만 남긴것
                batch_emb = self.embedding(batch)  # (B, d_w)
                batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

                outputs, hidden = self.rnn(batch_emb, hidden)  # (1, B, d_h), (1, B, d_h)

                attn_values, attn_scores = self.attention(hidden, encoder_outputs)  # (B, d_h), (B, S_L)
                concat_outputs = torch.cat((outputs, attn_values.unsqueeze(0)), dim=-1)  # (1, B, 2d_h)

                return self.output_linear(concat_outputs).squeeze(0), hidden  # (B, V), (1, B, d_h)

        decoder = Decoder(dot_attn)
        ```
    - Seq2Seq with attention 모델 구축
        ```python
        class Seq2seq(nn.Module):
            def __init__(self, encoder, decoder):
                super(Seq2seq, self).__init__()

                self.encoder = encoder
                self.decoder = decoder

            def forward(self, src_batch, src_batch_lens, trg_batch, teacher_forcing_prob=0.5):
                # src_batch: (B, S_L), src_batch_lens: (B), trg_batch: (B, T_L)

                encoder_outputs, hidden = self.encoder(src_batch, src_batch_lens)  # encoder_outputs: (S_L, B, d_h), hidden: (1, B, d_h)

                input_ids = trg_batch[:, 0]  # (B)
                batch_size = src_batch.shape[0]
                outputs = torch.zeros(trg_max_len, batch_size, vocab_size)  # (T_L, B, V)

                for t in range(1, trg_max_len):
                    decoder_outputs, hidden = self.decoder(input_ids, encoder_outputs, hidden)  # decoder_outputs: (B, V), hidden: (1, B, d_h)

                    outputs[t] = decoder_outputs
                    _, top_ids = torch.max(decoder_outputs, dim=-1)  # top_ids: (B)

                    input_ids = trg_batch[:, t] if random.random() > teacher_forcing_prob else top_ids

                return outputs
        seq2seq = Seq2seq(encoder, decoder)
        ```
    - 모델 학습해보기
        ```python
        # V: vocab size
        outputs = seq2seq(src_batch, src_batch_lens, trg_batch)  # (T_L, B, V)

        print(outputs.shape)
        '''
        torch.Size([22, 10, 100])
        '''
        ```
        ```python
        loss_function = nn.CrossEntropyLoss()

        preds = outputs[1:, :, :].transpose(0, 1)  # (B, T_L-1, V)
        loss = loss_function(preds.reshape(-1, vocab_size), trg_batch[:,1:].reshape(-1,1).squeeze(1)) # trg_batch -> (10,20)

        print(loss)
        '''
        tensor(4.6156, grad_fn=<NllLossBackward>)
        '''
        ```
    - 모델 추론하기
        ```python
        sample_sent = [4, 10, 88, 46, 72, 34, 14, 51]
        sample_len = len(sample_sent)

        sample_batch = torch.LongTensor(sample_sent).unsqueeze(0)  # (1, L)
        sample_batch_len = torch.LongTensor([sample_len])  # (1)

        encoder_output, hidden = seq2seq.encoder(sample_batch, sample_batch_len)  # hidden: (4, 1, d_h)
        ```
        ```python
        input_id = torch.LongTensor([sos_id]) # (1)
        output = []

        for t in range(1, trg_max_len):
            decoder_output, hidden = seq2seq.decoder(input_id, encoder_output, hidden)  # decoder_output: (1, V), hidden: (4, 1, d_h)

            _, top_id = torch.max(decoder_output, dim=-1)  # top_ids: (1)

            if top_id == eos_id:
                break
            else:
                output += top_id.tolist()
                input_id = top_id
        
        print(output)
        '''
        [93, 87, 42, 42, 42, 42, 42, 25, 86, 16, 1, 67, 16, 11, 80, 9, 99, 86, 16, 11, 80]
        '''
        ```
    - Concat Attention(Bahdanau Attention)
        - Attetion
            ```python
            class ConcatAttention(nn.Module):
                def __init__(self):
                    super().__init__()

                    self.w = nn.Linear(2*hidden_size, hidden_size, bias=False)
                    self.v = nn.Linear(hidden_size, 1, bias=False)

                def forward(self, decoder_hidden, encoder_outputs):  # (1, B, d_h), (S_L, B, d_h)
                    src_max_len = encoder_outputs.shape[0]
                    
                    # x = torch.tensor([[1],[2]])
                    # print(x.repeat(2,3,4))
                    ## tensor([[[1, 1, 1, 1],
                    ## [2, 2, 2, 2],
                    ## [1, 1, 1, 1],
                    ## [2, 2, 2, 2],
                    ## [1, 1, 1, 1],
                    ## [2, 2, 2, 2]],

                    ##[[1, 1, 1, 1],
                    ## [2, 2, 2, 2],
                    ## [1, 1, 1, 1],
                    ## [2, 2, 2, 2],
                    ## [1, 1, 1, 1],
                    ## [2, 2, 2, 2]]])
                    
                    decoder_hidden = decoder_hidden.transpose(0, 1).repeat(1, src_max_len, 1)  # (B, S_L, d_h)
                    encoder_outputs = encoder_outputs.transpose(0, 1)  # (B, S_L, d_h)

                    concat_hiddens = torch.cat((decoder_hidden, encoder_outputs), dim=2)  # (B, S_L, 2d_h)
                    energy = torch.tanh(self.w(concat_hiddens))  # (B, S_L, d_h)

                    attn_scores = F.softmax(self.v(energy), dim=1)  # (B, S_L, 1)
                    attn_values = torch.sum(torch.mul(encoder_outputs, attn_scores), dim=1)  # (B, d_h)

                    return attn_values, attn_scores
            concat_attn = ConcatAttention()
            ```
        - decoder
            ```python
            class Decoder(nn.Module):
                def __init__(self, attention):
                    super().__init__()

                    self.embedding = nn.Embedding(vocab_size, embedding_size)
                    self.attention = attention
                    self.rnn = nn.GRU(
                        embedding_size + hidden_size,
                        hidden_size
                    )
                    self.output_linear = nn.Linear(hidden_size, vocab_size)

                def forward(self, batch, encoder_outputs, hidden):  # batch: (B), encoder_outputs: (S_L, B, d_h), hidden: (1, B, d_h)  
                    batch_emb = self.embedding(batch)  # (B, d_w)
                    batch_emb = batch_emb.unsqueeze(0)  # (1, B, d_w)

                    attn_values, attn_scores = self.attention(hidden, encoder_outputs)  # (B, d_h), (B, S_L)

                    concat_emb = torch.cat((batch_emb, attn_values.unsqueeze(0)), dim=-1)  # (1, B, d_w+d_h)

                    outputs, hidden = self.rnn(concat_emb, hidden)  # (1, B, d_h), (1, B, d_h)

                    return self.output_linear(outputs).squeeze(0), hidden  # (B, V), (1, B, d_h)
            decoder = Decoder(concat_attn)
            ```
        - 코드 실행
            ```python
            seq2seq = Seq2seq(encoder, decoder)
            outputs = seq2seq(src_batch, src_batch_lens, trg_batch)

            print(outputs.shape)
            '''
            torch.Size([22, 10, 100])
            '''
            ```

#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)

---

## #6

### Beam Search
- 문장을 decoding 하는데에 사용하는 대표적인 알고리즘
- k라는 beam size를 가지고 타임스텝마다 스코어가 가장 높은 k개의 candidate를 우리가 고려하고 추적하는 것이 beam search의 핵심 아이디어    
( $k^T$ 가 아니라 매 타임스텝마다 그중에서 가장 높은 k개의 candidate만 뽑아가는것(= 계속 k개의 후보만 존재))
- Beam size가 2일 경우 예시    
    ![](./img/beamsearch.gif)

#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)

---

## #7

### Transformer
- Transformer : Attention만을 Sequence 데이터를 입력으로 받고 Sequence 형태의 데이터를 예측할 수 있는 모델 구조
- 모델 구조 이미지
    ![](./img/Transformer.jpg)
- Positional Encoder
    ![](./img/Transformer1.jpg)
    ![](./img/Transformer2.jpg)
    - 코드
        ```python
        class PositionalEncoding(nn.Module):
            """
            compute sinusoid encoding.
            """
            def __init__(self, d_model, max_len, device):
                """
                constructor of sinusoid encoding class

                :param d_model: dimension of model
                :param max_len: max sequence length
                :param device: hardware device setting
                """
                super(PositionalEncoding, self).__init__()

                # same size with input matrix (for adding with input matrix)
                self.encoding = torch.zeros(max_len, d_model, device=device)
                self.encoding.requires_grad = False  # we don't need to compute gradient

                pos = torch.arange(0, max_len, device=device)
                pos = pos.float().unsqueeze(dim=1)
                # 1D => 2D unsqueeze to represent word's position

                _2i = torch.arange(0, d_model, step=2, device=device).float()
                # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
                # "step=2" means 'i' multiplied with two (same with 2 * i)

                self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
                self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
                # compute positional encoding to consider positional information of words

            def forward(self, x):
                # self.encoding
                # [max_len = 512, d_model = 512]

                batch_size, seq_len = x.size()
                # [batch_size = 128, seq_len = 30]

                return self.encoding[:seq_len, :]
                # [seq_len = 30, d_model = 512]
                # it will add with tok_emb : [128, 30, 512]  
        ```

#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)
- https://github.com/hyunwoongko/transformer