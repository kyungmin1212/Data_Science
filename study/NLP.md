## Table of Contents

- [Bag-of-words 와 NaiveBayes Classifier](#1)
- [Word Embedding(Word2Vec, Glove)(cf. CBOW, Skip-gram)](#2)

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
        <img src="./img/Bag-of-words.jpg" width="70%" height="70%">
- NaiveBayes Classifier : Bag-of-words를 활용한 대표적인 문서 분류 기법
    - 이론    
        <img src="./img/NaiveBayes_Classifier.jpg" width="70%" height="70%">
    - 예시    
        <img src="./img/NaiveBayes_Classifier1.jpg" width="50%" height="50%">
        <img src="./img/NaiveBayes_Classifier2.jpg" width="50%" height="50%">

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
        <img src="./img/Word2Vec.jpg" width="70%" height="70%">
        (우리는 이 중에서 input 단어가 들어갔을때 W1을 거쳐 hidden layer에 존재하는 벡터를 임베딩 벡터로 사용)
    - Property of Word2Vec
        - 단어들 사이에 관계를 표현해보면 비슷한 관계는 같은 벡터를 가짐
            <img src="./img/Word2Vec1.jpg" width="70%" height="70%">
- GloVe
    - Word2Vec과 더불어 많이 쓰이는 워드 임베딩 방법
    - GloVe는 어떠한 단어쌍이 동시에 등장하는 횟수를 미리 계산해서 중복되는 계산을 줄여줄수 있다는 장점이 존재하여 상대적으로 더 빠르고 더 적은 데이터에 대해서도 더 잘 동작하는 특성을 보임
    - Glove 과정
        - 각 입출력 단어 쌍들에 대해 그 학습 데이터에서 그 두 단어가 한 윈도우 내에서 총 몇 번 동시에 등장했는지를 사전에 미리 계산 $P_{ij}$
        - 입력워드의 임베딩 벡터 $u_i$ 와 출력워드의 임베딩 벡터 $v_j$의 내적값이 한 윈도우 안에서 두 단어가 동시에 나타난 횟수인 $P_{ij}$에 가까워질 수 있도록 학습 진행함
    - Glove 수식    
        <img src="./img/glove.jpg" width="50%" height="50%">

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