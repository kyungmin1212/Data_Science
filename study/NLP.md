## Table of Contents

- [](#1)

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
- NaiveBayes Classifier : Bage-of-words를 활용한 대표적인 문서 분류 기법
    - 이론    
        <img src="./img/NaiveBayes_Classifier.jpg" width="50%" height="50%">
    - 예시    
        <img src="./img/NaiveBayes_Classifier1.jpg" width="100%" height="100%">
        <img src="./img/NaiveBayes_Classifier2.jpg" width="100%" height="100%">

#### References
- [boostcamp AI Tech](https://boostcamp.connect.or.kr/program_ai.html)