## Table of Contents

- [LabelEncoder](#1)

---

## #1

### LabelEncoder
- 예시 코드
    ```python
    from sklearn.preprocessing import LabelEncoder

    data = ['A','B','C']

    le = LabelEncoder()
    le = le.fit(data)
    print(le.transform(['A','B','C']))
    print(le.inverse_transform([0,1,2]))
    '''
    [0 1 2]
    ['A' 'B' 'C']
    '''
    ```

#### References

---