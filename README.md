# Torchtext Tutorial

- torchtext == 0.11.0

---

## 본 레포지토리는 "PyTorch로 시작하는 딥러닝 입문"의 "토치텍스트 튜토리얼"을 참조하였습니다.

- Reference) https://wikidocs.net/60314

---

## 0. 토치텍스트(torchtext)란?

- 토치텍스트(torchtext)는 텍스트에 대해 여러 추상화 기능을 제공하는 자연어 처리 라이브러리입니다.

---

## 1. 토치텍스트(torchtext)의 기능

- 파일 로드하기(File Loading) : 다양한 형태의 코퍼스를 로드합니다.
- 토큰화(Tokenization) : 문장을 단어 단위로 분리해줍니다.
- 단어 집합(Vocab) : 단어 집합을 만들어줍니다.
- 정수 인코딩(Integer Encoding) : 전체 코퍼스의 단어들을 각각의 고유한 정수로 맵핑합니다.
- 단어 벡터(Word Vector) : 단어 집합들의 단어들에 고유한 임베딩 벡터를 만들어 줍니다. 랜덤값으로 초기화한 값일 수 도 있고, 사전 훈련된 임베딩 벡터들을 로드할수도 있습니다.
- 배치화(Batching) : 훈련 샘플들의 배치를 만들어 줍니다. + Padding 작업도 같이 이루어 집니다.

---

## 2. Train/Test 데이터로 분리하기

### 2-0. 라이브러리 임포트

- 본 Torchtext Tutorial에서는 IMDB리뷰 데이터를 통해 튜토리얼을 진행합니다.

```
import urllib.request
import pandas as pd
```

### 2-1. 데이터 다운로드

```
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
```

### 2-2. 데이터 불러오기 & Train/Test 데이터 분리하기

```
df=pd.read_csv('IMDb_Reviews.csv',encoding='latin1') # 전체 샘플 갯수 50000개
train=df[:25000] # 훈련 샘플 25000개
test=df[25000:] # 테스트 샘플 25000개
train.to_csv('./data/train_data.csv',index=False)
test.to_csv('./data/test_data.csv',index=False)
```

---

## 3. torchtext를 활용하여 필드(Field) 정의하기

### 3-0. 라이브러리 임포트

```
from torchtext.legacy import data
```

### 3-1. 필드(Field) 정의하기

- sequential : 시퀀스 데이터 여부 (Default == True)
- use_vocab : 단어집 생성 여부 (Default == True)
- tokenize : 사용할 토큰화 함수 (Default == str.split)
- lower : 영어 시퀀스 소문자 여부 (Default == False)
- batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올건지 여부 (Default == False)
- is_target : 레이블 데이터 여부 (Default == False)
- fix_length : 시퀀스 최대 허용 길이, 이 길이에 따라서 패딩(padding) 작업이 진행된다.

```
TEXT=data.Field(
    sequential=True,
    use_vocab=True,
    tokenize=str.split,
    lower=True,
    batch_first=True,
    is_target=False,
    fix_length=20
)

LABEL=data.Field(
    sequential=False,
    use_vocab=False,
    batch_first=False,
    is_target=True
)
```

---

## 4. 데이터셋 만들기

### 4-0. 라이브러리 임포트

```
from torchtext.legacy.data import TabularDataset
```

### 4-1. 데이터셋 만들기

- path : 데이터 파일 경로
- train/test : train/test 파일 이름
- format : 데이터 포맷 (csv,tsv,json)
- fields : 필드 정의
- skip_header : 데이터의 첫번째 줄 생략 (csv 파일의 경우 첫번째 줄은 컬럼명)

```
train_data,test_data=Tabular.split(
    path='./data',
    train='train_data.csv',
    test='test_data.csv',
    format='csv',
    fields=[('text',TEXT),('label',LABEL)],
    skip_header=True
)
```

### 4-2. 추가적인 기능

- vars() : 주어진 인덱스의 샘플 확인 가능
- TabularDataset.fields.items() : field 구성 확인

```
print(vars(train_data[0]))
print(train_data.fields.items())

###
{'text': ['my', 'family', 'and', 'i', 'normally', 'do', ... ,'and', 'claudine!!'], 'label': '1'}
dict_items([('text', <torchtext.legacy.data.field.Field object at 0x7fe4add8f090>), ('label', <torchtext.legacy.data.field.Field object at 0x7fe4add8f050>)])
###
```

---

## 5. 단어 집합 만들기

- 토큰화 작업이 끝난 후에는 단어를 고유한 정수로 맵핑해주는 정수 인코딩(Integer Encoding)작업이 필요합니다. 이를 위해서는 단어 집합(vocab)이 필요합니다.
- `<Field Name>.build_vocab(dataset,min_freq,max_size)`
- min_freq : 최소 등장 빈도 조건 , max_size : 단어 집합의 최대 크기

### 5-1. 단어 집합 생성

```
TEXT.build_vocab(train_data,min_freq=10,max_size=10000)
```

### 5-2. 단어 집합 확인

- 단어 집합의 크기는 max_size로 지정한 10000보다 2개가 많게 나온다 그 이유는 unk,pad 와 같은 special token이 추가되어 있기 때문입니다.

```
print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
### 10002
```

- vocab을 확인하기 위해서는 stoi를 사용하여 확인할 수 있습니다.

```
print(TEXT.vocab.stoi)
```

---

## 6. torchtext 데이터로더 만들기

- 데이터로더는 데이터셋에서 미니 배치만큼 데이터를 로드하는 역할을 합니다.

### 6-0. 라이브러리 임포트

```
from torchtext.legacy.data import Iterator
```

### 6-1. 데이터로더 생성

```
batch_size=10
train_loader=Iterator(dataset=train_data,batch_size=batch_size)
test_loader=Iterator(dataset=test_data,batch_size=batch_size)
```

### 6-2. 데이터로더 확인

- 데이터로더의 미니배치의 갯수

```
print('훈련 데이터 미니배치의 갯수 : {}'.format(len(train_loader)))
print('테스트 데이터 미니배치의 갯수 : {}'.format(len(test_loader)))

###
훈련 데이터의 미니배치의 갯수 : 2500
테스트 데이터의 미니배치의 갯수 : 2500
###
```

- 데이터로더에서 미니배치 하나씩 꺼내보는 방법 : next,iter 사용

```
batch=next(iter(train_loader))
```

- torchtext의 데이터로더는 일반적인 데이터로더와 다르게 torchtext.legacy.batch.Batch 객체이기 때문에 필드를 통해서 접근해야지만 tensor로 나타납니다.

```
print(type(batch))
print(batch.text)

###
<class 'torchtext.legacy.data.batch.Batch'>

tensor([[  29,   48,  251,  114,    3,   25,  369,   38,   96,   43,    0, 1648,
            8,   43,    0,    0, 1321,   47,   15,    3],
        [   9,   61,  465,   10,  664,   34, 3871,    0,    2,  119,  850,    9,
           91,  207,   63,    6, 1654,   11,   17,    7]])
###
```
