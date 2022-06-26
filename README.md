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
