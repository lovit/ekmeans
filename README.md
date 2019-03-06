# Epsilon constrained k-means for document clustering with noise removal

Use [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset) for demo.

```python
from navernews_10days import get_bow

X, idx_to_vocab, vocab_to_idx = get_bow(date='2016-10-20', tokenize='noun')
X.shape # (30091, 9774)
```

Import ek-means package

```python
from ekmeans import EKMeans

n_clusters = 300
epsilon = 0.5
max_depth = 10
min_size = 5

ekmeans = EKMeans(n_clusters, epsilon, max_depth, min_size)
labels = ekmeans.fit_predict(X)
```

To label clusters

```python
from ekmeans import proportion_keywords
import numpy as np

centers = ekmeans.cluster_centers_
labels = ekmeans.labels_
labels = labels[np.where(labels >= 0)[0]]
cluster_size = np.bincount(labels, minlength=centers.shape[0])

keywords = proportion_keywords(centers, labels, topk=20,
    candidates_topk=50, index2word=idx_to_vocab, min_score=0.3)
```

```
...

[cluster 4, 621 docs]: 연합뉴스, 고양, 킨텍스, 수상, 2016, 20일, 서울, 국회, 제공, 기자
[cluster 5, 471 docs]: 김정민, 이수혁, 로맨틱, 수애, 김영광, 월화드라마, 홍나리, 고난길, 제작발표회, 연하, 로맨스, 조보아, 타임스퀘어, 아빠, 남자, 원작, 24일, 우리, 드라마, 코미디
[cluster 6, 294 docs]: 토론, 트럼프, 대선후보, 클린턴, 대선, 라스베이거스, 공화당, 3차, 도널드, 힐러리, 후보, 선거, 주장, 결과, 이라고, 미국, 민주당, 19일, 현지시간, 대통령
[cluster 7, 714 docs]: 압구정, 주지홍, 제작보고회, 김윤혜, 박근형, 성동일, 김유정, 차태현, 힐링, 사람들, 사랑, 때문, 영화, 서현진, 배우, 감독, 박보검, 오전, 20일, 서울
[cluster 8, 474 docs]: 100여개, 21개, 수주회, 유수, 이번달, 김세, 찾았다, 뉴미디어, 보도자료, 마이데일리, 유진, 헤라서울패션위크, 기사, 컬렉션, 실시간, 패션쇼, 17일, 구매, 디자이너, 41개

[cluster 9, 576 docs]: 이데일리, 머니투데이, 로이터, 미디어, 종합, 경향신문, 경제, 10월, 재배포, 금지, 무단, 2016년, 기자
[cluster 10, 171 docs]: 박세, 일간스포츠, 오후, 금지, 재배포, 무단
[cluster 11, 173 docs]: 아시아경제, 문호, 서울중앙지방검찰청, 서초구, 보는, 받기, 인턴기자, 토론회, 신분, 아우디폭스바겐코리아, 참고인, 총괄대표, 트레, 세계, 경제, 폭스바겐, 이지은, 배출가스, 조작, 대선후보
[cluster 12, 140 docs]: 김현우, 엔터온뉴스, 전자신문, 금지, 재배포, 무단, 기자
[cluster 13, 381 docs]: 노컷뉴스, 속보, 연예, 강남구, 스타뉴스, 역삼동, 론칭, 스타, 리얼, 김창, 저작권자, 재배포, 무단, 금지, 20일, 기자, 함께, 서울, 참석

[cluster 14, 275 docs]: 프로듀스, 소라, 키미, 형은, 세이, 불독, 101, 마포구, 쇼케이스, 롤링, 본명, 걸그룹, 걸크러쉬, 데뷔, 김민지, 박소라, 이진희, 콘셉트, 헤럴드, 홍대
[cluster 15, 130 docs]: 동행명령장, 사유서, 운영위, 증인, 수석, 불출석, 국감, 동행명령, 발부, 비서실장, 민정수석, 우병우, 정진석, 운영위원회, 출석, 야당, 원내대표, 이유, 국정감사, 제출
[cluster 16, 174 docs]: 오패산, 살인, 폭죽, 강북경찰서, 성병대, 성씨, 총격, 사제, 범행, 총기, 화약, 대상자, 전자발찌, 경찰관, 인근, 경찰, 경위, 범죄, 현장, 출동
[cluster 17, 182 docs]: 누구라, 불법행위, 유용, 재단들, 퇴임, 미르, 재단, 기업들, 처벌받, 스포츠재단, 위기, 엄정, 설립, 최씨, 제기, 모금, 확산, 문화, 자금, 감사
[cluster 18, 126 docs]: 고성희, 나리, 수영, 정원, 화신, 질투, 고경표, 공효진, 조정석, 자신, 등장, 출연, 모습, 함께, 방송, 사진, 배우, 기자
...
```
