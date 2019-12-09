# Epsilon constrained k-means for document clustering with noise removal

Use [lovit_textmining_dataset](https://github.com/lovit/textmining_dataset) for demo.

```python
from lovit_textmining_dataset.navernews_10days import get_bow

X, idx_to_vocab, vocab_to_idx = get_bow(date='2016-10-20', tokenize='noun')
X.shape # (30091, 9774)
```

Import ek-means package

```python
from ekmeans import EKMeans

model = EKMeans(n_clusters = 100, metric = 'cosine',
    epsilon = 0.6, min_size = 3, max_depth = 5,
    max_iter = 4, coarse_iter = 6,
    random_state = 0
)
model.fit(X)
```

```
[round: 1/5 full-iter: 1/10] #changes: 10568, diff: 0.2443, inner: 0.3146, #assigned: 10568, #clusters: 81, time: (-16s)
[round: 1/5 full-iter: 2/10] #changes: 4031, diff: 0.08416, inner: 0.2711, #assigned: 13760, #clusters: 95, time: 3s (-14s)
[round: 1/5 full-iter: 3/10] #changes: 2111, diff: 0.0261, inner: 0.2704, #assigned: 15207, #clusters: 100, time: 5s (-11s)
[round: 1/5 full-iter: 4/10] #changes: 1103, diff: 0.01185, inner: 0.275, #assigned: 15900, #clusters: 100, time: 6s (-9s)
...
[round: 5/5 full-iter: 2/4] #changes: 151, diff: 0.001965, inner: 0.2723, #assigned: 23660, #clusters: 500, time: 15s (-15s)
[round: 5/5 full-iter: 3/4] #changes: 102, diff: 0.0005417, inner: 0.2718, #assigned: 23665, #clusters: 500, time: 22s (-7s)
[round: 5/5 full-iter: 4/4] #changes: 47, diff: 0.0003335, inner: 0.2715, #assigned: 23665, #clusters: 500, time: 31s
[round: 5/5] #assigned: 23665 (78.64%), time: 1m 45s
```

To label clusters

```python
from ekmeans import proportion_keywords
import numpy as np

centers = model.cluster_centers_
labels = model.labels_
labels = labels[np.where(labels >= 0)[0]]
cluster_size = np.bincount(labels, minlength=centers.shape[0])

keywords = proportion_keywords(centers, labels, topk=20,
    candidates_topk=50, index2word=idx_to_vocab, min_score=0.3)
```

```
[cluster 0, 26 docs]: 행정처분, 성분, 검출, 적정성, 살균제, 가습기, 보건당국, 30개, 품목, 물질, 수거, 화장품, 유아, 식품의약품안전처, 유해성, 대형마트, 가습기살균제, 식약처, 회수, 의뢰
[cluster 1, 238 docs]: 윤동주, 문호, 노해, 서울중앙지방검찰청, 아시아경제, 보는, 남대문시장, 특가, 인턴기자, 세계, 아우디폭스바겐코리아, 배출가스, 경제, 받기, 공시, 서초구, 신분, 조작, 출석, 참여
[cluster 2, 185 docs]: 저수지, 방탄복, 성병대, 발견, 경찰, 구속영장, 살인, 범행, 충돌, 출동, 강북경찰서, 사제, 총격, 경찰관, 피의자, 총기, 신청, 신고, 전자발찌, 경위
[cluster 6, 23 docs]: 학칙, 사회부총리, 만나서, 학사관리, 교육부, 행복교육박람회, 성적, 부총리, 이대, 사실관계, 자료, 정유라씨, 면담, 입학, 감사, 이화여대,
[cluster 15, 87 docs]: 학내, 농성, 점거, 교수들, 총장, 최경희, 본관, 사임, 이화, 학사, 선출, 사퇴, 특혜, 이화여대, 입학, 이대, 학생들, 정유라, 시위, 규명
[cluster 18, 335 docs]: 동대문, 방지, 스포츠동아, 서울패션위크, 동아닷컴, 헤라, 멤버, 상추, 다이아, 스포츠조선, 효민, 동대문디자인플라자, 2017, 티아라, 스타투데이, 고우리, 소윙바운더리스, 손나은, 현아, 씨스타
[cluster 19, 39 docs]: 우크라이나, 베를린, 페트, 블라디미르, 정부군, 해킹, 러시아, 휴전, 회담, 총리, 시리아, 파리, 알레포, 프랑스, 푸틴, 체포, 반군, 연장, 독일, 사태
[cluster 20, 90 docs]: 서울가정법원, 성남지원, 가사소송법, 임우재, 이부진, 친권, 관할권, 수원지법, 이혼소송, 주소지, 삼성전기, 주소, 호텔신라, 1심, 항소심, 이혼, 승소, 관할, 무효, 살았던
[cluster 21, 131 docs]: 11월호, 매거진, 화보, 입술, 박수진, 메이크업, 시크, 몸매, 오연서, 아나운서, 하이, 소화, 면모, 후문, 부드, 돋보, 발산, 가을, 에이핑크, 패션
[cluster 22, 28 docs]: 대우조선, 대우조선해양, 회계, 사옥, 수사관, 정기, 재무, 국세청, 전담, 해양, 법인세, 흑자, 도로, 적자, 조사, 기획, 특별, 201, 2014년, 해석
...
[cluster 125, 23 docs]: 석유, 저유가, 국채, 사우디, 발행, 만기, 아르헨티나, 신흥국, 165, 50년, 30년, 달러화, 채권, 조달, 재정, 수익률, 금리, 사우디아라비아, 달러, 사상
[cluster 127, 60 docs]: 기뢰, 소해헬기, 소해함, 진해, 8개국, 뉴질랜드, 태국, 병력, 380, 무인기, 함정, 탐색, 해군, 캐나다, 실전, 훈련, 해상, 폭발물, 53, 호주
[cluster 129, 31 docs]: 자금조달, 코넥스, 보통주, 상장기업, 거래소, 한국거래소, 조달, 전환사채, 금액, 유상증자, 사회적, 기업, 발행, 전환, 활성화, 자금, 고용, 책임, 48, 2013년
[cluster 130, 54 docs]: 빅브레인, 갓세븐, 오블리스, 너무너무너무, 신용재, 산들, 펜타곤, 다비치, 몬스타엑스, 아이오아이, 엠카운트다운, 세븐, 박진영, 완전체, 중독성, 잠깐, 보컬, 코드, 엠넷, 진영
[cluster 132, 42 docs]: 썬코어, 왕자, 외아들, 두바이, 왈리드, 회관, 최대주주, 상장사, 베어링, 000만, 오일, 최규, 유상증자, 제3자, 전경련회관, 사우디아라비아, 기자간담회, 2030, 중동, 배정
...
```
