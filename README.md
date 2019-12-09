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

model = EKMeans(
    n_clusters = 100,
    epsilon = 0.6,
    min_size = 3,
    metric = 'cosine',
    max_iter = 4,
    coarse_iter = 6,
    max_depth = 5
)
model.fit(X)
```

```
[round: 1/5 full-iter: 1/10] #changes: 11053, diff: 0.2222, inner: 0.3039, #assigned: 11053, #clusters: 84, time: (-16s)
[round: 1/5 full-iter: 2/10] #changes: 3031, diff: 0.07349, inner: 0.2611, #assigned: 13318, #clusters: 95, time: 3s (-13s)
[round: 1/5 full-iter: 3/10] #changes: 1824, diff: 0.02872, inner: 0.269, #assigned: 14431, #clusters: 99, time: 4s (-11s)
[round: 1/5 full-iter: 4/10] #changes: 1195, diff: 0.0117, inner: 0.2736, #assigned: 15117, #clusters: 100, time: 6s (-9s)
[round: 1/5 full-iter: 5/10] #changes: 690, diff: 0.006144, inner: 0.2755, #assigned: 15402, #clusters: 100, time: 7s (-7s)
...
[round: 5/5 full-iter: 3/4] #changes: 73, diff: 0.001483, inner: 0.2747, #assigned: 23304, #clusters: 499, time: 25s (-8s)
[round: 5/5 full-iter: 4/4] #changes: 96, diff: 0.0005618, inner: 0.2747, #assigned: 23329, #clusters: 500, time: 33s
[round: 5/5] #assigned: 23329 (77.53%), time: 1m 48s
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
[cluster 1, 92 docs]: 성남지원, 가사소송법, 서울가정법원, 관할권, 이부진, 임우재, 이혼소송, 친권, 수원지법, 삼성전기, 호텔신라, 주소지, 1심, 항소심, 관할, 주소, 이혼, 무효, 승소, 살았던
[cluster 3, 173 docs]: 재단들, 누구라, 불법행위, 퇴임, 처벌받, 재단, 엄정, 기업들, 유용, 미르, 최씨, 스포츠재단, 설립, 확산, 의혹, 자금, 스포츠, 최순실씨, 문화, 대통령
[cluster 4, 21 docs]: 인턴, 김은혜, 배우는, 권영, 견해, 자세, 김종인, 결정적, 공원, 바랍니다, 98, 기획, 있습니다, 스튜디오, 뉴스, 최순실, 어제, 내용, 먼저, 콘텐츠
[cluster 5, 56 docs]: 상실, 신사동, 갈아, 끊임없이, 작곡, 마이데일리, 뉴미디어, 힐링, 제작보고회, 보도자료, 압구정, 기억, 코미디, 실시간, 차태현, 구매, 기사, 사랑, 김유정, 사람들
[cluster 6, 339 docs]: 윤창, 노컷뉴스, 경향신문, 서울경제, 정책조정회의, 확실, 조선일보, 우상호, 정론관에서, 원내수석부대표, 한경닷컴, 상임고문, 공시, 헤럴드경제, 동아일보, 운영위원회, 정진석, 방법, 헤럴드, 정계복귀
[cluster 9, 21 docs]: 테슬라, 자율주행, 자율주행차, 하드웨어, 장착, 레이더, 신차, 탑재, 운전자, 소프트웨어, 운전, 장치, 360, 완전, 250, 모델, 카메라, 센서, 감지, 차량
...
[cluster 64, 269 docs]: 포토콜, 양윤영, 김진솔, 손예진, 역삼동, 서인영, 미샤, 라움, 박희, 강소영, 이탈, 런칭, 론칭, 미스코리아, 시리즈, 기념, 강남구, 포토, 즐기는, 행사
[cluster 65, 146 docs]: 프로듀스, 연습생, 세이, 5인조, 키미, 프로듀스101, 소라, 본명, 형은, 걸크러쉬, 엠넷, 불독, 101, 싱글, 롤링, 당당, 서교동, 강렬, 쇼케이스, 마포구
[cluster 66, 262 docs]: 승복, 트럼프, 클린턴, 불복, 오바마, 힐러리, 도널드, 공화당, 토론, 여론조사, 대선후보, 후보, 대선, 3차, 선거, 라스베이거스, 지지, 2차, 조작, 주장
[cluster 67, 1014 docs]: 147, 사천비행장, 고양, 430, 사천, 122, 백승, 21, 행복교육, 165, 킨텍스, 박람회, 특수, 자유학기제, 2016, 2015, 14, 박정, 300, 공군
[cluster 68, 51 docs]: 집단대출, 보금자리론, 주택담보대출, 중도금, 대출, 금융당국, 시중은행, 한도, 가계부채, 서민, 은행들, 주택, 금리, 수요, 금융, 규제, 은행, 부동산, 분양, 대책
...
[cluster 399, 48 docs]: 너무너무너무, 신용재, 빅브레인, 오블리스, 갓세븐, 박진영, 아이오아이, 세븐, 엠카운트다운, 완전체, 중독성, 다비치, 잠깐, 산들, 펜타곤, 코드, 보컬, 엠넷, 엑스, 발랄
[cluster 402, 40 docs]: 투기자본감시센터, 형사8부, 대검찰청, 문체부, 시민단체, 소환, 고발, 항의, 서울중앙지검, 허가, 문화체육관광부, 참고인, 모금, 미르재단, 미르, 설립, 신분, 재단, 스포츠재단, 2명
[cluster 404, 53 docs]: 집행유예, 피고인, 종업원, 총괄회장, 부장판사, 선고, 넘겨, 징역, 기소, 횡령, 성폭행, 롯데그룹, 재판, 배당, 저지, 항소심, 허위, 혐의, 형사, 사기
[cluster 413, 22 docs]: 한국인터넷진흥원, 클라우드, 물리적, 획득, 기술적, 공공기관, 모의, 정보보호, 인증, 점검, 공공, 민간, 사업자, 보안, 미래창조과학부, 부여, 미래부, 시스템, 이용, 수행
[cluster 421, 20 docs]: 노벨문학상, 박경리문학상, 문학, 케냐, 악마, 언어, 꽃잎, 시옹오, 감옥, 프레스, 시인, 소설, 수상자, 아프리카, 토지, 원주, 선생, 영어, 작가, 기자간담회
...
```
