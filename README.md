# Kotra-Project
KOTRA-Project는 최종적으로 코트라와 계약을 맺게 되어 최종 결과물을 Private으로 전환했습니다.<br>
해당 Repository와 하단 회의록에서는 프로젝트의 일부 작업만 확인할 수 있습니다. <br>
회의록: https://special-scabiosa-ef6.notion.site/KOTRA-40707eb743ca4ae78fc4a1cabccfa8c3<br>

#
### 프로젝트 소개<br>
Kotra project는 대한무역투자진흥공사인 Kotra와 연계하여 진행한 프로젝트로 2021년에 발생한 요소수 대란을 기점으로 대두된 글로벌 공급망 위기에 대처하기 위해 구축한 세관 고시 분석 및 모니터링 시스템입니다.


시스템 구축을 위해 수출입 상위 10개 교역국 중 각 나라별 품목들의 세부 의존도를 기준으로 모니터링 5개 국가(중국, 미국, 일본, 베트남, 호주)를 선정하였으며 각 국가별 특수 요건을 고려하여 크롤링을 진행하였습니다. 또한 모니터링 품목을 선정하기 위해 MTI 4단위 코드를 기준으로 각 나라별 TOP100 품목을 수집하였고 이에 해당하는 MTI 6단위, KSIC, HS code를 매핑해 모니터링 품목 매칭표를 완성하였습니다.

전처리 과정에서는 수집한 모든 데이터(세관 고시, 모니터링 품목)를 영어로 통일했고 특수문자와 개행문자 제거, 소문자로 변경, nltk 불용어 제거 및 국가별 자체 불용어 리스트를 생성하여 키워드 추출 과정에서 출력되는 불필요한 단어들을 제거하여 정확도를 높였습니다.


문장 임베딩을 위한 NLP 모델로는 구글에서 공개한 BERT 모델을 미세 조정하여 Sentence Embedding 추출의 성능을 향상 시킨 SBERT 모델을 사용했고 다양한 미세 조정 모델 중 보편적인 주제에 대한 성능이 가장 우수하다고 알려진 all-mpnet-base-v2를 사용했습니다.


### System Flow Chart
![image](https://user-images.githubusercontent.com/52529935/194687321-e4234302-7e0e-4806-aad7-e414db7c09bf.png)

### System View
![image](https://user-images.githubusercontent.com/52529935/194688702-f692321f-8ecb-42bd-8c5b-68d0ead3a19a.png)
시연 영상 링크 : https://www.youtube.com/watch?app=desktop&v=89itkBMtdCQ&feature=youtu.be

### 기대효과
이번 프로젝트를 통해 데이터 수집을 자동화 했으며 수집한 데이터에서 추출한 키워드와 매칭표와의 연계를 통해 해당 고시 정보가 우리나라 산업에 미치는 영향을 파악할 수 있을 것으로 예상됩니다. 또한 현재 이를 웹 서비스로 제공하여 사용자가 직관적으로 정보를 확인할 수 있도록 했으며 메일 서비스를 바탕으로 추출된 키워드와 연관있는 산업 관계자들에게 신속한 정보를 제공해 대응 방안을 및 시간을 확보할 수 있는 기회를 제공할 것으로 기대됩니다.
