### Core Concept
외국인이 한국에 여행 혹은 방문했을 때 적용되는 국내법을 알려주는 모델

### MVP submit plan
- 외국인이 한국에 여행 혹은 방문했을 때 적용되는 국내법 3~5개 찾기
- ChatGPT API를 사용해서 해당 법에 대한 답변을 잘 수행하는지 확인하기
- 답변이 미흡할 경우, 3개 정도의 질문과 답변 페어를 구성하여 해당 법에 대해서 답변할 수 있도록 few-shot prompt engineering 작업 진행하기
- 3~5개에 대해서 답변을 수행할 수 있는 모델을 streamlit 등에 배포
- 테스트 주소, 소개영상과 참가신청서 작성하여 제출 (~9/3)

## Dockerfile 실행 명령
- Dockerfile에서 docker image 생성
```bash
docker build -t prompterday-mvp .
```
- 이미지에서 컨테이너 생성
```bash
docker run -p 8501:8501 prompterday-mvp
```