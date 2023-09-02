### Core Concept
외국인이 한국에 여행 혹은 방문했을 때 적용되는 국내법을 알려주는 모델

### MVP submit plan
- 외국인이 한국에 여행 혹은 방문했을 때 적용되는 국내법 3~5개 찾기
- ChatGPT API를 사용해서 해당 법에 대한 답변을 잘 수행하는지 확인하기
- 답변이 미흡할 경우, 3개 정도의 질문과 답변 페어를 구성하여 해당 법에 대해서 답변할 수 있도록 few-shot prompt engineering 작업 진행하기
- 3~5개에 대해서 답변을 수행할 수 있는 모델을 streamlit 등에 배포
- 테스트 주소, 소개영상과 참가신청서 작성하여 제출 (~9/3)

# app 호스팅을 위한 docker 및 GCP 사용법
## Dockerfile 실행 명령
- Dockerfile에서 docker image 생성
```bash
docker build -t {dockerhub id}/prompterday-mvp .
```
- dockerhub에 업로드
```bash
docker login
docker push {dockerhub_id}/prompterday-mvp
```
- 이미지에서 컨테이너 생성
```bash
docker pull {dockerhub_id}/prompterday-mvp
docker run -p 8501:8501 -d {dockerhub id}/prompterday-mvp
```
## GCP compute engine 사용
- [dockerhub에 업로드한 image 지정해서 compute engine 실행](https://cloud.google.com/compute/docs/containers?hl=ko)
- [고정 IP 할당](https://datainsider.tistory.com/108)
- [외부접속 포트 (streamlit의 경우 8501) 할당](https://minimin2.tistory.com/173)
- [현재 app 접속 가능 주소](http://34.16.0.128:8501/)