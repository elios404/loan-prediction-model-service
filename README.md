# 인공지능을 이용한 대출 심사 서비스

모델 구현과 구현된 모델을 서비스할 백엔드 fastAPI 코드 

환경 설정 방법
`conda env create -f environment.yml`
를 통해서 설치된 패키지와 의존성을 그대로 가져올 수 있다.

더하여 추가로 2개 설치
`pip install fastapi`
`pip install "uvicorn[standard]"`

`uvicorn main:app --reload` 를 통해서 서버 실행