# **IR & RS 대회**

## 개요

> - kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
> - 2024.10.02.수 10:00 ~ 2024.10.23.수 19:00

## 파일 소개

- ir/
    - Information Retriever 내용만 모아둔 폴더이다.
    - kkh-1-data-meta.py
        - meta.csv 파일 내용에 한국어 내용을 추가하여 meta_kr.csv 파일로 만들어 줌
- rs/
    - Recommender Systems 내용만 모아둔 폴더이다.
    - kkh-1-data-meta.py
        - meta.csv 파일 내용에 한국어 내용을 추가하여 meta_kr.csv 파일로 만들어 줌
- kkh-util-disk.py
    - 서버 사용량 확인
- font/
    - 폰트 파일
- pyproject.toml
    - 프로젝트 패키지 관리를 위한 poetry 설정 파일
- jupyter_to_python.sh
    - 주피터 파일을 파이썬 파일로 변환하는 리눅스 스크립트

## 우분투에 git 세팅

- apt update
- apt install -y git wget htop curl vim libgl1-mesa-glx libglib2.0-0 openjdk-11-jdk
- git --version
- git config --global user.email "helpotcreator@gmail.com"
- git config --global user.name "helpotcreator"
- cd /
- git clone https://{개인 토큰}@github.com/UpstageAILab3/upstage-ai-advanced-ir7.git
- mv upstage-ai-advanced-ir7 kkh
- cd kkh
- git remote -v
- git checkout -b kimkihong origin/kimkihong
- git branch -a
- mkdir ir
- mkdir rc

## data.tar.gz 세팅(ir 대회)

- cd /kkh/ir
- wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000322/data/data.tar.gz
- tar -xzvf data.tar.gz
- rm data.tar.gz

## data.tar.gz 세팅(rs 대회)

- 대회 참석한 팀원에게 부탁하여 file url 전달 받아서 진행해야 함

## 우분투에 miniconda3 설치

- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
- chmod +x Miniconda3-latest-Linux-x86_64.sh
- ./Miniconda3-latest-Linux-x86_64.sh
- conda create -n irs python=3.10
- conda init
- source ~/.bashrc
- conda activate irs
- rm Miniconda3-latest-Linux-x86_64.sh
- pip install jupyter nbconvert numpy matplotlib seaborn scikit-learn timm torch==2.3.1 pyyaml tqdm torch pytorch-lightning rouge transformers transformers[torch] evaluate konlpy fastapi uvicorn bitsandbytes faiss-cpu google-generativeai

## 우분투에 elasticsearch 설치

- cd /kkh
- wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.8.0-linux-x86_64.tar.gz
- tar -xzf elasticsearch-8.8.0-linux-x86_64.tar.gz
- chown -R daemon:daemon elasticsearch-8.8.0/
- cd elasticsearch-8.8.0/bin
- ./elasticsearch-plugin install analysis-nori
- cd /kkh
- rm elasticsearch-8.8.0-linux-x86_64.tar.gz
- 파이썬 가상환경 실행
- pip install elasticsearch==8.8.0 openai==1.7.2 sentence-transformers==2.2.2
- source ~/.bashrc
- /kkh/elasticsearch-8.8.0/bin/elasticsearch-setup-passwords auto -url "https://localhost:9200"
- 위 명령 실행 후 "Please confirm that you would like to continue"에서 y 입력 필요
- 마지막에 출력되는 password를 복사하여, 파이썬 소스코드에서 활용하면 됨(이것이 접속용 최고 권한 비밀번호임)
- 파이썬 가상환경 실행

## 우분투 터미널에서 elasticsearch 실행

- 새로운 터미널 창을 연다.
- sudo chmod 644 /kkh/elasticsearch-8.8.0/config/certs/http_ca.crt
- sudo -u daemon -s bash -c "bash /kkh/elasticsearch-8.8.0/bin/elasticsearch"

## wandb

- pip install wandb
- 콘다 실행한 환경에서 wandb login
- wandb 토큰 입력

## miniconda3 세팅_우분투_bash(선택)

우분투 bash 쉘 시작할 때, nlp 가상환경이 기본으로 실행되도록 하는 방법임.

- vim ~/.bashrc
- 가장 아래에 다음 두 줄 추가 후 저장
    - conda deactivate
    - conda activate irs
- source ~/.bashrc

## miniconda3 세팅_윈도우_cmd(선택)
cmd 시작할 때, 어떤 가상환경도 실행되지 않도록 하는 방법임.

- conda config --set auto_activate_base false

## jupyter_to_python.sh 파일 작성(선택)

```bash
#!/bin/bash

# 주피터 노트북 파일명을 인자로 받음
NOTEBOOK_FILE="$1"

# 파일명이 주어지지 않으면 에러 메시지를 출력하고 종료
if [ -z "$NOTEBOOK_FILE" ]; then
    echo "Usage: $0 <notebook-file>"
    exit 1
fi

# 주어진 파일이 .ipynb 확장자를 가지고 있는지 확인
if [[ "$NOTEBOOK_FILE" != *.ipynb ]]; then
    echo "Error: The input file must have a .ipynb extension"
    exit 1
fi

# jupyter nbconvert 명령어를 사용하여 노트북 파일을 Python 스크립트로 변환
python -m jupyter nbconvert --to script "$NOTEBOOK_FILE"

# 변환 결과 확인
if [ $? -eq 0 ]; then
    echo "Conversion successful: ${NOTEBOOK_FILE%.ipynb}.py"
else
    echo "Conversion failed"
    exit 1
fi
```

## jupyter_to_python.sh 파일 세팅

- chmod +x jupyter_to_python.sh
- poetry run ./jupyter_to_python.sh {주피터 파일명}.ipynb
- poetry run python {만들어진 파이썬 파일}.py