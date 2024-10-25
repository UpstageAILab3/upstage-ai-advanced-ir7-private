[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Tm6AYAOm)
# Title (Please modify the title)
## Team

![image](https://github.com/user-attachments/assets/57586433-261e-4f14-b811-8861ba1757cd)


## 1. Competiton Info

### Overview

![image](https://github.com/user-attachments/assets/9911a936-967b-448d-87e7-4a217f27f0d5)

과학 상식을 질문하는 시나리오를 가정하고 과학 상식 문서 4200여개를 미리 검색엔진에 색인합니다.
대화 메시지 또는 질문이 들어오면 과학 상식에 대한 질문 의도인지 그렇지 않은 지 판단 후에 과학 상식 질문이라면 검색엔진으로부터 적합한 문서들을 추출하고 이를 기반으로 답변을 생성합니다. 

만일 과학 상식 이외의 질문이라면 검색엔진을 활용할 필요 없이 적절한 답을 바로 생성합니다.

마지막으로, 본 프로젝트는 모델링에 중점을 둔 대회가 아니라 RAG(Retrieval Augmented Generation) 시스템의 개발에 집중하고 있습니다. 이 대회는 여러 모델과 다양한 기법, 그리고 앙상블을 활용하여 모델의 성능을 향상시키는 일반적인 모델링 대회와는 다릅니다. 대신에 검색 엔진이 올바른 문서를 색인했는지, 그리고 생성된 답변이 적절한지 직접 확인하는 것이 중요한 대회입니다.

### Timeline

프로젝트 전체 기간 (4주) : 10월 2일 (수) 10:00 ~ 10월 24일 (목) 19:00

## 2. Components ( 이 부분 나중에 채우기)

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

- 학습 데이터 개요
이번 대회는 머신러닝 모델을 학습하는 것 보다는 임베딩 생성 모델, 검색엔진, LLM 등을 활용하여 레퍼런스를 잘 추출하고 이를 토대로 얼마나 답변을 잘 생성하는지 판단하는 대회입니다.

따라서 모델 학습을 위한 학습데이터를 별도로 제공하지 않고, 과학 상식 정보를 담고 있는 순수 색인 대상 문서 4200여개가 제공됩니다. 

문서의 예시는 아래와 같습니다. 'doc_id'에는 uuid로 문서별 id가 부여되어 있고 'src'는 출처를 나타내는 필드입니다. 그리고 실제 RAG에서 레퍼런스로 참고할 지식 정보는 'content' 필드에 저장되어 있습니다.

참고로  데이터를 Open Ko LLM Leaderboard에 들어가는 Ko-H4 데이터 중 MMLU, ARC 데이터를 기반으로 생성했기 때문에 출처도 두가지 카테고리를 가집니다.

- 평가 데이터 개요
이번 대회는 RAG 시스템이 사용자 질문에 적합한 레퍼런스를 찾고 이를 토대로 답을 잘하는 지 보는 대회입니다.

이 때 사용자 입력은 짧은 키워드 보다는 자연어로 된 질문 형태를 띄게 됩니다.

특히 멀티턴 대화 시나리오를 가정하기 때문에 리스트 형태의 사용자와 시스템이 주고 받은 메시지 형태로 평가 데이터가 구성되어 있습니다. 아래 예시 중 세번째("eval_id": 2)를 보시면 메시지 히스토리가 리스트 형태로 되어 있는 것을 확인할 수 있습니다.

따라서 LLM을 활용하여 standalone query를 잘 추출해 내는 것이 매우 중요합니다.

### EDA

![image](https://github.com/user-attachments/assets/4af60c24-dd93-4c25-87a8-056a0e29601d)

![image](https://github.com/user-attachments/assets/0714c5bc-7e44-4b00-875c-9c5e76156205)


## 4. RAG

![image](https://github.com/user-attachments/assets/74256ab5-a70a-4937-a3eb-6dd15ff166a0)

![image](https://github.com/user-attachments/assets/89a16e13-4d05-4e31-bf5c-4c7eb680dd61)

![image](https://github.com/user-attachments/assets/7a641738-031f-49ff-b136-cd5e2bd733d7)


## 5. Result

### Leader Board

![image](https://github.com/user-attachments/assets/38da8a88-0738-4e50-ac7e-a854fe04cb99)



### Presentation

- _Insert your presentaion file(pdf) link_ (이 부분 나중에 채우기)


