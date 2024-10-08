# SentenceEmbeddings + FaissDB + OpenAI_GPT

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import traceback
# from subprocess import Popen, PIPE, STDOUT
# import time


class SentenceEmbeddings:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        # 임베딩 모델을 초기화
        self.model = SentenceTransformer(model_name)

    # 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
    def get_embeddings_in_batches(self, docs, batch_size):
        batch_embeddings = []

        # 총 학습 문서 수와 배치 크기에 따라 배치별로 임베딩 생성
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            contents = [doc["content"] for doc in batch]
            embeddings = self.model.encode(contents)
            batch_embeddings.extend(embeddings)

        # 모든 문서에 대한 임베딩 반환
        return batch_embeddings


class FaissDB:
    def __init__(self, d):
        self.index = faiss.IndexFlatL2(d)  # L2 거리(유클리드 거리)를 사용한 평면 인덱스
        self.documents = []  # 문서를 저장할 리스트

    def add_embeddings(self, embeddings, docs):
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)  # 벡터를 FAISS 인덱스에 추가
        self.documents.extend(docs)  # 원본 문서 저장

    def search(self, query_embedding, k=3):
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)  # 벡터 검색
        results = [{"doc": self.documents[i], "distance": distances[0][idx]} for idx, i in enumerate(indices[0])]
        return results


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(react_agent, eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            if idx > 5:
                break
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = react_agent.answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1


class OpenAI_GPT:
    def __init__(self, llm_model="gpt-3.5-turbo-1106", key_file='./ex-key/openai-helpotcreator-key-all.txt', faiss_db=None):
        self.llm_model = llm_model
        self.api_key = self.load_api_key(key_file)
        self.client = OpenAI()
        self.faiss_db = faiss_db
        self.persona_qa = self.get_persona_qa_prompt()
        self.persona_function_calling = self.get_persona_function_calling_prompt()

    def load_api_key(self, file_path):
        try:
            with open(file_path, 'r') as file:
                api_key = file.readline().strip()
            if not api_key:
                raise ValueError("API key is empty.")
            os.environ["OPENAI_API_KEY"] = api_key
            return api_key
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {file_path} not found.")
        except ValueError as e:
            raise ValueError(f"Error: {e}")

    def get_persona_qa_prompt(self):
        return """
        ## Role: 과학 상식 전문가

        ## Instructions
        - 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
        - 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
        - 한국어로 답변을 생성한다.
        """

    def get_persona_function_calling_prompt(self):
        return """
        ## Role: 과학 상식 전문가

        ## Instructions
        - 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
        - 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
        """

    def answer_question(self, messages):
        response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

        # LLM을 통한 질의 분석 및 함수 호출
        msg = [{"role": "system", "content": self.persona_function_calling}] + messages
        try:
            result = self.client.chat.completions.create(
                model=self.llm_model,
                messages=msg,
                tools=[],
                temperature=0,
                seed=1,
                timeout=10
            )
        except Exception as e:
            traceback.print_exc()
            return response

        response["answer"] = result.choices[0].message.content
        return response


def start():
    embedding_module = SentenceEmbeddings()

    with open("./data/documents.jsonl") as f:
        docs = [json.loads(line) for line in f]
    embeddings = embedding_module.get_embeddings_in_batches(docs, 100)

    # FAISS DB에 임베딩을 추가하고 검색 기능 테스트
    faiss_db = FaissDB(768)
    faiss_db.add_embeddings(embeddings, docs)

    # 검색엔진에 색인이 잘 되었는지 테스트하기 위한 질의
    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    query_embedding = embedding_module.model.encode([test_query])[0]
    search_result = faiss_db.search(query_embedding, 3)

    # 결과 출력 테스트
    for result in search_result:
        print('_distance:', result["distance"], '_content:', result["doc"]["content"])

    # RAG 구현
    llm_model = "gpt-3.5-turbo-1106"
    openai_key_file = './ex-key/openai-helpotcreator-key-all.txt'
    react_agent = OpenAI_GPT(llm_model, openai_key_file, faiss_db)

    eval_rag(react_agent, "./data/eval.jsonl", "./sample_submission.csv")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start()
