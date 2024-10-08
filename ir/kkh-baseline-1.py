# SentenceEmbeddings 클래스화 완료
# 엘라스틱서치 클래스화 진행중
# OpenAI_GPT 클래스화 완료

import os
import json
from elasticsearch import Elasticsearch, helpers
from subprocess import Popen, PIPE, STDOUT
import time
from openai import OpenAI
import traceback



class SentenceEmbeddings:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"):
        # 임베딩 모델을 초기화
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    # 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
    # docs: 학습 데이터 전체 문서, batch_size: 배치 크기
    def get_embeddings_in_batches(self, docs, batch_size):
        batch_embeddings = []

        # 총 학습 문서 수와 배치 크기에 따라 배치별로 임베딩 생성
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            contents = [doc["content"] for doc in batch]
            # 배치 단위로 임베딩 생성
            embeddings = self.model.encode(contents)
            batch_embeddings.extend(embeddings)

        # 모든 문서에 대한 임베딩 반환
        return batch_embeddings





def interface_embedding_db():
    return get_elastic_server()

def get_elastic_server():
    es_password = 'r7ymfymLEmqoEPaUYlGy'
    es_username = 'elastic'
    es = Elasticsearch(['https://localhost:9200'],
                       basic_auth=(es_username, es_password),
                       ca_certs="/kkh/elasticsearch-8.8.0/config/certs/http_ca.crt",
                       verify_certs=True)

    # 먼저 Elasticsearch가 이미 실행 중인지 확인
    try:
        if es.ping():
            print("Elasticsearch is already running!")
            return es
    except Exception:
        print("Elasticsearch is not running. Starting it now...")

    # Elasticsearch가 실행 중이 아니면 시작
    es_server = Popen(['/kkh/elasticsearch-8.8.0/bin/elasticsearch'],
                      stdout=PIPE, stderr=STDOUT,
                      preexec_fn=lambda: os.setuid(1))

    # Elasticsearch 서버가 준비될 때까지 대기
    timeout = 300  # 최대 대기 시간 (초)
    interval = 1  # 확인 간격 (초)
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            if es.ping():
                print("Elasticsearch is ready!")
                return es
        except Exception as e:
            print(f"Waiting for Elasticsearch to be ready... ({e})")
        time.sleep(interval)

    # 시간 초과 시 예외 발생
    raise TimeoutError("Elasticsearch did not become ready in time")



# 새로운 엘라스틱서치_색인 생성
def create_es_index(index, es):
    settings = {
        "analysis": {
            "analyzer": { # 분석기로 nori를 선택함(한국어 이기 때문)
                "nori": { 
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed", # 복합어 분리 방식: mixed는 복합어를 분리한 결과 및 복합어 원본을 모두 저장
                    "filter": ["nori_posfilter"] # 필요없는 품사는 생략하도록 지정한다. 지정한 필터 이름이 "nori_posfilter" 이다.
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"] # 어미, 조사, 구분자, 줄임표, 지정사, 보조 용언 등
                }
            }
        }
    }

    # 색인을 위한 mapping 설정 (역색인 필드, 임베딩 필드 모두 설정)
    mappings = {
        "properties": {
            "content": { # 엘라스틱서치에서 text 유형의 필드는 기본적으로 색인됨
                "type": "text",
                "analyzer": "nori"
            },
            "embeddings": {
                "type": "dense_vector",
                "dims": 768, # 벡터의 차원
                "index": True, # "embeddings" 필드도 색인하겠다는 뜻임
                "similarity": "l2_norm" # 유클리드 거리 방식으로 유사도 계산함
                # "similarity": "hnsw" # 유클리드 거리 방식으로 유사도 계산함
            }
        }
    }
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)
    return es



# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(index, docs, es):
    actions = []
    for doc in docs:
        action = {
            '_index': index,
            '_source': doc
        }
        actions.append(action)
    return helpers.bulk(es, actions)

# 역색인을 이용한 검색
def sparse_retrieve(query_str, size, es):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index="test", query=query, size=size, sort="_score")

# Vector 유사도를 이용한 검색
def dense_retrieve(embedding_module, es, query_str, size):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = embedding_module.model.encode([query_str])[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index="test", knn=knn)

# LLM과 검색엔진을 활용한 RAG 구현
# messages: [{"role": "user", "content": "식물이 빛을 에너지로 변환하는 과정에 대해 설명해줘."}]
def answer_question(messages, es, persona_function_calling, client, llm_model, tools):
    # 함수 출력 초기화
    # 순서대로, 사용자의 질문, 리트리버가 찾아온 관련 문서 개수, 관련 문서 내용, llm의 답변
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    #msg = [{"role": "system", "content": persona_function_calling}, {"role": "user", "content": "식물이 빛을 에너지로 변환하는 과정에 대해 설명해줘."}]
    
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        search_result = sparse_retrieve(standalone_query, 3, es)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response

# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
# def eval_rag(react_agent, eval_filename, output_filename, es, persona_function_calling, client, llm_model, tools):
def eval_rag(react_agent, eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            if idx > 5:
              break
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            # response = answer_question(j["msg"], es, persona_function_calling, client, llm_model, tools)
            response = react_agent.answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1


class OpenAI_GPT:
    def __init__(self, llm_model="gpt-3.5-turbo-1106", key_file='./ex-key/openai-helpotcreator-key-all.txt', es=None):
        self.llm_model = llm_model
        self.api_key = self.load_api_key(key_file)  # 키 로딩 중 오류 발생 시 예외 발생
        self.client = OpenAI()
        self.es = es  # Elasticsearch 또는 다른 검색 엔진 객체
        self.persona_qa = self.get_persona_qa_prompt()
        self.persona_function_calling = self.get_persona_function_calling_prompt()
        self.tools = self.get_tools()

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

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search relevant documents",
                    "parameters": {
                        "properties": {
                            "standalone_query": {
                                "type": "string",
                                "description": "Final query suitable for use in search from the user messages history."
                            }
                        },
                        "required": ["standalone_query"],
                        "type": "object"
                    }
                }
            }
        ]

    def answer_question(self, messages):
        """질문에 대한 답변을 생성하고 필요한 경우 검색을 수행하는 함수"""
        response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

        # LLM을 통한 질의 분석 및 함수 호출
        msg = [{"role": "system", "content": self.persona_function_calling}] + messages
        try:
            result = self.client.chat.completions.create(
                model=self.llm_model,
                messages=msg,
                tools=self.tools,
                temperature=0,
                seed=1,
                timeout=10
            )
        except Exception as e:
            traceback.print_exc()
            return response

        # 검색 호출이 필요한 경우
        if result.choices[0].message.tool_calls:
            tool_call = result.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            standalone_query = function_args.get("standalone_query")

            # Baseline으로는 sparse_retrieve 함수를 외부에서 호출하여 검색 결과 추출
            search_result = sparse_retrieve(standalone_query, 3, self.es)

            response["standalone_query"] = standalone_query
            retrieved_context = []
            for i, rst in enumerate(search_result['hits']['hits']):
                retrieved_context.append(rst["_source"]["content"])
                response["topk"].append(rst["_source"]["docid"])
                response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

            # 검색 결과를 사용하여 다시 LLM 호출
            content = json.dumps(retrieved_context)
            messages.append({"role": "assistant", "content": content})
            msg = [{"role": "system", "content": self.persona_qa}] + messages
            try:
                qa_result = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
            except Exception as e:
                traceback.print_exc()
                return response

            response["answer"] = qa_result.choices[0].message.content

        # 검색 없이 바로 답변 생성
        else:
            response["answer"] = result.choices[0].message.content

        return response


def start():
    # embedding_module = interface_embedding_module()
    embedding_module = SentenceEmbeddings()

    # 문서의 content 필드 내용을 임베딩 형태로 바꾼다.
    with open("./data/documents.jsonl") as f:
        docs = [json.loads(line) for line in f]
    embeddings = embedding_module.get_embeddings_in_batches(docs, 100)

    # 원본 데이터에, "embeddings"라는 필드를 추가한다.
    index_docs = []
    for doc, embedding in zip(docs, embeddings):
        doc["embeddings"] = embedding.tolist()
        index_docs.append(doc) # 색인 완료된 문서
    


    # 임베딩을 저장할 DB을 실행한다.
    es = interface_embedding_db()

    create_es_index("test", es)

    # DB 인덱스에 (원본 + embeddings) 내용을 저장한다.
    ret = bulk_add("test", index_docs, es)



    # 검색엔진에 색인이 잘 되었는지 테스트하기 위한 질의
    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

    # 역색인을 사용하는 검색 예제
    search_result_retrieve = sparse_retrieve(test_query, 3, es)

    # 결과 출력 테스트
    for rst in search_result_retrieve['hits']['hits']:
        print('_score:', rst['_score'], '_source:', rst['_source']["content"])

    # Vector 유사도 사용한 검색 예제
    search_result_retrieve = dense_retrieve(embedding_module, es, test_query, 3)

    # 결과 출력 테스트
    for rst in search_result_retrieve['hits']['hits']:
        print('_score:', rst['_score'], '_source:', rst['_source']["content"])



    # ## RAG 구현
    # 준비된 검색엔진과 LLM을 활용하여 대화형 RAG 구현
    llm_model = "gpt-3.5-turbo-1106"
    openai_key_file = './ex-key/openai-helpotcreator-key-all.txt'
    react_agent = OpenAI_GPT(llm_model, openai_key_file, es)


    eval_rag(react_agent, "./data/eval.jsonl", "./sample_submission.csv")

if __name__ == "__main__":
    start()
