import json
import torch
import time
import numpy as np
import faiss
from scipy import stats
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F

from elasticsearch import Elasticsearch, helpers

import os
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

from openai import OpenAI

class Solar:
    def __init__(self, api_key_path, model="solar-pro"):
        self.api_key_path = api_key_path
        self.model = model
        self.api_key = self.load_api_key()
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.upstage.ai/v1/solar")

    def load_api_key(self):
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.readline().strip()
            if not api_key:
                raise ValueError("API key is empty.")
            # os.environ["OPENAI_API_KEY"] = api_key
            return api_key
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {self.api_key_path} not found.")
        except ValueError as e:
            raise ValueError(f"Error: {e}")

    def chat(self, system_message, human_message):
        set_stream = False
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": human_message}
            ],
            stream=set_stream,
        )

        if set_stream:
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    # print(chunk.choices[0].delta.content, end="")
                    response = chunk.choices[0].delta.content
        else:
            response = stream.choices[0].message.content # stream=False인 경우
        return response

    def prompt_1(self, human_message):
        system_message="""
            사용자가 내용을 입력하면, 핵심 내용을 질문 형태로 요약해줘.
            최대한 사용자가 작성한 단어를 그대로 사용하고, 임의로 새로운 단어 생성은 최대한 자제해줘.
            반말이면, 요약도 반말로 해주고, 존댓말이면, 요약도 존댓말로 해줘.
            영어 단어가 나와서, 요약할 때 해당 영단어를 적는것은 좋지만, 문장 전체를 영어로 적는 것은 자제해줘.
            원문보다 더 길게 요약하지 말아줘.
            쌍따옴표 사용하지 말아줘.
            """
        response = self.chat(system_message, human_message)
        return response

class Gemini:
    def __init__(self, api_key_path, model="gemini-1.5-flash"):
        self.api_key_path = api_key_path
        self.model = None
        self.load_api_key()
        self.configure_model(model)

    def load_api_key(self):
        """Load API key from a specified file."""
        try:
            with open(self.api_key_path, 'r') as file:
                api_key = file.readline().strip()
            if not api_key:
                raise ValueError("API key is empty.")
            os.environ["OPENAI_API_KEY"] = api_key
            return api_key
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {self.api_key_path} not found.")
        except ValueError as e:
            raise ValueError(f"Error: {e}")

    def configure_model(self, model):
        """Configure the generative model."""
        genai.configure(api_key=os.environ["OPENAI_API_KEY"])
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(m.name)
        self.model = genai.GenerativeModel(model)

    @staticmethod
    def to_markdown(text):
        """Convert text to Markdown format."""
        text = text.replace("•", "  *")
        return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))

    def embed_content(self, content, task_type="retrieval_document", max_dim=768):
        """Embed content and ensure the embedding dimension does not exceed max_dim."""
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=content,
            task_type=task_type,
            title="Embedding of single string"
        )

        # 임베딩 벡터를 768 차원으로 제한
        embedding = result['embedding']
        if len(embedding) > max_dim:
            embedding = embedding[:max_dim]  # 768차원으로 자름

        # 임베딩 결과 반환
        return embedding

    def prompt_1(self, content):
        response = self.model.generate_content(f"""
            - 내용: {content}\n
            \n
            #######\n

            위 내용을 요약해줘.
            핵심적인 단어는 최대한 살려줘.
            결과만 적어주고, 다른 내용은 적지 말아줘.
            """,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=["x"],
                max_output_tokens=100,
                temperature=0.1,
            )
        )

        # display(response.text)
        # display(response.candidates)
        return response.text

    def prompt_2(self, content):
        response = self.model.generate_content(f"""
            - 내용: {content}\n
            \n
            #######\n

            위 내용을 질문 형태로 요약해줘.
            핵심적인 단어는 최대한 살려줘.
            결과만 적어주고, 다른 내용은 적지 말아줘.
            """,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                stop_sequences=["x"],
                max_output_tokens=100,
                temperature=0.1,
            )
        )

        # display(response.text)
        # display(response.candidates)
        return response.text


class Embedder:
    def __init__(self, model_name="klue/roberta-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # embed 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

        # summarize 모델 초기화
        self.summary_tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained("digit82/kobart-summarization")
        self.summary_model.to(self.device)

    def summarize(self, text, text_max_length):
        """
        문장이 최대 글자수를 초과할 경우, 요약해서 리턴
        :param text: 요약할 문장 (str)
        :param text_max_length: 최대 글자수 (int)
        :return: 요약된 문장 (str)
        """
        # print(f'=== 요약 시작:\n{text}')
        inputs = self.summary_tokenizer.encode(text, return_tensors="pt").to(self.device)
        summary_ids = self.summary_model.generate(inputs, max_length=text_max_length, length_penalty=1.0)
        summary = self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # print(f'=== 요약 완료:\n{summary}')
        return summary

    def embed(self, texts, batch_size=32, text_max_length=500):
        """
        문장 리스트를 배치 단위로 묶어서 임베딩으로 변환
        :param texts: 요약할 문장 리스트 (list of str)
        :param batch_size: 배치 크기 (int)
        :return: 임베딩된 리스트 (list of torch.Tensor)
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # 500글자 초과일 경우만 요약
            # batch_texts = [self.summarize(text, text_max_length) if len(text) > text_max_length else text for text in batch_texts]
            # batch_texts = [self.gemini.prompt_1(text) if len(text) > text_max_length else text for text in batch_texts]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # 평균 풀링
                
                # 임베딩 차원 조정
                if batch_embeddings.shape[1] != 768:
                    batch_embeddings = F.interpolate(batch_embeddings.unsqueeze(1), size=768, mode='linear', align_corners=False).squeeze(1)
                
                embeddings.append(batch_embeddings.cpu())
            print(f'=== 임베딩 완료: {i:4d} ~ {(i + batch_size - 1):4d}번 ')
        return torch.cat(embeddings, dim=0)


    def custom_embed_in_file_documents(self, docs):
        """
        각 문서의 'content' 필드를 요약하고, 임베딩화한 결과를 반환
        :param docs: 문서 리스트 (list of dict)
        :return: 임베딩 리스트 (list of torch.Tensor)
        """
        texts = [doc["content"] for doc in docs]  # 문서의 content 추출
        embeddings = self.embed(texts, batch_size=32, text_max_length=500)  # 임베딩 수행
        return embeddings

    def custom_embed_in_file_eval(self, docs):
        """
        각 평가 문서의 임베딩화한 결과를 반환
        :param docs: 평가 문서 리스트 (list of dict)
        :return: 임베딩 리스트 (list of torch.Tensor)
        """
        texts = []
        for doc in docs:
            summary = doc["summary"]
            texts.append(summary)
        embeddings = self.embed(texts, batch_size=32, text_max_length=500)  
        return embeddings


class ElasticDB:
    def __init__(self):
        es_password = 'r7ymfymLEmqoEPaUYlGy'
        es_username = 'elastic'
        self.es = None

        try:
            # Elasticsearch 인스턴스 초기화
            self.es = Elasticsearch(
                ['https://localhost:9200'],
                basic_auth=(es_username, es_password),
                ca_certs="/kkh/elasticsearch-8.8.0/config/certs/http_ca.crt",
                verify_certs=True,
                http_compress=True  # HTTP 압축 활성화
            )

            # 서버가 실행 중인지 확인
            if self.es.ping():
                print("Elasticsearch is already running!")
                return  # 이미 실행 중이므로 종료

        except Exception as e:
            print(f"Failed to connect to Elasticsearch: {e}")
            return

        # Elasticsearch 서버가 준비될 때까지 대기
        timeout = 300  # 최대 대기 시간 (초)
        interval = 5  # 확인 간격 (초)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if self.es.ping():
                    print("Elasticsearch is ready!")
                    return  # 서버가 준비되면 종료
            except Exception as e:
                print(f"Waiting for Elasticsearch to be ready... ({e})")
            time.sleep(interval)

        raise TimeoutError("Elasticsearch did not become ready in time")

    # 새로운 엘라스틱서치_색인 생성
    def create_es_index(self, index):
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
                    "dims": 768, # 벡터의 차원. 문장이 매우 길면 뒷 부분이 삭제되어 저장된다.
                    "index": True, # "embeddings" 필드도 색인하겠다는 뜻임
                    # "similarity": "l2_norm" # 유클리드 거리 방식으로 유사도 계산함
                    "similarity": "cosine"
                }
            }
        }
        if self.es.indices.exists(index=index):
            self.es.indices.delete(index=index)
        self.es.indices.create(index=index, settings=settings, mappings=mappings)
        return self.es


    # Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
    def bulk_add(self, index, docs, chunk_size=1000):
        for i in range(0, len(docs), chunk_size):
            chunk = docs[i:i + chunk_size]
            actions = [
                {
                    '_index': index,
                    '_source': doc
                    # '_source': {
                    #     'content': doc['content'],  # 문서의 내용을 저장
                    #     'embeddings_dense': doc['embeddings'],  # 밀집 임베딩 저장
                    #     'embeddings_sparse': doc['embeddings']   # 희소 임베딩 저장
                    #     # 여기 적지 않은 나머지 필드는 자동으로 채워진다.
                    # }
                } for doc in chunk
            ]
            helpers.bulk(self.es, actions)
    

    def sparse_retrieve(self, query_str, size=200):
        query = {
            "query": {
                "match": {
                    "content": {
                        "query": query_str,
                        "analyzer": "nori",
                        "boost": 1.0
                    }
                }
            }
        }
        return self.es.search(index="test", body=query, size=size)


    def dense_retrieve(self, query_vector, size=200):
        query = {
            "knn": {
                "field": "embeddings",
                "query_vector": query_vector,
                "k": size,
                "num_candidates": 1000
            },
            "_source": ["content"]  # 필요한 필드만 반환
        }
        return self.es.search(index="test", body=query)


class FaissDB:
    def __init__(self, dim=768):
        self.index = faiss.IndexHNSWFlat(dim, 32)  # 768차원, HNSW 설정(32개의 연결 수)
        self.index.hnsw.efSearch = 512  # 검색 효율을 위해 탐색할 이웃 수를 512로 설정
        self.index.hnsw.efConstruction = 256  # 그래프 생성 시 사용할 이웃 수를 256으로 증가
        self.data = []
        self.id_map = {}

    # FAISS에 벡터 추가 (대량으로도 가능)
    def add(self, docs):
        vectors = []
        for i, doc in enumerate(docs):
            embeddings = np.array(doc['embeddings'], dtype='float32')
            vectors.append(embeddings)
            self.data.append(doc['content'])
            self.id_map[i] = len(self.data) - 1

        vectors = np.array(vectors, dtype='float32')
        self.index.add(vectors)  # 벡터 추가

    # Dense HNSW 방식 검색
    def dense_retrieve(self, query_vector, k=200):
        query_vector = np.array(query_vector, dtype='float32').reshape(1, -1)
        D, I = self.index.search(query_vector, k)  # k개의 가장 가까운 이웃 검색

        results = []
        for rank, i in enumerate(I[0]):
            if i != -1:
                result = {
                    'rank': rank + 1,  # 순위는 1부터 시작
                    'distance': D[0][rank],  # 유사도 점수를 거리로 저장
                    'content': self.data[self.id_map[i]],
                }
                results.append(result)

        return results


class Reranker:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to('cuda')
        self.model.eval()

    def exp_normalize(self, x):
        b = x.max()
        y = np.exp(x - b)
        return y / y.sum()

    def rerank(self, eval_item, docs, k=3):
        pairs = [[eval_item['summary'], doc['content']] for doc in docs]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = self.exp_normalize(scores.cpu().numpy())
        
        # 스코어와 인덱스 페어 생성
        scored_docs = [(score, idx) for idx, score in enumerate(scores)]
        scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True) # 내림차순 정렬

        topk = []
        references = []
        for score, idx in scored_docs[:k]:
            topk.append(docs[idx]["docid"])
            references.append({"score": float(score), "content": docs[idx]['content']})
        
        result = {"eval_id": eval_item["eval_id"], "standalone_query": eval_item["summary"], "topk": topk, "answer": references[0]["content"], "references": references}
        return result


def tfidf_compare(query, docs, k=10):
    vectorizer = TfidfVectorizer()

    # 문서들의 content 필드만 추출
    contents = [doc['content'] for doc in docs]
    doc_tfidf_matrix = vectorizer.fit_transform(contents) # 문서들은 contents 리스트에 추가된 순서대로 인덱스로 구분하면 됨.
    
    # 쿼리만 따로 벡터화 (벡터화된 문서들과 동일한 TF-IDF 벡터 공간에서 처리)
    query_tfidf_vector = vectorizer.transform([query])
    
    # 코사인 유사도를 계산하여 문서들 중에서 가장 유사한 k개의 문서 선택
    similarities = (doc_tfidf_matrix * query_tfidf_vector.T).toarray().flatten()
    sorted_indices = similarities.argsort()[::-1][:k]  # 유사도 높은 순서대로 정렬
    
    results = []
    for rank, i in enumerate(sorted_indices):
        result = {
            'rank': rank + 1,  # 순위
            # 'similarity': similarities[i],  # 유사도 점수
            # 'docid': docs[i]['docid'],  # 문서의 docid
            # 'src': docs[i]['src'],  # 문서의 src
            'content': docs[i]['content']  # 문서의 content
        }
        print(result)
        results.append(result)
    return results


def save_file(docs, file_path):
    """
    문서를 지정한 경로에 JSONL 형식으로 저장
    :param docs: 수정된 문서 리스트 (list of dict)
    :param file_path: 저장할 파일 경로 (str)
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def add_embeddings_to_docs(docs, embeddings):
    """
    문서 리스트에 임베딩을 추가하여 반환
    :param docs: 문서 리스트 (list of dict)
    :param embeddings: 임베딩 리스트 (list of torch.Tensor)
    :return: 임베딩이 추가된 문서 리스트 (list of dict)
    """
    for doc, embedding in zip(docs, embeddings):
        doc["embeddings"] = embedding.tolist()
    return docs


def make_summary_eval(docs, summary_model):
    texts = []
    for doc in docs:
        messages = doc["msg"]
        summarized_text = messages[0]["content"]
        if len(messages) > 1:  # 대화문이 2개 이상인 경우
            full_conversation = "  ".join([f"{msg['content']}" for msg in messages])
            print(f'=== 요약 시작(eval):\n{full_conversation}')
            summarized_text = summary_model.prompt_1(full_conversation)
            import time
            time.sleep(2)
            # summarized_text = full_conversation
            print(f'=== 요약 완료(eval):\n{summarized_text}')
        doc["summary"] = summarized_text
        texts.append(doc)  # 최종 텍스트 리스트에 추가
    return texts


def read_jsonl_file(file_path):
    doc_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc_list.append(json.loads(line))
    return doc_list


def db_es_test_sparse(db, test_query):
    # 벡터DB_테스트_sparse
    sparse_result = db.sparse_retrieve(test_query, size=7)
    # 벡터DB_테스트_sparse_결과
    print()
    print('=== sparse =========')
    print('====================')
    print()
    for rst in sparse_result['hits']['hits']:
        print('_score: ', rst['_score'], '   content: ', rst['_source']["content"])


def db_es_test_dense(db, embeddings_test_query):
    # 벡터DB_테스트_dense
    dense_result = db.dense_retrieve(embeddings_test_query, size=7)
    # 벡터DB_테스트_dense_결과
    print()
    print('=== dense =========')
    print('====================')
    print()
    for rst in dense_result['hits']['hits']:
        print('_score: ', rst['_score'], '   content: ', rst['_source']["content"])


def db_es_test_fusion(db, test_query, embeddings_test_query):
    # 벡터DB_테스트_sparse
    sparse_result = db.sparse_retrieve(test_query, size=7)
    # 벡터DB_테스트_sparse_결과
    print()
    print('=== sparse =========')
    print('====================')
    print()
    for rst in sparse_result['hits']['hits']:
        print('_score: ', rst['_score'], '   content: ', rst['_source']["content"])

    # 벡터DB_테스트_dense
    dense_result = db.dense_retrieve(embeddings_test_query, size=7)
    # 벡터DB_테스트_dense_결과
    print()
    print('=== dense =========')
    print('====================')
    print()
    for rst in dense_result['hits']['hits']:
        print('_score: ', rst['_score'], '   content: ', rst['_source']["content"])

    # 벡터DB_테스트_fusion
    sparse_scores = [hit['_score'] for hit in sparse_result['hits']['hits']]
    dense_scores = [hit['_score'] for hit in dense_result['hits']['hits']]

    # Z-score로 정규화
    sparse_z_scores = stats.zscore(sparse_scores) if len(sparse_scores) > 1 else sparse_scores
    dense_z_scores = stats.zscore(dense_scores) if len(dense_scores) > 1 else dense_scores

    # 결과 병합 (문서 ID를 기준으로 중복 체크)
    merged_results = {}
    # Sparse 결과 병합
    for idx, hit in enumerate(sparse_result['hits']['hits']):
        doc_id = hit['_id']
        merged_results[doc_id] = {
            'document': hit['_source'],
            'z_score': sparse_z_scores[idx]
        }

    # Dense 결과 병합
    for idx, hit in enumerate(dense_result['hits']['hits']):
        doc_id = hit['_id']
        if doc_id in merged_results:
            # 중복된 문서일 경우 Z-score 합산
            merged_results[doc_id]['z_score'] += dense_z_scores[idx]
        else:
            # 새 문서일 경우 추가
            merged_results[doc_id] = {
                'document': hit['_source'],
                'z_score': dense_z_scores[idx]
            }

    # Z-score 기준으로 결과 정렬
    # sorted_results = sorted(merged_results.items(), key=lambda x: x[1]['z_score'], reverse=True)
    # 1. merged_results.items()는 사전의 (key, value) 쌍들을 리스트 형태로 반환합니다.
    # key는 문서의 ID, value는 문서와 z_score 정보를 가진 또 다른 사전입니다.
    items_list = list(merged_results.items())

    # 2. 리스트에서 각 요소는 (key, value) 쌍이고, 여기서 value['z_score'] 값을 기준으로 정렬합니다.
    def get_z_score(item):
        return item[1]['z_score']

    # 3. 각 요소의 z_score를 기준으로 내림차순으로 정렬합니다.
    sorted_results = sorted(items_list, key=get_z_score, reverse=True)

    # 7. 최종 결과 리턴 (정렬된 순으로 문서 및 Z-score 반환)
    final_results = [{'document': result[1]['document'], 'z_score': result[1]['z_score']} for result in sorted_results]
    print()
    print('=== fusion =========')
    print('===========================')
    print()
    for doc in final_results:
        print('_score: ', doc['z_score'], '   content: ', doc['document']['content'])
        print('@@@===@@@')


def db_faiss_test_dense(db, embeddings_test_query):
    # 벡터DB_테스트_dense
    dense_result = db.dense_retrieve(embeddings_test_query, k=7)
    # 벡터DB_테스트_dense_결과
    print()
    print('=== dense =========')
    print('====================')
    print()
    for rst in dense_result:
        print('rank: ', rst['rank'], '   content: ', rst["content"])


def final_eval(db, reranker, eval_list, docs, output_filename):
    output = []
    for index, eval_item in enumerate(eval_list):
        eval_id = eval_item['eval_id']
        eval_content = eval_item['summary']

        if eval_id in [276, 261, 283, 32, 94, 90, 220, 245, 229, 247, 67, 57, 227, 2, 301, 222, 83, 64, 103, 218]:
            output.append({"eval_id": eval_item["eval_id"], "standalone_query": eval_item["summary"], "topk": [], "answer": "과학 지식을 벗어난 질문이라서, 답변 드리기 어렵습니다.", "references": []})
        else:
            # BM25
            bm25_results = db.sparse_retrieve(eval_content, size=200)

            # 리랭킹
            reranked_results = reranker.rerank(eval_item, [hit['_source'] for hit in bm25_results['hits']['hits']], k=3)
            output.append(reranked_results)
        print(f'최종 평가 {index} 번 완료')
    save_file(output, output_filename)
        

def embedding_in_documents(embedder, docs, save_file_path):
    new_docs = []
    for index, doc in enumerate(docs):
        doc["embeddings"] = embedder.embed_content(doc["content"], task_type="retrieval_document", max_dim=768)
        print(f'=== {index}번 문서_임베딩 시작 ===')
        print(f'- 원문: {doc["content"]}')
        print(f'=== {index}번 문서_임베딩 완료 ===')
        new_docs.append(doc)
    save_file(new_docs, save_file_path)


def embedding_in_eval(embedder, docs, save_file_path):
    new_docs = []
    for index, doc in enumerate(docs):
        doc["embeddings"] = embedder.embed_content(doc["summary"], task_type="retrieval_document", max_dim=768)
        print(f'=== {index}번 문서_임베딩 시작 ===')
        print(f'- 원문: {doc["summary"]}')
        print(f'=== {index}번 문서_임베딩 완료 ===')
        new_docs.append(doc)
    save_file(new_docs, save_file_path)


def start():
    # [문서 로딩]
    file_documents_list = read_jsonl_file("./data/documents.jsonl")
    file_eval_list = read_jsonl_file("./data/eval.jsonl")


    # [모델 생성]
    solar = Solar(api_key_path="./ex-key/upstage-helpotcreator-key.txt", model="solar-pro")
    embedder = Embedder("klue/roberta-large")
    gemini = Gemini(api_key_path='./ex-key/google-aistudio-helpotcreator-key.txt', model="gemini-1.5-flash")
    db_es = ElasticDB()
    db_faiss = FaissDB(dim=768)
    reranker = Reranker(model_name='Dongjin-kr/ko-reranker')
    

    # [요약]
    # file_eval_list_summary = make_summary_eval(docs=file_eval_list, summary_model=solar)
    # save_file(file_eval_list_summary, "./data/eval_summary.jsonl")


    # [저장한 파일들 로딩]
    file_eval_list_summary = read_jsonl_file("./data/eval_summary.jsonl")


    # [임베딩_test(gemini)]
    # embedding_in_documents(embedder=gemini, docs=file_documents_list, save_file_path="./data/documents_embed.jsonl")
    # embedding_in_eval(embedder=gemini, docs=file_eval_list_summary, save_file_path="./data/eval_embed.jsonl")



    # [임베딩_documents]
    # embeddings_documents = embedder.custom_embed_in_file_documents(file_documents_list)
    # file_documents_list_embed = add_embeddings_to_docs(file_documents_list, embeddings_documents)
    # save_file(file_documents_list_embed, "./data/documents_embed.jsonl")


    # [임베딩_eval]
    # embeddings_eval = embedder.custom_embed_in_file_eval(file_eval_list_summary)
    # file_eval_list_embed = add_embeddings_to_docs(file_eval_list_summary, embeddings_eval)
    # save_file(file_eval_list_embed, "./data/eval_embed.jsonl")


    # [저장한 파일들 로딩]
    file_documents_list_embed = read_jsonl_file("./data/documents_embed_gemini.jsonl")
    file_eval_list_embed = read_jsonl_file("./data/eval_embed_gemini.jsonl")


    # [DB_es_색인 생성]
    db_es.create_es_index("test")
    db_es.bulk_add("test", file_documents_list_embed) # DB에 (원본 + embeddings) 내용을 저장한다.


    # [DB_faiss_색인 생성]
    # db_faiss.add(docs=file_documents_list_embed)


    # [DB_테스트]
    # test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    # embeddings_test_query = gemini.embed_content(test_query, task_type="retrieval_document", max_dim=768)
    # embeddings_test_query = embedder.embed(texts=test_query , batch_size=1)
    # db_es_test_sparse(db=db_es, test_query=test_query)
    # db_es_test_dense(db=db_es, embeddings_test_query=embeddings_test_query)
    # db_es_test_fusion(db=db_es, embeddings_test_query=embeddings_test_query[0])
    # db_faiss_test_dense(db=db_faiss, embeddings_test_query=embeddings_test_query)
    # tfidf_compare(query=test_query, docs=file_documents_list_embed, k=7)


    # [최종 평가(+rerank)]
    final_eval(db=db_es, reranker=reranker, eval_list=file_eval_list_summary, docs=file_documents_list, output_filename='./data/submit.csv')

if __name__ == "__main__":
    start()