import os
import pickle
import faiss
import numpy as np
import unicodedata
import logging
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSDatabaseManager:
    def __init__(self, db_path, chunk_strategy="paragraph", search_strategy="knn_best_field", embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        초기화 메서드.
        
        :param db_path: FAISS DB 저장 경로.
        :param chunk_strategy: 텍스트를 청크로 나누는 방법을 선택. ("paragraph" 또는 "serm")
        :param search_strategy: 검색 전략을 선택. ("knn_best_field")
        :param embedding_model_name: 임베딩 모델의 이름.
        """
        self.db_path = db_path
        self.chunk_strategy = chunk_strategy
        self.search_strategy = search_strategy
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def save_faiss_db(self, faiss_index, documents):
        """FAISS 인덱스와 관련 문서 데이터를 파일로 저장"""
        faiss.write_index(faiss_index, self.db_path)
        with open(self.db_path + '_docs.pkl', 'wb') as f:
            pickle.dump(documents, f)
        logger.info(f"FAISS DB and documents saved to {self.db_path}")

    def load_faiss_db(self):
        """FAISS 인덱스와 관련 문서 데이터를 파일에서 로드"""
        faiss_index = faiss.read_index(self.db_path)
        with open(self.db_path + '_docs.pkl', 'rb') as f:
            documents = pickle.load(f)
        logger.info(f"FAISS DB and documents loaded from {self.db_path}")
        return faiss_index, documents

    def normalize_string(self, s):
        """유니코드 정규화"""
        return unicodedata.normalize('NFC', s)

    def chunk_text(self, text, max_length=512):
        """텍스트를 선택한 청크 전략에 따라 분할"""
        if self.chunk_strategy == "paragraph":
            return self.paragraph_based_chunking(text, max_length)
        else:
            raise ValueError("Invalid chunk strategy")

    def paragraph_based_chunking(self, text, max_length=512):
        """문단 단위로 텍스트를 청크로 분할"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            if current_length + len(paragraph) > max_length:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks

    def extract_best_field(self, doc):
        """문서에서 가장 중요한 필드 추출"""
        return doc.metadata.get('title', '') + " " + doc.page_content

    def create_faiss_index(self, texts, embedding_dim=None):
        """임베딩을 생성하고 FAISS 인덱스를 구축"""
        embeddings = self.embedding_model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')
        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]

        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(embeddings)
        return faiss_index

    def process_documents(self, df, max_length=512):
        """데이터프레임에서 문서를 로드하고 청크로 분할"""
        documents = []
        pdf_files = df['Source_path'].unique()
        logger.info(f"Loading PDF files from: {len(pdf_files)}")
        
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pdf_documents = loader.load()
            
            for pdf_document in pdf_documents:
                pdf_document.page_content = self.normalize_string(pdf_document.page_content.replace("\x07", ""))
                chunks = self.chunk_text(pdf_document.page_content, max_length)
                
                for chunk in chunks:
                    doc = Document(page_content=chunk, metadata=pdf_document.metadata)
                    documents.append(doc)
        
        return documents

    def make_db(self, df, max_length=512, embedding_dim=None, fewshot=False):
        """
        DB를 생성하는 메서드. fewshot이 True이면 Few-shot DB 생성 방식 적용.
        
        :param df: 데이터프레임.
        :param max_length: 텍스트 청크의 최대 길이.
        :param embedding_dim: 임베딩 차원.
        :param fewshot: Few-shot DB 생성 여부.
        """
        if os.path.exists(self.db_path):
            self.load_faiss_db()
            return

        if fewshot:
            df = df.drop('SAMPLE_ID', axis=1)
            records = df.to_dict(orient='records')
            texts = ["\n\n".join(self.normalize_string(value) for value in record.values()) for record in records]
        else:
            documents = self.process_documents(df, max_length)
            texts = [self.extract_best_field(doc) for doc in documents]

        faiss_index = self.create_faiss_index(texts, embedding_dim)
        self.save_faiss_db(faiss_index, documents if not fewshot else records)

    def search_documents(self, query, top_k=10):
        """
        선택된 검색 전략에 따라 검색을 수행하는 메서드.
        
        :param query: 검색 쿼리.
        :param top_k: 반환할 상위 k개의 결과 수.
        :return: 검색된 문서 또는 예제 리스트.
        """
        faiss_index, documents = self.load_faiss_db()
        query_embedding = self.embedding_model.encode([query])
        _, indices = faiss_index.search(np.array(query_embedding).astype('float32'), top_k)

        if self.search_strategy == "knn_best_field":
            results = self._apply_best_field_strategy(documents, indices[0], query)
        else:
            results = [documents[i] for i in indices[0]]

        return results

    def _apply_best_field_strategy(self, documents, indices, query):
        """
        Best Field 전략을 적용하여 검색 결과를 처리하는 메서드.
        
        :param documents: 검색된 문서 리스트.
        :param indices: 검색된 문서의 인덱스 리스트.
        :param query: 검색 쿼리.
        :return: Best Field 전략이 적용된 검색 결과 리스트.
        """
        results = []
        for i in indices:
            doc = documents[i]
            if query.lower() in doc.metadata.get('title', '').lower():
                results.insert(0, doc)
            else:
                results.append(doc)
        return results
