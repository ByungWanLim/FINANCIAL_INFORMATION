import os
import pickle
import faiss
import numpy as np
import unicodedata
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class FAISSDatabaseManager:
    def __init__(self, chunk_strategy="paragraph", search_strategy="knn_best_field", embedding_model_name='intfloat/multilingual-e5-small'):
        """
        초기화 메서드.
        
        :param chunk_strategy: 텍스트를 청크로 나누는 방법을 선택. ("paragraph" 또는 "serm")
        :param search_strategy: 검색 전략을 선택. ("knn_best_field" 또는 "serm_best_field")
        :param embedding_model_name: 임베딩 모델의 이름.
        """
        self.chunk_strategy = chunk_strategy
        self.search_strategy = search_strategy
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def save_faiss_db(self, faiss_index, db_path, documents):
        """FAISS 인덱스와 관련 문서 데이터를 파일로 저장"""
        faiss.write_index(faiss_index, db_path)
        with open(db_path + '_docs.pkl', 'wb') as f:
            pickle.dump(documents, f)
        print(f"FAISS DB and documents saved to {db_path}")

    def load_faiss_db(self, db_path):
        """FAISS 인덱스와 관련 문서 데이터를 파일에서 로드"""
        faiss_index = faiss.read_index(db_path)
        with open(db_path + '_docs.pkl', 'rb') as f:
            documents = pickle.load(f)
        print(f"FAISS DB and documents loaded from {db_path}")
        return faiss_index, documents

    def normalize_string(self, s):
        """유니코드 정규화"""
        return unicodedata.normalize('NFC', s)

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

    def serm_based_chunking(self, text, max_length=512):
        """SERM 기반의 텍스트 청크 생성"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def extract_best_field(self, doc):
        """문서에서 가장 중요한 필드 추출"""
        if 'title' in doc.metadata:
            return doc.metadata['title'] + " " + doc.page_content
        else:
            return doc.page_content

    def create_faiss_index(self, texts, embedding_dim=None):
        """임베딩을 생성하고 FAISS 인덱스를 구축"""
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings).astype('float32')
        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]

        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_index.add(embeddings)
        return faiss_index

    def make_db(self, df, db_path, max_length=512, embedding_dim=None, fewshot=False):
        """
        DB를 생성하는 메서드. fewshot이 True이면 Few-shot DB 생성 방식 적용.
        
        :param df: 데이터프레임.
        :param db_path: FAISS DB 저장 경로.
        :param max_length: 텍스트 청크의 최대 길이.
        :param embedding_dim: 임베딩 차원.
        :param fewshot: Few-shot DB 생성 여부.
        """
        if db_path is not None and os.path.exists(db_path):
            print(f"Loading FAISS DB from: {db_path}")
            self.load_faiss_db(db_path)
            return

        if fewshot:
            # Few-shot DB 생성
            df = df.drop('SAMPLE_ID', axis=1)
            records = df.to_dict(orient='records')
            texts = ["\n\n".join(self.normalize_string(value) for value in record.values()) for record in records]
            faiss_index = self.create_faiss_index(texts, embedding_dim)
            self.save_faiss_db(faiss_index, db_path, records)
            return

        # 일반 DB 생성
        documents = []
        pdf_files = df['Source_path'].unique()
        print(f"Loading PDF files from: {len(pdf_files)}")
        
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            pdf_documents = loader.load()
            
            for pdf_document in pdf_documents:
                pdf_document.page_content = self.normalize_string(pdf_document.page_content.replace("\x07", ""))
                
                if self.chunk_strategy == "paragraph":
                    document_chunks = self.paragraph_based_chunking(pdf_document.page_content, max_length)
                elif self.chunk_strategy == "serm":
                    document_chunks = self.serm_based_chunking(pdf_document.page_content, max_length)
                
                for chunk in document_chunks:
                    doc = Document(page_content=chunk, metadata=pdf_document.metadata)
                    documents.append(doc)
        
        texts = [self.extract_best_field(doc) for doc in documents]
        faiss_index = self.create_faiss_index(texts, embedding_dim)
        self.save_faiss_db(faiss_index, db_path, documents)

    def search_documents(self, db_path, query, top_k=10):
        """
        선택된 검색 전략에 따라 검색을 수행하는 메서드.
        
        :param db_path: FAISS DB 경로.
        :param query: 검색 쿼리.
        :param top_k: 반환할 상위 k개의 결과 수.
        """
        faiss_index, documents = self.load_faiss_db(db_path)
        query_embedding = self.embedding_model.embed_query(query)
        _, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), top_k)

        if self.search_strategy in ["knn_best_field", "serm_best_field"]:
            results = [documents[i] for i in indices[0]]
            # for i, doc in enumerate(results):
            #     if isinstance(doc, dict):  # Few-shot 방식의 결과
            #         print(f"Result {i+1}: {doc}...")
            #     else:  # 일반 문서 검색 결과
            #         print(f"Result {i+1}: {doc.page_content[:200]}...")
        return results