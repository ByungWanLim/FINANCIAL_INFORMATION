import os
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import unicodedata
def normalize_string(s):
    """유니코드 정규화"""
    return unicodedata.normalize('NFC', s)

def get_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-small',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def save_faiss_db(faiss_index, db_path):
    faiss.write_index(faiss_index, db_path)
    print(f"FAISS DB saved to {db_path}")

def load_faiss_db(db_path):
    faiss_index = faiss.read_index(db_path)
    print(f"FAISS DB loaded from {db_path}")
    return faiss_index

def extract_best_field(doc):
    # 문서에서 Best Field를 선택 (예: 제목을 선택)
    if 'title' in doc.metadata:
        return doc.metadata['title'] + " " + doc.page_content
    else:
        return doc.page_content

def make_db(df, db_path, embedding_dim=768):
    if db_path is not None and os.path.exists(db_path):
        print(f"Loading FAISS DB from: {db_path}")
        db = load_faiss_db(db_path)
        return db

    documents = []
    pdf_files = df['Source_path'].unique()
    print(f"Loading PDF files from: {len(pdf_files)}")
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        for pdf_document in pdf_documents:
            pdf_document.page_content = pdf_document.page_content.replace("\x07", "")
        documents.extend(pdf_documents)
    
    # 유니코드 정규화
    for doc in documents:
        doc.page_content = normalize_string(doc.page_content)
    
    # 문서 분할
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
    chunks = chunk_splitter.split_documents(documents)
    print(f"Done. {len(chunks)} chunks created.")
    
    # Best Field를 사용한 FAISS 인덱스 생성
    print("Creating FAISS DB with Best Field")
    best_field_texts = [extract_best_field(doc) for doc in chunks]
    embeddings = get_embedding().embed_documents(best_field_texts)
    
    # FAISS 인덱스 생성 (L2 거리 기반 KNN)
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(np.array(embeddings).astype('float32'))
    
    # FAISS 인덱스 저장
    if db_path is not None:
        save_faiss_db(faiss_index, db_path)
    print("Done.")
    
    return faiss_index
