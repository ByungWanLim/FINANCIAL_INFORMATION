
from langchain_community.vectorstores import FAISS

import os

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document


from langchain_community.retrievers import KNNRetriever
from langchain_huggingface import HuggingFaceEmbeddings

import pandas as pd

import pickle

import unicodedata
import faiss
import numpy as np

from langchain.schema import Document

def normalize_string(s):
    """유니코드 정규화"""
    return unicodedata.normalize('NFC', s)

def get_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-base',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
        )
    return embeddings

# 문서에서 "Best Field" 추출 함수
def extract_best_field(doc: Document) -> str:
    if 'title' in doc.metadata:
        return doc.metadata['title'] + " " + doc.page_content
    else:
        return doc.page_content
    
    # FAISS 인덱스 생성 및 문서 임베딩 추가
def create_faiss_index_with_best_field(documents: list[Document], embedding_dim: int = 768) -> faiss.Index:
    embedding_model = get_embedding()
    
    # Best Field를 사용하여 문서 임베딩 생성
    best_field_texts = [extract_best_field(doc) for doc in documents]
    embeddings = embedding_model.embed_documents(best_field_texts)
    
    # FAISS 인덱스 생성 (L2 거리 기반)
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    faiss_index.add(np.array(embeddings).astype('float32'))
    
    return faiss_index

def save_faiss_db(db, db_path):
    db.save_local(db_path)
    with open(db_path + '_metadata.pkl', 'wb') as f:
        pickle.dump(db, f)

def load_faiss_db(db_path):
    db = FAISS.load_local(db_path, embeddings=get_embedding(), allow_dangerous_deserialization=True)
    with open(db_path + '_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return db, metadata

def make_db(df, db_path):
    if db_path is not None:
        if os.path.exists(db_path) and os.path.exists(db_path + '_metadata.pkl'):
            print("Loading FAISS DB from:", db_path)
            db, metadata = load_faiss_db(db_path)
            return db

    documents = []
    pdf_files = df['Source_path'].unique()
    print("Loading PDF files from:", len(pdf_files))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        for pdf_document in pdf_documents:
            pdf_document.page_content = pdf_document.page_content.replace("\x07","")
        documents.extend(pdf_documents)
    # 유니코드 정규화
    for doc in documents:
        doc.page_content = normalize_string(doc.page_content)
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = chunk_splitter.split_documents(documents)
    print("Done.", len(chunks), "chunks")
    
    print("Creating FAISS DB")
    # FAISS DB 생성 및 저장
    db = FAISS.from_documents(chunks, embedding=get_embedding())
    if db_path is not None:
        save_faiss_db(db, db_path)
    print("Done.")
    
    return db

def make_fewshot_db(df, db_path):
    if db_path is not None:
        if os.path.exists(db_path) and os.path.exists(db_path + '_metadata.pkl'):
            print("Loading FAISS DB from:", db_path)
            db, metadata = load_faiss_db(db_path)
            return db

    df = df.drop('SAMPLE_ID', axis=1)
    df = df.to_dict(orient='records')
    print("Loaded Fewshot Set:", df[:1])
    # 벡터화할 텍스트 생성 및 유니코드 정규화
    to_vectorize = ["\n\n".join(normalize_string(value) for value in example.values()) for example in df]
    
    # 벡터화 및 FAISS DB 생성
    print("Creating FAISS DB")
    fewshot_vectordb = FAISS.from_texts(to_vectorize, embedding=get_embedding(), metadatas=df)
        # FAISS DB 저장
    if db_path is not None:
        save_faiss_db(fewshot_vectordb, db_path)
    print("Done.")
    return fewshot_vectordb

def knn_db(df):
    df = df.drop('SAMPLE_ID', axis=1)
    documents = []
    pdf_files = df['Source_path'].unique()
    print("Loading PDF files from:", len(pdf_files))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        for pdf_document in pdf_documents:
            pdf_document.page_content = pdf_document.page_content.replace("\x07","")
        documents.extend(pdf_documents)
    # 유니코드 정규화
    for doc in documents:
        doc.page_content = normalize_string(doc.page_content)
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = chunk_splitter.split_documents(documents)
    print("Done.", len(chunks), "chunks")
    retriever = KNNRetriever.from_documents(chunks, embedding=get_embedding())
    
    return retriever

def best_field(text):
    return text