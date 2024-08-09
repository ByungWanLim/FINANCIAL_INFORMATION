from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.prompts import (
    FewShotPromptTemplate
)
import pickle
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
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

def load_and_vectorize(csv_path, db_path):
    if os.path.exists(db_path) and os.path.exists(db_path + '_metadata.pkl'):
        print("Loading FAISS DB from:", db_path)
        db, metadata = load_faiss_db(db_path)
        return db

    # CSV 파일을 불러와 데이터프레임 생성
    train_df = pd.read_csv(csv_path)
    train_df.drop('SAMPLE_ID', axis=1, inplace=True)
    trainset = train_df.to_dict(orient='records')
    print("Loaded Fewshot Set:", trainset[:3])
    
    # 벡터화할 텍스트 생성 및 유니코드 정규화
    to_vectorize = ["\n\n".join(normalize_string(value) for value in example.values()) for example in trainset]
    
    # 벡터화 및 FAISS DB 생성
    fewshot_vectordb = FAISS.from_texts(to_vectorize, embedding=get_embedding(), metadatas=trainset)
    
    # FAISS DB 저장
    save_faiss_db(fewshot_vectordb, db_path)
    
    return fewshot_vectordb

def save_faiss_db(db, db_path):
    db.save_local(db_path)
    with open(db_path + '_metadata.pkl', 'wb') as f:
        pickle.dump(db, f)

def load_faiss_db(db_path):
    db = FAISS.load_local(db_path, embeddings=get_embedding(), allow_dangerous_deserialization=True)
    with open(db_path + '_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return db, metadata

def load_chunks_make_docdb(pdf_directory, db_path):
    if os.path.exists(db_path) and os.path.exists(db_path + '_metadata.pkl'):
        print("Loading FAISS DB from:", db_path)
        db, metadata = load_faiss_db(db_path)
        return db

    print("Loading PDF files from:", pdf_directory)
    documents = []

    # PDF 파일들을 로드하여 분할
    pdf_files = glob(os.path.join(pdf_directory, '*.pdf').replace('\\', '/'))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        documents.extend(pdf_documents)
    
    # 유니코드 정규화
    for doc in documents:
        doc.page_content = normalize_string(doc.page_content)

    # 분할된 텍스트를 벡터로 변환하여 FAISS DB에 저장
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    chunks = chunk_splitter.split_documents(documents)
    print("Done.", len(chunks), "chunks")
    
    print("Creating FAISS DB")
    # FAISS DB 생성 및 저장
    db = FAISS.from_documents(chunks, embedding=get_embedding())
    save_faiss_db(db, db_path)
    print("Done.")
    
    return db

def make_db(df):
    documents = []
    pdf_files = df['Source_path'].unique()
    print("Loading PDF files from:", len(pdf_files))
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pdf_documents = loader.load()
        documents.extend(pdf_documents)
    # 유니코드 정규화
    for doc in documents:
        doc.page_content = normalize_string(doc.page_content)
    chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    chunks = chunk_splitter.split_documents(documents)
    print("Done.", len(chunks), "chunks")
    
    print("Creating FAISS DB")
    # FAISS DB 생성 및 저장
    db = FAISS.from_documents(chunks, embedding=get_embedding())
    print("Done.")
    return db

def make_fewshot_db(df):
    df = df.drop('SAMPLE_ID', axis=1)
    df = df.to_dict(orient='records')
    # 벡터화할 텍스트 생성 및 유니코드 정규화
    to_vectorize = ["\n\n".join(normalize_string(value) for value in example.values()) for example in df]
    
    # 벡터화 및 FAISS DB 생성
    fewshot_vectordb = FAISS.from_texts(to_vectorize, embedding=get_embedding(), metadatas=df)
    return fewshot_vectordb