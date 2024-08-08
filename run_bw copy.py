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
from langchain_huggingface import HuggingFacePipeline ,HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    FewShotPromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
import bitsandbytes as bnb
import pickle
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from langchain_community.document_transformers import LongContextReorder

from faiss_module import load_and_vectorize,load_chunks_make_docdb
from model_bw import setup_llm_pipeline
from save_bw import save
from seed import seed_everything

seed_everything(42)

def make_dict(dir='train.csv'):
    df = pd.read_csv(dir)
    df.drop('SAMPLE_ID', axis=1, inplace=True)
    
    return df.to_dict(orient='records')

def make_fewshot_prompt(fewshot_vectordb, k = 3):
    # Semantic Similarity Example Selector 설정
    example_prompt = PromptTemplate.from_template("<|start_header_id|>user<|end_header_id|>: {Question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>: <|begin_of_text|>{Answer}<|end_of_text|>")

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=fewshot_vectordb,
        k=k,
    )

    # FewShotPromptTemplate 생성
    fewshot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )
    return fewshot_prompt

def make_fewshot_string(fewshot_prompt, train_retriever, buff):
    ex_qa = fewshot_prompt.invoke({"input": buff['Question']}).to_string()
    fewshot_list = ex_qa.split('\n\n')[:-1]
    for i, entry in enumerate(fewshot_list):
        question = entry.split('\n')[0]
        question = question.replace('Question: ', '')
        retrieved_docs = train_retriever.invoke(question)
        num = "Example {}\n".format(i+1)
        fewshot_list[i] = num + "context: " + retrieved_docs[0].page_content + entry + '\n\n'
    return str(fewshot_list)

def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    context = ""
    for doc in docs:
        context += doc.page_content
        context += '\n\n'
    return context

def extract_answer(response):
    # AI: 로 시작하는 줄을 찾아 그 이후의 텍스트만 추출
    lines = response.split('\n')
    for line in lines:
        if line.startswith('Answer:'):
            return line.replace('Answer:', '').strip()
        if line.startswith('assistant:'):
            return line.replace('assistant:', '').strip()
    return response.strip()  # AI: 를 찾지 못한 경우 전체 응답을 정리해서 반환
    return response


def run(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    fewshot_db = load_and_vectorize('train.csv', './fewshot_faiss_db')
    fewshot_prompt = make_fewshot_prompt(fewshot_db)
    
    train_db = load_chunks_make_docdb('./train_source', './train_faiss_db')
    train_retriever = train_db.as_retriever(search_type = "mmr",search_kwargs={'k': 1})
    
    test_db = load_chunks_make_docdb('./test_source', './test_faiss_db')
    test_retriver = test_db.as_retriever(search_type = "mmr",search_kwargs={'k': 3})
    
    train_dict = make_dict('train.csv')
    test_dict = make_dict('test.csv')
    
    llm = setup_llm_pipeline(model_id)
    # reordering = LongContextReorder()
    results =[]
    
    for i in tqdm(range(len(test_dict))):
        
        if i % 30 == 0:
            print("Clearing cache")
            torch.cuda.empty_cache()

        fewshot_str = make_fewshot_string(fewshot_prompt, train_retriever, test_dict[i])
        # print(fewshot_str)
        
        full_template = """
"""+f"""
{fewshot_str}
"""+"""
다음 정보를 바탕으로 질문에 답하세요:

context: {context}

user: {input}

assistant:

주어진 질문에만 답변하세요. 
짧고 간결한 하나의 문장으로만 답변해주세요.
같은 내용을 반복하지 마세요.
"""
        prompt = PromptTemplate.from_template(full_template)
        qa_chain = (
        {
            "context": test_retriver | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        )
        # print("================================================")
        print("Questions: ",test_dict[i]['Question'])
        answer = qa_chain.invoke(test_dict[i]['Question'])
        answer = extract_answer(answer)
        results.append({
            "Question": test_dict[i]['Question'],
            "Answer": answer,
            "Source": test_dict[i]['Source']
            })
        print("Answer: ",results[-1]['Answer'])
        #print(results[-1]['Source'])
    
    save(results)

    
    
if __name__ == "__main__":
    # EleutherAI/polyglot-ko-1.3b
    #"meta-llama/Meta-Llama-3.1-8B-Instruct"
    # maywell/TinyWand-kiqu
    # yanolja/EEVE-Korean-Instruct-2.8B-v1.0
    # aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored 
    run(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct")