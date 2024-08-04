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

from faiss_module import load_and_vectorize,load_chunks_make_docdb
from model import setup_llm_pipeline
from save import save

def make_dict(dir='train.csv'):
    df = pd.read_csv(dir)
    df.drop('SAMPLE_ID', axis=1, inplace=True)
    
    return df.to_dict(orient='records')

def make_fewshot_prompt(fewshot_vectordb, k = 3):
    # Semantic Similarity Example Selector 설정
    example_prompt = PromptTemplate.from_template("Question: {Question}\nAnswer: {Answer}\nSource: {Source}")

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
        fewshot_list[i] = num + retrieved_docs[0].page_content + entry + '\n\n'
    return str(fewshot_list)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def fewshot_rag(llm, fewshot_db,test_retriver,question):
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=fewshot_db,
        k=3,
    )

    # The prompt template will load examples by passing the input do the `select_examples` method
    # example_selector.select_examples({"input": "horse"})
    # Define the few-shot prompt.
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=["input"],
        example_selector=example_selector,
        # Define how each example will be formatted.
        # In this case, each example will become 2 messages:
        # 1 human, and 1 AI
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{Question}"), ("ai", "{Answer}")]
        ),
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You will be my Q&A helper.
Refer to the Context and the example above and output the answer to the question in one sentence.
At this time, there are rules for writing answers. These rules must be followed.
Rule 1: Write the answer by fully considering the Context information.
Rule 2: The answer does not include the "Context" or "Source".
Rule 3: The answer must be written in one sentence as concisely as possible.
Rule 4: In the answer, phrases such as "Answer:" and "Answer:" are excluded.
Rule 5: In addition to the phrases mentioned in Rule 3, phrases unrelated to the answer are excluded.
Rule 6: The answer format follows A is B.
Rule 7: The answer must be in Korean. 
Rule 8: Expected answer length is 1 sentences.

{context}
"""),

            ("human", "\n\n{input}"),
            ("ai", ""),
        ]
    )

    chain = (
                {
                "context": test_retriver | format_docs,
                "input": RunnablePassthrough(),
            }
        | final_prompt 
        | llm
        | StrOutputParser()
        )

    return chain.invoke(question)

def extract_answer(response):
    # AI: 로 시작하는 줄을 찾아 그 이후의 텍스트만 추출
    lines = response.split('\n')
    for line in lines:
        if line.startswith('AI:'):
            return line.replace('AI:', '').strip()
    return response.strip()  # AI: 를 찾지 못한 경우 전체 응답을 정리해서 반환

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
    results =[]
    for i in tqdm(range(len(test_dict))):
        answer = extract_answer(fewshot_rag(llm, fewshot_db, test_retriver, test_dict[i]['Question']))
        results.append({
            "Question": test_dict[i]['Question'],
            "Answer": answer,
            "Source": test_dict[i]['Source']
        
            })
        print("================================================")
        print("Questions: ",results[-1]['Question'])
        print("Answer: ",results[-1]['Answer'])
        #print(results[-1]['Source'])
        
    save(results)
if __name__ == "__main__":
    # EleutherAI/polyglot-ko-1.3b
    #"meta-llama/Meta-Llama-3.1-8B-Instruct"
    # maywell/TinyWand-kiqu
    # yanolja/EEVE-Korean-Instruct-2.8B-v1.0
    # MLP-KTLim/llama-3-Korean-Bllossom-8B
    run(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct")