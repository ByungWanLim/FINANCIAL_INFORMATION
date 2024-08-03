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

def get_embedding():
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-small',
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
        )
    return embeddings

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


def setup_llm_pipeline(model_id):
    # 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False
    
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
    )
    # 모델 로드 및 양자화 설정 적용
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        #trust_remote_code=True 
        )

    # HuggingFacePipeline 객체 생성
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        #model_kwargs={"torch_dtype": torch.bfloat16},
        do_sample = True,
        temperature=0.6,
        top_p=0.9,
        return_full_text=False,
        # eos_token_id=terminators,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id  # 패딩 토큰을 EOS 토큰 ID로 설정
    )


    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm

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
            ("system", "You are a helpful QA assistant. {context}"),
            few_shot_prompt,
            ("human", "\n\n{input}"),
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
        print(test_dict[i]['Question'])
        
        results.append({
            "Question": test_dict[i]['Question'],
            "Answer": fewshot_rag(llm, fewshot_db,test_retriver,test_dict[i]['Question']),
            "Source": test_dict[i]['Source']
        
            })
        print("================================================")
        print("Questions: ",results[-1]['Question'])
        print("Answer: ",results[-1]['Answer'])
        #print(results[-1]['Source'])
        
    # 제출용 샘플 파일 로드
    submit_df = pd.read_csv("./sample_submission.csv")
    # 생성된 답변을 제출 DataFrame에 추가
    submit_df['Answer'] = [item['Answer'] for item in results]
    submit_df['Answer'] = submit_df['Answer'].fillna("데이콘")     # 모델에서 빈 값 (NaN) 생성 시 채점에 오류가 날 수 있음 [ 주의 ]

    # 결과를 CSV 파일로 저장
    submit_df.to_csv("./baseline_submission.csv", encoding='UTF-8-sig', index=False)
        
if __name__ == "__main__":
    # EleutherAI/polyglot-ko-1.3b
    #"meta-llama/Meta-Llama-3.1-8B-Instruct"
    # maywell/TinyWand-kiqu
    # yanolja/EEVE-Korean-Instruct-2.8B-v1.0
    run(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct")