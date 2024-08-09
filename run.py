from tqdm import tqdm
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from faiss_module import load_and_vectorize,load_chunks_make_docdb
from model import setup_llm_pipeline
from fewshot_module import fewshot_ex
from save_module import save
from seed_module import seed_everything
from utils_module import make_dict, extract_answer, format_docs
seed_everything(52)

def run(train_retriever,test_retriver,fewshot_db, dataset ,llm):
    # reordering = LongContextReorder()
    results =[]
    for i in tqdm(range(len(dataset))):
        # train_retriever가 있으면 context를 포함한 fewshot prompt 생성
        # 없으면 fewshot prompt만 생성
        fewshot_str = fewshot_ex(fewshot_db, dataset[i],train_retriever= train_retriever, fewshot_num = 3)
        #print(fewshot_str)
        full_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Today Date: 8 Aug 2024
1,000,000 원= 100 만원
10 백만원 = 10,000,000 원
100 백만원 = 100,000,000 원
You are the financial expert who helps me with my financial information Q&As.
You earn 10 points when you answer me and follow the rules and lose 7 points when you don't.

Here are some rules you should follow.
- Please use contexts to answer the question.
- Please your answers should be concise.
- Please answers must be written in Korean.
- Please answer the question in 1-3 sentences.

Please answer like the example below.<|eot_id|>
""" +f"{fewshot_str}" + """<|start_header_id|>system<|end_header_id|>
Now Do it for me.
Given the following contexts about Question:
{context}<|start_header_id|>user<|end_header_id|>
{input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n
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
        print("\nQuestion: ",dataset[i]['Question'])
        answer = qa_chain.invoke(dataset[i]['Question'])
        answer = extract_answer(answer)
        results.append({
            "Question": dataset[i]['Question'],
            "Answer": answer,
            "Source": dataset[i]['Source']
            })
        print("Answer: ",results[-1]['Answer'])
        #print(results[-1]['Source'])
    return results
    
    
if __name__ == "__main__":
    # EleutherAI/polyglot-ko-1.3b
    #"meta-llama/Meta-Llama-3.1-8B-Instruct"
    # maywell/TinyWand-kiqu
    # yanolja/EEVE-Korean-Instruct-2.8B-v1.0
    # MLP-KTLim/llama-3-Korean-Bllossom-8B
    
    # train에도 RAG를 쓸 때 사용
    train_db = load_chunks_make_docdb('./train_source', './train_faiss_db')
    train_retriever = train_db.as_retriever(search_kwargs={'k': 1})
    # train_dict = make_dict('train.csv')
    
    test_db = load_chunks_make_docdb('./test_source', './test_faiss_db')
    test_retriver = test_db.as_retriever(search_kwargs={'k': 3})
    test_dict = make_dict('test.csv')
    
    fewshot_db = load_and_vectorize('train.csv', './fewshot_faiss_db')
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = setup_llm_pipeline(model_id)
    
    results = run(train_retriever,
        test_retriver,
        fewshot_db, 
        test_dict ,
        llm)
    save(results)