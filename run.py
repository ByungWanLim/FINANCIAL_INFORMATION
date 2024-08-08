from tqdm import tqdm
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from faiss_module import load_and_vectorize,load_chunks_make_docdb
from model import setup_llm_pipeline
from fewshot_module import make_fewshot_prompt, fewshot_ex
from save_module import save
from seed_module import seed_everything
from utils_module import make_dict, extract_answer, format_docs
seed_everything(52)

def run(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    # train에도 RAG를 쓸 때 사용
    # train_db = load_chunks_make_docdb('./train_source', './train_faiss_db')
    # train_retriever = train_db.as_retriever(search_type = "mmr",search_kwargs={'k': 2})
    # train_dict = make_dict('train.csv')
    
    test_db = load_chunks_make_docdb('./test_source', './test_faiss_db')
    test_retriver = test_db.as_retriever(search_kwargs={'k': 3})
    test_dict = make_dict('test.csv')
    
    fewshot_db = load_and_vectorize('train.csv', './fewshot_faiss_db')
    fewshot_prompt = make_fewshot_prompt(fewshot_db, k=5)
    llm = setup_llm_pipeline(model_id)
    # reordering = LongContextReorder()
    results =[]
    for i in tqdm(range(len(test_dict))):
        fewshot_str = ""
        fewshot_str = fewshot_ex(fewshot_prompt, test_dict[i])
        # fewshot_str = ex_with_context(fewshot_str, train_retriever)
        # print(fewshot_str)
        
        full_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
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

Please answer like the example below.
""" +f"{fewshot_str}" + """<|eot_id|><|start_header_id|>system<|end_header_id|>
Given the following contexts about Question:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
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
        print("\nQuestion: ",test_dict[i]['Question'])
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
    # MLP-KTLim/llama-3-Korean-Bllossom-8B
    run(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct")