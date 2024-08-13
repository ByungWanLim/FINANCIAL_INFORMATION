from tqdm import tqdm
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from faiss_module import  make_db, make_fewshot_db
from model import setup_llm_pipeline
from fewshot_module import fewshot_ex
from save_module import save
from seed_module import seed_everything
from utils_module import make_dict, extract_answer, format_docs
seed_everything(52)

def run(train_db,test_db,fewshot_db, dataset ,llm, verbose = False):
    # reordering = LongContextReorder()
    results =[]
    for i in tqdm(range(len(dataset))):
        test_retriver = test_db.as_retriever(search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.77,'k':2})
        # train_retriever가 있으면 context를 포함한 fewshot prompt 생성
        # 없으면 fewshot prompt만 생성
        # fewshot_str = fewshot_ex(fewshot_db, dataset[i],train_db= train_db, fewshot_num = 3)
        #print(fewshot_str)
        full_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the financial expert who helps me with my financial information Q&As.
You earn 10 points when you answer me and follow the rules and lose 7 points when you don't.

12,500 백만원 = 125 억원 = 12,500,000,000 원
5,400 백만원 = 54 억원 = 5,400,000,000 원

Here are some rules you should follow.
- Please use contexts to answer the question.
- Please your answers should be concise.
- Please answers must be written in Korean.
- Please answer the question in 1-3 sentences.

- Use the three examples below to learn how to follow the rules and reference information in context.<|eot_id|>
""" +"""
<|start_header_id|>user<|end_header_id|>
Question: {input}\n\nContext: {context}<|eot_id|>
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
        if verbose:
            print("\nQuestion: ",dataset[i]['Question'])
        answer = qa_chain.invoke(dataset[i]['Question'])
        answer = extract_answer(answer)
        results.append({
            "Question": dataset[i]['Question'],
            "Answer": answer,
            "Source": dataset[i]['Source']
            })
        if verbose:
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
    train_df = pd.read_csv('train.csv')
    train_db = make_db(train_df, './train_faiss_db')

    # train_dict = make_dict('train.csv')

    test_df = pd.read_csv('test.csv')
    test_db = make_db(test_df, './test_faiss_db')

    test_dict = make_dict('test.csv')
    
    fewshot_db = make_fewshot_db(train_df, './fewshot_faiss_db')
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = setup_llm_pipeline(model_id)
    
    results = run(train_db= train_db,
        test_db= test_db,
        fewshot_db=fewshot_db, 
        dataset= test_df.to_dict(orient='records') ,
        llm=llm,
        verbose=True)
    save(results)