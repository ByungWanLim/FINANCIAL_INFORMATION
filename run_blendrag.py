from tqdm import tqdm
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

from model import setup_llm_pipeline
from vectordb_module import FAISSDatabaseManager
from save_module import save
from seed_module import seed_everything
from utils_module import make_dict, extract_answer, format_docs
seed_everything(52)

def fewshot_ex(fewshot_db, buff, train_db=FAISSDatabaseManager, fewshot_num=3, context_num=1):
    results = fewshot_db.search_documents(buff, top_k=fewshot_num)
    fewshot_str = ""
    for i, entry in enumerate(results):
        buff_string = "<|start_header_id|>user<|end_header_id|>\n"
        question = entry['Question']
        buff_string += f"Question\n{question}\n\n"
        if train_db is not None:
            contexts = train_db.search_documents(question, top_k=context_num)
            contexts = format_docs(contexts)
            buff_string += f"Context\n{contexts}\n<|eot_id|>"
        buff_string += f"<|start_header_id|>assistant<|end_header_id>\n{entry['Answer']}<|eot_id|>"
        fewshot_str += buff_string
    return fewshot_str

def run(train_db, test_db, fewshot_db, dataset, llm, verbose=False):
    
    results = []
    for i in tqdm(range(len(dataset))):
        fewshot_str = fewshot_ex(fewshot_db, dataset[i]['Question'], train_db=train_db, fewshot_num=3, context_num=1)
        full_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the financial expert who helps me with my financial information Q&As.
You earn 10 points when you answer me and follow the rules and lose 7 points when you don't.
Here are some rules you should follow.
- Please use contexts to answer the question.
- Please your answers should be concise.
- Please answers must be written in Korean.
- Please answer the question in 1-3 sentences.

- Use the three examples below to learn how to follow the rules and reference information in context.<|eot_id|>
"""
        full_template += f"{fewshot_str}"
        full_template +="\n\n"
        contexts = test_db.search_documents(dataset[i]['Question'], top_k=3)
        contexts = format_docs(contexts)
        full_template += """<|start_header_id|>user<|end_header_id|>\nQuestion\n{input}\n\n"""
        full_template += f"""Context\n{contexts}\n<|eot_id|>"""
        full_template += """<|start_header_id|>assistant<|end_header_id>\n"""

        prompt = PromptTemplate.from_template(full_template)
        qa_chain = (
        {
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        )
        if verbose:
            print("\nQuestion: ", dataset[i]['Question'])
        answer = qa_chain.invoke(input=dataset[i]['Question'])
        answer = extract_answer(answer)
        results.append({
            "Question": dataset[i]['Question'],
            "Answer": answer,
            "Source": dataset[i]['Source']
        })
        if verbose:
            print("Answer: ", results[-1]['Answer'])
    return results


        
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    # EleutherAI/polyglot-ko-1.3b
    #"meta-llama/Meta-Llama-3.1-8B-Instruct"
    # maywell/TinyWand-kiqu
    # yanolja/EEVE-Korean-Instruct-2.8B-v1.0
    # MLP-KTLim/llama-3-Korean-Bllossom-8B
    # train에도 RAG를 쓸 때 사용
    train_df = pd.read_csv('train.csv')
    train_db = FAISSDatabaseManager(db_path='./knn_train', chunk_strategy="recursive", search_strategy="knn_best_field")
    train_db.make_db(train_df)
    
    test_df = pd.read_csv('test.csv')
    test_db = FAISSDatabaseManager(db_path='./knn_test', chunk_strategy="recursive", search_strategy="knn_best_field")
    test_db.make_db(test_df)
    test_dict = make_dict('test.csv')
    
    fewshot_db = FAISSDatabaseManager(db_path='./fewshot_db',chunk_strategy='paragraph',search_strategy="knn_best_field")
    fewshot_db.make_db(train_df,fewshot=True)
    
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = setup_llm_pipeline(model_id)
    
    results = run(train_db= train_db,
        test_db= test_db,
        fewshot_db=fewshot_db, 
        dataset= test_df.to_dict(orient='records') ,
        llm=llm,
        verbose=True)
    save(results)