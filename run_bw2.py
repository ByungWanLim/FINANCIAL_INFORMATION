from tqdm import tqdm
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from faiss_module import load_and_vectorize,load_chunks_make_docdb
from model_bw import setup_llm_pipeline
from fewshot_module import fewshot_ex
from save_module_bw import save
from seed_module import seed_everything
from utils_module import make_dict, extract_answer, format_docs
seed_everything(52)


def run(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    # train에도 RAG를 쓸 때 사용
    train_db = load_chunks_make_docdb('./train_source', './train_faiss_db')
    train_retriever = train_db.as_retriever(search_kwargs={'k': 1})
    # train_dict = make_dict('train.csv')

    test_db = load_chunks_make_docdb('./test_source', './test_faiss_db')
    test_retriver = test_db.as_retriever(search_kwargs={'k': 3})
    test_dict = make_dict('test.csv')

    fewshot_db = load_and_vectorize('train.csv', './fewshot_faiss_db')

    llm = setup_llm_pipeline(model_id)
    # reordering = LongContextReorder()
    results =[]
    for i in tqdm(range(len(test_dict))):
        # train_retriever가 있으면 context를 포함한 fewshot prompt 생성
        # 없으면 fewshot prompt만 생성
        fewshot_str = fewshot_ex(fewshot_db, test_dict[i],train_retriever= train_retriever, fewshot_num = 3)
        #print(fewshot_str)
        
        full_template = """
    다음 정보를 바탕으로 질문에 답하세요:
    {context}

    질문: {input}
    
    주어진 질문에만 답변하세요. 문장으로 답변해주세요. 답변할 때 질문의 주어를 써주세요.
    답변:
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
    # "rtzr/ko-gemma-2-9b-it"
    run(model_id = "rtzr/ko-gemma-2-9b-it")