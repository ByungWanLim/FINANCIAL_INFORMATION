import os
import pandas as pd
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from faiss_module import load_and_vectorize, load_chunks_make_docdb
from utils_module import format_docs, sys_token, user_token, context_token


def fewshot_ex(fewshot_db, buff, train_retriever = None, fewshot_num = 3):
    template ="""<|start_header_id|>user<|end_header_id|>\n{Question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{Answer}<|eot_id|>\n\n"""

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=fewshot_db,
        k=fewshot_num,
        
    )
    ex = example_selector.select_examples({'Question': buff['Question']}) # buff['Question']에 해당하는 fewshot_num개의 문서를 선택 리턴: [{'Question': '질문', 'Answer': '답변'}]
    fewshot_str = ""
    for i , entry in enumerate(ex):
        
        question = entry['Question']
        if train_retriever is not None:
            fewshot_prompt = PromptTemplate.from_template("""<|start_header_id|>system<|end_header_id|>
Given the following contexts about Question:\n{contents}"""+template)
            retrieved_docs = train_retriever.invoke(question)
            ex[i]['contents'] = format_docs(retrieved_docs) # [{'Question': '질문', 'Answer': '답변', 'contents': '문서'}]
        else:
            fewshot_prompt = PromptTemplate.from_template(template) # [{'Question': '질문', 'Answer': '답변'}]
        fewshot_str += fewshot_prompt.format(**ex[i]) 
    return fewshot_str
