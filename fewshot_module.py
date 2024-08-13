import os
import pandas as pd
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from utils_module import format_docs, sys_token, user_token, context_token


def fewshot_ex(fewshot_db, buff, train_db = None, fewshot_num = 3):
    #template ="""<|start_header_id|>user<|end_header_id|>\nQuestion: {Question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{Answer}<|eot_id|>\n\n"""

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=fewshot_db,
        k=fewshot_num,
        
    )
    ex = example_selector.select_examples({'Question': buff['Question']}) # buff['Question']에 해당하는 fewshot_num개의 문서를 선택 리턴: [{'Question': '질문', 'Answer': '답변'}]
    fewshot_str = ""
    for i , entry in enumerate(ex):
        train_retriever = train_db.as_retriever(search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.77,'k':1})
        question = entry['Question']
        retrieved_docs = train_retriever.invoke(question)
        if train_db is not None and len(retrieved_docs) > 0:

            fewshot_prompt = PromptTemplate.from_template("""<|start_header_id|>user<|end_header_id|>\nQuestion\n{Question}\n\nContext\n{contents}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{Answer}<|eot_id|>\n\n""")
            
            ex[i]['contents'] = format_docs(retrieved_docs) # [{'Question': '질문', 'Answer': '답변', 'contents': '문서'}]
        else:
            fewshot_prompt = PromptTemplate.from_template("""<|start_header_id|>user<|end_header_id|>\nQuestion\n{Question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{Answer}<|eot_id|>\n\n""") # [{'Question': '질문', 'Answer': '답변'}]
        fewshot_str += fewshot_prompt.format(**ex[i]) 
    return fewshot_str
