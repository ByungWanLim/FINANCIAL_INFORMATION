import os
import pandas as pd
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from faiss_module import load_and_vectorize, load_chunks_make_docdb

def make_fewshot_prompt(fewshot_vectordb, k = 3):
    # Semantic Similarity Example Selector 설정
    example_prompt = PromptTemplate.from_template("""
        <|eot_id|><|start_header_id|>user<|end_header_id|>\n{Question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{Answer}""")
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

def fewshot_ex(fewshot_prompt, buff):
    ex_qa = fewshot_prompt.invoke({"input": buff['Question']}).to_string()
    #print(ex_qa)
    fewshot_list = ex_qa.split('\n\n')[:-1]
    
    return '\n\n'.join(fewshot_list)

