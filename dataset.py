import os
import pandas as pd
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from faiss_module import load_and_vectorize, load_chunks_make_docdb

def make_fewshot_prompt(fewshot_vectordb, k = 3):
    # Semantic Similarity Example Selector 설정
    example_prompt = PromptTemplate.from_template("<|start_header_id|>user<|end_header_id|>: {Question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>: <|begin_of_text|>{Answer}<|end_of_text|>")
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
        fewshot_list[i] = num + "context: " + retrieved_docs[0].page_content + entry + '\n\n'
    return str(fewshot_list)

if __name__ == "__main__":
    train_df = pd.read_csv("train.csv")
    #test_df = pd.read_csv("test.csv")
    
    fewshot_db = load_and_vectorize('train.csv', './fewshot_faiss_db')
    fewshot_prompt = make_fewshot_prompt(fewshot_db)
    
    train_db = load_chunks_make_docdb('train.csv', './train_faiss_db')
    train_retriever = train_db.as_retriever(search_type = "mmr",search_kwargs={'k': 3})

    
    train_data = []
    for _, row in train_df.iterrows():
        train_contexts = train_retriever.invoke(row["Question"])
        train_data.append({
            "SAMPLE_ID": row["SAMPLE_ID"],
            "Source_path": row["Source_path"],
            "Question": row["Question"],
            "Answer": row["Answer"],
            "context_0": train_contexts[0].page_content,
            "context_1": train_contexts[1].page_content,
            "context_2": train_contexts[2].page_content})
        # print(f"Question: {question}")
        # print(f"Source: {row["Source_path"]}")
        # print(train_contexts[0].page_content)
        # print(train_contexts[0].metadata)
        # print(train_contexts[1].page_content)
        # print(train_contexts[2].page_content)
    train_data_df = pd.DataFrame(train_data)
    train_data_df.to_csv("train_data.csv", index=False)
    