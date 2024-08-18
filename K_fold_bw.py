import torch
import numpy as np
import pandas as pd
from collections import Counter
from faiss_module_bw import load_and_vectorize,load_chunks_make_docdb, make_db, make_fewshot_db
from model_bw2 import setup_llm_pipeline
from fewshot_module import fewshot_ex
from save_module_bw import save
from seed_module import seed_everything
from utils_module import make_dict, extract_answer, format_docs
from run import run
seed_everything(52)
from sklearn.model_selection import KFold

def calculate_f1_score(true_sentence, predicted_sentence, sum_mode=True):

    #공백 제거
    true_sentence = ''.join(true_sentence.split())
    predicted_sentence = ''.join(predicted_sentence.split())
    
    true_counter = Counter(true_sentence)
    predicted_counter = Counter(predicted_sentence)

    #문자가 등장한 개수도 고려
    if sum_mode:
        true_positive = sum((true_counter & predicted_counter).values())
        predicted_positive = sum(predicted_counter.values())
        actual_positive = sum(true_counter.values())

    #문자 자체가 있는 것에 focus를 맞춤
    else:
        true_positive = len((true_counter & predicted_counter).values())
        predicted_positive = len(predicted_counter.values())
        actual_positive = len(true_counter.values())

    #f1 score 계산
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0
    recall = true_positive / actual_positive if actual_positive > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_average_f1_score(true_sentences, predicted_sentences):
    
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    
    for true_sentence, predicted_sentence in zip(true_sentences, predicted_sentences):
        precision, recall, f1_score = calculate_f1_score(true_sentence, predicted_sentence)
        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score
    
    avg_precision = total_precision / len(true_sentences)
    avg_recall = total_recall / len(true_sentences)
    avg_f1_score = total_f1_score / len(true_sentences)
    
    return {
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1_score': avg_f1_score
    }

k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=52)
train_df = pd.read_csv('train.csv')
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# llm = setup_llm_pipeline(model_id)
model_path = "./saved_model"
llm = setup_llm_pipeline(model_path)
fold_result = []

for fold, (train_index, val_index) in enumerate(kf.split(train_df)):
    fold_results = []
    print(f"\nFold {fold + 1}/{k_folds}")
    # 수정된 부분: .iloc[] 사용
    train_set = train_df.iloc[train_index]
    val_set = train_df.iloc[val_index]
    train_db = load_chunks_make_docdb('./train_source', './train_faiss_db')
    
    fewshot_db = make_fewshot_db(train_set)
    train_retriever = train_db.as_retriever(search_kwargs={'k': 1})
    val_retriever = train_db.as_retriever(search_kwargs={'k': 3})
    pred = run(
        train_retriever,
        val_retriever,
        fewshot_db, 
        val_set.to_dict(orient='records'),
        llm,
        varbose=False)
    
    result = pd.DataFrame()
    result['pred'] = [result['Answer'] for result in pred]
    val_set.index = range(len(val_set))
    result['gt'] = val_set['Answer']
        
    result = calculate_average_f1_score(result['gt'], result['pred'])
    print(result)
    fold_result.append(result)
    
    fewshot_db = None
    train_db = None
    val_db = None
    torch.cuda.empty_cache()
    
# 모든 fold의 결과를 집계하고 메트릭 계산
all_results = [result for fold_result in fold_results for result in fold_result]
print(f"Average F1 Score: {np.mean([result['average_f1_score'] for result in all_results])}")
print(f"Average Precision: {np.mean([result['average_precision'] for result in all_results])}")
print(f"Average Recall: {np.mean([result['average_recall'] for result in all_results])}")