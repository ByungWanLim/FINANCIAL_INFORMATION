from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from faiss_module import make_db
import pandas as pd
def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    context = ""
    # context = "<|start_header_id|>system<|end_header_id|>\nContext\n"
    for i, doc in enumerate(docs):
        #context += f"Document {i+1}\n"
        context += doc.page_content
        context += '\n\n'
    # context += "<|eot_id|>"
    return context 

def format_docs_wnum(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    context = ""
    # context = "<|start_header_id|>system<|end_header_id|>\nContext\n"
    for i, doc in enumerate(docs):
        context += f"Document {i+1}\n"
        context += doc.page_content
        context += '\n\n'
    # context += "<|eot_id|>"
    return context

train_df = pd.read_csv('train.csv')
train_db = make_db(train_df,'./train_faiss_db')
answer_list = []
context_list = []
context_list_wnum = []
for i, entry in enumerate(train_df.to_dict(orient='records')):
    question = entry['Question']
    answer = entry['Answer']+"<|eot_id|>"
    # print(question)
    # print(answer)
    train_retriever = train_db.as_retriever(search_type="similarity_score_threshold",
                search_kwargs={'filter': {'source':entry['Source_path']},'score_threshold': 0.76,'k':3})
    docs = train_retriever.invoke(question)
    if len(docs) == 0:
        context_list.append("None")
        context_list_wnum.append("None")
    else:
        #print(format_docs(docs))
        context_list.append(format_docs(docs))
        context_list_wnum.append(format_docs_wnum(docs))
    answer_list.append(answer)
    
train_df['Answer'] = answer_list
train_df['context'] = context_list
train_df['context_wnum'] = context_list_wnum

# 프롬프트 생성
def create_prompt(row):
    context = row['context']
    question = row['Question']
    prompt  =f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are a Korean Q&A Assistant.<|eot_id|>
<|start_header_id|>system<|end_header_id|>
{context}
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

train_df['prompt'] = train_df.apply(create_prompt, axis=1)

from sklearn.model_selection import train_test_split

# 데이터셋을 학습셋과 검증셋으로 나누기
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=52)

# train_df는 학습셋, val_df는 검증셋

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import get_peft_model, LoraConfig, TaskType

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)
#만약 GPU에서 amp(Automatic Mixed Precision)를 지원한다면, fp16 대신 bfloat16을 사용하는 것이 더욱 안정적일 수 있습니다.

# 모델 로드 및 양자화 설정 적용
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# LoRA 어댑터 설정 추가
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
# task_type: 적용할 작업의 유형(GPT 계열의 경우 CAUSAL_LM).
# inference_mode: 학습 모드(False) 또는 추론 모드(True).
# r: LoRA의 병목 차원.
# lora_alpha: 스케일링 인자.
# lora_dropout: 드롭아웃 확률.

# 모델에 LoRA 어댑터 적용
model = get_peft_model(model, peft_config)

# 이후 Trainer를 사용해 파인튜닝을 진행


from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset


# 토크나이즈 함수 정의
def tokenize_function(examples):
    return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=256)
# 모델의 정답 라벨 설정
def add_labels(examples):
    labels = tokenizer(examples['Answer'], padding='max_length', truncation=True, max_length=256).input_ids
    examples['labels'] = labels
    return examples


# 예시로, 기존 데이터셋 생성
train_dataset = Dataset.from_pandas(train_df[['prompt', 'Answer']])
val_dataset = Dataset.from_pandas(val_df[['prompt', 'Answer']])

# 토크나이징과 라벨 추가는 동일하게 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.map(add_labels, batched=True)

val_dataset = val_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(add_labels, batched=True)

# DistributedSampler 생성
# train_sampler = DistributedSampler(train_dataset)
# val_sampler = DistributedSampler(val_dataset)

# DataLoader에 샘플러 적용
# train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
# val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=2)

from collections import Counter
import numpy as np
# F1 점수를 계산하는 함수
def calculate_f1_score(true_sentence, predicted_sentence, sum_mode=True):
    true_sentence = ''.join(true_sentence.split())
    predicted_sentence = ''.join(predicted_sentence.split())
    
    true_counter = Counter(true_sentence)
    predicted_counter = Counter(predicted_sentence)
    
    if sum_mode:
        true_positive = sum((true_counter & predicted_counter).values())
        predicted_positive = sum(predicted_counter.values())
        actual_positive = sum(true_counter.values())
    else:
        true_positive = len((true_counter & predicted_counter).values())
        predicted_positive = len(predicted_counter.values())
        actual_positive = len(true_counter.values())

    precision = true_positive / predicted_positive if predicted_positive > 0 else 0
    recall = true_positive / actual_positive if actual_positive > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score
# compute_metrics 함수 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    predicted_sentences = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    true_sentences = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    
    f1_scores = [calculate_f1_score(true, pred)[2] for true, pred in zip(true_sentences, predicted_sentences)]
    avg_f1_score = np.mean(f1_scores)
    
    return {"f1": avg_f1_score}

from transformers import Trainer, TrainingArguments

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir='./results',                  # 결과를 저장할 디렉토리
    evaluation_strategy="epoch",             # 에포크마다 평가
    per_device_train_batch_size=1,           # GPU 하나당 배치 크기 (DataLoader에서 설정한 것과 일치시킴)
    per_device_eval_batch_size=1,            # GPU 하나당 평가 배치 크기 (DataLoader에서 설정한 것과 일치시킴)
    num_train_epochs=3,                      # 학습할 에포크 수
    learning_rate=5e-5,                      # 학습률
    weight_decay=0.01,                       # 가중치 감소
    fp16=True,                               # Mixed Precision 사용 (메모리 절약 및 학습 속도 증가)
    dataloader_pin_memory=True,              # DataLoader가 메모리 핀 설정을 하도록 설정
    dataloader_drop_last=False,              # 배치가 나누어 떨어지지 않을 경우 마지막 배치 드롭 여부
    report_to="none",                        # 로깅이나 기타 리포팅 도구를 사용하지 않음 (필요시 변경)
)

# Trainer 설정
trainer = Trainer(
    model=model,                             # 학습할 모델
    args=training_args,                      # 위에서 설정한 TrainingArguments
    #train_dataloader=train_dataset,       # 커스텀 DataLoader (DistributedSampler 사용)
    #eval_dataloader=val_dataset,          # 커스텀 DataLoader (DistributedSampler 사용)
    train_dataset=train_dataset,            # 학습 데이터셋
    eval_dataset=val_dataset,                # 검증 데이터셋
    tokenizer=tokenizer,                     # 사용 중인 토크나이저
    compute_metrics=compute_metrics          # 평가 메트릭 함수 (선택 사항)
)

# 학습 수행
trainer.train()

# 모델 저장
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
