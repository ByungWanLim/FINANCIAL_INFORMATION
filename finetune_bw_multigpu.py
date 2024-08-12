import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def prepare_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # DistributedSampler에서 셔플을 처리합니다
        sampler=DistributedSampler(dataset, shuffle=shuffle)
    )

def train(rank, world_size):
    setup(rank, world_size)
    
    # 토크나이즈 함수 정의
    def tokenize_function(examples):
        return tokenizer(examples['prompt'], padding='max_length', truncation=True, max_length=512)
    
    # 모델의 정답 라벨 설정
    def add_labels(examples):
        labels = tokenizer(examples['Answer'], padding='max_length', truncation=True, max_length=512).input_ids
        examples['labels'] = labels
        return examples

    # 데이터셋 생성 및 전처리
    train_dataset = Dataset.from_pandas(train_df[['prompt', 'Answer']])
    val_dataset = Dataset.from_pandas(val_df[['prompt', 'Answer']])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.map(add_labels, batched=True)

    val_dataset = val_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(add_labels, batched=True)

    # DataLoader 생성
    train_dataloader = prepare_dataloader(train_dataset, batch_size=1)  # 배치 사이즈를 GPU 수로 나눕니다
    val_dataloader = prepare_dataloader(val_dataset, batch_size=2, shuffle=False)

    # 모델을 GPU로 이동하고 DDP로 래핑
    #model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 학습 루프
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # 검증
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = {k: v.to(rank) for k, v in batch.items()}
                outputs = model(**inputs)
                # 검증 메트릭 계산 등

    cleanup()

if __name__ == "__main__":
    world_size = 1  # 4개의 GPU 사용
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
