from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline ,HuggingFaceEmbeddings
import torch

def setup_llm_pipeline(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    # 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              use_default_system_prompt = False
                                              )
    
    # 패딩 토큰 추가
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")

    # terminators 설정
    terminators = tokenizer.eos_token_id

    # 양자화 설정 적용
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 모델 로드 및 설정
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 모델의 패딩 토큰 ID 설정
    model.config.pad_token_id = pad_token_id
    
    # 토크나이저에 추가된 토큰에 맞게 모델의 임베딩 레이어 재초기화
    model.resize_token_embeddings(len(tokenizer))
    # HuggingFacePipeline 객체 생성
     # HuggingFacePipeline 객체 생성
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.6,
        top_p=0.6,
        return_full_text=False,
        eos_token_id=terminators,
        max_new_tokens=512,
        pad_token_id=pad_token_id  # 패딩 토큰 ID 설정
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm