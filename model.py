from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline ,HuggingFaceEmbeddings
import torch

def setup_llm_pipeline(model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    # 토크나이저 로드 및 설정
        # 양자화 설정 적용
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.4,
        do_sample=True,
        top_p = 0.6,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=512,
        eos_token_id = terminators,
        pad_token_id = tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm


