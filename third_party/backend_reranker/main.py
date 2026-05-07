import os
import math
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reranker-service")

app = FastAPI(title="Specialized Reranker Service (Qwen3-Reranker-0.6B)")

# Load configurations from environment
MODEL_ID = os.environ.get("RERANKER_MODEL_ID", "Qwen/Qwen3-Reranker-0.6B")
GPU_ID = os.environ.get("RERANKER_GPU_ID", "0")
MAX_MODEL_LEN = int(os.environ.get("RERANKER_MAX_MODEL_LEN", "4096"))
GPU_UTIL = float(os.environ.get("RERANKER_GPU_UTIL", "0.3"))
ATTENTION_BACKEND = os.environ.get("RERANKER_ATTENTION_BACKEND", "FLASHINFER")
DEFAULT_TASK = os.environ.get("RERANKER_DEFAULT_TASK", "Given a search query, retrieve relevant passages that answer the query")
LOCAL_FILES_ONLY = os.environ.get("RERANKER_LOCAL_FILES_ONLY", "True").lower() == "true"

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# Global model and tokenizer
model = None
tokenizer = None
true_token = None
false_token = None
prefix_tokens = None
suffix_tokens = None
sampling_params = None

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: str = DEFAULT_TASK

@app.on_event("startup")
async def load_model():
    global model, tokenizer, true_token, false_token, suffix_tokens, sampling_params
    logger.info(f"Loading Reranker Model: {MODEL_ID} on GPU {GPU_ID}...")
    
    tokenizer_kwargs: Dict[str, Any] = {"local_files_only": LOCAL_FILES_ONLY}
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **tokenizer_kwargs)
    except Exception:
        if not LOCAL_FILES_ONLY:
            raise
        logger.warning(
            "Local-only tokenizer load failed; retrying without local_files_only for %s.",
            MODEL_ID,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = LLM(
        model=MODEL_ID,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        gpu_memory_utilization=GPU_UTIL,
        attention_backend=ATTENTION_BACKEND,
        trust_remote_code=True,
    )
    
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Use the exact prefix/suffix pattern from the Qwen3-Reranker model card
    # (raw string concatenation, NOT apply_chat_template). Going through
    # apply_chat_template + manual suffix produced a near-constant ~0.0001
    # 'no' verdict regardless of relevance — the model collapses because the
    # template inserts unexpected control tokens.
    global prefix_tokens
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=1,
        logprobs=20, 
        allowed_token_ids=[true_token, false_token],
    )
    logger.info(f"Reranker Model Loaded. True token: {true_token}, False token: {false_token}")

def format_instruction(instruction, query, doc):
    """Build the user-content string used between the fixed prefix/suffix."""
    return f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"

@app.post("/v1/rerank")
async def rerank(request: RerankRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.documents:
        return {"scores": []}

    # 1. Process Inputs (1 Query vs N Documents)
    contents = [format_instruction(request.instruction, request.query, doc) for doc in request.documents]
    content_token_lists = [tokenizer.encode(text, add_special_tokens=False) for text in contents]

    # Max length management (Safety): reserve generation tokens and a small margin.
    reserve_tokens = max(1, int(getattr(sampling_params, "max_tokens", 1)))
    safety_margin = 8
    max_content_len = max(
        1,
        MAX_MODEL_LEN - len(prefix_tokens) - len(suffix_tokens) - reserve_tokens - safety_margin,
    )
    truncated_count = 0
    final_inputs = []
    for ele in content_token_lists:
        if len(ele) > max_content_len:
            truncated_count += 1
            ele = ele[:max_content_len]
        full = list(prefix_tokens) + list(ele) + list(suffix_tokens)
        final_inputs.append(TokensPrompt(prompt_token_ids=full))
    if truncated_count:
        logger.warning(
            "Truncated %d/%d rerank prompts to content_max_len=%d (model_max=%d).",
            truncated_count,
            len(contents),
            max_content_len,
            MAX_MODEL_LEN,
        )
    
    # 2. Compute Logits
    try:
        outputs = model.generate(final_inputs, sampling_params, use_tqdm=False)
    except ValueError as e:
        error_text = str(e)
        if "maximum model length" not in error_text.lower():
            raise HTTPException(status_code=500, detail=f"Reranker ValueError: {error_text}") from e

        fallback_max_len = max(1, max_content_len - 128)
        logger.warning(
            "Rerank overflow detected; retrying with stricter truncation (fallback_max_content_len=%d).",
            fallback_max_len,
        )
        fallback_inputs = [
            TokensPrompt(prompt_token_ids=list(prefix_tokens) + list(ele[:fallback_max_len]) + list(suffix_tokens))
            for ele in content_token_lists
        ]
        outputs = model.generate(fallback_inputs, sampling_params, use_tqdm=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranker generation error: {e}") from e

    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        
        # Logit handling following the provided logic
        true_logit = final_logits[true_token].logprob if true_token in final_logits else -10.0
        false_logit = final_logits[false_token].logprob if false_token in final_logits else -10.0
        
        try:
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
        except ZeroDivisionError:
            score = 0.0
            
        scores.append(score)
    
    return {"scores": scores}

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
